"""원·원통 복원: STL 메시에서 원통면을 찾아 진짜 원통면 + 원/원호/타원 모서리로 B-rep 을 만든다.

방법 (CAD 가 내보낸 메시 전제):
1. 꼭짓점을 합치고(weld) 같은 평면 삼각형을 묶는다(평면 조각, patch).
2. 원통을 삼각형으로 쪼개면 조각들은 축과 평행한 모선(generator) 모서리로 이어진 '띠'가 된다.
   같은 방향 모선으로 이어진 조각 줄을 축에 수직인 평면에 투영하면 2D 꺾은선이 되고,
   그 꺾은선을 원호/직선으로 나눈다(원 맞춤 잔차 < fit_tol, 마디당 각 <= 50도, 마디 3개 이상).
   -> 원호 = 원통 (반지름·축), 직선 = 평면.
3. 원통 조각과 그 이웃 평면 조각만 OCC 면으로 직접 만든다. 경계는 원통∩평면이면 원/원호/타원,
   모선이면 직선, 그 밖은 꺾은선. 꼭짓점은 만나는 면들 위로 끌어다 붙인다(snap).
   나머지 삼각형은 예전처럼 STL 경로로 읽어 꿰맨 뒤 같은 평면끼리 합친다.
4. 면 검사 실패 원통은 평면으로 되돌리고(최대 3회) 다시 만든다. 최종 형상이 검사 불통과이거나
   부피 차가 너무 크면 None 을 돌려주고 호출 쪽이 예전(삼각형) 변환을 쓴다.
"""
import math
import os
import struct
import tempfile

import numpy as np

from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeSolid,
                                BRepBuilderAPI_MakeVertex, BRepBuilderAPI_Sewing)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.Geom import Geom_Circle, Geom_CylindricalSurface, Geom_Ellipse, Geom_Plane
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Ax2, gp_Ax3, gp_Circ, gp_Dir, gp_Elips, gp_Pln, gp_Pnt
from OCP.ShapeFix import ShapeFix_Face, ShapeFix_Solid
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.StlAPI import StlAPI_Reader
from OCP.TopAbs import TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Face, TopoDS_Shape, TopoDS_Wire

THETA_MAX = math.radians(50.0)   # 원호 한 마디 최대 각 (6각형 60도는 원으로 안 봄)
MIN_CHORDS = 3                   # 원호로 인정할 최소 마디 수
NARROW = 1.5                     # 축 길이 < NARROW x 마디 폭 이면 좁은 원통
ANG_TOL = math.radians(0.1)      # 같은 평면으로 볼 법선 차이


# ---------------------------------------------------------------- STL 읽기
def load_stl(path):
    """(V float64 (n,3), F int64 (m,3)) — 같은 좌표 꼭짓점은 하나로 합침."""
    with open(path, "rb") as fh:
        data = fh.read()
    tris = None
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                                     ("a", "<u2")]), count=n, offset=84)
            tris = rec["v"].astype(np.float64)
    if tris is None:  # ASCII
        txt = data.decode("latin-1").split()
        vals = [txt[i + 1:i + 4] for i, w in enumerate(txt) if w == "vertex"]
        tris = np.array(vals, dtype=np.float64).reshape(-1, 3, 3)
    pts = tris.reshape(-1, 3)
    V, inv = np.unique(pts.astype(np.float32), axis=0, return_inverse=True)
    F = inv.reshape(-1, 3).astype(np.int64)
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    return V.astype(np.float64), F[ok]


def mesh_volume(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _components(n, a, b):
    """간선 (a,b) 로 이어진 0..n-1 의 연결 성분 번호 (numpy union-find)."""
    lab = np.arange(n)
    while True:
        la, lb = lab[a], lab[b]
        m = la != lb
        if not m.any():
            break
        lo, hi = np.minimum(la[m], lb[m]), np.maximum(la[m], lb[m])
        np.minimum.at(lab, hi, lo)
        while True:
            nl = lab[lab]
            if np.array_equal(nl, lab):
                break
            lab = nl
    return np.unique(lab, return_inverse=True)[1]


def _basis(d):
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(d, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(d, e1)


def fit_circle(P):
    """2D 점들에 원 맞춤 -> (center(2), r, max_residual)."""
    x, y = P[:, 0], P[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(P))]
    sol = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
    c = sol[:2]
    r = math.sqrt(max(sol[2] + c @ c, 1e-30))
    for _ in range(5):  # 기하 잔차 Gauss-Newton
        d = P - c
        dist = np.linalg.norm(d, axis=1)
        dist[dist == 0] = 1e-30
        J = np.c_[-d / dist[:, None], -np.ones(len(P))]
        res = dist - r
        step = np.linalg.lstsq(J, -res, rcond=None)[0]
        c = c + step[:2]
        r = r + step[2]
        if np.abs(step).max() < 1e-12:
            break
    res = np.abs(np.linalg.norm(P - c, axis=1) - r)
    return c, abs(r), float(res.max())


# ---------------------------------------------------------------- 메시 분석
class Mesh:
    def __init__(self, V, F, fit_tol):
        self.V, self.F = V, F
        self.diag = float(np.linalg.norm(V.max(0) - V.min(0)))
        self.fit_tol = fit_tol
        self.ptol = max(fit_tol / 10.0, 1e-5 * self.diag)
        cr = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        ar = np.linalg.norm(cr, axis=1)
        self.area = ar / 2
        self.N = cr / np.where(ar > 0, ar, 1)[:, None]
        # 모서리
        E = np.sort(np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1).reshape(-1, 2), 1)
        key = E[:, 0] * (len(V) + 1) + E[:, 1]
        order = np.argsort(key, kind="stable")
        ks = key[order]
        uniq, start, cnt = np.unique(ks, return_index=True, return_counts=True)
        self.manifold = bool((cnt == 2).all())
        self.ok = self.manifold
        if not self.manifold:
            return
        t = order // 3
        self.et = np.stack([t[start], t[start + 1]], 1)        # 모서리 양쪽 삼각형
        self.ev = E[order[start]]                               # 모서리 두 꼭짓점
        # 방향 일관성 (두 삼각형이 모서리를 반대 방향으로 돎)
        def dirsign(tri, a, b):
            f = F[tri]
            ia = np.argmax(f == a[:, None], 1)
            return f[np.arange(len(tri)), (ia + 1) % 3] == b
        s0 = dirsign(self.et[:, 0], self.ev[:, 0], self.ev[:, 1])
        s1 = dirsign(self.et[:, 1], self.ev[:, 0], self.ev[:, 1])
        self.ok = bool((s0 != s1).all())
        self.nedge = len(self.ev)

    # -- 평면 조각
    def patches(self):
        V, F, N = self.V, self.F, self.N
        ta, tb = self.et[:, 0], self.et[:, 1]
        same = np.einsum("ij,ij->i", N[ta], N[tb]) > math.cos(ANG_TOL)
        # b 의 남은 꼭짓점이 a 평면 위인가
        fb = F[tb]
        mask = (fb != self.ev[:, [0]]) & (fb != self.ev[:, [1]])
        ob = fb[np.arange(len(fb)), np.argmax(mask, 1)]
        dist = np.abs(np.einsum("ij,ij->i", V[ob] - V[F[ta, 0]], N[ta]))
        same &= dist < self.ptol
        pid = _components(len(F), ta[same], tb[same])
        # 평면성 검사: 조각별 평면 맞춤 후 최대 거리
        P = pid.max() + 1
        w = self.area
        nsum = np.zeros((P, 3))
        np.add.at(nsum, pid, N * w[:, None])
        nrm = np.linalg.norm(nsum, axis=1)
        pn = nsum / np.where(nrm > 0, nrm, 1)[:, None]
        cen = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3
        csum = np.zeros((P, 3))
        np.add.at(csum, pid, cen * w[:, None])
        asum = np.bincount(pid, w, P)
        pc = csum / np.where(asum > 0, asum, 1)[:, None]
        pd = np.einsum("ij,ij->i", pn, pc)
        dev = np.zeros(P)
        for k in range(3):
            np.maximum.at(dev, pid, np.abs(np.einsum("ij,ij->i", V[F[:, k]], pn[pid]) - pd[pid]))
        bad = dev[pid] > self.ptol
        if bad.any():  # 휘어진 조각은 삼각형 하나씩으로 쪼갬
            pid = pid.copy()
            pid[bad] = P + np.arange(bad.sum())
            pid = np.unique(pid, return_inverse=True)[1]
            P = pid.max() + 1
            nsum = np.zeros((P, 3))
            np.add.at(nsum, pid, N * w[:, None])
            nrm = np.linalg.norm(nsum, axis=1)
            pn = nsum / np.where(nrm > 0, nrm, 1)[:, None]
            csum = np.zeros((P, 3))
            np.add.at(csum, pid, cen * w[:, None])
            asum = np.bincount(pid, w, P)
            pc = csum / np.where(asum > 0, asum, 1)[:, None]
            pd = np.einsum("ij,ij->i", pn, pc)
        self.pid, self.npatch = pid, P
        self.pn, self.pd = pn, pd
        self.ptri_count = np.bincount(pid, minlength=P)
        self.parea = asum
        # 조각별 꼭짓점 목록
        pv = np.unique(np.c_[np.repeat(pid, 3), F.reshape(-1)], axis=0)
        cut = np.searchsorted(pv[:, 0], np.arange(P + 1))
        self.pverts = [pv[cut[i]:cut[i + 1], 1] for i in range(P)]

    # -- 원통 찾기
    def cylinders(self):
        V, pid = self.V, self.pid
        pa, pb = pid[self.et[:, 0]], pid[self.et[:, 1]]
        inter = pa != pb
        ei = np.nonzero(inter)[0]
        if len(ei) == 0:
            return []
        lo, hi = np.minimum(pa[ei], pb[ei]), np.maximum(pa[ei], pb[ei])
        key = lo * (self.npatch + 1) + hi
        order = np.argsort(key, kind="stable")
        ei, key, lo, hi = ei[order], key[order], lo[order], hi[order]
        uk, st = np.unique(key, return_index=True)
        grp = np.searchsorted(uk, key)
        # 묶음(조각쌍)별: 기준 직선 = 첫 모서리
        e0 = ei[st]
        A0, B0 = V[self.ev[e0, 0]], V[self.ev[e0, 1]]
        dvec = B0 - A0
        L = np.linalg.norm(dvec, axis=1)
        dvec = dvec / np.where(L > 0, L, 1)[:, None]
        dev = np.zeros(len(uk))
        for k in range(2):
            P = V[self.ev[ei, k]] - A0[grp]
            perp = P - np.einsum("ij,ij->i", P, dvec[grp])[:, None] * dvec[grp]
            np.maximum.at(dev, grp, np.linalg.norm(perp, axis=1))
        # 조각 사이 모든 모서리 (원통 끝이 매끈하게 이어지는지 볼 때 씀)
        ed = V[self.ev[ei, 1]] - V[self.ev[ei, 0]]
        self.ip_a, self.ip_b = lo, hi
        self.ip_dir = ed / np.linalg.norm(ed, axis=1)[:, None]
        self.ip_ang = np.arccos(np.clip(np.einsum("ij,ij->i", self.pn[lo], self.pn[hi]), -1, 1))
        glo, ghi = lo[st], hi[st]
        cosang = np.clip(np.einsum("ij,ij->i", self.pn[glo], self.pn[ghi]), -1, 1)
        ang = np.arccos(cosang)
        gen = (dev < self.ptol) & (ang > ANG_TOL * 0.5) & (ang <= THETA_MAX + 1e-6) & (L > 0)
        gi = np.nonzero(gen)[0]
        if len(gi) == 0:
            return []
        gdir, gpt = dvec[gi], A0[gi]
        g_lo, g_hi = glo[gi], ghi[gi]
        adj = {}
        for k in range(len(gi)):
            adj.setdefault(int(g_lo[k]), []).append(k)
            adj.setdefault(int(g_hi[k]), []).append(k)
        cosd = math.cos(math.radians(1.0))
        seen = np.zeros(len(gi), bool)
        cyls = []
        for s in range(len(gi)):
            if seen[s]:
                continue
            d0 = gdir[s]
            comp, stack = [], [s]
            seen[s] = True
            while stack:
                g = stack.pop()
                comp.append(g)
                for p in (int(g_lo[g]), int(g_hi[g])):
                    for g2 in adj[p]:
                        if not seen[g2] and abs(gdir[g2] @ d0) > cosd:
                            seen[g2] = True
                            stack.append(g2)
            if len(comp) < MIN_CHORDS - 1:
                continue
            dsum = sum(gdir[g] * np.sign(gdir[g] @ d0) for g in comp)
            d = dsum / np.linalg.norm(dsum)
            cyls += self._fit_component(comp, d, g_lo, g_hi, gpt)
        return self._dedupe(cyls)

    def _paths(self, comp, g_lo, g_hi):
        """조각 그래프(모선 쌍)를 경로/고리로 분해. 각 경로 = (조각들, 쌍들, 닫힘)."""
        nb = {}
        for g in comp:
            a, b = int(g_lo[g]), int(g_hi[g])
            nb.setdefault(a, []).append((g, b))
            nb.setdefault(b, []).append((g, a))
        used, out = set(), []
        def walk(p0, g, p1):
            pats, prs = [p0, p1], [g]
            used.add(g)
            while len(nb[pats[-1]]) == 2:
                nxt = [(g2, q) for g2, q in nb[pats[-1]] if g2 not in used]
                if not nxt:
                    break
                g2, q = nxt[0]
                used.add(g2)
                prs.append(g2)
                pats.append(q)
            return pats, prs
        for p, lst in nb.items():
            if len(lst) != 2:
                for g, q in lst:
                    if g not in used:
                        pats, prs = walk(p, g, q)
                        out.append((pats, prs, False))
        for p, lst in nb.items():
            for g, q in lst:
                if g not in used:
                    pats, prs = walk(p, g, q)
                    closed = pats[-1] == pats[0]
                    if closed:
                        pats = pats[:-1]
                    out.append((pats, prs, closed))
        return out, nb

    def _fit_component(self, comp, d, g_lo, g_hi, gpt):
        e1, e2 = _basis(d)
        to2 = lambda X: np.c_[X @ e1, X @ e2]  # noqa: E731
        res = []
        paths, nb = self._paths(comp, g_lo, g_hi)
        for pats, prs, closed in paths:
            pts = [to2(gpt[g][None])[0] for g in prs]
            chord_p = []   # 마디 i 의 조각
            if closed:
                # 고리: pats[i] 는 prs[i-1] 와 prs[i] 사이
                chord_p = [pats[(i + 1) % len(pats)] for i in range(len(prs))]
                Q = np.array(pts)
            else:
                chord_p = list(pats[1:-1])
                Q = list(pts)
                # 끝 조각이 띠(차수 1)면 먼 쪽 끝점 추가
                for end, pos in ((pats[0], 0), (pats[-1], -1)):
                    if len(nb[end]) == 1:
                        X = to2(self.V[self.pverts[end]])
                        ref = Q[pos]
                        far = X[np.argmax(np.linalg.norm(X - ref, axis=1))]
                        if np.linalg.norm(far - ref) > self.ptol:
                            if pos == 0:
                                Q.insert(0, far)
                                chord_p.insert(0, end)
                            else:
                                Q.append(far)
                                chord_p.append(end)
                Q = np.array(Q)
            arcs = self._segment(Q, closed)
            for (i, j, c, r) in arcs:
                k = len(chord_p)
                patches = [chord_p[m % k] for m in range(i, j)]
                res.append(dict(c2=c, r=r, d=d, e1=e1, e2=e2, patches=set(patches)))
        return res

    def _segment(self, Q, closed):
        """2D 꺾은선을 원호로 나눔 -> [(i, j, center, r)] (마디 i..j-1)."""
        tol = self.fit_tol
        n = len(Q)
        if closed:
            if n >= 8:
                c, r, e = fit_circle(Q)
                if e < tol and self._arc_ok(Q, r, True):
                    return [(0, n, c, r)]
            # 가장 긴 마디에서 시작하도록 돌림
            seg = np.linalg.norm(np.roll(Q, -1, 0) - Q, axis=1)
            s = int(np.argmax(seg))
            Qe = np.vstack([Q[s:], Q[:s], Q[s:s + 1]])
            arcs = self._greedy(Qe)
            return [(i + s, j + s, c, r) for i, j, c, r in arcs]
        return self._greedy(Q)

    def _sag_ok(self, P, r):
        """원호 전체의 활꼴 높이가 맞춤 오차의 3배 이상이어야 곡률이 믿을 만함."""
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        span = float(2 * np.arcsin(np.minimum(seg / (2 * r), 1)).sum())
        sag = r if span >= math.pi else r * (1 - math.cos(span / 2))
        return sag >= 3 * self.fit_tol

    def _arc_ok(self, P, r, closed=False):
        """원 위 점열인지: 마디 각 <= THETA_MAX, 한쪽으로만 돎, 꺾임각 = 마디 길이로 정해지는 값."""
        if r > self.diag:  # 부품보다 큰 원은 믿지 않음
            return False
        Q = np.vstack([P, P[:2]]) if closed else P
        v = np.diff(Q, axis=0)
        seg = np.linalg.norm(v, axis=1)
        if not ((seg > 0).all() and (seg <= 2 * r * math.sin(THETA_MAX / 2) * 1.0001).all()):
            return False
        cr = v[:-1, 0] * v[1:, 1] - v[:-1, 1] * v[1:, 0]
        if not ((cr > 0).all() or (cr < 0).all()):
            return False
        dt = np.einsum("ij,ij->i", v[:-1], v[1:]) / (seg[:-1] * seg[1:])
        turn = np.arctan2(np.abs(cr) / (seg[:-1] * seg[1:]), dt)
        half = np.arcsin(np.minimum(seg / (2 * r), 1.0))
        expect = half[:-1] + half[1:]
        return bool((np.abs(turn - expect) <= np.maximum(math.radians(0.05), 0.1 * expect)).all())

    def _greedy(self, Q):
        tol, n = self.fit_tol, len(Q)
        arcs, i = [], 0
        while i + MIN_CHORDS < n:
            j, best = i + MIN_CHORDS, None
            while j < n:
                P = Q[i:j + 1]
                c, r, e = fit_circle(P)
                if e < tol and self._arc_ok(P, r):
                    best = (i, j, c, r)
                    j += 1
                    continue
                break
            if best and self._sag_ok(Q[best[0]:best[1] + 1], best[3]):
                arcs.append(best)
                i = best[1]
            else:
                i += 1
        # 같은 원 이웃은 합침
        merged = []
        for a in arcs:
            if merged and merged[-1][1] == a[0] and np.linalg.norm(merged[-1][2] - a[2]) < tol * 5                     and abs(merged[-1][3] - a[3]) < tol * 5:
                m = merged[-1]
                merged[-1] = (m[0], a[1], m[2], m[3])
            else:
                merged.append(a)
        # CAD 메셔는 원호를 같은 각도로 나눈다 -> 안쪽 마디 길이가 같아야 함(2 %), 양 끝 마디는 같거나 짧음.
        # 스플라인 윤곽·날개 단면 같은 자유곡선은 마디 길이가 들쭉날쭉해서 여기서 걸러짐.
        seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        out = []
        for a in merged:
            sg = seg[a[0]:a[1]]
            inner = sg[1:-1] if len(sg) > 2 else sg
            if inner.max() > 1.02 * inner.min() or sg.max() > 1.02 * inner.max():
                continue
            out.append(a)
        # 사이에 직선 없이 맞붙은 원호가 서로 다른 원이면 자유곡선으로 보고 버림
        res = []
        for k, a in enumerate(out):
            bad = False
            for b in (out[k - 1] if k else None, out[k + 1] if k + 1 < len(out) else None):
                if b is not None and (b[1] == a[0] or a[1] == b[0]):
                    bad = True  # 같은 원이면 위에서 이미 합쳐졌음
            if not bad:
                res.append(a)
        return res

    def _narrow_smooth(self, c, axp):
        """축 방향으로 좁고(길이 < 마디 폭 x3) 끝이 다른 곡면과 매끈하게 이어지면 가짜 원통
        (구·자유곡면을 띠 하나씩 잘라 본 것)으로 본다."""
        d = c["d"]
        pl = list(c["patches"])
        vs = np.unique(np.concatenate([self.pverts[p] for p in pl]))
        h = (self.V[vs] - axp) @ d
        width = 0.0
        for p in pl:
            X = self.V[self.pverts[p]]
            Y = np.c_[X @ c["e1"], X @ c["e2"]]
            width = max(width, float(np.linalg.norm(Y.max(0) - Y.min(0))))
        if np.ptp(h) >= NARROW * width:
            return False
        ina, inb = np.isin(self.ip_a, pl), np.isin(self.ip_b, pl)
        m = ina ^ inb
        side = m & (np.abs(self.ip_dir @ d) < math.cos(math.radians(5)))             & (self.ip_ang > ANG_TOL) & (self.ip_ang < math.radians(30))
        return bool(side.any())

    def _dedupe(self, cyls):
        """같은 축·반지름 원통 합치고, 3D 로 모든 꼭짓점이 원통 위인지 확인."""
        out = []
        for c in cyls:
            d, e1, e2 = c["d"], c["e1"], c["e2"]
            axp = c["c2"][0] * e1 + c["c2"][1] * e2
            c["p"] = axp
            ok_p, rad_p = set(), set()
            for p in c["patches"]:
                X = self.V[self.pverts[p]] - axp
                rad = np.linalg.norm(X - np.outer(X @ d, d), axis=1)
                if np.abs(rad - c["r"]).max() >= self.fit_tol * 1.5:
                    continue
                # 띠 모양인지: 꼭짓점이 모선 두 줄(2D 두 점) 위에만 있어야 함
                Y = np.c_[X @ e1, X @ e2]
                a0 = Y[np.argmax(np.linalg.norm(Y - Y[0], axis=1))]
                a1 = Y[np.argmax(np.linalg.norm(Y - a0, axis=1))]
                dmin = np.minimum(np.linalg.norm(Y - a0, axis=1), np.linalg.norm(Y - a1, axis=1))
                rad_p.add(p)
                if dmin.max() < self.fit_tol and np.linalg.norm(a1 - a0) > 2 * self.fit_tol:
                    ok_p.add(p)
            # 띠가 충분하면 반지름만 맞는 조각(끝에서 이웃 면 삼각형이 섞인 띠)도 넣음
            if len(ok_p) >= MIN_CHORDS:
                ok_p = rad_p
            if len(ok_p) < MIN_CHORDS:
                continue
            c["patches"] = ok_p
            if self._narrow_smooth(c, axp):
                continue
            for o in out:
                if abs(o["d"] @ d) > math.cos(math.radians(0.5)) and abs(o["r"] - c["r"]) < self.fit_tol:
                    w = o["p"] - axp
                    if np.linalg.norm(w - (w @ d) * d) < self.fit_tol:
                        o["patches"] |= ok_p
                        break
            else:
                out.append(c)
        # 한 조각이 두 원통에 들면 먼저(큰 쪽) 것만
        out.sort(key=lambda c: -len(c["patches"]))
        taken = set()
        for c in out:
            c["patches"] -= taken
            taken |= c["patches"]
        out = [c for c in out if len(c["patches"]) >= MIN_CHORDS]
        return self._drop_sweeps(out)

    def _drop_sweeps(self, cyls):
        """굽은 경로를 따라 쓴 필렛·관(토러스 등)은 짧은 원통 여러 개로 잘려 보인다.
        같은 반지름, 다른 축의 원통과 끝이 매끈하게 붙은 짧은 원통(축 길이 < 2r)은 뺀다."""
        if len(cyls) < 2:
            return cyls
        owner = -np.ones(self.npatch, int)
        for k, c in enumerate(cyls):
            owner[list(c["patches"])] = k
        oa, ob = owner[self.ip_a], owner[self.ip_b]
        m = (oa >= 0) & (ob >= 0) & (oa != ob) & (self.ip_ang < math.radians(30))
        bad = set()
        for a, b in set(zip(oa[m].tolist(), ob[m].tolist())):
            ca, cb = cyls[a], cyls[b]
            if abs(ca["r"] - cb["r"]) > 3 * self.fit_tol or abs(ca["d"] @ cb["d"]) > math.cos(math.radians(0.5)):
                continue
            for k in (a, b):
                c = cyls[k]
                vs = np.unique(np.concatenate([self.pverts[p] for p in c["patches"]]))
                if np.ptp((self.V[vs] - c["p"]) @ c["d"]) < 2 * c["r"]:
                    bad.add(k)
        return [c for k, c in enumerate(cyls) if k not in bad]


# ---------------------------------------------------------------- 면 만들기
def _pnt(x):
    return gp_Pnt(float(x[0]), float(x[1]), float(x[2]))


def _dir(x):
    return gp_Dir(float(x[0]), float(x[1]), float(x[2]))


class Builder:
    def __init__(self, mesh, cyls):
        self.m = mesh
        self.cyls = cyls

    def build(self):
        m = self.m
        F = m.F
        P = m.npatch
        cyl_of_patch = -np.ones(P, int)
        for k, c in enumerate(self.cyls):
            for p in c["patches"]:
                cyl_of_patch[p] = k
        tcyl = cyl_of_patch[m.pid]
        # 원통 영역 = 같은 원통 조각끼리 이어진 덩어리
        ta, tb = m.et[:, 0], m.et[:, 1]
        joint = (tcyl[ta] >= 0) & (tcyl[ta] == tcyl[tb])
        joint |= m.pid[ta] == m.pid[tb]
        comp = _components(len(F), ta[joint], tb[joint])
        # 영역: 원통 영역 + 원통에 닿은 평면 조각
        region = -np.ones(len(F), int)
        self.surf = []  # ('cyl', k) | ('pln', patch)
        cylcomp = np.unique(comp[tcyl >= 0])
        cmap = {}
        for cc in cylcomp:
            cmap[cc] = len(self.surf)
            k = int(tcyl[np.argmax(comp == cc)])
            self.surf.append(("cyl", k))
        sel = tcyl >= 0
        region[sel] = np.array([cmap[c] for c in comp[sel]])
        # 원통 영역과 닿은 평면 조각
        ra, rb = region[ta], region[tb]
        touch = set(m.pid[tb[(ra >= 0) & (rb < 0)]].tolist()) | set(m.pid[ta[(rb >= 0) & (ra < 0)]].tolist())
        pmap = {}
        for p in sorted(touch):
            pmap[p] = len(self.surf)
            self.surf.append(("pln", p))
        selp = (region < 0) & np.isin(m.pid, list(touch))
        region[selp] = np.array([pmap[p] for p in m.pid[selp]])
        self.region = region
        self.nreg = len(self.surf)
        lab = np.where(region >= 0, region, -1 - np.arange(len(F)))
        la, lb = lab[ta], lab[tb]
        bnd = (la != lb) & ((la >= 0) | (lb >= 0))
        self.bedges = np.nonzero(bnd)[0]
        self.lab = lab
        self._snap()
        self._chains()
        faces, bad = self._faces()
        return faces, bad

    # 꼭짓점별 제약 면
    def _surfaces_of_vertex(self, v, tris):
        m = self.m
        out = []
        seen = set()
        for t in tris:
            r = self.region[t]
            if r >= 0:
                key = ("r", r)
                if key not in seen:
                    seen.add(key)
                    kind, k = self.surf[r]
                    out.append(("cyl", self.cyls[k]) if kind == "cyl" else ("pln", (m.pn[k], m.pd[k])))
            else:
                p = m.pid[t]
                if m.ptri_count[p] >= 2 and ("p", p) not in seen:
                    seen.add(("p", p))
                    out.append(("pln", (m.pn[p], m.pd[p])))
        return out

    def _snap(self):
        m = self.m
        bv = np.unique(m.ev[self.bedges].reshape(-1))
        # 꼭짓점 -> 삼각형
        Ff = m.F.reshape(-1)
        order = np.argsort(Ff, kind="stable")
        cut = np.searchsorted(Ff[order], np.arange(len(m.V) + 1))
        self.vtris = lambda v: order[cut[v]:cut[v + 1]] // 3  # noqa: E731
        X = m.V.copy()
        self.maxsnap = 0.0
        for v in bv:
            surfs = self._surfaces_of_vertex(v, self.vtris(v))
            p = X[v].copy()
            for _ in range(60):
                p0 = p.copy()
                for kind, s in surfs:
                    if kind == "pln":
                        n, dd = s
                        p = p - (p @ n - dd) * n
                    else:
                        w = p - s["p"]
                        h = w @ s["d"]
                        rv = w - h * s["d"]
                        nr = np.linalg.norm(rv)
                        if nr > 0:
                            p = s["p"] + h * s["d"] + rv * (s["r"] / nr)
                if np.abs(p - p0).max() < 1e-12:
                    break
            self.maxsnap = max(self.maxsnap, float(np.linalg.norm(p - m.V[v])))
            X[v] = p
        self.X = X

    def _chains(self):
        m = self.m
        ev = m.ev[self.bedges]
        la, lb = self.lab[m.et[self.bedges, 0]], self.lab[m.et[self.bedges, 1]]
        keys = list(zip(np.minimum(la, lb).tolist(), np.maximum(la, lb).tolist()))
        inc = {}
        for i, (a, b) in enumerate(ev.tolist()):
            inc.setdefault(a, []).append(i)
            inc.setdefault(b, []).append(i)
        corner = set()
        for v, lst in inc.items():
            if len(lst) != 2 or keys[lst[0]] != keys[lst[1]]:
                corner.add(v)
        used = np.zeros(len(ev), bool)
        chains, first = [], []

        def walk(v0, e0):
            verts, cur, e = [v0], v0, e0
            first.append(self.bedges[e0])
            while True:
                used[e] = True
                a, b = ev[e]
                nxt = b if a == cur else a
                verts.append(int(nxt))
                if nxt in corner or nxt == v0:
                    break
                cand = [e2 for e2 in inc[nxt] if not used[e2]]
                if not cand:
                    break
                cur, e = nxt, cand[0]
            return verts

        for v in corner:
            for e in inc[v]:
                if not used[e]:
                    chains.append((keys[e], walk(v, e)))
        for e in range(len(ev)):
            if not used[e]:
                v0 = int(ev[e][0])
                corner.add(v0)
                chains.append((keys[e], walk(v0, e)))
        self.chains = [(k, v, e) for (k, v), e in zip(chains, first)]
        self.corner = corner

    # 모서리 기하
    def _vertex(self, v):
        if v not in self.vmap:
            self.vmap[v] = BRepBuilderAPI_MakeVertex(_pnt(self.X[v])).Vertex()
        return self.vmap[v]

    def _surf_of_label(self, lab):
        if lab < 0:
            return None
        kind, k = self.surf[lab]
        return ("cyl", self.cyls[k]) if kind == "cyl" else ("pln", (self.m.pn[k], self.m.pd[k]))

    def _conic_edge(self, verts, P, cyl, pln):
        """원통∩평면 위 점열 -> 원/원호/타원 TopoDS_Edge (안 맞으면 None)."""
        d, r = cyl["d"], cyl["r"]
        tol = max(self.m.fit_tol, 1e-3 * r) * 2
        closed = verts[0] == verts[-1]
        if pln is not None:
            n, dd = pln
            cphi = n @ d
            if abs(cphi) < math.sin(math.radians(0.5)):
                return None
            # 축과 평면의 교점
            tpar = (dd - cyl["p"] @ n) / cphi
            ctr = cyl["p"] + tpar * d
        else:
            h = (P - cyl["p"]) @ d
            if np.ptp(h) > tol:
                return None
            ctr = cyl["p"] + h.mean() * d
            n, cphi = d, 1.0
        if abs(cphi) > math.cos(math.radians(0.5)):
            ax = gp_Ax2(_pnt(ctr), _dir(d), _dir(cyl["e1"]))
            geo = Geom_Circle(gp_Circ(ax, float(r)))
            xa, ya, R1, R2 = cyl["e1"], np.cross(d, cyl["e1"]), r, r
        else:
            nn = n * np.sign(cphi)
            maj = d - (d @ nn) * nn
            maj /= np.linalg.norm(maj)
            R = r / abs(cphi)
            ax = gp_Ax2(_pnt(ctr), _dir(nn), _dir(maj))
            geo = Geom_Ellipse(gp_Elips(ax, float(R), float(r)))
            xa, ya, R1, R2 = maj, np.cross(nn, maj), R, r
        W = P - ctr
        x, y = W @ xa / R1, W @ ya / R2
        if np.abs(np.hypot(x, y) - 1).max() * r > tol:
            return None
        t = np.unwrap(np.arctan2(y, x))
        if closed:
            # 꼭짓점을 원통 이음선(u=0) 위에 둔다 (ShapeFix 가 이음선에서 모서리를 쪼개지 않게)
            g0 = cyl["p"] + r * cyl["e1"]
            sx = ((dd - g0 @ n) / cphi) if pln is not None else float(h.mean())
            q0 = g0 + sx * d
            w0 = q0 - ctr
            t0 = math.atan2(w0 @ ya / R2, w0 @ xa / R1)
            Vx = BRepBuilderAPI_MakeVertex(_pnt(q0)).Vertex()
            mk = BRepBuilderAPI_MakeEdge(geo, Vx, Vx, t0, t0 + 2 * math.pi)
            return (mk.Edge(), bool(t[-1] < t[0])) if mk.IsDone() else None
        t0, t1 = float(t[0]), float(t[-1])
        if abs(t1 - t0) < 1e-9:
            return None
        if t1 > t0:
            mk = BRepBuilderAPI_MakeEdge(geo, self._vertex(verts[0]), self._vertex(verts[-1]), t0, t1)
            rev = False
        else:
            mk = BRepBuilderAPI_MakeEdge(geo, self._vertex(verts[-1]), self._vertex(verts[0]), t1, t0)
            rev = True
        return (mk.Edge(), rev) if mk.IsDone() else None

    def _chain_edges(self, key, verts):
        """체인 -> [(edge, reversed)] (체인 방향 기준)."""
        P = self.X[verts]
        sa, sb = self._surf_of_label(key[0]), self._surf_of_label(key[1])
        kinds = [s for s in (sa, sb) if s is not None]
        cyl = [s[1] for s in kinds if s[0] == "cyl"]
        pln = [s[1] for s in kinds if s[0] == "pln"]
        closed = verts[0] == verts[-1]
        if len(verts) > 2 or closed:
            for c in cyl:
                r = None
                if pln:
                    r = self._conic_edge(verts, P, c, pln[0])
                elif len(cyl) == 1 or True:
                    r = self._conic_edge(verts, P, c, None)
                if r and r[0] is not None:
                    self.n_conic += 1
                    return [r]
        if not closed:
            a, b = P[0], P[-1]
            u = b - a
            Lu = np.linalg.norm(u)
            if Lu > 0:
                w = P - a
                perp = w - np.outer(w @ u / Lu ** 2, u)
                if np.linalg.norm(perp, axis=1).max() < self.m.ptol * 5:
                    mk = BRepBuilderAPI_MakeEdge(self._vertex(verts[0]), self._vertex(verts[-1]))
                    if mk.IsDone():
                        return [(mk.Edge(), False)]
        out = []
        for i in range(len(verts) - 1):
            mk = BRepBuilderAPI_MakeEdge(self._vertex(verts[i]), self._vertex(verts[i + 1]))
            if mk.IsDone():
                out.append((mk.Edge(), False))
        self.n_poly += 1
        return out

    def _faces(self):
        self.vmap = {}
        self.n_conic = self.n_poly = 0
        reg_chains = [[] for _ in range(self.nreg)]
        m = self.m
        for key, verts, e in self.chains:
            edges = self._chain_edges(key, verts)
            for lab in key:
                if lab >= 0:
                    # 이 영역 삼각형이 v0->v1 방향으로 도는가 (바깥 법선 기준 반시계)
                    t = m.et[e, 0] if self.lab[m.et[e, 0]] == lab else m.et[e, 1]
                    f = m.F[t].tolist()
                    i0 = f.index(verts[0])
                    fwd = f[(i0 + 1) % 3] == verts[1]
                    reg_chains[lab].append((verts[0], verts[-1], edges, fwd))
        bld = BRep_Builder()
        faces, bad = [], []
        for r in range(self.nreg):
            kind, k = self.surf[r]
            flip = False
            if kind == "cyl":
                c = self.cyls[k]
                surf = Geom_CylindricalSurface(gp_Ax3(_pnt(c["p"]), _dir(c["d"]), _dir(c["e1"])), float(c["r"]))
                tr = np.nonzero(self.region == r)[0]
                cen = self.m.V[self.m.F[tr]].mean(1) - c["p"]
                rad = cen - np.outer(cen @ c["d"], c["d"])
                flip = float(np.einsum("ij,ij->i", rad, self.m.N[tr]) @ self.m.area[tr]) < 0
            else:
                surf = Geom_Plane(gp_Pln(_pnt(self.m.pn[k] * self.m.pd[k]), _dir(self.m.pn[k])))
            wires = self._wires(reg_chains[r])
            if wires is None:
                bad.append(r)
                continue
            f = TopoDS_Face()
            bld.MakeFace(f, surf, 1e-7)
            for w in wires:
                bld.Add(f, w.Reversed() if flip else w)
            sf = ShapeFix_Face(f)
            sf.SetPrecision(1e-6)
            sf.SetMaxTolerance(max(self.m.fit_tol * 5, 0.05))
            sf.Perform()
            f2 = sf.Face()
            if f2.IsNull() or not BRepCheck_Analyzer(f2).IsValid():
                bad.append(r)
                continue
            faces.append(f2.Reversed() if flip else f2)
        return faces, bad

    def _wires(self, chains):
        """영역 경계 체인 -> 닫힌 와이어들 (바깥 법선 기준 반시계 방향)."""
        bld = BRep_Builder()
        wires, darts = [], []
        for a, b, edges, fwd in chains:
            # 방향 있는 체인: (시작, 끝, 방향 맞춘 모서리들)
            seq = []
            for e, rv in (edges if fwd else edges[::-1]):
                want_rev = rv if fwd else (not rv)
                seq.append(e.Reversed() if want_rev else e)
            s0, s1 = (a, b) if fwd else (b, a)
            if s0 == s1:
                w = TopoDS_Wire()
                bld.MakeWire(w)
                for e in seq:
                    bld.Add(w, e)
                wires.append(w)
            else:
                darts.append((s0, s1, seq))
        out_of = {}
        for i, (s0, _, _) in enumerate(darts):
            out_of.setdefault(s0, []).append(i)
        used = [False] * len(darts)
        for s in range(len(darts)):
            if used[s]:
                continue
            w = TopoDS_Wire()
            bld.MakeWire(w)
            start, i, guard = darts[s][0], s, 0
            while True:
                used[i] = True
                for e in darts[i][2]:
                    bld.Add(w, e)
                cur = darts[i][1]
                if cur == start:
                    break
                nxt = [j for j in out_of.get(cur, []) if not used[j]]
                guard += 1
                if not nxt or guard > len(darts):
                    return None
                i = nxt[0]
            wires.append(w)
        return wires


# ---------------------------------------------------------------- 진입점
def _write_stl(path, X, F):
    tri = X[F].astype(np.float32)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    rec = np.zeros(len(F), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80 + struct.pack("<I", len(F)))
        fh.write(rec.tobytes())


def _shells_to_solids(sewn):
    solids, open_shells = [], 0
    ex = TopExp_Explorer(sewn, TopAbs_SHELL)
    while ex.More():
        shell = TopoDS.Shell_s(ex.Current())  # 면 방향은 메시에서 맞춰 둠 (ShapeFix_Shell 은 큰 메시에서 느림)
        if shell.Closed():
            mk = BRepBuilderAPI_MakeSolid(shell)
            fix = ShapeFix_Solid(mk.Solid())
            fix.Perform()
            solids.append(fix.Solid())
        else:
            open_shells += 1
            solids.append(shell)
        ex.Next()
    return solids, open_shells


def recover(src, tol=0.01, fit_tol=0.01, log=None):
    """원통 복원 시도. 성공하면 (shape, info), 할 일이 없거나 실패면 (None, info)."""
    say = log or (lambda s: None)
    info = dict(cyl=0, diameters=[], reason="")
    V, F = load_stl(src)
    mesh = Mesh(V, F, fit_tol)
    if not mesh.manifold:
        info["reason"] = "닫힌 다양체 메시가 아님"
        return None, info
    if not mesh.ok:
        info["reason"] = "삼각형 방향이 뒤섞임"
        return None, info
    mesh.patches()
    cyls = mesh.cylinders()
    vmesh = mesh_volume(V, F)
    info["vol_mesh"] = vmesh / 1000.0
    info["tri"] = len(F)
    for attempt in range(4):
        if not cyls:
            info["reason"] = info["reason"] or "원통 없음"
            return None, info
        b = Builder(mesh, cyls)
        faces, bad = b.build()
        if not bad:
            break
        drop = set()
        for r in bad:
            kind, k = b.surf[r]
            if kind == "cyl":
                drop.add(k)
            else:  # 평면 면이 깨지면 닿은 원통을 모두 뺌
                tris = np.nonzero(b.region == r)[0]
                ta, tb = mesh.et[:, 0], mesh.et[:, 1]
                nbr = set(b.region[tb[np.isin(ta, tris)]].tolist()) | set(b.region[ta[np.isin(tb, tris)]].tolist())
                for q in nbr:
                    if q >= 0 and b.surf[q][0] == "cyl":
                        drop.add(b.surf[q][1])
        say(f"  원통 {len(drop)}개 면 만들기 실패 -> 평면으로 되돌림")
        info["reason"] = f"원통 {len(drop)}개 되돌림"
        cyls = [c for k, c in enumerate(cyls) if k not in drop]
    else:
        info["reason"] = "면 만들기 반복 실패"
        return None, info
    # 나머지 삼각형은 STL 경로로
    rest = np.nonzero(b.region < 0)[0]
    sew = BRepBuilderAPI_Sewing(tol)
    for f in faces:
        sew.Add(f)
    tmp = None
    if len(rest):
        fd, tmp = tempfile.mkstemp(suffix=".stl")
        os.close(fd)
        _write_stl(tmp, b.X, F[rest])
        raw = TopoDS_Shape()
        if not StlAPI_Reader().Read(raw, tmp):
            os.remove(tmp)
            info["reason"] = "임시 STL 읽기 실패"
            return None, info
        sew.Add(raw)
    sew.Perform()
    if tmp:
        os.remove(tmp)
    solids, open_shells = _shells_to_solids(sew.SewedShape())
    comp = TopoDS_Compound()
    bb = BRep_Builder()
    bb.MakeCompound(comp)
    for s in solids:
        bb.Add(comp, s)
    u = ShapeUpgrade_UnifySameDomain(comp, True, True, False)
    u.Build()
    shape = u.Shape()
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    vb = props.Mass()
    # 허용 부피 차: 원통 면적 x 최대 활꼴 높이 (+0.1%)
    bound = 1e-3 * abs(vmesh)
    for c in cyls:
        tris = np.isin(mesh.pid, list(c["patches"]))
        area = mesh.area[tris].sum()
        bound += area * c["r"] * (1 - math.cos(THETA_MAX / 2))
    info.update(cyl=len(cyls), diameters=sorted({round(float(2 * c["r"]), 3) for c in cyls}),
                vol_brep=vb / 1000.0, vol_err=(vb - vmesh) / abs(vmesh) if vmesh else 0.0,
                open=open_shells, bodies=len(solids), maxsnap=b.maxsnap,
                conic_edges=b.n_conic, poly_chains=b.n_poly)
    valid = BRepCheck_Analyzer(shape).IsValid()
    if open_shells:
        info["reason"] = "복원 후 열린 껍질"
        return None, info
    if not valid:
        info["reason"] = "복원 형상 검사 불통과"
        return None, info
    if abs(vb - vmesh) > bound:
        info["reason"] = f"부피 차 {100 * (vb - vmesh) / vmesh:.2f}% 가 허용 밖"
        return None, info
    info["reason"] = ""
    return shape, info
