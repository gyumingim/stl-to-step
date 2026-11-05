"""build123d 로 정답을 아는 시험 부품을 만들어 STL(거침/고움)로 내보낸다.
+ Fusion 360 식 '엇갈린 고리' 메시(위·아래 고리 꼭짓점 각도·개수가 다름)를 numpy 로 직접 만든다.

출력: tests/out/<name>_<coarse|fine>.stl  +  tests/out/truth.json (참 지름·부피)
"""
import json
import math
import os
import struct

import numpy as np

from build123d import (Align, Axis, Box, Cylinder, Pos, Rot, SlotOverall,
                       extrude, export_stl, fillet)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
TESS = {"coarse": (0.05, 0.5), "fine": (0.001, 0.1)}  # (선형 mm, 각도 rad)


def parts():
    # 1) 판 + 지름 다른 관통 구멍 4개
    plate = Box(80, 40, 5)
    holes = [(3.0, -28, 0), (5.0, -12, 0), (8.0, 6, 0), (12.0, 26, 0)]
    for d, x, y in holes:
        plate -= Pos(x, y, 0) * Cylinder(d / 2, 10)
    yield "plate_holes", plate, [d for d, *_ in holes]

    # 2) 상자 위 원통 보스 (지름 20)
    boss = Box(40, 40, 10) + Pos(0, 0, 5) * Cylinder(10, 12, align=(Align.CENTER, Align.CENTER, Align.MIN))
    yield "box_boss", boss, [20.0]

    # 3) 모서리를 둥글린 상자 (필렛 반지름 4 -> 지름 8)
    b = Box(30, 20, 10)
    b = fillet(b.edges(), 4)
    yield "box_fillet", b, [8.0]

    # 4) 긴 구멍(슬롯) 판: 폭 6 -> 반원 지름 6
    p = Box(50, 20, 4)
    p -= extrude(SlotOverall(30, 6), 10, both=True)
    yield "plate_slot", p, [6.0]

    # 5) 관 (바깥 지름 20, 안 지름 14)
    t = Cylinder(10, 30) - Cylinder(7, 40)
    yield "tube", t, [20.0, 14.0]

    # 6) 세로 필렛만 둥근 상자 (모서리 4개, R3) + 구멍 하나 (지름 4)
    c = Box(30, 20, 8)
    c = fillet(c.edges().filter_by(Axis.Z), 3)
    c -= Cylinder(2, 20)
    yield "box_vfillet_hole", c, [6.0, 4.0]

    # 7) 비스듬한 구멍 (축이 기울어진 원통, 지름 6)
    s = Box(40, 40, 10) - Rot(30, 0, 0) * Cylinder(3, 60)
    yield "plate_tilted_hole", s, [6.0]


def ring(r, L, n, shift):
    """2D 고리 점 + 둘레 매개변수(0..1). L=0: 원(n 마디, shift 마디만큼 돌림).
    L>0: 긴 구멍 모양(반원 2 개, 반원당 n 마디, 끝점은 접점)."""
    pts = []
    if L == 0:
        for i in range(n):
            a = 2 * math.pi * (i + shift) / n
            pts.append((r * math.cos(a), r * math.sin(a)))
    else:
        for cx, a0 in ((L / 2, -math.pi / 2), (-L / 2, math.pi / 2)):
            for i in range(n + 1):
                a = a0 + math.pi * i / n
                pts.append((cx + r * math.cos(a), r * math.sin(a)))
    P = np.array(pts)
    seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
    t = np.r_[0, np.cumsum(seg)[:-1]] / seg.sum()
    if L == 0:
        t = ((np.arctan2(P[:, 1], P[:, 0]) / (2 * math.pi)) % 1.0)
    order = np.argsort(t)
    return P[order], t[order]


def zigzag_prism(fn, r, L, h, n_top, n_bot, shift):
    """엇갈린 고리 기둥: 위 고리 n_top, 아래 고리 n_bot 마디 (+ shift). 옆면은 둘레 매개변수 순서 지그재그.
    뚜껑은 가운데 점 부채꼴(볼록이라 평면 그대로)."""
    Pt, tt = ring(r, L, n_top, 0.0)
    Pb, tb = ring(r, L, n_bot, shift)
    nt, nb = len(Pt), len(Pb)
    V = [(x, y, h) for x, y in Pt] + [(x, y, 0.0) for x, y in Pb] + [(0.0, 0.0, h), (0.0, 0.0, 0.0)]
    ct, cb = nt + nb, nt + nb + 1
    T = [(ct, i, (i + 1) % nt) for i in range(nt)] + [(cb, nt + (i + 1) % nb, nt + i) for i in range(nb)]
    i = j = 0
    while i < nt or j < nb:
        nxt_t = tt[i + 1] if i + 1 < nt else 1.0 + tt[0]
        nxt_b = tb[j + 1] if j + 1 < nb else 1.0 + tb[0]
        if j >= nb or (i < nt and nxt_t <= nxt_b):
            T.append((i % nt, nt + j % nb, (i + 1) % nt))
            i += 1
        else:
            T.append((i % nt, nt + j % nb, nt + (j + 1) % nb))
            j += 1
    V, T = np.array(V), np.array(T)
    tri = V[T]
    if np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:
        T = T[:, ::-1]
    tri = V[T].astype(np.float32)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    with open(fn, "wb") as fh:
        fh.write(b"zigzag test".ljust(80, b" ") + struct.pack("<I", len(T)))
        for k in range(len(T)):
            fh.write(struct.pack("<12fH", *nrm[k], *tri[k].reshape(-1), 0))
    return (math.pi * r * r + 2 * r * L) * h


def main():
    os.makedirs(OUT, exist_ok=True)
    truth = {}
    for name, shape, diams in parts():
        for tag, (lt, at) in TESS.items():
            fn = os.path.join(OUT, f"{name}_{tag}.stl")
            export_stl(shape, fn, tolerance=lt, angular_tolerance=at)
            truth[f"{name}_{tag}"] = dict(diams=diams, volume=shape.volume, stl=fn)
        print(f"{name}: 부피 {shape.volume:.3f} mm3, 지름 {diams}")
    # 엇갈린 고리 (Fusion 식): (이름, 반지름, 직선 길이, 높이, 위 마디, 아래 마디, 돌림)
    for name, r, L, h, nt, nb, sh in (("zz_cyl_half", 10, 0, 20, 24, 24, 0.5),
                                       ("zz_cyl_mixed", 17, 0, 40, 30, 27, 0.3),
                                       ("zz_stadium", 5, 20, 4, 12, 11, 0.0)):
        fn = os.path.join(OUT, f"{name}.stl")
        vol = zigzag_prism(fn, r, L, h, nt, nb, sh)
        truth[name] = dict(diams=[2.0 * r], volume=vol, stl=fn)
        print(f"{name}: 부피 {vol:.3f} mm3, 지름 {[2.0 * r]}")
    with open(os.path.join(OUT, "truth.json"), "w", encoding="utf-8") as fh:
        json.dump(truth, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
