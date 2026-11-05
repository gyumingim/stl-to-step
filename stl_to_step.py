"""STL -> STEP 변환기 (OCP / OpenCascade).

삼각형 메시를 면으로 꿰매(sew) 닫힌 솔리드로 만들고, 같은 평면 위의 삼각형을
큰 면으로 합친 뒤(--no-merge 로 끔) STEP 으로 쓴다.
원·원통 복원(기본 켬, --no-circles 로 끔): 구멍·보스·필렛·긴 구멍 같은 원통면을 찾아
진짜 원통면과 원/원호 모서리로 바꾼다 (recover.py). 실패하면 예전 방식(삼각형)으로 저장.
그 밖의 곡면(구·원뿔·자유곡면)은 여전히 삼각형 조각이다.

사용:
  python stl_to_step.py part.stl                 # part.step 생성
  python stl_to_step.py a.stl b.stl -o out_dir/  # 여러 개
  python stl_to_step.py "parts/*.stl" -o step_out/
옵션:
  --tol 0.01     꿰매기 허용 오차 (mm)
  --no-merge     면 합치기 생략 (빠름, 면 수 = 삼각형 수)
  --no-circles   원·원통 복원 끔
  --fit-tol 0.01 원 맞춤 허용 오차 (mm)
"""
import argparse
import glob
import os
import sys
import time

from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.ShapeFix import ShapeFix_Solid
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.StlAPI import StlAPI_Reader
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder
from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Shape

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recover  # noqa: E402


def count(shape, kind):
    n, ex = 0, TopExp_Explorer(shape, kind)
    while ex.More():
        n += 1
        ex.Next()
    return n


def cyl_faces(shape):
    n, ex = 0, TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        if BRepAdaptor_Surface(TopoDS.Face_s(ex.Current())).GetType() == GeomAbs_Cylinder:
            n += 1
        ex.Next()
    return n


def convert(src, dst, tol=0.01, merge=True, circles=True, fit_tol=0.01):
    t0 = time.time()
    info = None
    if circles:
        try:
            shape, info = recover.recover(src, tol=tol, fit_tol=fit_tol)
        except Exception as e:  # 복원 쪽 버그도 예전 변환으로 넘어감
            shape, info = None, dict(reason=f"복원 오류: {e}", cyl=0, diameters=[])
        if shape is not None:
            _write_step(shape, dst)
            return dict(
                tri=info["tri"], faces=count(shape, TopAbs_FACE), bodies=info["bodies"],
                open=0, valid=True, vol=info["vol_brep"], vol_mesh=info["vol_mesh"],
                vol_err=info["vol_err"], cyl=cyl_faces(shape), diameters=info["diameters"],
                circles="ok", note="", sec=time.time() - t0)
    r = _faceted(src, dst, tol, merge)
    r.update(cyl=0, diameters=[], sec=time.time() - t0,
             circles="off" if not circles else "fallback",
             note="" if not circles else info.get("reason", ""))
    return r


def _write_step(shape, dst):
    Interface_Static.SetCVal_s("write.step.schema", "AP214")
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    if w.Write(dst) != IFSelect_RetDone:
        raise RuntimeError(f"STEP 쓰기 실패: {dst}")


def _faceted(src, dst, tol, merge):
    """예전 방식: 삼각형을 꿰매고 같은 평면끼리 합침."""
    raw = TopoDS_Shape()
    if not StlAPI_Reader().Read(raw, src):
        raise RuntimeError(f"STL 읽기 실패: {src}")
    n_tri = count(raw, TopAbs_FACE)
    if n_tri > 200_000:
        print(f"  주의: 삼각형 {n_tri:,}개 — 수십 분 걸릴 수 있음 "
              f"(슬라이서/메시 도구로 먼저 줄이기 권장)", flush=True)

    sew = BRepBuilderAPI_Sewing(tol)
    sew.Add(raw)
    sew.Perform()
    sewn = sew.SewedShape()

    # 껍질마다 솔리드로 (여러 덩어리 STL 대응)
    solids, open_shells = [], 0
    ex = TopExp_Explorer(sewn, TopAbs_SHELL)
    while ex.More():
        shell = TopoDS.Shell_s(ex.Current())
        if shell.Closed():
            mk = BRepBuilderAPI_MakeSolid(shell)
            fix = ShapeFix_Solid(mk.Solid())
            fix.Perform()
            solids.append(fix.Solid())
        else:
            open_shells += 1
            solids.append(shell)  # 열린 껍질은 그대로 (솔리드 아님)
        ex.Next()
    if not solids:
        solids = [sewn]

    comp = TopoDS_Compound()
    bld = BRep_Builder()
    bld.MakeCompound(comp)
    for s in solids:
        bld.Add(comp, s)
    shape = comp

    if merge:
        u = ShapeUpgrade_UnifySameDomain(shape, True, True, False)
        u.Build()
        shape = u.Shape()

    _write_step(shape, dst)
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return dict(
        tri=n_tri, faces=count(shape, TopAbs_FACE), bodies=len(solids),
        open=open_shells, valid=BRepCheck_Analyzer(shape).IsValid(),
        vol=props.Mass() / 1000.0,
    )


def main():
    ap = argparse.ArgumentParser(description="STL -> STEP 변환")
    ap.add_argument("inputs", nargs="+", help="STL 파일 (글롭 가능)")
    ap.add_argument("-o", "--out", help="출력 파일(.step) 또는 폴더")
    ap.add_argument("--tol", type=float, default=0.01, help="꿰매기 오차 mm")
    ap.add_argument("--no-merge", action="store_true", help="면 합치기 생략")
    ap.add_argument("--no-circles", action="store_true", help="원·원통 복원 끔")
    ap.add_argument("--fit-tol", type=float, default=0.01, help="원 맞춤 허용 오차 mm")
    a = ap.parse_args()

    files = []
    for p in a.inputs:
        files += sorted(glob.glob(p)) or [p]
    if not files:
        sys.exit("입력 없음")
    single_out = a.out and a.out.lower().endswith((".step", ".stp")) and len(files) == 1
    if a.out and not single_out:
        os.makedirs(a.out, exist_ok=True)

    fail = 0
    for f in files:
        if single_out:
            dst = a.out
        else:
            base = os.path.splitext(os.path.basename(f))[0] + ".step"
            dst = os.path.join(a.out or os.path.dirname(f) or ".", base)
        try:
            r = convert(f, dst, a.tol, not a.no_merge, not a.no_circles, a.fit_tol)
            warn = []
            if r["open"]:
                warn.append(f"열린 껍질 {r['open']}개(구멍 있는 메시)")
            if not r["valid"]:
                warn.append("형상 검사 불통과")
            if r["circles"] == "ok":
                ds = ", ".join(f"{d:g}" for d in r["diameters"][:12])
                circ = (f" | 원통면 {r['cyl']}개 (지름 {ds}{' …' if len(r['diameters']) > 12 else ''}) "
                        f"| 메시 대비 부피 {100 * r['vol_err']:+.2f}%")
            elif r["circles"] == "fallback":
                circ = f" | 원 복원 안 함: {r['note']}"
            else:
                circ = ""
            print(f"OK  {f} -> {dst} | 삼각형 {r['tri']} -> 면 {r['faces']} | "
                  f"덩어리 {r['bodies']} | 부피 {r['vol']:.1f} cm3 | {r['sec']:.1f}s" + circ
                  + (" | 경고: " + ", ".join(warn) if warn else ""), flush=True)
        except Exception as e:
            fail += 1
            print(f"FAIL {f}: {e}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
