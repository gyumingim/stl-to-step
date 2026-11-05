"""STL -> STEP 변환기 (OCP / OpenCascade).

삼각형 메시를 면으로 꿰매(sew) 닫힌 솔리드로 만들고, 같은 평면·같은 원통 위의
삼각형을 큰 면으로 합친 뒤(--no-merge 로 끔) STEP 으로 쓴다.
주의: 곡면은 여전히 삼각형 조각(평면) 모음이다 — NURBS 로 복원하지 않는다.

사용:
  python stl_to_step.py part.stl                 # part.step 생성
  python stl_to_step.py a.stl b.stl -o out_dir/  # 여러 개
  python stl_to_step.py "parts/*.stl" -o step_out/
옵션:
  --tol 0.01     꿰매기 허용 오차 (mm)
  --no-merge     면 합치기 생략 (빠름, 면 수 = 삼각형 수)
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
from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Shape


def count(shape, kind):
    n, ex = 0, TopExp_Explorer(shape, kind)
    while ex.More():
        n += 1
        ex.Next()
    return n


def convert(src, dst, tol=0.01, merge=True):
    t0 = time.time()
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

    Interface_Static.SetCVal_s("write.step.schema", "AP214")
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    if w.Write(dst) != IFSelect_RetDone:
        raise RuntimeError(f"STEP 쓰기 실패: {dst}")

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return dict(
        tri=n_tri, faces=count(shape, TopAbs_FACE), bodies=len(solids),
        open=open_shells, valid=BRepCheck_Analyzer(shape).IsValid(),
        vol=props.Mass() / 1000.0, sec=time.time() - t0,
    )


def main():
    ap = argparse.ArgumentParser(description="STL -> STEP 변환")
    ap.add_argument("inputs", nargs="+", help="STL 파일 (글롭 가능)")
    ap.add_argument("-o", "--out", help="출력 파일(.step) 또는 폴더")
    ap.add_argument("--tol", type=float, default=0.01, help="꿰매기 오차 mm")
    ap.add_argument("--no-merge", action="store_true", help="면 합치기 생략")
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
            r = convert(f, dst, a.tol, not a.no_merge)
            warn = []
            if r["open"]:
                warn.append(f"열린 껍질 {r['open']}개(구멍 있는 메시)")
            if not r["valid"]:
                warn.append("형상 검사 불통과")
            print(f"OK  {f} -> {dst} | 삼각형 {r['tri']} -> 면 {r['faces']} | "
                  f"덩어리 {r['bodies']} | 부피 {r['vol']:.1f} cm3 | {r['sec']:.1f}s"
                  + (" | 경고: " + ", ".join(warn) if warn else ""), flush=True)
        except Exception as e:
            fail += 1
            print(f"FAIL {f}: {e}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
