"""STEP 을 다시 읽어 면 종류(평면/원통/기타)·원통 지름·검사·부피를 센다."""
import sys
from collections import Counter

from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCP.GProp import GProp_GProps
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS


def analyze(path):
    rd = STEPControl_Reader()
    rd.ReadFile(path)
    rd.TransferRoots()
    shp = rd.OneShape()
    kinds, diams = Counter(), Counter()
    ex = TopExp_Explorer(shp, TopAbs_FACE)
    while ex.More():
        s = BRepAdaptor_Surface(TopoDS.Face_s(ex.Current()))
        t = s.GetType()
        if t == GeomAbs_Plane:
            kinds["plane"] += 1
        elif t == GeomAbs_Cylinder:
            kinds["cyl"] += 1
            diams[round(2 * s.Cylinder().Radius(), 3)] += 1
        else:
            kinds["other"] += 1
        ex.Next()
    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(shp, p)
    return dict(kinds=dict(kinds), diams=dict(sorted(diams.items())),
                valid=BRepCheck_Analyzer(shp).IsValid(), vol=p.Mass())


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(f, analyze(f))
