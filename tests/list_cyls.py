"""빠른 확인: 파일마다 찾은 원통 지름 목록 (B-rep 안 만듦)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import recover  # noqa: E402

fs = sys.argv[1:]
tj = os.path.join(HERE, "out", "truth.json")
if not fs and os.path.exists(tj):
    fs = [t["stl"] for t in json.load(open(tj, encoding="utf-8")).values()]
for f in fs:
    V, F = recover.load_stl(f)
    m = recover.Mesh(V, F, 0.01)
    m.patches()
    c = m.cylinders()
    print(os.path.basename(f)[:28].ljust(28), sorted(round(float(2 * x["r"]), 3) for x in c))
