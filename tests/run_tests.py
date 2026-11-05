"""시험: 정답 아는 부품 + 실제 부품을 변환하고 STEP 을 다시 읽어 확인한다.

  python tests/make_test_parts.py      # 먼저 시험 STL 만들기
  python tests/run_tests.py [실제 STL ...]
결과: 표 출력 + tests/out/results.json
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from check_step import analyze  # noqa: E402
from stl_to_step import convert  # noqa: E402

OUT = os.path.join(HERE, "out")


def run(stl, truth=None):
    name = os.path.splitext(os.path.basename(stl))[0]
    row = dict(part=name, tri=None)
    for mode, circ in (("old", False), ("new", True)):
        dst = os.path.join(OUT, "step", f"{name}_{mode}.step")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        r = convert(stl, dst, circles=circ)
        a = analyze(dst)
        row["tri"] = r["tri"]
        row[mode] = dict(sec=round(r["sec"], 2), faces=sum(a["kinds"].values()), kinds=a["kinds"],
                         diams=a["diams"], valid=a["valid"], vol=a["vol"],
                         circles=r["circles"], note=r.get("note", ""), vol_err_mesh=r.get("vol_err"))
    if truth:
        row["true_diams"] = truth["diams"]
        row["true_vol"] = truth["volume"]
        for mode in ("old", "new"):
            row[mode]["vol_err_true_pct"] = 100 * (row[mode]["vol"] - truth["volume"]) / truth["volume"]
        got = list(row["new"]["diams"])
        row["diam_err"] = max((min(abs(g - d) for g in got) for d in truth["diams"]), default=None) if got else None
    return row


def main():
    rows = []
    tj = os.path.join(OUT, "truth.json")
    truth = json.load(open(tj, encoding="utf-8")) if os.path.exists(tj) else {}
    for key, t in truth.items():
        rows.append(run(t["stl"], t))
    for f in sys.argv[1:]:
        if os.path.getsize(f) > 5 * 1024 * 1024:
            print("건너뜀(5 MB 초과):", f)
            continue
        rows.append(run(f))
    print(f"{'part':28s} {'tri':>6s} | {'true d':18s} | {'got d (faces)':34s} | cyl | valid | "
          f"Verr new/old % | sec new/old | note")
    for r in rows:
        n, o = r["new"], r["old"]
        got = " ".join(f"{d:g}({c})" for d, c in n["diams"].items())
        td = " ".join(f"{d:g}" for d in r.get("true_diams", [])) or "-"
        if "true_vol" in r:
            ve = f"{n['vol_err_true_pct']:+.3f}/{o['vol_err_true_pct']:+.3f}"
        else:
            ve = f"{100 * (n['vol'] - o['vol']) / o['vol']:+.3f} vs old"
        print(f"{r['part']:28s} {r['tri']:6d} | {td:18s} | {got[:34]:34s} | {n['kinds'].get('cyl', 0):3d} | "
              f"{str(n['valid']):5s} | {ve:14s} | {n['sec']:5.2f}/{o['sec']:5.2f} | "
              f"{n['circles']} {n['note']}")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(created=stamp, rows=rows), fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
