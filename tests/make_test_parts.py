"""build123d 로 정답을 아는 시험 부품을 만들어 STL(거침/고움)로 내보낸다.

출력: tests/out/<name>_<coarse|fine>.stl  +  tests/out/truth.json (참 지름·부피)
"""
import json
import os

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


def main():
    os.makedirs(OUT, exist_ok=True)
    truth = {}
    for name, shape, diams in parts():
        for tag, (lt, at) in TESS.items():
            fn = os.path.join(OUT, f"{name}_{tag}.stl")
            export_stl(shape, fn, tolerance=lt, angular_tolerance=at)
            truth[f"{name}_{tag}"] = dict(diams=diams, volume=shape.volume, stl=fn)
        print(f"{name}: 부피 {shape.volume:.3f} mm3, 지름 {diams}")
    with open(os.path.join(OUT, "truth.json"), "w", encoding="utf-8") as fh:
        json.dump(truth, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
