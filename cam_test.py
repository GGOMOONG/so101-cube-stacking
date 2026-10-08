#!/usr/bin/env python3
"""카메라 분리 시험: 한 대씩 / 두 대 동시 / 여는 순서 바꿔서 5초씩 읽고 프레임 수를 센다.
사용: python ~/robotics/cam_test.py            (카메라 0,1 기본)
      python ~/robotics/cam_test.py 0 1 8      (인덱스 0,1을 8초씩)
"""
import sys, time
import cv2

idx = [int(a) for a in sys.argv[1:3]] or [0, 1]
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0


def open_cam(i):
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def run(order):
    caps = {}
    t_open = {}
    for i in order:
        t0 = time.time()
        caps[i] = open_cam(i)
        t_open[i] = time.time() - t0
    ok = {i: 0 for i in caps}
    fail = {i: 0 for i in caps}
    first_ok = {i: None for i in caps}
    t0 = time.time()
    while time.time() - t0 < secs:
        for i, c in caps.items():
            r, _ = c.read()
            if r:
                ok[i] += 1
                if first_ok[i] is None:
                    first_ok[i] = round(time.time() - t0, 2)
            else:
                fail[i] += 1
    for c in caps.values():
        c.release()
    parts = []
    for i in order:
        parts.append(
            f"cam{i}: open {t_open[i]:.1f}s, first frame {first_ok[i]}s, ok {ok[i]}, fail {fail[i]}"
        )
    return " | ".join(parts)


tests = [
    ("A 단독", [idx[0]]),
    ("B 단독", [idx[1]]),
    ("C 동시 (0→1 순)", [idx[0], idx[1]]),
    ("D 동시 (1→0 순)", [idx[1], idx[0]]),
]
for name, order in tests:
    print(f"== {name} ==", flush=True)
    print(run(order), flush=True)
    time.sleep(1.0)
print("done")
