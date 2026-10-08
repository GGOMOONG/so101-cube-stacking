#!/usr/bin/env python3
"""한 카메라를 세 가지 프로파일(기본 / 1280x720 / 640x480)로 각각 열어 5초 동안 프레임 수를 센다.
사용: python ~/robotics/cam_test2.py 1
"""
import sys, time
import cv2

i = int(sys.argv[1]) if len(sys.argv) > 1 else 1
secs = 5.0


def trial(name, w=None, h=None):
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if w:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 30)
    ok = fail = 0
    first = None
    t0 = time.time()
    while time.time() - t0 < secs:
        r, f = cap.read()
        if r:
            ok += 1
            if first is None:
                first = round(time.time() - t0, 2)
                print(f"   실제 크기 {f.shape[1]}x{f.shape[0]}", flush=True)
        else:
            fail += 1
    cap.release()
    print(f"{name}: first {first}s, ok {ok}, fail {fail}", flush=True)
    time.sleep(1.0)


print(f"camera {i}")
trial("기본 프로파일")
trial("1280x720", 1280, 720)
trial("640x480", 640, 480)
trial("기본 프로파일(재)")
print("done")
