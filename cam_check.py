#!/usr/bin/env python3
"""카메라 밝기 검사: 탑(0)·손목(1)을 한 장씩 찍어 평균 밝기, 어두운/날아간 픽셀 비율을 출력하고 calib/cam_check_{top,wrist}.png 로 저장.
사용: python ~/robotics/cam_check.py            (기본 0,1)
      python ~/robotics/cam_check.py 0 1 --ref  (--ref: 이전 촬영본 calib/cam_ref_*.png 과 비교)
"""
import sys, time, os, glob
import cv2, numpy as np

os.chdir(os.path.expanduser("~/robotics"))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
idx = [int(a) for a in args[:2]] or [0, 1]
names = {idx[0]: "top", idx[1]: "wrist"}


def stats(f):
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return g.mean(), (g < 30).mean() * 100, (g > 240).mean() * 100


for i in idx:
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); cap.set(cv2.CAP_PROP_FPS, 30)
    t0 = time.time(); f = None
    while time.time() - t0 < 3.0:  # 3초 워밍업(자동 노출 안정)
        r, fr = cap.read()
        if r: f = fr
    cap.release()
    if f is None:
        print(f"cam{i} ({names[i]}): 프레임 없음"); continue
    m, dark, sat = stats(f)
    path = f"calib/cam_check_{names[i]}.png"; cv2.imwrite(path, f)
    line = f"cam{i} ({names[i]}): 평균 밝기 {m:.0f}/255, 어두운 픽셀 {dark:.1f}%, 날아간 픽셀 {sat:.1f}%  -> {path}"
    ref = f"calib/cam_ref_{names[i]}.png"
    if os.path.exists(ref):
        rm, _, _ = stats(cv2.imread(ref))
        line += f"  | 기준 {rm:.0f} (차이 {m - rm:+.0f})"
    print(line, flush=True)
print("done")
