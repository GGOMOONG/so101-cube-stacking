#!/usr/bin/env python3
"""체스판 사진들로 카메라 내부 파라미터(초점거리·주점·왜곡)를 구한다.
사용: python ~/robotics/calib_intrinsics.py            (기본 ~/robotics/calib/chess → ~/robotics/calib/top_intrinsics.npz)
      python ~/robotics/calib_intrinsics.py ~/robotics/calib/chess_wrist ~/robotics/calib/wrist_intrinsics.npz
판정: 재투영 오차(RMS) 0.5 픽셀 이하면 좋음, 1.0 이하면 사용 가능, 그 이상이면 사진을 다시 찍는다.
"""
import sys, os, glob
import numpy as np
import cv2

src = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/robotics/calib/chess")
dst = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/robotics/calib/top_intrinsics.npz")
PATTERN = (6, 9)
SQUARE_MM = 25.0

objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM
obj_pts, img_pts, used = [], [], []
size = None
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
for f in sorted(glob.glob(os.path.join(src, "*.png"))):
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    found, corners = cv2.findChessboardCorners(gray, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH)
    if not found:
        print("skip (not found):", os.path.basename(f))
        continue
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    obj_pts.append(objp)
    img_pts.append(corners)
    used.append(f)
print(f"사용한 사진 {len(used)}장 / 크기 {size}")
if len(used) < 8:
    sys.exit("사진이 8장 미만입니다. 더 찍으세요.")
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
print("재투영 오차 RMS (픽셀):", round(rms, 3))
print("K (내부 행렬):\n", np.round(K, 2))
print("왜곡 계수:", np.round(dist.ravel(), 4))
fx, fy = K[0, 0], K[1, 1]
print(f"초점거리 fx={fx:.1f} fy={fy:.1f} 픽셀, 주점 ({K[0,2]:.1f}, {K[1,2]:.1f})")
# 사진별 오차: 튀는 사진이 있으면 지우고 다시 돌린다
for f, o, i, r, t in zip(used, obj_pts, img_pts, rvecs, tvecs):
    proj, _ = cv2.projectPoints(o, r, t, K, dist)
    e = float(np.sqrt(np.mean(np.sum((i.reshape(-1, 2) - proj.reshape(-1, 2)) ** 2, axis=1))))
    flag = "  <-- 크다" if e > 1.0 else ""
    print(f"  {os.path.basename(f)}: {e:.3f}{flag}")
np.savez(dst, K=K, dist=dist, size=size, rms=rms)
print("저장:", dst)
