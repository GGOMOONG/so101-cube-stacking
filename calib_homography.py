#!/usr/bin/env python3
"""격자(2쪽) 사진 한 장으로 탑 카메라 픽셀 → 테이블 mm 변환(호모그래피)을 구한다.
ArUco 마커(4×4_50, id 0~3, 40 mm)를 자동으로 찾아 16개 모서리를 격자 좌표와 짝짓는다.
사용: python ~/robotics/calib_homography.py 0                      (카메라 0에서 바로 촬영)
      python ~/robotics/calib_homography.py ~/robotics/calib/grid.png (이미 찍은 사진 사용)
출력: ~/robotics/calib/top_homography.npz (H, 내부 보정 적용 여부), 확인용 그림 ~/robotics/calib/top_homography_check.png
판정: 마커 모서리 재투영 오차 평균 1 mm 이하면 좋음, 2 mm 이하면 사용 가능.
"""
import sys, os
import numpy as np
import cv2

arg = sys.argv[1] if len(sys.argv) > 1 else "0"
CAL = os.path.expanduser("~/robotics/calib/top_intrinsics.npz")
OUT = os.path.expanduser("~/robotics/calib/top_homography.npz")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# 마커 위치 (격자 원점 기준 mm, 좌하 모서리), 크기 40 mm. make_calibration_targets.py와 같아야 함.
MS = 40.0
LOWER_LEFT = {0: (20, 20), 1: (120, 20), 2: (20, 180), 3: (120, 180)}

# ---- 사진 얻기 ----
if arg.isdigit():
    cap = cv2.VideoCapture(int(arg), cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); cap.set(cv2.CAP_PROP_FPS, 30)
    print("카메라 창에서 격자가 잘 보이면 스페이스, q는 취소", flush=True)
    img = None
    while True:
        ok, f = cap.read()
        if not ok:
            continue
        cv2.imshow("grid capture", f)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            img = f; break
        if k == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()
    if img is None:
        sys.exit("취소")
    cv2.imwrite(os.path.expanduser("~/robotics/calib/grid.png"), img)
else:
    img = cv2.imread(os.path.expanduser(arg))
    if img is None:
        sys.exit("사진을 열 수 없음: " + arg)

# ---- 내부 보정 적용 (있으면) ----
use_intr = os.path.exists(CAL)
if use_intr:
    c = np.load(CAL)
    K, dist = c["K"], c["dist"]
    img_u = cv2.undistort(img, K, dist)
    print("내부 보정 적용 (RMS %.3f px)" % float(c["rms"]))
else:
    img_u = img
    print("내부 보정 파일 없음 → 왜곡 보정 없이 진행")

# ---- 마커 검출 ----
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
corners, ids, _ = det.detectMarkers(cv2.cvtColor(img_u, cv2.COLOR_BGR2GRAY))
if ids is None or len(ids) < 2:
    sys.exit("마커가 2개 미만 검출됨. 격자가 화면에 다 들어오고 초점이 맞는지 확인.")
ids = ids.flatten()
print("검출된 마커 id:", sorted(ids.tolist()))

# 마커 모서리 순서(OpenCV): 좌상, 우상, 우하, 좌하 (마커 자체 기준). 인쇄물에서는 y가 위로 증가하므로
# 좌하(x,y) 기준으로: 좌상=(x, y+MS), 우상=(x+MS, y+MS), 우하=(x+MS, y), 좌하=(x, y)
src, dst = [], []
for c4, mid in zip(corners, ids):
    if mid not in LOWER_LEFT:
        continue
    x, y = LOWER_LEFT[int(mid)]
    world = [(x, y + MS), (x + MS, y + MS), (x + MS, y), (x, y)]
    for p, w in zip(c4.reshape(4, 2), world):
        src.append(p); dst.append(w)
src, dst = np.float32(src), np.float32(dst)
H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
err = np.linalg.norm(proj - dst, axis=1)
print("마커 모서리 %d개, 재투영 오차 평균 %.2f mm, 최대 %.2f mm" % (len(err), err.mean(), err.max()))
# 픽셀 크기(mm/px) 대략치: 격자 중앙 근처에서 1 px 이동이 몇 mm인가
Hinv = np.linalg.inv(H)
c_px = cv2.perspectiveTransform(np.float32([[[90, 120]]]), Hinv)[0, 0]
p1 = cv2.perspectiveTransform(np.float32([[c_px], [c_px + [1, 0]]]), H).reshape(-1, 2)
print("격자 중앙에서 1 px ≈ %.2f mm" % np.linalg.norm(p1[1] - p1[0]))

# ---- 카메라 자세 (테이블 평면 z=0 기준): 높이가 있는 물체(큐브 윗면)의 시차 보정에 필요 ----
K_use = K if use_intr else np.array([[700, 0, 320], [0, 700, 240], [0, 0, 1]], np.float64)
obj3d = np.hstack([dst, np.zeros((len(dst), 1), np.float32)]).astype(np.float64)
okp, rvec, tvec = cv2.solvePnP(obj3d, src.astype(np.float64), K_use, None, flags=cv2.SOLVEPNP_ITERATIVE)
R, _ = cv2.Rodrigues(rvec)
cam_pos = (-R.T @ tvec).ravel()   # 격자 좌표계에서 본 카메라 위치 (mm)
print("카메라 위치(격자 기준 mm): x %.0f, y %.0f, 높이 %.0f" % tuple(cam_pos))
np.savez(OUT, H=H, use_intrinsics=use_intr, K=K_use, dist=(dist if use_intr else np.zeros(5)),
         rvec=rvec, tvec=tvec, cam_pos=cam_pos)
print("저장:", OUT)

# ---- 확인 그림: 격자선을 mm 좌표에서 픽셀로 되투영 ----
vis = img_u.copy()
for gx in range(0, 181, 20):
    pts = np.float32([[[gx, 0]], [[gx, 240]]]); p = cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2).astype(int)
    cv2.line(vis, tuple(p[0]), tuple(p[1]), (0, 255, 0), 1)
for gy in range(0, 241, 20):
    pts = np.float32([[[0, gy]], [[180, gy]]]); p = cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2).astype(int)
    cv2.line(vis, tuple(p[0]), tuple(p[1]), (0, 255, 0), 1)
cv2.aruco.drawDetectedMarkers(vis, corners, ids.reshape(-1, 1))
o = cv2.perspectiveTransform(np.float32([[[0, 0]]]), Hinv)[0, 0].astype(int)
cv2.circle(vis, tuple(o), 6, (0, 0, 255), -1)
chk = os.path.expanduser("~/robotics/calib/top_homography_check.png")
cv2.imwrite(chk, vis)
print("확인 그림:", chk, "(초록 격자가 인쇄된 격자와 겹치면 정상)")
