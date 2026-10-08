#!/usr/bin/env python3
"""탑 카메라 영상에서 주황·빨강 큐브를 찾아 테이블 mm 좌표로 바꾼다.
사용: python ~/robotics/cube_locate.py 0                 (카메라 0, 스페이스로 측정, q 종료; 창에 실시간 표시)
      python ~/robotics/cube_locate.py ~/robotics/calib/grid.png   (사진 한 장)
필요: ~/robotics/calib/top_intrinsics.npz, ~/robotics/calib/top_homography.npz (rvec/tvec 포함)
원리: 큐브 윗면 중심 픽셀 → 왜곡 보정 → 카메라에서 나가는 광선 → 높이 z=CUBE_H 평면과 만나는 점 = 큐브 윗면 중심의 (x, y).
      (z=0 평면으로 계산하면 카메라에서 멀수록 큐브 높이만큼 바깥으로 밀려 보이는 시차 오차가 생긴다. 둘 다 출력해 비교.)
"""
import sys, os
import numpy as np
import cv2

CUBE_H = 30.0  # mm, 큐브 한 변 (윗면 높이)
# HSV 범위 (조명에 따라 조정). H: 0~180
# 2026-09-23 실측: 주황 H 5~12, 빨강 H 174~180 및 0~1 (가까운 곳은 S 165~185, 먼 곳은 S 110~130까지 떨어짐)
COLORS = {
    "orange": [((4, 90, 90), (22, 255, 255))],
    "red":    [((0, 90, 90), (2, 255, 255)), ((170, 90, 90), (180, 255, 255))],
}
MIN_AREA = 150  # px

c = np.load(os.path.expanduser("~/robotics/calib/top_homography.npz"))
K, dist, H, rvec, tvec = c["K"], c["dist"], c["H"], c["rvec"], c["tvec"]
R, _ = cv2.Rodrigues(rvec)
cam = (-R.T @ tvec).ravel()
Kinv = np.linalg.inv(K)


def pixel_to_plane(u, v, z):
    """왜곡 보정된 픽셀 (u,v) → 광선 → 평면 z와의 교점 (격자 좌표 mm)."""
    ray_cam = Kinv @ np.array([u, v, 1.0])
    ray_w = R.T @ ray_cam
    t = (z - cam[2]) / ray_w[2]
    p = cam + t * ray_w
    return float(p[0]), float(p[1])


def locate(img_bgr):
    if img_bgr.shape[1] != 640 or img_bgr.shape[0] != 480:   # 보정은 640×480 모드 기준. 다른 크기(예: 1280×720)는 비율이 달라 무효.
        raise SystemExit(f"사진 크기 {img_bgr.shape[1]}x{img_bgr.shape[0]}: 보정은 640x480 전용. 카메라 모드로 다시 찍으세요.")
    u = cv2.undistort(img_bgr, K, dist)
    hsv = cv2.cvtColor(u, cv2.COLOR_BGR2HSV)
    out = {}
    vis = u.copy()
    for name, ranges in COLORS.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [cc for cc in cnts if cv2.contourArea(cc) >= MIN_AREA]
        if not cnts:
            out[name] = None
            continue
        cnt = max(cnts, key=cv2.contourArea)
        m = cv2.moments(cnt)
        cx_all, cy_all = m["m10"] / m["m00"], m["m01"] / m["m00"]
        # 윗면만 고르기: 색 영역 안에서 밝은 부분(윗면은 위에서 조명을 받아 옆면보다 밝다). Otsu로 밝기 이분.
        blob = np.zeros_like(mask); cv2.drawContours(blob, [cnt], -1, 255, -1)
        v = hsv[:, :, 2]
        vals = v[blob > 0]
        thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        top = ((v > thr) & (blob > 0)).astype(np.uint8) * 255
        top = cv2.morphologyEx(top, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        tc, _ = cv2.findContours(top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if tc and cv2.contourArea(max(tc, key=cv2.contourArea)) >= 0.3 * cv2.contourArea(cnt):
            tcnt = max(tc, key=cv2.contourArea); mm_ = cv2.moments(tcnt)
            cx, cy = mm_["m10"] / mm_["m00"], mm_["m01"] / mm_["m00"]
            cv2.drawContours(vis, [tcnt], -1, (255, 255, 0), 1)
            ang_cnt = tcnt
        else:
            cx, cy = cx_all, cy_all; ang_cnt = cnt
        # 큐브 방향: 윗면 최소 외접 사각형의 꼭짓점을 격자 평면(z=큐브 높이)으로 옮겨 한 변의 각도(격자 x축 기준, −45~45°)
        box = cv2.boxPoints(cv2.minAreaRect(ang_cnt))
        bg = np.array([pixel_to_plane(float(bx), float(by), CUBE_H) for bx, by in box])
        e = bg[1] - bg[0]; angle_deg = float(((np.degrees(np.arctan2(e[1], e[0])) + 45) % 90) - 45)
        xy_top = pixel_to_plane(cx, cy, CUBE_H)
        xy_flat = cv2.perspectiveTransform(np.float32([[[cx, cy]]]), H)[0, 0]
        xy_all = pixel_to_plane(cx_all, cy_all, CUBE_H)
        out[name] = dict(px=(cx, cy), xy=xy_top, xy_flat=(float(xy_flat[0]), float(xy_flat[1])), xy_all=xy_all, area=cv2.contourArea(cnt), angle_deg=angle_deg)
        col = (0, 140, 255) if name == "orange" else (0, 0, 255)
        cv2.drawContours(vis, [cnt], -1, col, 2)
        cv2.circle(vis, (int(cx), int(cy)), 4, (255, 255, 255), -1)
        cv2.putText(vis, f"{name} ({xy_top[0]:.0f},{xy_top[1]:.0f})mm", (int(cx) + 8, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return out, vis


def report(out):
    for name, r in out.items():
        if r is None:
            print(f"{name}: 못 찾음")
        else:
            print(f"{name}: 윗면 중심 ({r['xy'][0]:.1f}, {r['xy'][1]:.1f}) mm   |  전체 무게중심+높이보정 ({r['xy_all'][0]:.1f}, {r['xy_all'][1]:.1f})   |  평면 가정 ({r['xy_flat'][0]:.1f}, {r['xy_flat'][1]:.1f})   면적 {r['area']:.0f}px")


if __name__ == "__main__":   # import 시에는 실행하지 않음(pick_stack.py가 import함)
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    print(f"카메라 위치(격자 기준): x {cam[0]:.0f}, y {cam[1]:.0f}, 높이 {cam[2]:.0f} mm; 큐브 높이 {CUBE_H} mm")
    if arg.isdigit():
        cap = cv2.VideoCapture(int(arg), cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); cap.set(cv2.CAP_PROP_FPS, 30)
        print("스페이스=측정값 출력, s=사진 저장, q=종료", flush=True)
        n = 0
        while True:
            ok, f = cap.read()
            if not ok:
                continue
            out, vis = locate(f)
            cv2.imshow("cube locate", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(" "):
                report(out)
            elif k == ord("s"):
                import time
                p = os.path.expanduser(time.strftime("~/robotics/calib/cube_%m%d_%H%M%S.png")); cv2.imwrite(p, f); print("saved", p, flush=True)
            elif k == ord("q"):
                break
        cap.release(); cv2.destroyAllWindows()
    else:
        img = cv2.imread(os.path.expanduser(arg))
        out, vis = locate(img)
        report(out)
        p = os.path.expanduser(arg).rsplit(".", 1)[0] + "_located.png"
        cv2.imwrite(p, vis); print("그림:", p)
