#!/usr/bin/env python3
"""손목 카메라 hand-eye 보정: ArUco 격자 종이(4×4_50 id 0~3, 40 mm, 어제까지 쓴 그 종이)를 책상에 고정하고, 토크를 끈 팔을 손으로 여러 자세로 옮기며
손목 카메라 사진 + 관절 읽기값을 모은다. 그 뒤 (1) 손목 카메라 내부 파라미터, (2) 카메라의 gripper_link 기준 자세(hand-eye)를 구해 저장한다.
사용:
  python ~/robotics/wrist_cam_calib.py capture     # 토크 OFF → 팔을 잡고 체스판이 손목 화면에 크게·기울여 보이게 두고 Enter (10~15자세, q 종료)
  python ~/robotics/wrist_cam_calib.py solve       # calib/wrist_calib/*.png + poses.csv → calib/wrist_intrinsics.npz, calib/wrist_cam.json
결과 wrist_cam.json: K, dist, R_cam_in_gripper, t_cam_in_gripper_mm, mjcf(카메라 한 줄, sim_cam.py가 읽음), 재투영 오차.
판정: 재투영 RMS 1 px 이하, hand-eye 잔차(체스판 위치 흩어짐) 5 mm 이하면 좋음.
"""
import os, sys, glob, json, time
import numpy as np, cv2
sys.path.insert(0, os.path.expanduser("~/robotics"))
DIR = os.path.expanduser("~/robotics/calib/wrist_calib"); os.makedirs(DIR, exist_ok=True)
PATTERN, SQUARE_MM = (6, 9), 25.0
LOWER_LEFT = {0: (20.0, 20.0), 1: (120.0, 20.0), 2: (20.0, 180.0), 3: (120.0, 180.0)}; MS = 40.0   # 마커 좌하 모서리(격자 mm), 한 변
_ad = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())


def detect_markers(img_bgr, min_markers=1):
    """검출된 마커 모서리 → (객체점 mm(z=0), 이미지점). 마커가 min_markers 미만이면 None."""
    corners, ids, _ = _ad.detectMarkers(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    if ids is None: return None
    obj, img = [], []
    for c4, mid in zip(corners, ids.flatten()):
        if int(mid) not in LOWER_LEFT: continue
        x, y = LOWER_LEFT[int(mid)]
        for p, w in zip(c4.reshape(4, 2), [(x, y + MS), (x + MS, y + MS), (x + MS, y), (x, y)]): obj.append((w[0], w[1], 0.0)); img.append(p)
    if len(obj) < 4 * min_markers: return None
    return np.float32(obj), np.float32(img), sorted(ids.flatten().tolist())
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
mode = (sys.argv[1] if len(sys.argv) > 1 else "solve") if __name__ == "__main__" else None

if mode == "capture":
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from wrist_view import open_wrist, grab
    robot = SO101Follower(SO101FollowerConfig(port="/dev/tty.usbmodem5B790156431", id="my_follower_arm", cameras={}))
    robot.connect(calibrate=False); robot.bus.disable_torque(); cap = open_wrist()
    print("토크 OFF. ArUco 격자 종이가 손목 화면에 보이게(마커 1개 이상, 2~4개면 더 좋음) 팔을 잡고 Enter. 높이·기울임·손목 회전을 바꿔 가며 10~15장. q 종료.")
    n = len(glob.glob(os.path.join(DIR, "w*.png")))
    while True:
        s = input(f"[{n}장] Enter=촬영, q=종료: ").strip()
        if s == "q": break
        f = grab(cap)
        if f is None: print("프레임 없음"); continue
        det = detect_markers(f); obs = robot.get_observation(); q = [obs[f"{j}.pos"] for j in JOINTS]
        if det is None:
            cv2.imwrite(os.path.join(DIR, "last_fail.png"), f); print("  마커 없음 → 자세 조정 후 다시 (마지막 화면: calib/wrist_calib/last_fail.png)"); continue
        found = True; print(f"  마커 {det[2]}", end="")
        cv2.imwrite(os.path.join(DIR, f"w{n:02d}.png"), f)
        with open(os.path.join(DIR, "poses.csv"), "a") as fh: fh.write(f"w{n:02d}," + ",".join(f"{v:.3f}" for v in q) + "\n")
        print(f"  저장 w{n:02d}: 관절 {np.round(q,1)}"); n += 1
    cap.release(); robot.disconnect(); sys.exit()

# ---- solve ---- (import 시에는 실행되지 않음)
if mode == "solve":
    import mujoco
    from sim_phys import PhysSim, quat_of, _fmt
    poses = {l.split(",")[0]: np.array(l.strip().split(",")[1:], float) for l in open(os.path.join(DIR, "poses.csv"))}
    c0 = np.load(os.path.expanduser("~/robotics/calib/top_intrinsics.npz")); K0, d0 = c0["K"], c0["dist"]     # 손목 렌즈 = 탑과 같은 모듈로 가정(마커만으로는 렌즈 추정이 불안정)
    s = PhysSim({}, kp=25); gl = s.m.body("gripper_link").id; views = []
    for f in sorted(glob.glob(os.path.join(DIR, "w*.png"))):
        nm = os.path.basename(f)[:-4]
        if nm not in poses: continue
        det = detect_markers(cv2.imread(f))
        if det is None: print("skip", nm); continue
        obj, img, ids = det
        ok, rv, tv = cv2.solvePnP(obj.astype(np.float64), img.astype(np.float64), K0, d0, flags=cv2.SOLVEPNP_IPPE if len(obj) == 4 else cv2.SOLVEPNP_ITERATIVE)
        proj, _ = cv2.projectPoints(obj.astype(np.float64), rv, tv, K0, d0); err = float(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1).mean())
        s.set_state(poses[nm]); mujoco.mj_kinematics(s.m, s.d)
        views.append(dict(nm=nm, n=len(ids), Rc=cv2.Rodrigues(rv)[0], tc=tv.ravel(), err=err, Rg=s.d.xmat[gl].reshape(3, 3).copy(), tg=s.d.xpos[gl].copy() * 1000.0))
    print(f"사용 {len(views)}장 (마커 2개 이상 {sum(v['n'] >= 2 for v in views)}장, 회차별 PnP 재투영 평균 {np.mean([v['err'] for v in views]):.2f} px)")
    def handeye(vs, method):
        R, t = cv2.calibrateHandEye([v['Rg'] for v in vs], [v['tg'] for v in vs], [v['Rc'] for v in vs], [v['tc'] for v in vs], method=method); t = t.ravel()
        P = np.array([v['Rg'] @ (R @ v['tc'] + t) + v['tg'] for v in vs]); sp = np.linalg.norm(P - P.mean(0), axis=1); return R, t, sp
    best = None
    for label, sel in [("전체", views), ("마커≥2", [v for v in views if v['n'] >= 2]), ("마커≥2·재투영<2px", [v for v in views if v['n'] >= 2 and v['err'] < 2.0])]:
        if len(sel) < 5: continue
        for mn, m in {"PARK": cv2.CALIB_HAND_EYE_PARK, "HORAUD": cv2.CALIB_HAND_EYE_HORAUD, "TSAI": cv2.CALIB_HAND_EYE_TSAI}.items():
            try: R, t, sp = handeye(sel, m)
            except Exception: continue
            print(f"  {label:14s} {len(sel):2d}장 {mn:7s} 카메라 위치 {np.round(t,1)} 종이 원점 흩어짐 평균 {sp.mean():5.1f} 최대 {sp.max():5.1f} mm")
            if best is None or sp.mean() < best[0]: best = (sp.mean(), sp.max(), label, mn, len(sel), R, t)
    sp_mean, sp_max, label, mn, n, R, t = best
    print(f"선택: {label} / {mn} / {n}장 → 흩어짐 평균 {sp_mean:.1f} 최대 {sp_max:.1f} mm (5 mm 이하면 좋음), 카메라 위치(gripper_link mm) {np.round(t,1)}")
    W, H = 640, 480; R_mj = R @ np.diag([1, -1, -1])
    mjcf = (f'<camera name="wrist" pos="{_fmt(t/1000)}" quat="{_fmt(quat_of(R_mj))}" resolution="{W} {H}" focalpixel="{K0[0,0]:.2f} {K0[1,1]:.2f}" principalpixel="{W/2-K0[0,2]:.2f} {H/2-K0[1,2]:.2f}" sensorsize="0.0036 0.0027"/>')
    json.dump({"K": K0.tolist(), "dist": d0.ravel().tolist(), "intrinsics": "top camera values assumed", "R_cam_in_gripper": R.tolist(), "t_cam_in_gripper_mm": t.tolist(),
               "spread_mm": {"mean": float(sp_mean), "max": float(sp_max)}, "n_views": n, "method": mn, "selection": label, "mjcf": mjcf}, open(os.path.expanduser("~/robotics/calib/wrist_cam.json"), "w"), indent=2)
    np.savez(os.path.expanduser("~/robotics/calib/wrist_intrinsics.npz"), K=K0, dist=d0, size=(W, H), rms=float("nan"))
    print("저장: calib/wrist_cam.json\n", mjcf)
