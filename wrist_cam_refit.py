#!/usr/bin/env python3
"""손목 카메라 hand-eye 재풀이: 카메라 위치(gripper_link)를 자 실측(렌즈–턱 끝 110, 축 수직 65, 축 따라 90 mm)으로 고정하고 회전만 최적화.
ArUco 30장(calib/wrist_calib)의 종이 원점·종이 자세가 모든 뷰에서 일치하도록 R을 찾는다(scipy least_squares).
사용: python wrist_cam_refit.py [--apply]   → 흩어짐 비교 출력, --apply면 calib/wrist_cam.json의 mjcf 갱신(mjcf_handeye는 보존)"""
import os, sys, glob, json, numpy as np, cv2, mujoco
os.chdir(os.path.expanduser("~/robotics")); sys.path.insert(0, ".")
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
import wrist_cam_calib as wc
from sim_phys import PhysSim, quat_of, _fmt
from gripper_geom import GripperGeom

poses = {l.split(",")[0]: np.array(l.strip().split(",")[1:], float) for l in open(os.path.join(wc.DIR, "poses.csv"))}
c0 = np.load("calib/top_intrinsics.npz"); K0, d0 = c0["K"], c0["dist"]
s = PhysSim({}, kp=25); gl = s.m.body("gripper_link").id; views = []
for f in sorted(glob.glob(os.path.join(wc.DIR, "w*.png"))):
    nm = os.path.basename(f)[:-4]
    if nm not in poses: continue
    det = wc.detect_markers(cv2.imread(f))
    if det is None: continue
    obj, img, ids = det
    ok, rv, tv = cv2.solvePnP(obj.astype(np.float64), img.astype(np.float64), K0, d0, flags=cv2.SOLVEPNP_IPPE if len(obj) == 4 else cv2.SOLVEPNP_ITERATIVE)
    s.set_state(poses[nm]); mujoco.mj_kinematics(s.m, s.d)
    views.append(dict(nm=nm, n=len(ids), Rc=cv2.Rodrigues(rv)[0], tc=tv.ravel(), Rg=s.d.xmat[gl].reshape(3, 3).copy(), tg=s.d.xpos[gl].copy() * 1000.0))
print(f"뷰 {len(views)}장")

wj = json.load(open("calib/wrist_cam.json")); R0 = np.array(wj["R_cam_in_gripper"]); t0 = np.array(wj["t_cam_in_gripper_mm"])
gg = GripperGeom(); tip = gg.tip_gl * 1000; u = gg.lever / np.linalg.norm(gg.lever)
v = t0 - tip; n = v - (v @ u) * u; n /= np.linalg.norm(n)
t_meas = tip - 90.0 * u + 65.0 * n                         # 자 실측: 축 따라 90 mm 뒤, 축에서 65 mm(카메라 쪽)
print("hand-eye t", np.round(t0, 1), "→ 실측 고정 t", np.round(t_meas, 1))

def spread(R, t, vs):
    P = np.array([v['Rg'] @ (R @ v['tc'] + t) + v['tg'] for v in vs]); Rp = [v['Rg'] @ R @ v['Rc'] for v in vs]
    zs = np.array([r[:, 2] for r in Rp]); ang = np.degrees(np.arccos(np.clip(zs @ zs.mean(0) / np.linalg.norm(zs.mean(0)), -1, 1)))
    return np.linalg.norm(P - P.mean(0), axis=1), ang
def resid(x, t, vs):
    R = Rot.from_rotvec(x).as_matrix() @ R0
    P = np.array([v['Rg'] @ (R @ v['tc'] + t) + v['tg'] for v in vs]); zs = np.array([(v['Rg'] @ R @ v['Rc'])[:, 2] for v in vs])
    return np.r_[(P - P.mean(0)).ravel(), 50.0 * (zs - zs.mean(0)).ravel()]      # 종이 원점(mm) + 종이 법선(×50 → mm 상당)
sel = [v for v in views if v['n'] >= 2] if sum(v['n'] >= 2 for v in views) >= 6 else views
sp0, an0 = spread(R0, t0, sel); print(f"원래 hand-eye: 원점 흩어짐 평균 {sp0.mean():.1f} 최대 {sp0.max():.1f} mm, 법선 흩어짐 {an0.mean():.1f}°")
sp1, an1 = spread(R0, t_meas, sel); print(f"위치만 실측으로: 평균 {sp1.mean():.1f} 최대 {sp1.max():.1f} mm, 법선 {an1.mean():.1f}°")
res = least_squares(resid, np.zeros(3), args=(t_meas, sel)); R1 = Rot.from_rotvec(res.x).as_matrix() @ R0
sp2, an2 = spread(R1, t_meas, sel); dang = np.degrees(np.linalg.norm(res.x))
print(f"위치 고정+회전 재풀이: 평균 {sp2.mean():.1f} 최대 {sp2.max():.1f} mm, 법선 {an2.mean():.1f}°, 회전 변화 {dang:.1f}°")
if "--apply" in sys.argv:
    R_mj = R1 @ np.diag([1, -1, -1]); W, H = 640, 480
    mjcf = (f'<camera name="wrist" pos="{_fmt(t_meas/1000)}" quat="{_fmt(quat_of(R_mj))}" resolution="{W} {H}" focalpixel="{K0[0,0]:.2f} {K0[1,1]:.2f}" principalpixel="{W/2-K0[0,2]:.2f} {H/2-K0[1,2]:.2f}" sensorsize="0.0036 0.0027"/>')
    wj.setdefault("mjcf_handeye", wj["mjcf"]); wj["mjcf"] = mjcf; wj["R_refit"] = R1.tolist(); wj["t_refit_mm"] = t_meas.tolist()
    wj["refit"] = dict(spread_mean=float(sp2.mean()), spread_max=float(sp2.max()), normal_deg=float(an2.mean()), rot_change_deg=float(dang), note="10-01 자 실측 위치 고정, 회전 재풀이; tilt +3° 미적용")
    json.dump(wj, open("calib/wrist_cam.json", "w"), indent=1); print("wrist_cam.json 갱신:", mjcf)
