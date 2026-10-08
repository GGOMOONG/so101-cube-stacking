#!/usr/bin/env python3
"""손목 카메라 자세 맞춤(결정판): 큐브를 턱 끝에 맞춰 물린 실물 손목 사진 3장(calib/wrist_pose, 큐브 자세가 그리퍼 기준으로 정확히 알려짐)과
시뮬 렌더의 주황 마스크 IoU가 최대가 되도록 카메라의 위치(3)·회전(3)·초점거리 배율(1)을 최적화한다. 모델을 다시 컴파일하지 않고 cam_pos/cam_quat/cam_intrinsic을 직접 바꿔 빠르게 평가.
사용: python wrist_cam_fit_cube.py [--apply]   → 결과 출력·비교 그림 calib/sim/wrist_cube_fit.png, --apply면 wrist_cam.json mjcf 갱신"""
import os, sys, json, re, numpy as np, cv2, mujoco
os.chdir(os.path.expanduser("~/robotics")); sys.path.insert(0, ".")
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
import sim_phys as sp, sim_demo
from sim_cam import distort, top_cam_xml
from gripper_geom import GripperGeom
sp.PhysSim.CUBE_CONTACT = sp.PhysSim.STIFF_CUBE; gg = GripperGeom(); u = gg.lever / np.linalg.norm(gg.lever)
LATERAL_MM = float(os.environ.get("CUBE_LATERAL", "15"))        # 10-02 수정: 처음 맞춤은 큐브를 고정 턱 면 선 위(0 mm)에 놓는 오류가 있었음 → 카메라를 옆으로 9 mm 잘못 옮김
_ax = gg.axis / np.linalg.norm(gg.axis); N_OPEN = np.cross(_ax, u); N_OPEN /= np.linalg.norm(N_OPEN); RP = np.column_stack([u, N_OPEN, _ax])
if np.linalg.det(RP) < 0: RP[:, 2] *= -1
CUBE_IN_HAND_MM = float(os.environ.get("CUBE_DEPTH", "15"))      # 큐브 중심 = 턱 끝에서 축을 따라 15 mm 안쪽(바닥이 턱 끝)

def orange_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); return cv2.inRange(hsv, (9, 120, 80), (22, 255, 255)) > 0
poses = []
for k in (1, 2, 3):
    f = f"calib/wrist_pose/p{k}_wrist.png"
    if not os.path.exists(f): continue
    j = json.load(open(f"calib/wrist_pose/p{k}.json")); poses.append((np.array(j["state"]), orange_mask(cv2.imread(f)), cv2.imread(f)))
print(f"실물 사진 {len(poses)}장")

wj = json.load(open(os.environ.get("CAM_JSON", "calib/wrist_cam.json"))); base = wj[os.environ.get("CAM_KEY", "mjcf")]
s = sp.PhysSim({"orange": (100, 100, 0), "red": (0, 100, 0)}, extra_xml=top_cam_xml(sp.Kin()), gripper_extra_xml=base)
for nm in ("pad_fixed", "pad_moving"): s.m.geom(nm).rgba[3] = 0
cam = s.m.camera("wrist").id; pos0 = s.m.cam_pos[cam].copy(); quat0 = s.m.cam_quat[cam].copy(); intr0 = s.m.cam_intrinsic[cam].copy()
gl = s.m.body("gripper_link").id; ao = s.m.joint("orange_free").qposadr[0]; ar = s.m.joint("red_free").qposadr[0]

def set_cam(x):
    d = x[:3] / 1000.0; rv = np.deg2rad(x[3:6]); fs = x[6]
    R0 = np.empty(9); mujoco.mju_quat2Mat(R0, quat0); R0 = R0.reshape(3, 3)
    s.m.cam_pos[cam] = pos0 + R0 @ d                                   # 카메라 자체 좌표계에서 이동(mm)
    Rn = R0 @ Rot.from_rotvec(rv).as_matrix(); q = np.empty(4); mujoco.mju_mat2Quat(q, Rn.flatten()); s.m.cam_quat[cam] = q
    s.m.cam_intrinsic[cam] = intr0 * np.array([fs, fs, 1, 1])
def render(st):
    s.set_state(st); s.set_ctrl(st); mujoco.mj_forward(s.m, s.d)
    Rg = s.d.xmat[gl].reshape(3, 3); pg = s.d.xpos[gl]
    c_w = Rg @ (gg.tip_gl - CUBE_IN_HAND_MM / 1000.0 * u + LATERAL_MM / 1000.0 * N_OPEN) + pg          # 큐브는 고정 턱 안쪽 면에 붙어 있음 → 중심은 열림 방향으로 반 변(15 mm)
    Rc = Rg @ RP; q = np.empty(4); mujoco.mju_mat2Quat(q, Rc.flatten()); s.d.qpos[ao:ao+3] = c_w; s.d.qpos[ao+3:ao+7] = q; s.d.qpos[ar+2] = -1.0   # 큐브 면을 턱 면에 맞춤
    mujoco.mj_forward(s.m, s.d); return cv2.cvtColor(distort(s.render_cam("wrist")), cv2.COLOR_RGB2BGR)
def cost(x):
    set_cam(x); tot = 0.0
    for st, m_real, _ in poses:
        m_sim = orange_mask(render(st)); inter = np.logical_and(m_real, m_sim).sum(); union = np.logical_or(m_real, m_sim).sum()
        tot += 1.0 - inter / max(union, 1)
    return tot / len(poses)

FS = float(os.environ.get("FOCAL_SCALE", "0.93"))                # ArUco 30장으로 추정한 손목 초점거리 ≈ 650 px (탑 가정값 703의 0.93); 거리와 겹치므로 고정
x0 = np.array([0, 0, 0, 0, 0, 0, FS]); c0 = cost(x0); print(f"현재 카메라(초점 ×{FS}): 평균 IoU {1-c0:.3f}")
if os.environ.get('EVAL_ONLY'): sys.exit()
def cost6(y): return cost(np.r_[y, FS])
best = (c0, x0)
for start in [np.zeros(6), np.array([0, 0, 15, 0, 6, 0.]), np.array([10, 0, 15, 0, -6, 0.]), np.array([-10, 5, 20, 5, 6, 0.]), np.array([-10, -5, 10, -5, 0, 5.])]:
    if os.environ.get("ROT_ONLY"):      # 위치는 자 실측(110/65/90 mm)에 고정, 회전 3자유도만
        start = start.copy(); start[:3] = 0
        f3 = lambda r: cost6(np.r_[0, 0, 0, r]); s3 = np.vstack([start[3:]] + [start[3:] + np.eye(3)[i] * 4.0 for i in range(3)])
        r = minimize(f3, start[3:], method="Nelder-Mead", options=dict(xatol=0.2, fatol=1e-4, maxiter=400, initial_simplex=s3))
        if r.fun < best[0]: best = (r.fun, np.r_[0, 0, 0, r.x, FS])
        print(f"  회전만, 시작 {np.round(start[3:],1)} → IoU {1-r.fun:.3f} at {np.round(r.x,1)}"); continue
    simplex = np.vstack([start] + [start + np.eye(6)[i] * (8.0 if i < 3 else 4.0) for i in range(6)])
    r = minimize(cost6, start, method="Nelder-Mead", options=dict(xatol=0.3, fatol=1e-4, maxiter=800, initial_simplex=simplex))
    if r.fun < best[0]: best = (r.fun, np.r_[r.x, FS])
    print(f"  시작 {np.round(start,1)} → IoU {1-r.fun:.3f} at {np.round(r.x,1)}")
c, x = best; print(f"최적: 이동(카메라 좌표 mm) {np.round(x[:3],1)}, 회전(°) {np.round(x[3:6],1)}, 초점 배율 {x[6]:.3f} → 평균 IoU {1-c:.3f} (전 {1-c0:.3f})")
set_cam(x); rows = []
for st, m_real, real in poses:
    sim = render(st); ov = real.copy(); e = cv2.Canny(orange_mask(sim).astype(np.uint8) * 255, 50, 150); ov[e > 0] = (255, 0, 0)
    rows.append(np.hstack([real, sim, ov]))
OUTP = os.environ.get("OUT_PNG", "calib/sim/wrist_cube_fit.png"); cv2.imwrite(OUTP, np.vstack(rows)); print("그림", OUTP)
if "--apply" in sys.argv:
    q = s.m.cam_quat[cam]; p = s.m.cam_pos[cam]; fx, fy = s.m.cam_intrinsic[cam][:2]
    mjcf = re.sub(r'pos="[^"]+"', f'pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"', base, count=1); mjcf = re.sub(r'quat="[^"]+"', f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"', mjcf, count=1)
    mjcf = re.sub(r'focalpixel="[^"]+"', f'focalpixel="{fx/0.0036*640/640:.2f} {fy/0.0027*480/480:.2f}"', mjcf, count=1) if False else re.sub(r'focalpixel="([0-9.]+) ([0-9.]+)"', lambda m: f'focalpixel="{float(m.group(1))*x[6]:.2f} {float(m.group(2))*x[6]:.2f}"', mjcf, count=1)
    wj.setdefault("mjcf_handeye", base); wj["mjcf_before_cube_fit"] = base; wj["mjcf"] = mjcf
    wj["cube_fit"] = dict(shift_cam_mm=x[:3].tolist(), rot_deg=x[3:6].tolist(), focal_scale=float(x[6]), iou_after=float(1-c), iou_before=float(1-c0), cube_depth_mm=CUBE_IN_HAND_MM, n=len(poses))
    OUTJ = os.environ.get("OUT_JSON", "calib/wrist_cam_v4.json"); wj["cube_fit"]["cube_lateral_mm"] = LATERAL_MM; json.dump(wj, open(OUTJ, "w"), indent=1); print(OUTJ, "저장:", mjcf)
