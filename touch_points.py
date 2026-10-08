#!/usr/bin/env python3
"""격자 → 로봇 좌표 변환 구하기 (그리퍼로 격자 점 짚기).
팔의 토크를 끄고, 사람이 그리퍼 끝을 격자 교차점에 갖다 댄 뒤 그 점의 격자 좌표를 입력하면
관절 각도(도) → 정기구학(placo, URDF) → 로봇 베이스 기준 그리퍼 위치(mm)를 기록한다.
3점 이상 모으면 격자 평면 → 로봇 베이스 강체 변환(회전+이동)을 최소제곱으로 맞추고 저장한다.

사용: python ~/robotics/touch_points.py
      입력 예: "60 100"  (격자 x y, mm, 테이블 위)  /  "90 140 30" (높이 30 mm, 큐브 윗면)  /  "fit" 변환 계산·저장  /  "q" 종료
출력: ~/robotics/calib/grid_to_base.json, ~/robotics/calib/touch_points.csv
"""
import os, sys, json, time
import numpy as np

URDF = "/Users/m4pro/robotics/so101_model/so101_new_calib.urdf"
PORT = "/dev/tty.usbmodem5B790156431"
ROBOT_ID = "my_follower_arm"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
OUT_JSON = os.path.expanduser("~/robotics/calib/grid_to_base.json")
OUT_CSV = os.path.expanduser("~/robotics/calib/touch_points.csv")

os.chdir(os.path.dirname(URDF))
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

kin = RobotKinematics(URDF, target_frame_name="gripper_frame_link", joint_names=JOINTS)
robot = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID, cameras={}))
robot.connect(calibrate=False)
robot.bus.disable_torque()
print("토크 OFF. 팔을 손으로 움직여 그리퍼 끝(닫은 상태)을 격자 교차점에 대고, 그 점의 좌표를 입력하세요.")
print("예: 60 100   |  fit = 변환 계산  |  q = 종료  |  단, 팔을 놓으면 떨어지니 한 손으로 잡은 채 입력")

pts_grid, pts_base, rows = [], [], []
if os.path.exists(OUT_CSV):
    for line in open(OUT_CSV).read().strip().splitlines()[1:]:
        v = [float(x) for x in line.split(",")]
        pts_grid.append(v[:3]); pts_base.append(v[3:6]); rows.append(line)
    print(f"이전 기록 {len(pts_grid)}점 불러옴")


def read_pose():
    obs = robot.get_observation()
    q = np.array([obs[f"{j}.pos"] for j in JOINTS], float)
    T = kin.forward_kinematics(q)
    return q, T[:3, 3] * 1000.0


def fit(G, B):
    """격자 평면 점 (x,y,0) → 베이스 (x,y,z) 강체 변환. Kabsch."""
    G3 = np.asarray(G, float)
    B3 = np.asarray(B, float)
    gc, bc = G3.mean(0), B3.mean(0)
    Hm = (G3 - gc).T @ (B3 - bc)
    U, S, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = bc - R @ gc
    res = (R @ G3.T).T + t - B3
    return R, t, np.linalg.norm(res, axis=1)


try:
    while True:
        s = input("> ").strip()
        if s == "q":
            break
        if s == "fit":
            if len(pts_grid) < 3:
                print("3점 이상 필요"); continue
            R, t, res = fit(pts_grid, pts_base)
            print("점별 잔차(mm):", np.round(res, 1), " 평균 %.1f  최대 %.1f" % (res.mean(), res.max()))
            json.dump({"R": R.tolist(), "t_mm": t.tolist(), "n_points": len(pts_grid),
                       "residual_mean_mm": float(res.mean()), "residual_max_mm": float(res.max()),
                       "note": "p_base = R @ [gx, gy, gz] + t  (mm). gripper_frame_link 기준, 그리퍼 닫은 상태."},
                      open(OUT_JSON, "w"), ensure_ascii=False, indent=2)
            print("저장:", OUT_JSON)
            continue
        try:
            vals = [float(v) for v in s.split()]
            gx, gy = vals[0], vals[1]; gz = vals[2] if len(vals) > 2 else 0.0
        except Exception:
            print("형식: x y [z]  (예: 60 100 또는 90 140 30)"); continue
        q, p = read_pose()
        pts_grid.append([gx, gy, gz]); pts_base.append(p.tolist())
        row = f"{gx},{gy},{gz},{p[0]:.1f},{p[1]:.1f},{p[2]:.1f}," + ",".join(f"{v:.1f}" for v in q)
        rows.append(row)
        print(f"  격자 ({gx:.0f},{gy:.0f},{gz:.0f}) → 베이스 x={p[0]:.1f} y={p[1]:.1f} z={p[2]:.1f} mm   관절(도) {np.round(q,1)}")
        with open(OUT_CSV, "w") as f:
            f.write("gx,gy,gz,bx,by,bz," + ",".join(JOINTS) + "\n" + "\n".join(rows) + "\n")
finally:
    robot.disconnect()
    print("연결 해제")
