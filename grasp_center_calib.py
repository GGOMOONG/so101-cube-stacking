#!/usr/bin/env python3
"""그리퍼 잡기 중심·턱 방향 보정 (1회 실측).
절차: 토크 끔 → 사람이 팔을 종이 위 아무 데나 그리퍼가 아래를 보게 둠 → Enter → 토크 켜고 턱 열림(20)
      → 사람이 주황 큐브를 두 패드 사이 한가운데에 반듯하게 넣음 → Enter → 턱 닫힘(9, 물기) → Enter
      → 팔이 큐브 중심 높이 15 mm까지 수직 하강(큐브가 테이블에 닿음) → 열림 → 60 mm 상승
      → 탑 카메라(0)로 큐브 중심·방향 측정 → 팔 계산 기준점과 비교 → calib/grasp_center.json 저장
출력: pad_offset_mm(그리퍼 프레임 x,y: 계산 기준점 → 실제 잡기 중심), jaw_angle_offset_deg(계산 턱 축 → 실제 큐브 면 방향)
사용: python ~/robotics/grasp_center_calib.py
"""
import sys, os, time, json
import numpy as np, cv2
sys.path.insert(0, os.path.expanduser("~/robotics"))
from so101_ik import Kin
from gripper_geom import GripperGeom
import cube_locate as CL
from wrist_view import wrist_measure, open_wrist
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "/dev/tty.usbmodem5B790156431"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
OPEN, HOLD = 35.0, 12.0
k = Kin(); GG = GripperGeom()
EXTRA_HOLD = GG.mid_offset_mm(GG.reading_for_pad_gap(30.0))      # 계산상 "문 큐브의 중심" (손끝 기준)
robot = SO101Follower(SO101FollowerConfig(port=PORT, id="my_follower_arm", cameras={}, position_p_coefficient=48)); robot.connect(calibrate=False)


def cur(): o = robot.get_observation(); return np.array([o[f"{j}.pos"] for j in JOINTS], float)
def send(q): robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, q)})
def move(q_t, secs=None):
    q0 = cur(); dq = np.abs(q_t - q0); n = int(max(0.8, dq[:5].max() / 25.0 if secs is None else secs) * 30)
    for i in range(1, n + 1):
        send(q0 + (q_t - q0) * (0.5 - 0.5 * np.cos(np.pi * i / n))); time.sleep(1 / 30)
    time.sleep(0.4)


try:
    robot.bus.disable_torque()
    input("토크 꺼짐. 팔을 손으로 종이 위(가운데쯤, 큐브가 없는 곳) 15~20 cm 높이에 그리퍼가 아래를 보게 두고 Enter > ")
    q = cur(); robot.bus.enable_torque(); q[5] = OPEN; move(q, 1.0)
    input("턱이 열렸습니다. 주황 큐브를 두 패드 사이 한가운데, 면이 패드와 평행하게 넣고 손으로 받친 채 Enter > ")
    q[5] = HOLD; move(q, 1.0)
    input("턱이 닫혔습니다. 큐브를 놓고 손을 치운 뒤 Enter (팔이 수직으로 내려가 큐브를 테이블에 놓습니다) > ")
    q = cur(); u = k.reading_to_urdf(q[:5]); roll_now = u[4]
    p_now = k.fk_grid(q[:5], EXTRA_HOLD)
    q_dn, e = k.ik_grid((p_now[0], p_now[1], 15.0), roll_urdf_deg=roll_now, q0_reading=q[:5], tip_extra_mm=EXTRA_HOLD)
    print(f"  현재 계산 기준점 {np.round(p_now,1)} → 하강 목표 z 15 (IK 잔차 {e:.1f} mm)")
    move(np.r_[q_dn, HOLD])
    q_rel = cur(); p_ref = k.fk_grid(q_rel[:5], EXTRA_HOLD); _, R = k.fk_base(q_rel[:5])
    jaw_dir = k.R.T @ R[:, 0]; jaw_ang = np.degrees(np.arctan2(jaw_dir[1], jaw_dir[0]))
    print(f"  놓기 직전 계산 기준점(문 큐브 중심) {np.round(p_ref,1)}, 계산 턱 축 각도 {jaw_ang:.1f}°")
    q_o = q_rel.copy(); q_o[5] = OPEN; move(q_o, 1.0)
    q_up, _ = k.ik_grid((p_ref[0], p_ref[1], 75.0), roll_urdf_deg=roll_now, q0_reading=q_rel[:5], tip_extra_mm=EXTRA_HOLD)
    move(np.r_[q_up, OPEN]); time.sleep(0.5)
    # 대기 높이(75)에서 손목 사진: 이 순간 큐브는 잡기 중심 바로 아래 → 손목 화면에서의 큐브 위치·각도가 "제대로 잡을 때"의 목표
    wcap = open_wrist(); wm = wrist_measure(wcap, save=os.path.expanduser("~/robotics/calib/grasp_center_wrist.png")); wcap.release()
    if wm is None: print("  ⚠ 손목 카메라에서 주황 큐브를 못 찾음 (calib/grasp_center_wrist.png 확인)")
    else: print(f"  손목 목표: 픽셀 ({wm['u']:.0f},{wm['v']:.0f}), 각도 {wm['angle']:.1f}°, 큐브 폭 {wm['w_px']:.0f} px")
    # 팔을 로봇 쪽으로 물러나게(토크 유지) → 탑 카메라 시야에서 큐브가 가려지지 않게
    q_away, _ = k.ik_grid((p_ref[0], max(p_ref[1] - 80.0, 40.0), 120.0), roll_urdf_deg=roll_now, q0_reading=q_up, tip_extra_mm=EXTRA_HOLD)
    move(np.r_[q_away, OPEN]); time.sleep(0.5)
    print("  큐브를 놓고 팔이 물러났습니다(토크 유지). 탑 카메라로 측정합니다.")
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION); cap.set(3, 640); cap.set(4, 480)
    f = None
    for _ in range(15):
        ok, fr = cap.read()
        if ok: f = fr
    cap.release()
    out, vis = CL.locate(f); cv2.imwrite(os.path.expanduser("~/robotics/calib/grasp_center_top.png"), vis)
    if not out.get("orange"): sys.exit("주황 큐브 미검출: calib/grasp_center_top.png 확인")
    cx, cy = out["orange"]["xy"]; ang_c = out["orange"].get("angle_deg", None)
    d_grid = np.array([cx - p_ref[0], cy - p_ref[1], 0.0])
    d_grip = R.T @ (k.R @ d_grid)          # 격자 → 베이스 → 그리퍼 프레임
    res = {"pad_offset_mm": [float(d_grip[0]), float(d_grip[1])], "grid_offset_mm": [float(d_grid[0]), float(d_grid[1])],
           "jaw_angle_calc_deg": float(jaw_ang), "cube_angle_top_deg": (None if ang_c is None else float(ang_c)),
           "jaw_angle_offset_deg": (None if ang_c is None else float(((ang_c - jaw_ang) + 45) % 90 - 45)),
           "ref_point_grid": p_ref.tolist(), "cube_measured_grid": [float(cx), float(cy)],
           "wrist_target": (None if wm is None else {"u": float(wm["u"]), "v": float(wm["v"]), "angle_deg": float(wm["angle"]), "w_px": float(wm["w_px"]), "hover_z_mm": 75.0}),
           "note": "실제 잡기 중심 = 계산 기준점 + pad_offset(그리퍼 x,y); 실제 턱 축 = 계산 턱 축 + jaw_angle_offset; wrist_target = 대기 높이 75에서 큐브가 잡기 중심 바로 아래일 때 손목 화면 위치·각도"}
    json.dump(res, open(os.path.expanduser("~/robotics/calib/grasp_center.json"), "w"), ensure_ascii=False, indent=2)
    print(f"\n큐브 실측 ({cx:.1f}, {cy:.1f})  계산 기준점 ({p_ref[0]:.1f}, {p_ref[1]:.1f})  차이(격자) {np.round(d_grid[:2],1)} mm → 그리퍼 프레임 오프셋 {np.round(d_grip[:2],1)} mm")
    print(f"큐브 방향(탑 카메라) {ang_c}  계산 턱 축 {jaw_ang:.1f}° → 각도 오프셋 {res['jaw_angle_offset_deg']}")
    print("저장: calib/grasp_center.json")
finally:
    robot.disconnect(); print("연결 해제")
