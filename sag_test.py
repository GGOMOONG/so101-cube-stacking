#!/usr/bin/env python3
"""처짐(명령 위치 − 실제 위치) 측정: 모터 게인(P/I/D)을 바꿔 가며 같은 자세에서 비교한다.
사용: python ~/robotics/sag_test.py            (기본 P16 I0 D32 = LeRobot 기본값)
      python ~/robotics/sag_test.py --p 32     (P만 변경)   --i 4  --d 32 도 가능
절차: 자세 4개(멀리 대기/잡기, 중간 대기/잡기)로 이동(정착 보정 없음) → 도착 오차 → 그리퍼 닫기(12) 후 변화 → 열기(35) 후 변화 → 1초간 흔들림(표준편차).
결과: 표 출력 + calib/gain_tests.csv 추가. 끝나면 테이블 위 20 mm로 내려간 뒤 토크 끔.
"""
import sys, os, time, argparse
import numpy as np
sys.path.insert(0, os.path.expanduser("~/robotics"))
from so101_ik import Kin
from gripper_geom import GripperGeom
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

ap = argparse.ArgumentParser(); ap.add_argument("--p", type=int, default=16); ap.add_argument("--i", type=int, default=0); ap.add_argument("--d", type=int, default=32)
a = ap.parse_args()
PORT = "/dev/tty.usbmodem5B790156431"; JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
OPEN, HOLD, ROLL = 35.0, 12.0, -50.0
POSES = [("멀리 대기", (100, 200, 75)), ("멀리 잡기", (100, 200, 18)), ("중간 대기", (140, 160, 75)), ("중간 잡기", (140, 160, 18))]
k = Kin(); GG = GripperGeom(); EXTRA = GG.mid_offset_mm(OPEN)
robot = SO101Follower(SO101FollowerConfig(port=PORT, id="my_follower_arm", cameras={}, position_p_coefficient=a.p, position_i_coefficient=a.i, position_d_coefficient=a.d))
robot.connect(calibrate=False)
print(f"게인 P={a.p} I={a.i} D={a.d}  (모터에서 읽음: P={robot.bus.read('P_Coefficient','shoulder_lift')} I={robot.bus.read('I_Coefficient','shoulder_lift')} D={robot.bus.read('D_Coefficient','shoulder_lift')})")


def cur(): o = robot.get_observation(); return np.array([o[f"{j}.pos"] for j in JOINTS], float)
def send(q): robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, q)})
def move(q_t):
    q0 = cur(); n = int(max(0.8, np.abs(q_t - q0)[:5].max() / 25.0) * 30)
    for i in range(1, n + 1): send(q0 + (q_t - q0) * (0.5 - 0.5 * np.cos(np.pi * i / n))); time.sleep(1 / 30)
    time.sleep(0.5)
def fk(): return k.fk_grid(cur()[:5], EXTRA)
def jitter():
    s = np.array([cur()[:5] for _ in range(20)]); return float(np.abs(s - s.mean(0)).max())


rows = []
try:
    input("Enter를 누르면 토크를 켜고 4개 자세를 차례로 갑니다 (팔 주변을 비우세요) > ")
    robot.bus.enable_torque()
    for name, p in POSES:
        q, e = k.ik_grid(p, roll_urdf_deg=ROLL, q0_reading=cur()[:5], tip_extra_mm=EXTRA)
        move(np.r_[q, OPEN]); arr = fk(); err = np.array(p, float) - arr; jit = jitter()
        move(np.r_[cur()[:5], HOLD]); arr_c = fk(); d_close = arr_c - arr
        move(np.r_[cur()[:5], OPEN]); arr_o = fk(); d_open = arr_o - arr_c
        print(f"  {name:6s} 목표 {p}  도착 오차 (x{err[0]:+.1f}, y{err[1]:+.1f}, z{err[2]:+.1f}) mm  닫을 때 변화 z{d_close[2]:+.1f} y{d_close[1]:+.1f}  열 때 z{d_open[2]:+.1f} y{d_open[1]:+.1f}  흔들림 {jit:.2f}°")
        rows.append((name, err, d_close, d_open, jit))
    # 처짐 총평
    ez = np.mean([abs(r[1][2]) for r in rows]); exy = np.mean([np.linalg.norm(r[1][:2]) for r in rows]); dc = np.mean([abs(r[2][2]) for r in rows]); jm = max(r[4] for r in rows)
    print(f"\n요약 P={a.p}: 도착 z 오차 평균 {ez:.1f} mm, xy 오차 평균 {exy:.1f} mm, 닫을 때 z 변화 평균 {dc:.1f} mm, 최대 흔들림 {jm:.2f}°")
    path = os.path.expanduser("~/robotics/calib/gain_tests.csv"); new = not os.path.exists(path)
    with open(path, "a") as f:
        if new: f.write("time,P,I,D,pose,err_x,err_y,err_z,close_dz,close_dy,open_dz,open_dy,jitter_deg\n")
        for name, err, dc_, do_, jit in rows:
            f.write(f"{time.strftime('%m%d_%H%M%S')},{a.p},{a.i},{a.d},{name},{err[0]:.1f},{err[1]:.1f},{err[2]:.1f},{dc_[2]:.1f},{dc_[1]:.1f},{do_[2]:.1f},{do_[1]:.1f},{jit:.2f}\n")
    print("기록:", path)
    q_r, _ = k.ik_grid((100, 130, 20), roll_urdf_deg=ROLL, q0_reading=cur()[:5], tip_extra_mm=EXTRA); move(np.r_[q_r, OPEN])
except KeyboardInterrupt:
    print("\n중단")
finally:
    robot.disconnect(); print("연결 해제 (토크 꺼짐)")
