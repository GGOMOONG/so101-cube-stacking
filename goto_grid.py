#!/usr/bin/env python3
"""역기구학 이동 시험: 격자 좌표를 주면 팔이 그 위 60 mm(대기) → 지정 높이(기본 3 mm)로 내려간다.
사용: python ~/robotics/goto_grid.py 90 160          (격자 (90,160), 높이 3 mm)
      python ~/robotics/goto_grid.py 90 160 30       (높이 30 mm)
절차: 목표 관절각 출력 → Enter로 승인 → 대기 높이로 이동 → Enter → 내려감 → 자로 오차 측정 → Enter → 대기 높이로 복귀 → 종료.
안전: 최대 관절 속도 25°/s, 30 Hz 보간, 관절 한계 안에서만, Ctrl+C 시 그 자리에서 정지(토크 유지).
"""
import sys, os, time
import numpy as np
from so101_ik import Kin
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "/dev/tty.usbmodem5B790156431"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HOVER = 60.0          # mm
MAX_DEG_PER_S = 25.0
ROLL_URDF = -50.0     # 손목 회전: 수직으로 댔던 자세와 같게
GRIP = 2.0            # 그리퍼 닫음(읽기값)

gx, gy = float(sys.argv[1]), float(sys.argv[2])
gz = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0

k = Kin()
robot = SO101Follower(SO101FollowerConfig(port=PORT, id="my_follower_arm", cameras={}, position_p_coefficient=48))
robot.connect(calibrate=False)


def current():
    obs = robot.get_observation()
    return np.array([obs[f"{j}.pos"] for j in JOINTS], float)


def move_to(q_target6, label):
    q0 = current()
    dq = np.abs(q_target6 - q0); dur = max(1.0, dq.max() / MAX_DEG_PER_S)
    n = int(dur * 30)
    print(f"  {label}: {dur:.1f}초 동안 이동 (최대 관절 변화 {dq.max():.0f}°)")
    for i in range(1, n + 1):
        a = 0.5 - 0.5 * np.cos(np.pi * i / n)          # 부드러운 가감속
        q = q0 + (q_target6 - q0) * a
        robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, q)})
        time.sleep(1 / 30)
    time.sleep(0.5)
    qf = current(); pf = k.fk_grid(qf[:5])
    print(f"  도착: 관절 {np.round(qf[:5],1)} → FK 격자 ({pf[0]:.1f}, {pf[1]:.1f}, {pf[2]:.1f})")


try:
    q_now = current()
    print("현재 관절:", np.round(q_now, 1), "→ FK 격자", np.round(k.fk_grid(q_now[:5]), 1))
    q_h, e_h = k.ik_grid((gx, gy, gz + HOVER), roll_urdf_deg=ROLL_URDF, q0_reading=q_now[:5])
    q_t, e_t = k.ik_grid((gx, gy, gz), roll_urdf_deg=ROLL_URDF, q0_reading=q_h)
    print(f"목표 격자 ({gx:.0f},{gy:.0f},{gz:.0f})  대기 관절 {np.round(q_h,1)} (IK 잔차 {e_h:.1f} mm)  목표 관절 {np.round(q_t,1)} (IK 잔차 {e_t:.1f} mm)")
    if e_t > 3.0:
        sys.exit("IK 잔차가 3 mm를 넘음 → 닿지 않는 위치. 중단.")
    input("Enter를 누르면 토크를 켜고 대기 높이로 이동합니다 (팔 주변을 비우세요) > ")
    robot.bus.enable_torque()
    move_to(np.r_[q_h, GRIP], "대기 높이(+60 mm)로")
    input("Enter를 누르면 목표 높이로 내려갑니다 > ")
    move_to(np.r_[q_t, GRIP], f"목표 높이({gz:.0f} mm)로")
    input("자로 손끝과 교차점의 차이를 재세요. Enter를 누르면 대기 높이로 올라갑니다 > ")
    move_to(np.r_[q_h, GRIP], "대기 높이로 복귀")
    input("Enter를 누르면 테이블 위 15 mm까지 내려간 뒤 토크를 끄고 종료합니다 (팔이 살짝 내려앉음) > ")
    q_r, _ = k.ik_grid((gx, gy, 15.0), roll_urdf_deg=ROLL_URDF, q0_reading=q_h)
    move_to(np.r_[q_r, GRIP], "종료 자세(테이블 위 15 mm)로")
except KeyboardInterrupt:
    print("\n중단: 현재 위치에서 정지(토크 유지)")
finally:
    robot.disconnect()
    print("연결 해제")
