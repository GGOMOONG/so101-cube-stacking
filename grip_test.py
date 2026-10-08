#!/usr/bin/env python3
"""그리퍼 벌어짐 실측: 팔은 그 자리에 두고 그리퍼만 지정 값으로 연다. 자로 두 턱 끝 사이 간격을 잰다.
사용: python ~/robotics/grip_test.py 40      (40 % 열기 → 측정 → Enter → 2로 닫고 종료)
"""
import sys, time
import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
PORT = "/dev/tty.usbmodem5B790156431"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
val = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
robot = SO101Follower(SO101FollowerConfig(port=PORT, id="my_follower_arm", cameras={}, position_p_coefficient=48)); robot.connect(calibrate=False)
obs = robot.get_observation(); q = {f"{j}.pos": float(obs[f"{j}.pos"]) for j in JOINTS}
print("현재 그리퍼 읽기값 %.1f" % q["gripper.pos"])
try:
    robot.bus.enable_torque()
    for g in np.linspace(q["gripper.pos"], val, 30):
        q["gripper.pos"] = float(g); robot.send_action(q); time.sleep(1 / 30)
    time.sleep(0.5); print("그리퍼 %.0f 로 열림. 자로 두 턱 끝 사이 간격(mm)을 재세요. 어느 쪽 턱이 움직였는지도 보세요." % val)
    input("Enter를 누르면 닫고(2) 종료 > ")
    for g in np.linspace(val, 2.0, 30):
        q["gripper.pos"] = float(g); robot.send_action(q); time.sleep(1 / 30)
finally:
    robot.disconnect(); print("연결 해제")
