#!/usr/bin/env python3
"""팔로워 팔의 관절 각도(도)를 토크 OFF 상태에서 실시간으로 표시한다. 기준 자세 확인용.
사용: python ~/robotics/read_joints.py     (Ctrl+C로 종료)
"""
import time, os
import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "/dev/tty.usbmodem5B790156431"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
robot = SO101Follower(SO101FollowerConfig(port=PORT, id="my_follower_arm", cameras={}))
robot.connect(calibrate=False)
robot.bus.disable_torque()
print("토크 OFF. 팔을 원하는 자세로 잡고 숫자를 읽으세요. Ctrl+C로 종료.")
print("      " + "  ".join(f"{j[:10]:>10s}" for j in JOINTS))
try:
    while True:
        obs = robot.get_observation()
        q = [obs[f"{j}.pos"] for j in JOINTS]
        print("\r각도: " + "  ".join(f"{v:10.1f}" for v in q), end="", flush=True)
        time.sleep(0.2)
except KeyboardInterrupt:
    print()
finally:
    robot.disconnect()
    print("연결 해제")
