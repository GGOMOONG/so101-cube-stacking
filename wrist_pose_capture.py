#!/usr/bin/env python3
"""손목 카메라 5분 시험: 실물 놓기 자세(DAgger 3 보정 데이터의 열기 직전 상태)로 팔을 보내고,
큐브를 턱 끝에 맞춰(큐브 바닥 = 턱 끝) 물린 뒤 탑·손목 사진과 관절 상태를 저장한다 → 시뮬 같은 자세 렌더와 비교(카메라–턱–큐브 기하 직접 측정).
사용: python ~/robotics/wrist_pose_capture.py            (TOP=0 WRIST=1 환경변수로 카메라 번호 변경)
절차: 책상 위 큐브는 모두 치우고 시작(충돌 방지) → 팔이 자세 1로 감(그리퍼 열림) → 주황 큐브를 턱 사이에 끼우고 Enter → 닫고 촬영 → 자세 2 … → 종료 자세.
저장: calib/wrist_pose/pK_top.png, pK_wrist.png, pK.json(상태·명령), rest_top.png(시작 전 탑 사진: 빨강 위치용)"""
import os, sys, time, json
os.environ.setdefault("LEROBOT_GRIPPER_MAX_TORQUE", "300")   # 그리퍼 토크 한계(평가 때와 같음) — 없으면 큐브를 물다 Overload
import numpy as np, cv2
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
PORT = "/dev/tty.usbmodem5B790156431"; JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TOP, WRIST = int(os.environ.get("TOP", 0)), int(os.environ.get("WRIST", 1)); P = int(os.environ.get("P", 16))
OUT = os.path.expanduser("~/robotics/calib/wrist_pose"); os.makedirs(OUT, exist_ok=True)
POSES = {1: [-15.0, 130.9, -33.3, 84.2, -108.3], 2: [12.0, 116.2, -8.3, 78.0, -85.4], 3: [-27.2, 105.7, 7.1, 74.2, -118.1]}   # DAgger 3 ep 0·8·16 열기 직전
OPEN, CLOSE = 28.0, 0.0

def open_cam(i):
    c = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION); c.set(cv2.CAP_PROP_FRAME_WIDTH, 640); c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); c.set(cv2.CAP_PROP_FPS, 30); return c
def grab(cap, n=8):
    f = None; t0 = time.time()
    while time.time() - t0 < 4.0:                      # 손목 카메라가 멈췄다 돌아오는 경우 최대 4초 기다림
        for _ in range(n): ok, fr = cap.read(); f = fr if ok else f
        if f is not None: break
    if f is None: print("  !! 카메라 프레임 없음(손목 케이블?) — 검은 화면 저장"); f = np.zeros((480, 640, 3), np.uint8)
    return f

caps = {"top": open_cam(TOP), "wrist": open_cam(WRIST)}; time.sleep(2.0)
robot = SO101Follower(SO101FollowerConfig(port=PORT, id="my_follower_arm", cameras={}, position_p_coefficient=P)); robot.connect(calibrate=False)
obs = robot.get_observation(); cur = np.array([float(obs[f"{j}.pos"]) for j in JOINTS])
def send(q): robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, q)})
def move(target, dur=3.0):
    global cur
    t = np.asarray(target, float); n = int(dur * 30)
    for i in range(1, n + 1): send(cur + (t - cur) * (0.5 - 0.5 * np.cos(np.pi * i / n))); time.sleep(1 / 30)
    cur = t; time.sleep(0.5)
def state(): o = robot.get_observation(); return [float(o[f"{j}.pos"]) for j in JOINTS]
try:
    robot.bus.enable_torque()
    cv2.imwrite(f"{OUT}/rest_top.png", grab(caps["top"])); print("시작 탑 사진 저장. 팔이 자세 1로 갑니다(3초).")
    for k, pose in POSES.items():
        move(np.r_[pose, OPEN], dur=3.0)
        input(f"[자세 {k}] 주황 큐브를 턱 사이에 '바닥이 턱 끝과 같은 높이'가 되게 끼우고 손을 뗀 뒤 Enter > ")
        move(np.r_[pose, CLOSE], dur=1.0); time.sleep(1.0)
        st = state(); top = grab(caps["top"]); wr = grab(caps["wrist"])
        cv2.imwrite(f"{OUT}/p{k}_top.png", top); cv2.imwrite(f"{OUT}/p{k}_wrist.png", wr)
        json.dump({"pose_cmd": pose + [CLOSE], "state": st, "P": P}, open(f"{OUT}/p{k}.json", "w"))
        print(f"  저장: p{k}_top/wrist.png, 상태 {np.round(st,1).tolist()} (그리퍼 {st[5]:.1f})")
        move(np.r_[pose, OPEN], dur=1.0)
        input(f"  큐브를 빼고 Enter > ")
    move(np.r_[cur[:5], OPEN], dur=0.5); move([0, -5, 90, 75, -90, OPEN], dur=3.0); print("종료 자세. 완료 → calib/wrist_pose/")
finally:
    try: robot.disconnect()
    except Exception as e: print("연결 해제 중 오류(모터 Overload면 팔 전원을 껐다 켜세요):", str(e)[:80])
    for c in caps.values(): c.release()
