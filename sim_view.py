#!/usr/bin/env python3
"""MuJoCo로 SO-101 보기. 실제 보정값(calib/kinematic_calibration.json)으로 격자 종이를 로봇 베이스 기준 자리에 놓는다.
사용 (macOS는 mjpython 필요):
  mjpython ~/robotics/sim_view.py --joints 0 -100 90 60 -50 10        # 관절 읽기값 6개(도)로 정지 표시
  mjpython ~/robotics/sim_view.py --grid 90 160 3                     # 격자 좌표 → IK → 표시(관절값 출력)
  mjpython ~/robotics/sim_view.py --live                              # 실제 팔 읽기값을 10 Hz로 따라감(토크 끔)
  옵션: --cube gx gy [색]  (여러 번 가능, 30 mm 큐브 배치)   --png 파일  (창 대신 이미지 저장, python으로도 됨)
창에서: 마우스 드래그 회전, 스크롤 확대, 더블클릭 선택. 격자 종이 위 검은 사각형 4개가 ArUco 자리.
"""
import sys, os, time, argparse
import numpy as np
import mujoco
sys.path.insert(0, os.path.expanduser("~/robotics"))
from so101_ik import Kin

HERE = os.path.expanduser("~/robotics/sim")
ap = argparse.ArgumentParser()
ap.add_argument("--joints", nargs=6, type=float)
ap.add_argument("--grid", nargs="+", type=float)
ap.add_argument("--live", action="store_true")
ap.add_argument("--cube", nargs="+", action="append", default=[])
ap.add_argument("--png")
a = ap.parse_args()
k = Kin()


def rot_to_quat(R):
    q = np.empty(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel()); return q


def box(name, grid_xy, size_mm, z_mm, rgba, z_center=None):
    """격자 좌표(mm)에 놓인 상자 → 베이스 좌표 MJCF geom 문자열"""
    sx, sy, sz = size_mm
    zc = z_mm + sz / 2 if z_center is None else z_center
    p = k.grid_to_base((grid_xy[0], grid_xy[1], zc)) / 1000.0
    q = rot_to_quat(k.R)
    return (f'<geom name="{name}" type="box" size="{sx/2000} {sy/2000} {sz/2000}" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" '
            f'quat="{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}" rgba="{rgba}" contype="0" conaffinity="0"/>')


geoms = [box("table", (105, 148.5), (1200, 1200, 0.2), -0.3, "0.8 0.8 0.8 1"),      # 격자 평면과 같은 기울기의 테이블
         box("paper", (105, 148.5), (210, 297, 0.5), 0, "0.97 0.97 0.95 1")]
for i, (mx, my) in enumerate([(20, 20), (120, 20), (20, 180), (120, 180)]):
    geoms.append(box(f"aruco{i}", (mx + 20, my + 20), (40, 40, 0.6), 0, "0.05 0.05 0.05 1"))
geoms.append(box("origin", (0, 0), (6, 6, 0.7), 0, "1 0 0 1"))
COL = {"orange": "1 0.5 0 1", "red": "0.85 0.1 0.1 1", "주황": "1 0.5 0 1", "빨강": "0.85 0.1 0.1 1"}
for i, c in enumerate(a.cube):
    geoms.append(box(f"cube{i}", (float(c[0]), float(c[1])), (30, 30, 30), 0, COL.get(c[2] if len(c) > 2 else "orange", "1 0.5 0 1")))

scene = f"""<mujoco model="so101_scene">
  <include file="so101.xml"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/><global azimuth="150" elevation="-25" offwidth="1280" offheight="960"/></visual>
  <worldbody>
    <light pos="0.3 -0.3 1.2" dir="-0.2 0.2 -1" diffuse="0.7 0.7 0.7"/>
    {chr(10).join(geoms)}
  </worldbody>
</mujoco>"""
open(os.path.join(HERE, "scene.xml"), "w").write(scene)
m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "scene.xml")); d = mujoco.MjData(m)


def set_readings(r6):
    u = k.reading_to_urdf(r6[:5]); d.qpos[:5] = np.deg2rad(u); d.qpos[5] = np.deg2rad(1.275 * (r6[5] + 8.6)); mujoco.mj_forward(m, d)   # 그리퍼 읽기값 0~100 % → URDF −10°~100°
    return k.fk_grid(r6[:5])


if a.grid:
    gx, gy = a.grid[0], a.grid[1]; gz = a.grid[2] if len(a.grid) > 2 else 3.0
    q, e = k.ik_grid((gx, gy, gz), roll_urdf_deg=-50.0)
    r6 = np.r_[q, 2.0]; print(f"격자 ({gx},{gy},{gz}) → 관절 읽기값 {np.round(q,1)}  IK 잔차 {e:.1f} mm")
elif a.joints:
    r6 = np.array(a.joints, float)
else:
    r6 = np.array([0.0, -5.0, 95.0, 49.0, -96.0, 10.0])   # 실물 휴식 자세
p = set_readings(r6); print(f"손끝 격자 좌표 {np.round(p,1)} mm")

if a.png:
    r = mujoco.Renderer(m, 720, 960); cam = mujoco.MjvCamera(); cam.azimuth, cam.elevation, cam.distance = 135, -35, 0.75
    cam.lookat[:] = k.grid_to_base((100, 120, 20)) / 1000.0; r.update_scene(d, cam)
    import PIL.Image as I; I.fromarray(r.render()).save(a.png); print("저장:", a.png); sys.exit()

import mujoco.viewer   # macOS: launch()는 예외가 나서 launch_passive만 씀(2026-09-29)
if not a.live:
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            v.sync(); time.sleep(0.05)
    sys.exit()

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
robot = SO101Follower(SO101FollowerConfig(port="/dev/tty.usbmodem5B790156431", id="my_follower_arm", cameras={}, position_p_coefficient=48))
robot.connect(calibrate=False); robot.bus.disable_torque(); print("토크 끔. 팔을 손으로 움직이면 화면이 따라갑니다. 창을 닫으면 종료.")
try:
    last_print = 0.0
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            obs = robot.get_observation(); r6 = np.array([obs[f"{j}.pos"] for j in JOINTS], float)
            p = set_readings(r6); v.sync(); time.sleep(0.1)
            if time.time() - last_print > 1.0:      # 1초마다 한 줄 (앱 터미널은 \r 덮어쓰기가 안 됨)
                print(f"관절 {np.round(r6,1)}  손끝 격자 {np.round(p,0)}", flush=True); last_print = time.time()
finally:
    robot.disconnect(); print("\n연결 해제")
