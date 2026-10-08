#!/usr/bin/env python3
"""pick_stack 계획 미리보기: 큐브 위치를 주면 8단계 역기구학 계획을 MuJoCo 화면에서 재생한다(실물은 안 움직임).
사용 (macOS는 mjpython):
  mjpython ~/robotics/sim_plan.py 90 160 130 200        # 주황 (90,160), 빨강 (130,200)
  옵션: --loop (반복 재생)  --speed 2 (2배속)  --png 접두어 (단계별 이미지 저장, python으로도 됨)
화면: 주황·빨강 큐브, 손끝 경로(회색 점), 현재 단계 이름은 터미널에 출력. 창을 닫으면 종료.
계획은 pick_stack.py와 같은 단계·높이·속도 제한(25°/s)을 쓴다. IK 잔차 3 mm 초과 시 중단.
"""
import sys, os, time, argparse
import numpy as np
import mujoco
sys.path.insert(0, os.path.expanduser("~/robotics"))
from so101_ik import Kin

ap = argparse.ArgumentParser()
ap.add_argument("ox", type=float); ap.add_argument("oy", type=float); ap.add_argument("rx", type=float); ap.add_argument("ry", type=float)
ap.add_argument("--loop", action="store_true"); ap.add_argument("--speed", type=float, default=1.0); ap.add_argument("--png"); ap.add_argument("--roll", type=float, default=-50.0, help="손목 회전 URDF 각도(기본 −50 = pick_stack.py와 같음)")
a = ap.parse_args()

# pick_stack.py와 동일한 상수
HOVER, GRASP_Z, PLACE_Z = 60.0, 15.0, 30.0 + 15.0 + 5.0
ROLL_URDF = a.roll
GRIP_OPEN, GRIP_CLOSE, MAX_DEG_PER_S, Y_BIAS = 40.0, 2.0, 25.0, 2.0
ox, oy, rx, ry = a.ox, a.oy + Y_BIAS, a.rx, a.ry + Y_BIAS
HOME = np.array([0.0, -5.0, 95.0, 49.0, -96.0, GRIP_OPEN])   # 09-29 실물 휴식 자세(토크 끔 읽기값). 이전 임의값(-100)은 팔꿈치가 테이블을 뚫음

k = Kin()
plan = [("주황 위 대기", (ox, oy, GRASP_Z + HOVER), GRIP_OPEN), ("주황 잡기 높이", (ox, oy, GRASP_Z), GRIP_OPEN),
        ("닫기", (ox, oy, GRASP_Z), GRIP_CLOSE), ("들어올리기", (ox, oy, GRASP_Z + HOVER), GRIP_CLOSE),
        ("빨강 위 대기", (rx, ry, PLACE_Z + 40.0), GRIP_CLOSE), ("놓기 높이", (rx, ry, PLACE_Z), GRIP_CLOSE),
        ("열기", (rx, ry, PLACE_Z), GRIP_OPEN), ("올리기", (rx, ry, PLACE_Z + 40.0), GRIP_OPEN)]
qs, q_prev = [], None
for name, p, g in plan:
    q, e = k.ik_grid(p, roll_urdf_deg=ROLL_URDF, q0_reading=q_prev); qs.append(np.r_[q, g]); q_prev = q
    print(f"  {name:8s} 목표 {np.round(p,0)}  관절 {np.round(q,1)}  IK 잔차 {e:.1f} mm")
    if e > 3.0: sys.exit("닿지 않는 위치 → 중단")

# 장면: sim_view.py와 같은 방식(보정값으로 종이·큐브 배치)
def rot_to_quat(R):
    q = np.empty(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel()); return q
def box(name, gxy, size, z, rgba):
    p = k.grid_to_base((gxy[0], gxy[1], z + size[2] / 2)) / 1000.0; q = rot_to_quat(k.R)
    return (f'<geom name="{name}" type="box" size="{size[0]/2000} {size[1]/2000} {size[2]/2000}" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" '
            f'quat="{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}" rgba="{rgba}" contype="0" conaffinity="0"/>')
geoms = [box("table", (105, 148.5), (1200, 1200, 0.2), -0.3, "0.8 0.8 0.8 1"), box("paper", (105, 148.5), (210, 297, 0.5), 0, "0.97 0.97 0.95 1")]
for i, (mx, my) in enumerate([(20, 20), (120, 20), (20, 180), (120, 180)]):
    geoms.append(box(f"aruco{i}", (mx + 20, my + 20), (40, 40, 0.6), 0, "0.05 0.05 0.05 1"))
geoms += [box("origin", (0, 0), (6, 6, 0.7), 0, "1 0 0 1"), box("red", (a.rx, a.ry), (30, 30, 30), 0, "0.85 0.1 0.1 1").replace('contype="0" conaffinity="0"', 'contype="1" conaffinity="1"')]
scene = f"""<mujoco model="so101_plan"><include file="so101.xml"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/><global azimuth="135" elevation="-35" offwidth="1280" offheight="960"/></visual>
  <worldbody><light pos="0.3 -0.3 1.2" dir="-0.2 0.2 -1" diffuse="0.7 0.7 0.7"/>{chr(10).join(geoms)}
    <body name="orange" mocap="true"><geom name="orange" type="box" size="0.015 0.015 0.015" rgba="1 0.5 0 1" contype="0" conaffinity="0"/></body>
  </worldbody></mujoco>"""
HERE = os.path.expanduser("~/robotics/sim"); open(os.path.join(HERE, "plan_scene.xml"), "w").write(scene)
m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "plan_scene.xml")); d = mujoco.MjData(m)
mocap_id = m.body("orange").mocapid[0]; quat_grid = rot_to_quat(k.R)
orange_pos = k.grid_to_base((a.ox, a.oy, 15.0)) / 1000.0    # 주황 초기 위치(격자 중심 높이 15)

def grip_urdf_deg(r): return 1.275 * (r + 8.6)      # LeRobot 그리퍼 읽기값은 0~100 %(닫힘~열림), URDF 한계 −10°~100°
def set_readings(r6):
    d.qpos[:5] = np.deg2rad(k.reading_to_urdf(r6[:5])); d.qpos[5] = np.deg2rad(grip_urdf_deg(r6[5])); mujoco.mj_forward(m, d)
    return k.fk_grid(r6[:5])

def segments():
    """(단계 이름, 관절 궤적 프레임 목록, 큐브가 손에 붙어 있는지) 생성. pick_stack.move_to와 같은 코사인 보간."""
    q0 = HOME.copy(); holding = False
    for (name, p, g), qt in zip(plan, qs):
        dq = np.abs(qt - q0); dur = max(0.8, dq[:5].max() / MAX_DEG_PER_S); n = int(dur * 30)
        frames = [q0 + (qt - q0) * (0.5 - 0.5 * np.cos(np.pi * i / n)) for i in range(1, n + 1)]
        if name == "닫기": holding = True
        if name == "열기": holding = False
        yield name, frames, holding; q0 = qt

def place_orange(holding, tip_grid):
    if holding:
        d.mocap_pos[mocap_id] = k.grid_to_base((tip_grid[0], tip_grid[1], tip_grid[2])) / 1000.0
    d.mocap_quat[mocap_id] = quat_grid

d.mocap_pos[mocap_id] = orange_pos; d.mocap_quat[mocap_id] = quat_grid; set_readings(HOME)
trail = []

# ---- 간섭 검사 (창 열기 전에 전체 궤적을 훑음): 팔 메시 ↔ 빨강 큐브 최소 거리, 0 이하면 관통 ----
red_g = m.geom("red").id
arm_geoms = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
def dist_to_red():
    best = 1e9; who = None
    for g in arm_geoms:
        fromto = np.zeros(6); dd = mujoco.mj_geomDistance(m, d, g, red_g, 0.2, fromto)
        if dd < best: best = dd; who = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
    return best * 1000.0, who
print("\n간섭 검사 (팔 ↔ 빨강 큐브, 주황은 잡는 대상이라 제외):")
d.mocap_pos[mocap_id] = orange_pos; set_readings(HOME); worst = []
for name, frames, holding in segments():
    mn = (1e9, None)
    for q in frames:
        tip = set_readings(q); place_orange(holding, tip); dd, who = dist_to_red()
        if dd < mn[0]: mn = (dd, who)
    flag = "  ← 관통(실물이면 빨강을 밀거나 침)" if mn[0] <= 0 else ("  ← 5 mm 미만, 위험" if mn[0] < 5 else "")
    print(f"  {name:8s} 빨강과 최소 거리 {mn[0]:6.1f} mm ({mn[1]}){flag}")
    if mn[0] < 5: worst.append(name)
print("판정:", "간섭 없음" if not worst else "간섭 단계: " + ", ".join(worst) + " → 큐브 간격을 넓히거나 손목 회전(ROLL)을 바꿔야 함")
d.mocap_pos[mocap_id] = orange_pos; set_readings(HOME)

if a.png:
    r = mujoco.Renderer(m, 720, 960); cam = mujoco.MjvCamera(); cam.azimuth, cam.elevation, cam.distance = 135, -35, 0.75
    cam.lookat[:] = k.grid_to_base((100, 120, 20)) / 1000.0
    import PIL.Image as I
    for i, (name, frames, holding) in enumerate(segments()):
        for q in frames:
            tip = set_readings(q); place_orange(holding, tip)
        r.update_scene(d, cam); I.fromarray(r.render()).save(f"{a.png}_{i+1}_{name}.png"); print("저장:", f"{a.png}_{i+1}_{name}.png")
    sys.exit()

import mujoco.viewer
with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        d.mocap_pos[mocap_id] = orange_pos; set_readings(HOME); trail.clear()
        for name, frames, holding in segments():
            print(f"▶ {name}", flush=True)
            for q in frames:
                if not v.is_running(): break
                tip = set_readings(q); place_orange(holding, tip)
                trail.append(k.grid_to_base(tip) / 1000.0)
                with v.lock():
                    v.user_scn.ngeom = 0
                    for j, t in enumerate(trail[::3][:v.user_scn.maxgeom]):
                        mujoco.mjv_initGeom(v.user_scn.geoms[j], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.002, 0, 0]), t, np.eye(3).ravel(), np.array([0.3, 0.3, 0.3, 0.8], np.float32))
                        v.user_scn.ngeom = j + 1
                v.sync(); time.sleep(1 / 30 / a.speed)
            time.sleep(0.4 / a.speed)
        print("계획 재생 완료" + (" (반복)" if a.loop else " — 창을 닫으면 종료"), flush=True)
        if not a.loop:
            while v.is_running(): v.sync(); time.sleep(0.05)
            break
        time.sleep(1.0)
