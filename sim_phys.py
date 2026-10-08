#!/usr/bin/env python3
"""MuJoCo 물리 시뮬레이션 1단계: 구동기(위치 서보) + 자유 큐브 + 그리퍼 턱 안쪽 면 접촉(얇은 판 2장, 실물엔 패드 없음).
so101.xml(형상만)에서 so101_phys.xml(구동기·감쇠·패드 충돌체)을 만들고, 격자 좌표 큐브를 놓은 장면에서 물리를 돌린다.
사용:
  python ~/robotics/sim_phys.py --test lift            # 주황 (90,160) 집어 들어 올리기 → 빨강 (130,200) 위에 놓기. 단계별 처짐·큐브 높이 출력
  python ~/robotics/sim_phys.py --test sag --kp 25     # 실물 게인 시험 자세("멀리 대기" 등)에서 z 처짐 측정 → kp 보정용
  옵션: --kp N (팔 관절 위치 이득, 기본 25 ≈ 실물 P16 목표) --png 접두어 (단계별 이미지 저장)
모듈: from sim_phys import PhysSim; s = PhysSim({"orange": (90,160,0), "red": (130,200,0)}); s.goto(readings6); s.cube_grid("orange")
"""
import os, sys, re, json, argparse
from collections import deque
import numpy as np
import mujoco
sys.path.insert(0, os.path.expanduser("~/robotics"))
from so101_ik import Kin
from gripper_geom import GripperGeom, DEG_PER_PCT, CLOSED_READING

HERE = os.path.expanduser("~/robotics/sim")
KP_DEFAULT, KV, FORCE = 25.0, 1.0, 3.0          # 팔 관절: N·m/rad, N·m·s/rad, 토크 한계(STS3215 정격 ~3 N·m)
KP_GRIP, FORCE_GRIP = 3.0, 0.6                  # 그리퍼: LEROBOT_GRIPPER_MAX_TORQUE=300(약 30 %)에 맞춘 한계
PAD_LEN, PAD_W, PAD_T, PAD_INSET = 0.070, 0.020, 0.004, 0.035   # 턱 안쪽 면(판): 길이(턱 방향, 끝~70 mm), 폭(회전축 방향), 두께, 끝에서 중심까지. 09-30 사진: 별도 패드 없음, 완전 닫힘에서 끝이 만남
CUBE, CUBE_MASS = 0.030, 0.015                  # 30 mm, 15 g
CLOSED_GAP = 0.005                              # 손으로 완전히 닫았을 때 남는 턱 틈(실측 약 5 mm)
HOME = np.array([0.0, -5.0, 95.0, 49.0, -96.0, 35.0])
# 외관(10-01 실물 탑 사진과 수치로 맞춤: 책상 171, 주황 HSV (10,195,218), 빨강 (178,195,199), 그림자 최저/책상 0.71)
TABLE_RGBA = "0.72 0.72 0.72 1"
CUBE_RGBA = {"orange": "1 0.5 0 1", "red": "0.85 0.1 0.1 1"}
LIGHT = dict(head_diffuse=0.6, ambient=0.3, pos="0.3 -0.3 1.2", dir="-0.2 0.2 -1", diffuse=0.7, shadow="true", pos2="-0.3 0.4 1.0", dir2="0.3 -0.4 -1", diffuse2=0.0)
APPEARANCE_PATH = _ap = os.path.expanduser("~/robotics/calib/sim_appearance.json")
if os.path.exists(_ap):      # 10-01 실물 탑 사진 통계로 맞춘 값
    _a = json.load(open(_ap)); TABLE_RGBA = f"{_a['tb']} {_a['tb']} {_a['tb']} 1"; CUBE_RGBA = {"orange": _a["oc"], "red": _a["rc"]}
    LIGHT.update(head_diffuse=_a["hd"], ambient=_a["amb"], diffuse=_a["dif"], shadow=_a["sh"], diffuse2=_a["dif2"])


def apply_appearance(path):
    """다른 외관 json 적용(생성기 --appearance). 모델을 만들기 전에 호출."""
    global TABLE_RGBA, CUBE_RGBA, APPEARANCE_PATH
    APPEARANCE_PATH = os.path.expanduser(path); a = json.load(open(APPEARANCE_PATH)); TABLE_RGBA = f"{a['tb']} {a['tb']} {a['tb']} 1"; CUBE_RGBA = {"orange": a["oc"], "red": a["rc"]}
    LIGHT.update(head_diffuse=a["hd"], ambient=a["amb"], diffuse=a["dif"], shadow=a["sh"], diffuse2=a["dif2"])


def quat_of(R):
    q = np.empty(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel()); return q


def _fmt(v): return " ".join(f"{x:.6f}" for x in np.asarray(v).ravel())


def build_arm_xml(gripper_extra_xml="", compliance=None):
    """compliance: {"shoulder_lift": K, "elbow_flex": K} N·m/rad — 인코더에 안 보이는 수동 스프링 관절(같은 축)을 구동 관절 뒤에 추가"""
    """so101.xml → so101_phys.xml: 메시는 시각 전용, 관절 감쇠, 패드 충돌 상자, 위치 구동기."""
    gg = GripperGeom()
    lever_dir = gg.lever / np.linalg.norm(gg.lever); axis = gg.axis / np.linalg.norm(gg.axis)
    n_open = np.cross(axis, lever_dir); n_open /= np.linalg.norm(n_open)          # 작은 각에서 움직이는 턱 끝이 가는 방향(열림)
    Rp = np.column_stack([lever_dir, n_open, axis])                                 # 패드 상자 축: x=턱 길이, y=무는 방향, z=회전축
    if np.linalg.det(Rp) < 0: Rp[:, 2] *= -1
    tip = gg.tip_gl
    fixed_c = tip - PAD_INSET * lever_dir - (PAD_T / 2) * n_open                   # 고정 턱 안쪽 면: 접촉면이 끝선(θ=0에서 두 면이 만남), 판 몸체는 바깥쪽
    moving_c = tip - PAD_INSET * lever_dir + (PAD_T / 2 + CLOSED_GAP) * n_open     # 움직이는 턱 안쪽 면(열림 방향 쪽에 몸체). 완전 닫힘에서도 실물은 약 5 mm 틈(09-30 본인 확인)
    # 움직이는 턱 몸체 좌표계로 변환
    src = open(os.path.join(HERE, "so101.xml")).read()
    mb = re.search(r'<body name="moving_jaw_so101_v1_link" pos="([^"]+)" quat="([^"]+)">', src)
    p_b = np.array(mb.group(1).split(), float); q_b = np.array(mb.group(2).split(), float)
    R_b = np.empty(9); mujoco.mju_quat2Mat(R_b, q_b); R_b = R_b.reshape(3, 3)
    assert abs(R_b[:, 2] @ axis) > 0.99, "그리퍼 관절축이 URDF 축과 다름"
    moving_local = R_b.T @ (moving_c - p_b); Rp_local = R_b.T @ Rp
    size = _fmt([PAD_LEN / 2, PAD_T / 2, PAD_W / 2])
    pad_f = f'<geom name="pad_fixed" type="box" size="{size}" pos="{_fmt(fixed_c)}" quat="{_fmt(quat_of(Rp))}" rgba="0.2 0.2 0.8 0.6" friction="1.0 0.01 0.001" contype="1" conaffinity="1"/>'
    pad_m = f'<geom name="pad_moving" type="box" size="{size}" pos="{_fmt(moving_local)}" quat="{_fmt(quat_of(Rp_local))}" rgba="0.2 0.2 0.8 0.6" friction="1.0 0.01 0.001" contype="1" conaffinity="1"/>'
    x = src.replace('<mujoco model="so101_new_calib">', '<mujoco model="so101_phys">\n  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>')
    x = re.sub(r'(<geom [^>]*type="mesh"[^>]*)/>', r'\1 contype="0" conaffinity="0" group="1"/>', x)          # 메시는 보기만
    x = re.sub(r'<joint name="(shoulder_pan|shoulder_lift|elbow_flex|wrist_flex|wrist_roll)"', r'<joint name="\1" damping="0.3" armature="0.01"', x)
    x = x.replace('<joint name="gripper"', '<joint name="gripper" damping="0.05" armature="0.002"')
    if compliance:
        for jn, K in compliance.items():
            if K and K > 0:
                x = re.sub(r'(<joint name="%s"[^>]*/>)' % jn, r'\1\n        <joint name="%s_c" stiffness="%.2f" damping="0.3" armature="0.001" range="-0.35 0.35"/>' % (jn, K), x)
    x = x.replace('<body name="moving_jaw_so101_v1_link"', pad_f + '\n              ' + gripper_extra_xml + '\n              <body name="moving_jaw_so101_v1_link"')
    x = x.replace('<geom pos="0 0 0.0189" quat="1 0 0 0" type="mesh" rgba="1 0.82 0.12 1" mesh="moving_jaw_so101_v1" contype="0" conaffinity="0" group="1"/>',
                  '<geom pos="0 0 0.0189" quat="1 0 0 0" type="mesh" rgba="1 0.82 0.12 1" mesh="moving_jaw_so101_v1" contype="0" conaffinity="0" group="1"/>\n                ' + pad_m)
    assert "pad_moving" in x and "pad_fixed" in x
    x = x.replace('rgba="1 0.82 0.12 1"', 'rgba="0.13 0.13 0.13 1"')                                          # 실물 팔은 검정 출력물
    act = "\n  <actuator>\n"
    for j, lim in [("shoulder_pan", "-1.91986 1.91986"), ("shoulder_lift", "-1.74533 1.74533"), ("elbow_flex", "-1.69 1.69"), ("wrist_flex", "-1.65806 1.65806"), ("wrist_roll", "-2.74385 2.84121")]:
        act += f'    <position name="{j}" joint="{j}" kp="__KP__" kv="{KV}" ctrlrange="{lim}" forcerange="-{FORCE} {FORCE}"/>\n'
    act += f'    <position name="gripper" joint="gripper" kp="{KP_GRIP}" kv="0.1" ctrlrange="-0.174533 1.74533" forcerange="-{FORCE_GRIP} {FORCE_GRIP}"/>\n  </actuator>\n</mujoco>'
    x = x.replace("</mujoco>", act)
    return x


class PhysSim:
    ROLL_FIX_DEG = -36.3    # 손목 회전 영점 보정(도): URDF 턱 방향이 실물과 ~36° 어긋남(09-29 jaw_angle_offset). 0이면 기존 보정 그대로
    JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    KP_SCALE = 1.0; KP_GRIP_SCALE = 1.0; FRICTION_SCALE = 1.0; KP_OVERRIDE = None; KV_OVERRIDE = None
    STIFF_CUBE = ' solref="-50000 -300" priority="1"'; CUBE_CONTACT = ""   # CUBE_CONTACT=STIFF_CUBE로 단단한 큐브. 큐브 접촉 강성: 기본(0.02)이면 턱 판이 큐브에 5~6 mm 파고들어 읽기값이 12→4로 떨어지고 큐브가 서서히 빠짐(10-01 발견). 실물 나무 큐브는 단단함      # 맞춤 동역학에 곱하는 배율(요인 분리 실험, sim_policy_eval --kp-scale 등)
    def __init__(self, cubes, kp=None, extra_xml="", gripper_extra_xml="", compliance="auto"):
        """cubes: {"orange": (gx, gy, angle_deg), ...} 격자 mm. 큐브는 자유 물체, 책상은 격자 평면(z=0)."""
        self.k = Kin(); self.gg = GripperGeom(); self.kp = kp
        dj = os.path.expanduser("~/robotics/calib/sim_dynamics.json"); dyn = json.load(open(dj)) if os.path.exists(dj) else {}
        if compliance == "auto": compliance = dyn.get("compliance")           # 예: {"shoulder_lift": 70, "elbow_flex": 33}
        self.compliance = compliance
        arm = build_arm_xml(gripper_extra_xml, compliance).replace("__KP__", str(kp if kp is not None else KP_DEFAULT))
        if self.CUBE_CONTACT: arm = arm.replace('impratio="10"/>', 'impratio="10" noslip_iterations="10"/>')   # 단단한 큐브 모드: 쥔 큐브가 서서히 빠지는 MuJoCo 마찰 크리프 억제
        open(os.path.join(HERE, "so101_phys.xml"), "w").write(arm)
        qg = quat_of(self.k.R)
        def box(name, gxy, size_mm, z_bottom, rgba, col):
            sx, sy, sz = size_mm; p = self.k.grid_to_base((gxy[0], gxy[1], z_bottom + sz / 2)) / 1000.0
            return (f'<geom name="{name}" type="box" size="{sx/2000} {sy/2000} {sz/2000}" pos="{_fmt(p)}" quat="{_fmt(qg)}" rgba="{rgba}" '
                    f'contype="{1 if col else 0}" conaffinity="{1 if col else 0}" friction="0.8 0.01 0.001"/>')
        geoms = [box("table", (105, 148.5), (1200, 1200, 20), -20, TABLE_RGBA, True)]
        bodies = ""
        COL = dict(CUBE_RGBA)
        for name, (gx, gy, ang) in cubes.items():
            Rz = np.array([[np.cos(np.deg2rad(ang)), -np.sin(np.deg2rad(ang)), 0], [np.sin(np.deg2rad(ang)), np.cos(np.deg2rad(ang)), 0], [0, 0, 1]])
            p = self.k.grid_to_base((gx, gy, CUBE * 500)) / 1000.0; q = quat_of(self.k.R @ Rz)
            fr = "1.2 0.01 0.001" if name == "red" else "0.7 0.01 0.001"      # 빨강 바닥 미끄럼 방지 테이프
            bodies += (f'<body name="{name}" pos="{_fmt(p)}" quat="{_fmt(q)}"><freejoint name="{name}_free"/>'
                       f'<geom name="{name}" type="box" size="{CUBE/2} {CUBE/2} {CUBE/2}" mass="{CUBE_MASS}" rgba="{COL.get(name, "0.5 0.5 0.5 1")}" friction="{fr}" contype="1" conaffinity="1"{self.CUBE_CONTACT}/></body>\n')
        scene = f"""<mujoco model="so101_phys_scene">
  <include file="so101_phys.xml"/>
  <visual><headlight diffuse="{LIGHT['head_diffuse']} {LIGHT['head_diffuse']} {LIGHT['head_diffuse']}" ambient="{LIGHT['ambient']} {LIGHT['ambient']} {LIGHT['ambient']}"/><global azimuth="135" elevation="-35" offwidth="1280" offheight="960"/></visual>
  <worldbody>
    <light pos="{LIGHT['pos']}" dir="{LIGHT['dir']}" diffuse="{LIGHT['diffuse']} {LIGHT['diffuse']} {LIGHT['diffuse']}" castshadow="{LIGHT['shadow']}"/>
    <light pos="{LIGHT['pos2']}" dir="{LIGHT['dir2']}" diffuse="{LIGHT['diffuse2']} {LIGHT['diffuse2']} {LIGHT['diffuse2']}" castshadow="false"/>
    {chr(10).join(geoms)}
    {bodies}{extra_xml}
  </worldbody>
</mujoco>"""
        self.scene_path = os.path.join(HERE, "phys_scene.xml"); open(self.scene_path, "w").write(scene)
        self.m = mujoco.MjModel.from_xml_path(self.scene_path); self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.joint(n).qposadr[0] for n in self.JOINTS]); self.vadr = np.array([self.m.joint(n).dofadr[0] for n in self.JOINTS])
        self.cubes = list(cubes.keys()); self.renderer = None
        self.m.opt.timestep = 1.0 / (30 * 16); self.delay = 0; self._cq = None
        if kp is None and dyn:                     # 실물 시연 재생으로 맞춘 관절별 계수(sim_sysid.py). kp를 직접 주면 예전처럼 균일 게인
            j = dyn; self.delay = int(j["delay_frames"])
            for i, nm in enumerate(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]):
                kp_i = j["kp"][i] * (self.KP_SCALE if i < 5 else self.KP_GRIP_SCALE)      # 요인 분리 실험용 배율(기본 1)
                if self.KP_OVERRIDE is not None and i < 5 and self.KP_OVERRIDE[i] is not None: kp_i = self.KP_OVERRIDE[i]   # 관절별 직접 지정
                kv_i = j["kv"][i]
                if self.KV_OVERRIDE is not None and i < 5 and self.KV_OVERRIDE[i] is not None: kv_i = self.KV_OVERRIDE[i]
                self.m.actuator_gainprm[i, 0] = kp_i; self.m.actuator_biasprm[i, 1] = -kp_i; self.m.actuator_biasprm[i, 2] = -kv_i
                dof = self.m.joint(nm).dofadr[0]
                if "frictionloss" in j: self.m.dof_frictionloss[dof] = j["frictionloss"][i] * self.FRICTION_SCALE
                if "damping" in j: self.m.dof_damping[dof] = j["damping"][i]
        self.set_state(HOME); self.set_ctrl(HOME); mujoco.mj_forward(self.m, self.d)

    # ---- 읽기값 ↔ qpos ----
    def _q_of(self, r6):
        u = self.k.reading_to_urdf(r6[:5]); u[4] += self.ROLL_FIX_DEG
        return np.r_[np.deg2rad(u), np.deg2rad(DEG_PER_PCT * (r6[5] - CLOSED_READING))]
    def set_state(self, r6): self.d.qpos[self.qadr] = self._q_of(r6); self.d.qvel[self.vadr] = 0
    def set_ctrl(self, r6): self.d.ctrl[:6] = self._q_of(r6)
    def readings(self):
        q = self.d.qpos[self.qadr]
        u = np.rad2deg(q[:5]).copy(); u[4] -= self.ROLL_FIX_DEG
        return np.r_[self.k.urdf_to_reading(u), np.rad2deg(q[5]) / DEG_PER_PCT + CLOSED_READING]
    def tip_grid(self, extra=None): return self.k.fk_grid(self.readings()[:5], tip_extra_mm=extra)

    def hidden_sag_mm(self, p=(90, 160, 78)):
        """수동 스프링 관절로 인한 추가 처짐: 명령 자세 정착 후 (인코더 FK z) − (실제 턱 끝 z)"""
        q, _ = self.ik(p, -50, 35); self.set_state(np.r_[q, 35]); self.set_ctrl(np.r_[q, 35]); self.settle(2.0)
        z_enc = self.tip_grid(self.extra(35))[2]
        gid = self.m.geom("pad_fixed").id; c = self.d.geom_xpos[gid]; R = self.d.geom_xmat[gid].reshape(3, 3); sz = self.m.geom_size[gid]
        z_true = min(self.k.base_to_grid((c + R @ np.array([sx, sy, sz_])) * 1000)[2] for sx in (-sz[0], sz[0]) for sy in (-sz[1], sz[1]) for sz_ in (-sz[2], sz[2]))
        return z_enc, z_true
    # ---- 물리 ----
    def tick(self, r6, substeps=16):
        """제어 1틱(1/30 s): 명령을 지연 큐에 넣고(실물 명령→모터 지연) 가장 오래된 명령을 구동기에 적용한 뒤 물리 진행"""
        q = self._q_of(r6)
        if self._cq is None or self._cq.maxlen != self.delay + 1: self._cq = deque([q] * (self.delay + 1), maxlen=self.delay + 1)
        self._cq.append(q); self.d.ctrl[:6] = self._cq[0]; mujoco.mj_step(self.m, self.d, nstep=substeps)
    def step(self, n=1):
        for _ in range(n): mujoco.mj_step(self.m, self.d)
    def settle(self, sec=0.5): self._cq = None; self.step(int(sec / self.m.opt.timestep))
    def goto(self, r6_target, max_deg_s=25.0, settle=0.5):
        """실물 move_to처럼 코사인 보간으로 목표 읽기값까지 명령, 도착 후 settle초 정지. 반환: 실제 읽기값."""
        r0 = self.readings(); r0[5] = np.rad2deg(self.d.ctrl[5]) / DEG_PER_PCT + CLOSED_READING; rt = np.asarray(r6_target, float)
        dur = max(0.5, np.abs(rt[:5] - r0[:5]).max() / max_deg_s); n = int(dur / self.m.opt.timestep)
        for i in range(1, n + 1):
            self.set_ctrl(r0 + (rt - r0) * (0.5 - 0.5 * np.cos(np.pi * i / n))); self.step()
        self.settle(settle); return self.readings()
    def cube_grid(self, name):
        """큐브 중심 격자 좌표 mm와 격자 z축 대비 기울기(도)"""
        b = self.m.body(name).id; p = self.k.base_to_grid(self.d.xpos[b] * 1000.0)
        R = self.d.xmat[b].reshape(3, 3); up = self.k.R.T @ R[:, 2]
        return p, float(np.rad2deg(np.arccos(np.clip(abs(up[2]), -1, 1))))
    def pad_gap_mm(self): return self.pad_geometry(self.readings()[5])[0]
    def contacts(self, name):
        gid = self.m.geom(name).id; out = set()
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if gid in (c.geom1, c.geom2): out.add(mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2 if c.geom1 == gid else c.geom1))
        return sorted(out)
    def render_cam(self, cam_name, w=640, h=480):
        """이름 있는 카메라로 렌더 (RGB uint8, h×w×3)"""
        if getattr(self, "_cam_renderer", None) is None or self._cam_renderer.width != w: self._cam_renderer = mujoco.Renderer(self.m, h, w)
        self._cam_renderer.update_scene(self.d, camera=cam_name); return self._cam_renderer.render().copy()
    def snapshot(self, path, lookat=(100, 150, 20)):
        if self.renderer is None: self.renderer = mujoco.Renderer(self.m, 720, 960)
        cam = mujoco.MjvCamera(); cam.azimuth, cam.elevation, cam.distance = 135, -35, 0.7; cam.lookat[:] = self.k.grid_to_base(lookat) / 1000.0
        self.renderer.update_scene(self.d, cam); import PIL.Image as I; I.fromarray(self.renderer.render()).save(path)

    # ---- 계획 도우미 (pick_stack과 같은 기준점: 두 패드 중간점) ----
    def pad_geometry(self, grip_reading):
        """시뮬 모델 자체의 패드 기하: (패드 안쪽 면 간격 mm, 두 면 중간점의 gripper_frame_link 기준 오프셋 mm). 현재 상태는 건드리지 않음."""
        if not hasattr(self, "_scratch"): self._scratch = mujoco.MjData(self.m)
        d = self._scratch; d.qpos[:] = 0; d.qpos[self.qadr[5]] = np.deg2rad(DEG_PER_PCT * (grip_reading - CLOSED_READING)); mujoco.mj_kinematics(self.m, d)
        gl = self.m.body("gripper_link").id; Rg = d.xmat[gl].reshape(3, 3); pg = d.xpos[gl]
        pf, pm = self.m.geom("pad_fixed").id, self.m.geom("pad_moving").id
        Rf, Rm = d.geom_xmat[pf].reshape(3, 3), d.geom_xmat[pm].reshape(3, 3)
        faces_f = [d.geom_xpos[pf] + sgn * Rf[:, 1] * PAD_T / 2 for sgn in (1, -1)]; faces_m = [d.geom_xpos[pm] + sgn * Rm[:, 1] * PAD_T / 2 for sgn in (1, -1)]
        gap, a, b = min(((np.linalg.norm(a - b), a, b) for a in faces_f for b in faces_m), key=lambda t: t[0])
        mid_gl = Rg.T @ ((a + b) / 2 - pg) + PAD_INSET * (self.gg.lever / np.linalg.norm(self.gg.lever))   # 면 중앙의 중간점을 턱 끝 높이로 옮김(기준점 = 턱 끝 사이 중간)
        return gap * 1000.0, (self.gg.Rf.T @ (mid_gl - self.gg.pf)) * 1000.0
    def extra(self, grip_reading): return self.pad_geometry(grip_reading)[1]
    def reading_for_pad_gap(self, gap_mm):
        lo, hi = CLOSED_READING, 100.0
        for _ in range(30):
            mid = (lo + hi) / 2; lo, hi = (mid, hi) if self.pad_geometry(mid)[0] < gap_mm else (lo, mid)
        return (lo + hi) / 2
    def ik(self, p_grid, roll, grip_reading, q0=None):
        q, e = self.k.ik_grid(p_grid, roll_urdf_deg=roll, q0_reading=q0, tip_extra_mm=self.extra(grip_reading)); return q, e
    def roll_for_open_dir(self, p_grid, grip_reading, want_grid_dir=(1, 0, 0)):
        """열림 방향(n_open)이 격자 방향 want와 나란해지는 손목 회전 후보(URDF 도) 중 IK 잔차가 작은 것"""
        gg = self.gg; lever_dir = gg.lever / np.linalg.norm(gg.lever); n_open = np.cross(gg.axis / np.linalg.norm(gg.axis), lever_dir)
        best = None
        for roll in range(-180, 180, 5):
            q, e = self.ik(p_grid, roll, grip_reading)
            if e > 3: continue
            q_true = self.k.urdf_to_reading(self.k.reading_to_urdf(q) + np.array([0, 0, 0, 0, self.ROLL_FIX_DEG]))   # 시뮬 턱은 roll 영점 보정만큼 더 돌아가 있음
            _, Rm = self.k.fk_base(q_true); n_w = self.k.R.T @ (Rm @ (gg.Rf @ n_open))     # gripper_frame_link 기준 벡터를 격자로
            score = abs(n_w @ np.asarray(want_grid_dir, float))
            if best is None or score > best[0]: best = (score, roll, q, e)
        return best


def run_lift(kp, png):
    HOVER, GRASP_Z, PLACE_Z, OPEN, CLOSE = 60.0, 3.0, 48.0, 35.0, 12.0     # 기준점 = 턱 끝 사이 중간: 잡기는 턱 끝이 책상 3 mm 위, 놓기는 손안 큐브 높이로 보정
    ox, oy, rx, ry = 90.0, 160.0, 130.0, 200.0
    s = PhysSim({"orange": (ox, oy, 0), "red": (rx, ry, 0)}, kp=kp)
    s.settle(0.5)
    print(f"kp={kp}  초기 큐브: 주황 {np.round(s.cube_grid('orange')[0],1)}  빨강 {np.round(s.cube_grid('red')[0],1)}")
    sc, roll, q, e = s.roll_for_open_dir((ox, oy, GRASP_Z + HOVER), OPEN, (1, 0, 0))
    print(f"손목 회전 선택: roll {roll}° (열림 방향·격자x 정렬 {sc:.2f}, IK 잔차 {e:.1f} mm)")
    r30 = s.reading_for_pad_gap(30.0); print(f'30 mm 큐브 물림 읽기값(시뮬 기하) {r30:.0f}, 열림 35 패드 간격 {s.pad_geometry(35)[0]:.1f} mm')
    plan = [("주황 위 대기", (ox, oy, GRASP_Z + HOVER), OPEN), ("주황 잡기 높이", (ox, oy, GRASP_Z), OPEN), ("닫기", None, CLOSE),
            ("들어올리기", (ox, oy, GRASP_Z + HOVER), CLOSE), ("빨강 위 대기", (rx, ry, PLACE_Z + 40), CLOSE), ("놓기 높이", (rx, ry, PLACE_Z), CLOSE),
            ("열기", None, OPEN), ("올리기", (rx, ry, PLACE_Z + 40), OPEN)]
    q_prev = None; held = False; in_hand = np.zeros(3)
    for i, (name, p, g) in enumerate(plan):
        if p is not None and name in ('빨강 위 대기', '놓기 높이', '올리기'):
            p = np.asarray(p, float); p[:2] -= in_hand[:2]
            if name == '놓기 높이': p[2] = 45.0 + 2.0 - in_hand[2]      # 큐브 중심이 빨강 위 2 mm 여유에 오도록
            p = tuple(p)
        if p is None:
            tgt = np.r_[s.readings()[:5], g]; extra = s.extra(r30 if g == CLOSE else g)
            # 닫기/열기는 관절 유지, 그리퍼만
            tgt[:5] = np.rad2deg(s.d.ctrl[:5]); tgt[:5] = s.k.urdf_to_reading(tgt[:5])
        else:
            extra = s.extra(r30 if g == CLOSE else g)
            qq, e = s.ik(p, roll, r30 if g == CLOSE else g, q0=q_prev); q_prev = qq; tgt = np.r_[qq, g]
        got = s.goto(tgt, settle=0.6)
        tip = s.tip_grid(extra); tgt_p = np.asarray(p) if p is not None else tip
        o, tilt = s.cube_grid("orange"); r_, _ = s.cube_grid("red")
        cont = s.contacts("orange")
        if name == "들어올리기": held = ("pad_fixed" in cont and "pad_moving" in cont)
        if name == "들어올리기": in_hand = o - tip; print(f"   손안 큐브 오프셋(기준점 대비) {np.round(in_hand,1)} mm → 놓기 목표에 반영")
        print(f"{i+1}. {name:8s} 기준점 {np.round(tip,1)} (목표 {np.round(tgt_p,0)}, 오차 {np.round(tip - tgt_p,1)})  그리퍼 읽기 {got[5]:.0f} 패드간격 {s.pad_gap_mm():.1f} mm  "
              f"주황 {np.round(o,1)} 기울기 {tilt:.0f}°  접촉 {cont}")
        if png: s.snapshot(f"{png}_{i+1}_{name}.png")
    o, tilt = s.cube_grid("orange"); r_, _ = s.cube_grid("red")
    on_top = abs(o[2] - 45) < 6 and np.linalg.norm(o[:2] - r_[:2]) < 12 and tilt < 10
    print(f"결과: 주황 {np.round(o,1)} 빨강 {np.round(r_,1)} → {'쌓기 성공' if on_top else '실패'} (잡힘 판정: {held})")


def run_sag(kp):
    s = PhysSim({}, kp=kp)
    poses = {"멀리 대기": (90, 230, 78), "멀리 잡기": (90, 230, 18), "중간 대기": (90, 160, 78), "중간 잡기": (90, 160, 18)}
    print(f"kp={kp}: 실물 P16 z처짐 8.7/1.6/12.0/-0.4, P48 3.4/1.4/3.4/1.1 mm (09-29 gain_tests)")
    for name, p in poses.items():
        q, e = s.ik(p, -50, 35); s.set_state(np.r_[q, 35]); s.set_ctrl(np.r_[q, 35]); s.settle(1.5)
        tip = s.tip_grid(s.extra(35)); print(f"  {name}: 목표 {p} → 실제 {np.round(tip,1)}  z 처짐 {p[2]-tip[2]:.1f} mm  xy 오차 {np.round(tip[:2]-np.array(p[:2]),1)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", default="lift"); ap.add_argument("--kp", type=float, default=KP_DEFAULT); ap.add_argument("--png")
    a = ap.parse_args()
    if a.test == "lift": run_lift(a.kp, a.png)
    elif a.test == "sag": run_sag(a.kp)
