#!/usr/bin/env python3
"""MuJoCo 간섭 검사 모듈: 계획(관절 목표 목록)을 훑으며 팔 메시와 빨강 큐브의 최소 거리를 단계별로 돌려준다.
pick_stack.py가 손목 회전 후보를 고를 때 쓴다. 창은 열지 않는다.
    from sim_check import Checker; c = Checker(red_xy=(rx, ry))
    dists = c.sweep(q_start6, plan_names, qs6)   # [(단계 이름, 빨강과 최소 거리 mm)], 음수 = 관통
"""
import os, sys
import numpy as np
import mujoco
sys.path.insert(0, os.path.expanduser("~/robotics"))
from so101_ik import Kin

MAX_DEG_PER_S = 25.0
SIM_DIR = os.path.expanduser("~/robotics/sim")


class Checker:
    def __init__(self, red_xy, cube_mm=30.0):
        self.k = Kin(); k = self.k
        def quat(R):
            q = np.empty(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel()); return q
        p = k.grid_to_base((red_xy[0], red_xy[1], cube_mm / 2)) / 1000.0; q = quat(k.R); h = cube_mm / 2000
        xml = f"""<mujoco model="check"><include file="so101.xml"/><worldbody>
          <geom name="red" type="box" size="{h} {h} {h}" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" quat="{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}" rgba="0.85 0.1 0.1 1"/>
        </worldbody></mujoco>"""
        path = os.path.join(SIM_DIR, "check_scene.xml"); open(path, "w").write(xml)
        self.m = mujoco.MjModel.from_xml_path(path); self.d = mujoco.MjData(self.m)
        self.red = self.m.geom("red").id
        self.arm = [g for g in range(self.m.ngeom) if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]

    def _set(self, r6):
        self.d.qpos[:5] = np.deg2rad(self.k.reading_to_urdf(r6[:5])); self.d.qpos[5] = np.deg2rad(1.275 * (r6[5] + 8.6))
        mujoco.mj_forward(self.m, self.d)

    def _dist(self):
        best = 1e9; ft = np.zeros(6)
        for g in self.arm:
            dd = mujoco.mj_geomDistance(self.m, self.d, g, self.red, 0.2, ft)
            if dd < best: best = dd
        return best * 1000.0

    def sweep(self, q_start6, names, qs6, hz=10):
        out = []; q0 = np.asarray(q_start6, float)
        for name, qt in zip(names, qs6):
            qt = np.asarray(qt, float); dq = np.abs(qt - q0); n = max(4, int(max(0.8, dq[:5].max() / MAX_DEG_PER_S) * hz)); mn = 1e9
            for i in range(1, n + 1):
                self._set(q0 + (qt - q0) * (0.5 - 0.5 * np.cos(np.pi * i / n))); mn = min(mn, self._dist())
            out.append((name, mn)); q0 = qt
        return out
