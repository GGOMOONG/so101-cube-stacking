#!/usr/bin/env python3
"""그리퍼 턱 기하: 보정된 '손끝'(닫은 상태의 두 턱 끝점)과 URDF 턱 회전축으로, 턱을 r % 열었을 때
움직이는 턱 끝의 위치·두 턱 간격·두 턱 중간점을 gripper_frame_link 기준으로 계산한다.
잡기 목표는 손끝이 아니라 두 턱의 중간점을 큐브 중심에 맞춰야 한다(09-29 첫 실물 실행에서 고정 턱이 큐브 위에 얹힌 원인).
    from gripper_geom import GripperGeom; g = GripperGeom()
    g.gap_mm(40)            # 40 % 열림에서 턱 간격
    g.mid_offset_mm(40)     # 손끝 기준 중간점 오프셋 (gripper_frame_link, mm) → IK의 tip_extra_mm
    g.reading_for_gap(30)   # 30 mm 큐브를 물었을 때의 읽기값(간격 30 mm)
"""
import os, re, json
import numpy as np
from so101_fk import rpy_to_R

URDF = os.path.expanduser("~/robotics/so101_model/so101_new_calib.urdf")
CAL = os.path.expanduser("~/robotics/calib/kinematic_calibration.json")
# 09-29 실측(자, 턱 끝 사이): 읽기값 2 → 20 mm, 40 → 80 mm, 80 → 130 mm.
# 모터 1 %당 1.275°(범위 1451틱=127.5°), URDF 턱 길이 79 mm로 계산하면 18/79/129 mm → 일치. 읽기값 2에서 이미 13.5° 벌어져 있음(완전 닫힘은 −8.6 상당).
CLOSED_READING = -8.6   # 두 턱 끝이 만나는(가상) 읽기값. 실제 최소 읽기값 0에서도 약 15 mm 간격이 남음
DEG_PER_PCT = 1.275     # 읽기값 1 %당 턱 회전각
R_TIP_MM = 79.1         # 턱 회전축 → 턱 끝 유효 길이 (URDF 손끝 기준)
PAD_INSET_MM = 20.0     # (09-30 사진으로 근거 없음이 확인됨: 실물엔 패드가 없고 완전 닫힘에서 턱 끝이 만남. 실물 파이프라인은 grasp_center.json 실측 오프셋 + 손목 서보로 보정되어 있어 값은 유지. 시뮬(sim_phys.py)은 자체 기하 사용)
OPEN_SIGN = +1.0        # 턱이 열리는 회전 방향 (실물 확인으로 조정: 움직이는 턱이 격자 +x 쪽으로 열리면 +1)


def _rot(axis, th):
    k = axis / np.linalg.norm(axis); K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class GripperGeom:
    def __init__(self):
        u = open(URDF).read()
        j = re.search(r'<joint name="gripper" type="revolute">(.*?)</joint>', u, re.S).group(1)
        o = re.search(r'origin xyz="([^"]+)" rpy="([^"]+)"', j); ax = re.search(r'axis xyz="([^"]+)"', j).group(1)
        self.piv = np.array(o.group(1).split(), float)                                  # 턱 회전축 원점 (gripper_link)
        self.axis = rpy_to_R(*np.array(o.group(2).split(), float)) @ np.array(ax.split(), float)
        self.Rf = rpy_to_R(0, 3.14159, 0); self.pf = np.array([-0.0079, -0.000218121, -0.0981274])   # gripper_link → gripper_frame_link
        self.tip = np.array(json.load(open(CAL))["tip_offset_mm"]) / 1000.0
        self.tip_gl = self.Rf @ self.tip + self.pf                                      # 닫힌 손끝 (gripper_link)
        lever = self.tip_gl - self.piv; lever -= self.axis * (lever @ self.axis)         # 축에 수직인 성분만
        self.lever = lever / np.linalg.norm(lever) * (R_TIP_MM / 1000.0)               # 실측 유효 길이로
        self.closed_reading = CLOSED_READING

    def _theta(self, r): return OPEN_SIGN * np.deg2rad(max(0.0, r - CLOSED_READING) * DEG_PER_PCT)

    def _moving_disp_gl(self, r):
        """닫힘 대비 움직이는 턱 끝의 변위 (gripper_link, m)"""
        return (_rot(self.axis, self._theta(r)) - np.eye(3)) @ self.lever

    def gap_mm(self, r): return float(np.linalg.norm(self._moving_disp_gl(r)) * 1000.0)          # 턱 끝 간격
    def pad_gap_mm(self, r): return max(0.0, self.gap_mm(r) - PAD_INSET_MM)                       # 패드(무는 면) 간격

    def mid_offset_mm(self, r):
        """손끝(=닫힌 두 턱 끝) 기준, 두 패드 사이 중간점 오프셋 (gripper_frame_link, mm). 패드 변위 = 끝 변위 × (패드 간격/끝 간격)"""
        d = self._moving_disp_gl(r); g = self.gap_mm(r)
        scale = (self.pad_gap_mm(r) / g) if g > 1e-6 else 0.0
        return (self.Rf.T @ (d * scale / 2.0)) * 1000.0

    def reading_for_pad_gap(self, gap_mm):
        lo, hi = self.closed_reading, 100.0
        for _ in range(40):
            m = (lo + hi) / 2; lo, hi = (m, hi) if self.pad_gap_mm(m) < gap_mm else (lo, m)
        return (lo + hi) / 2

    def reading_for_gap(self, gap_mm):
        lo, hi = self.closed_reading, 100.0
        for _ in range(40):
            m = (lo + hi) / 2; lo, hi = (m, hi) if self.gap_mm(m) < gap_mm else (lo, m)
        return (lo + hi) / 2


if __name__ == "__main__":
    g = GripperGeom()
    for r in [2, 20, 35, 40, 80]:
        print(f"열림 {r:3d}%: 끝 간격 {g.gap_mm(r):5.1f} / 패드 간격 {g.pad_gap_mm(r):5.1f} mm  패드 중간점 오프셋 {np.round(g.mid_offset_mm(r),1)} mm")
    r30 = g.reading_for_pad_gap(30); print(f"30 mm 큐브를 물면 읽기값 ≈ {r30:.0f}, 중간점 오프셋 {np.round(g.mid_offset_mm(r30),1)} mm")
