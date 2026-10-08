#!/usr/bin/env python3
"""SO-101 역기구학 + 좌표 변환 (보정값 적용).
- 관절 약속: urdf_deg = sign * reading + offset  (calib/kinematic_calibration.json)
- 격자(mm) → 베이스(mm): p_base = R @ p_grid + t
- IK: 목표 손끝 위치(베이스 mm) + 그리퍼가 아래를 향하는 자세(기울기 pitch_deg, 0 = 수직) → 관절 읽기값(도)
     수치해: scipy least_squares, 위치 3식 + 그리퍼 축 방향 2식, 관절 5개.
사용 예:
    from so101_ik import Kin; k = Kin()
    q = k.ik_grid((90, 160, 20))          # 격자 (90,160) 위 20 mm, 그리퍼 수직
    k.fk_grid(q)                          # 확인: 관절 읽기값 → 격자 좌표
"""
import json, os
import numpy as np
from scipy.optimize import least_squares
from so101_fk import SO101FK

CAL = os.path.expanduser("~/robotics/calib/kinematic_calibration.json")
# URDF 관절 한계 (rad → deg), URDF 각도 기준
URDF_LIM = {"shoulder_pan": (-110, 110), "shoulder_lift": (-100, 100), "elbow_flex": (-96.8, 96.8),
            "wrist_flex": (-95, 95), "wrist_roll": (-157, 162)}
JOINTS5 = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def wrap(a): return (np.asarray(a) + 180) % 360 - 180


class Kin:
    def __init__(self, cal_path=CAL):
        d = json.load(open(cal_path))
        self.sign = np.array(d["sign"], float); self.off = np.array(d["offset_deg"], float)
        self.tip = np.array(d["tip_offset_mm"], float) / 1000.0
        self.R = np.array(d["R"]); self.t = np.array(d["t_mm"], float)
        self.fk = SO101FK()

    # ---- 좌표 변환 ----
    def grid_to_base(self, p): return self.R @ np.asarray(p, float) + self.t
    def base_to_grid(self, p): return self.R.T @ (np.asarray(p, float) - self.t)

    # ---- 관절 약속 ----
    def reading_to_urdf(self, q5): return wrap(self.sign * np.asarray(q5, float) + self.off)
    def urdf_to_reading(self, u5): return wrap((np.asarray(u5, float) - self.off) / self.sign)

    # ---- FK (읽기값 → 베이스 mm, 그리퍼 z축) ----
    def fk_base(self, q5_reading, tip_extra_mm=None):
        u = self.reading_to_urdf(q5_reading)
        T = self.fk.fk_deg(np.r_[u, 0.0])
        tip = self.tip + (np.zeros(3) if tip_extra_mm is None else np.asarray(tip_extra_mm, float) / 1000.0)
        p = (T[:3, :3] @ tip + T[:3, 3]) * 1000.0
        return p, T[:3, :3]

    def fk_grid(self, q5_reading, tip_extra_mm=None):
        p, Rm = self.fk_base(q5_reading, tip_extra_mm)
        return self.base_to_grid(p)

    # ---- IK ----
    def ik_base(self, p_target_mm, pitch_deg=0.0, roll_urdf_deg=0.0, q0_reading=None, tip_extra_mm=None):
        """p_target: 베이스 mm. 그리퍼 접근 방향: 수직 아래에서 pitch_deg만큼 로봇 쪽으로 기울임.
        반환: 관절 읽기값 5개(도), 잔차(mm)."""
        p_t = np.asarray(p_target_mm, float)
        tip = self.tip + (np.zeros(3) if tip_extra_mm is None else np.asarray(tip_extra_mm, float) / 1000.0)   # 손끝 대신 다른 기준점(예: 두 턱 중간점)
        # 원하는 그리퍼 접근축(베이스 좌표): 아래(-z)에서 pitch만큼 기울인 방향. 팔이 향하는 수평 방향을 기준으로 기울임.
        yaw = np.arctan2(p_t[1], p_t[0] - 38.8)  # 팬 축 기준 방향
        horiz = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        approach = -np.array([0, 0, 1.0]) * np.cos(np.deg2rad(pitch_deg)) + horiz * np.sin(np.deg2rad(pitch_deg))
        # URDF 그리퍼 프레임에서 손끝이 뻗는 축: 손목 롤 축(-x of wrist_roll frame) → gripper_frame의 어느 축인지 FK로 확인:
        # 확인 결과(2026-09-23): 0° 자세에서 gripper_frame의 +z축이 베이스 +x(손끝이 뻗는 방향)이고,
        # 수직으로 댄 기록에서 +z가 테이블을 향함 → 손끝 방향 = +z_gripper
        def resid(u5):
            T = self.fk.fk_deg(np.r_[u5, 0.0])
            p = (T[:3, :3] @ tip + T[:3, 3]) * 1000.0
            fwd = T[:3, :3][:, 2]
            return np.r_[(p - p_t) / 10.0, (fwd - approach) * 5.0, (u5[4] - roll_urdf_deg) / 50.0]
        lo = np.array([URDF_LIM[j][0] for j in JOINTS5]); hi = np.array([URDF_LIM[j][1] for j in JOINTS5])
        best = None
        starts = [self.reading_to_urdf(q0_reading)] if q0_reading is not None else []
        starts += [np.array([np.degrees(yaw), 30, 60, 30, roll_urdf_deg]), np.array([np.degrees(yaw), 60, 30, 0, roll_urdf_deg]),
                   np.array([np.degrees(yaw), 10, 80, 10, roll_urdf_deg])]
        for s in starts:
            s = np.clip(s, lo + 1, hi - 1)
            sol = least_squares(resid, s, bounds=(lo, hi), max_nfev=2000)
            T = self.fk.fk_deg(np.r_[sol.x, 0.0]); p = (T[:3, :3] @ tip + T[:3, 3]) * 1000.0
            err = np.linalg.norm(p - p_t)
            if best is None or err < best[1]:
                best = (sol.x, err)
        u, err = best
        return self.urdf_to_reading(u), err

    def ik_grid(self, p_grid_mm, **kw):
        return self.ik_base(self.grid_to_base(p_grid_mm), **kw)


if __name__ == "__main__":
    k = Kin()
    rows = np.loadtxt(os.path.expanduser("~/robotics/calib/touch_points.csv"), delimiter=",", skiprows=1, ndmin=2)
    print("보정점 재현: 기록된 관절 → FK → 격자 좌표 (정답과 비교)")
    for r in rows:
        g = r[0:3]; q = r[6:11]; p = k.fk_grid(q)
        print(f"  정답 {np.round(g,0)}  FK {np.round(p,1)}  오차 {np.linalg.norm(p-g):.1f} mm")
    print("\nIK 왕복: 격자 목표 → 관절 → FK")
    for tgt in [(90, 160, 20), (60, 140, 0), (120, 200, 30), (40, 220, 10)]:
        q, err = k.ik_grid(tgt)
        print(f"  목표 {tgt} → 관절(읽기값) {np.round(q,1)}  IK 잔차 {err:.2f} mm  FK 확인 {np.round(k.fk_grid(q),1)}")
