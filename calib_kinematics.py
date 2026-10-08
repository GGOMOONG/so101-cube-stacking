#!/usr/bin/env python3
"""로봇 기구학 보정: 관절 영점 오프셋 + 그리퍼 끝 위치 + 격자→베이스 변환을 한꺼번에 최소제곱으로 푼다.
입력: ~/robotics/calib/touch_points.csv (격자 좌표 gx gy gz, 관절 각도 6개)
기준 자세(참고용 제약): A 앞으로 수평 = URDF lift +90, elbow −90 / B 위로 수직 = lift 0, elbow −90
출력: ~/robotics/calib/kinematic_calibration.json
   joint_deg_urdf = sign * reading + offset  (pan, lift, elbow, wflex, roll)
   tip_offset_mm  : gripper_frame_link 기준 실제 손가락 끝의 위치 (x,y,z)
   R, t           : p_base = R @ p_grid + t
"""
import os, json, itertools
import numpy as np
from scipy.optimize import least_squares
from so101_fk import SO101FK

CSV = os.path.expanduser("~/robotics/calib/touch_points.csv")
OUT = os.path.expanduser("~/robotics/calib/kinematic_calibration.json")
REF = {"A": ([1.8, -177.3, -84.0, 0.6, -94.4], [None, 90.0, -90.0, 0.0, None]),   # 읽은 값, URDF 값(None=제약 없음)
       "B": ([0.0, 92.2, -84.2, 0.9, -94.4], [None, 0.0, -90.0, 0.0, None])}
REF_W = 0.3   # 기준 자세 제약 가중치 (눈대중이라 약하게)

fk = SO101FK()
rows = np.loadtxt(CSV, delimiter=",", skiprows=1, ndmin=2)
G = rows[:, 0:3]; Q = rows[:, 6:12]
print(f"점 {len(G)}개 (높이 {sorted(set(G[:,2].tolist()))})")


def wrap(a): return (a + 180) % 360 - 180


def rotvec_to_R(v):
    th = np.linalg.norm(v)
    if th < 1e-12: return np.eye(3)
    k = v / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def tip_positions(sign, off, tip):
    P = []
    for q in Q:
        qq = np.r_[wrap(sign * q[:5] + off), q[5]]
        T = fk.fk_deg(qq)
        P.append((T[:3, :3] @ (tip / 1000.0) + T[:3, 3]) * 1000.0)
    return np.array(P)


def residuals(x, sign):
    off = np.r_[0.0, x[0:4]]          # pan offset은 베이스 회전과 겹치므로 0 고정
    tip = x[4:7]; rv = x[7:10]; t = x[10:13]
    P = tip_positions(sign, off, tip)
    Rm = rotvec_to_R(rv)
    r = ((Rm @ G.T).T + t - P).ravel()
    ref = []
    for k, (rd, ur) in REF.items():
        for i in range(1, 5):
            if ur[i] is not None:
                ref.append(REF_W * wrap(sign[i] * rd[i] + off[i] - ur[i]))
    return np.r_[r, ref]


best = None
for sp, sw, sr in itertools.product([1, -1], repeat=3):
    sign = np.array([sp, 1, 1, sw, sr], float)
    x0 = np.r_[-92.2, -5.8, -1.0 * sw, sr * 94.4, 0, 0, 0, 0, 0, 0, 200, 0, 0]
    sol = least_squares(residuals, x0, args=(sign,), loss="soft_l1", f_scale=3.0, max_nfev=4000)
    P = tip_positions(sign, np.r_[0, sol.x[0:4]], sol.x[4:7])
    Rm = rotvec_to_R(sol.x[7:10]); res = np.linalg.norm((Rm @ G.T).T + sol.x[10:13] - P, axis=1)
    print(f"signs(pan,wflex,roll)={(sp,sw,sr)}: 점 잔차 평균 {res.mean():.2f} 최대 {res.max():.2f} mm | off lift={sol.x[0]:.1f} elbow={sol.x[1]:.1f} wflex={sol.x[2]:.1f} roll={sol.x[3]:.1f} | tip={np.round(sol.x[4:7],1)}")
    if best is None or res.mean() < best[0]:
        best = (res.mean(), sign, sol, res, Rm)
m, sign, sol, res, Rm = best
print("\n채택:", "signs", sign.tolist(), "점별 잔차", np.round(res, 1))
out = {"sign": sign.tolist(), "offset_deg": [0.0, *sol.x[0:4].tolist()], "tip_offset_mm": sol.x[4:7].tolist(),
       "R": Rm.tolist(), "t_mm": sol.x[10:13].tolist(), "residual_mean_mm": float(res.mean()), "residual_max_mm": float(res.max()),
       "n_points": int(len(G)), "joint_order": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
       "note": "urdf_deg = sign*reading + offset; tip in gripper_frame_link; p_base = R @ p_grid + t"}
json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)
print("저장:", OUT)
