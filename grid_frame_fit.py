#!/usr/bin/env python3
"""재보정용: 관절 영점·그리퍼 끝 보정은 그대로 두고(kinematic_calibration*.json), 새로 짚은 점들로
격자→베이스 변환(R, t)만 다시 맞춘다. 5점이면 충분.
사용: python ~/robotics/grid_frame_fit.py [touch_points.csv] [기준 json] [출력 json]
"""
import sys, os, json
import numpy as np
from so101_fk import SO101FK
CSV = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/robotics/calib/touch_points.csv")
REF = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/robotics/calib/kinematic_calibration_0923.json")
OUT = os.path.expanduser(sys.argv[3] if len(sys.argv) > 3 else "~/robotics/calib/kinematic_calibration.json")
d = json.load(open(REF)); sign = np.array(d["sign"]); off = np.array(d["offset_deg"]); tip = np.array(d["tip_offset_mm"]) / 1000
fk = SO101FK()
rows = np.loadtxt(CSV, delimiter=",", skiprows=1, ndmin=2); G = rows[:, 0:3]; Q = rows[:, 6:11]
def wrap(a): return (a + 180) % 360 - 180
B = np.array([(lambda T: (T[:3, :3] @ tip + T[:3, 3]) * 1000)(fk.fk_deg(np.r_[wrap(sign * q + off), 0.0])) for q in Q])
gc, bc = G.mean(0), B.mean(0); U, S, Vt = np.linalg.svd((G - gc).T @ (B - bc)); s = np.sign(np.linalg.det(Vt.T @ U.T))
R = Vt.T @ np.diag([1, 1, s]) @ U.T; t = bc - R @ gc; res = np.linalg.norm((R @ G.T).T + t - B, axis=1)
print(f"{len(G)}점, 점별 잔차 {np.round(res,1)} 평균 {res.mean():.1f} 최대 {res.max():.1f} mm")
print("격자에서 본 베이스 원점:", np.round(-R.T @ t, 1), " 테이블 기울기 %.1f°" % np.degrees(np.arccos(abs(R[2, 2]))))
d.update({"R": R.tolist(), "t_mm": t.tolist(), "n_points": int(len(G)), "residual_mean_mm": float(res.mean()),
          "residual_max_mm": float(res.max()), "note": d["note"] + f" | R,t refit from {os.path.basename(CSV)}"})
json.dump(d, open(OUT, "w"), ensure_ascii=False, indent=2); print("저장:", OUT)
