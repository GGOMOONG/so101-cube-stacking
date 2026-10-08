#!/usr/bin/env python3
"""실물 시연의 명령(action)을 시뮬에 재생해 실물 관절 궤적(observation.state)과 맞는 모터 계수를 찾는다 (real→sim 동역학 맞춤).
사용: python ~/robotics/sim_sysid.py            # 맞춤 실행 → calib/sim_dynamics.json
      python ~/robotics/sim_sysid.py --eval      # 현재 저장값으로 검증 에피소드 오차만 출력
방법: 에피소드마다 시뮬 상태를 실물 첫 프레임에 맞추고, 매 틱 명령을 (지연 d프레임 뒤) 구동기에 넣고 1/30 s 진행, 시뮬 읽기값과 실물 상태의 RMSE(도)를 관절별로 계산.
      kp·kv를 관절별로, 지연을 전체 공통으로 좌표 탐색. 그리퍼는 큐브에 막히는 구간(명령<20) 제외.
"""
import os, sys, glob, json, time
import numpy as np, pandas as pd, mujoco
sys.path.insert(0, os.path.expanduser("~/robotics")); os.chdir(os.path.expanduser("~/robotics"))
from sim_phys import PhysSim

DS = os.path.expanduser("~/.cache/huggingface/lerobot/moonbeards/color_cube_stack_unstack2/")
OUT = "calib/sim_dynamics.json"
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load_eps(idx):
    df = pd.concat([pd.read_parquet(x) for x in sorted(glob.glob(DS + "data/*/*.parquet"))])
    return [(np.stack(df[df.episode_index == e]["action"].values).astype(float), np.stack(df[df.episode_index == e]["observation.state"].values).astype(float)) for e in idx]


class Replayer:
    def __init__(self):
        self.s = PhysSim({}, kp=25); self.m, self.d = self.s.m, self.s.d; self.m.opt.timestep = 1.0 / (30 * 16)
    def set_gains(self, kp, kv):
        for i in range(6):
            self.m.actuator_gainprm[i, 0] = kp[i]; self.m.actuator_biasprm[i, 1] = -kp[i]; self.m.actuator_biasprm[i, 2] = -kv[i]
    def replay(self, A, S, delay):
        s = self.s; s.set_state(S[0]); self.d.qvel[:] = 0; s.set_ctrl(A[0]); mujoco.mj_forward(self.m, self.d)
        out = np.empty_like(S)
        for t in range(len(A)):
            s.set_ctrl(A[max(t - delay, 0)]); mujoco.mj_step(self.m, self.d, nstep=16); out[t] = s.readings()
        return out
    def rmse(self, eps, kp, kv, delay):
        self.set_gains(kp, kv); se = np.zeros(6); n = np.zeros(6)
        for A, S in eps:
            sim = self.replay(A, S, delay); e = sim - S
            se[:5] += (e[:, :5] ** 2).sum(0); n[:5] += len(e)
            ok = A[:, 5] >= 20; se[5] += (e[ok, 5] ** 2).sum(); n[5] += ok.sum()
        return np.sqrt(se / np.maximum(n, 1))


if __name__ == "__main__":
    rp = Replayer(); t0 = time.time()
    fit_eps = load_eps(list(range(0, 176, 15))); val_eps = load_eps(list(range(7, 176, 30)))
    print(f"맞춤 에피소드 {len(fit_eps)}개, 검증 {len(val_eps)}개 ({sum(len(a) for a, _ in fit_eps)} 프레임)")
    if "--eval" in sys.argv and os.path.exists(OUT):
        j = json.load(open(OUT)); print("검증 RMSE(도):", np.round(rp.rmse(val_eps, j["kp"], j["kv"], j["delay_frames"]), 2)); sys.exit()
    kp = np.array([25.0] * 5 + [3.0]); kv = np.array([1.0] * 5 + [0.1]); delay = 0
    base = rp.rmse(fit_eps, kp, kv, delay); print("기존값(kp 25, kv 1, 지연 0) RMSE(도):", np.round(base, 2), f"({time.time()-t0:.0f}s)")
    KP = [6, 9, 12, 16, 20, 25, 32, 40, 55, 75, 100]; KV = [0.2, 0.5, 1.0, 1.8, 3.0, 5.0]
    KPG = [1, 2, 3, 5, 8, 12]; KVG = [0.03, 0.1, 0.3, 0.6]
    for rnd in range(2):
        # 지연
        best = min(((rp.rmse(fit_eps, kp, kv, dl)[:5].mean(), dl) for dl in range(0, 6)), key=lambda t: t[0]); delay = best[1]
        # 관절별 kp, kv: 모든 관절에 같은 후보를 넣어 보고 관절마다 가장 좋은 것을 고름(관절 간 결합은 약함)
        table = {}
        for a in KP:
            for b in KV:
                table[(a, b)] = rp.rmse(fit_eps, np.r_[[a] * 5, kp[5]], np.r_[[b] * 5, kv[5]], delay)
        for j in range(5):
            (a, b) = min(table, key=lambda k_: table[k_][j]); kp[j], kv[j] = a, b
        tg = {(a, b): rp.rmse(fit_eps, np.r_[kp[:5], a], np.r_[kv[:5], b], delay)[5] for a in KPG for b in KVG}
        kp[5], kv[5] = min(tg, key=tg.get)
        print(f"라운드 {rnd+1}: 지연 {delay}프레임 kp {kp} kv {kv} → RMSE {np.round(rp.rmse(fit_eps, kp, kv, delay), 2)} ({time.time()-t0:.0f}s)", flush=True)
    fit = rp.rmse(fit_eps, kp, kv, delay); val = rp.rmse(val_eps, kp, kv, delay)
    print("최종  맞춤 RMSE(도):", np.round(fit, 2), "\n      검증 RMSE(도):", np.round(val, 2), "\n      기존 RMSE(도):", np.round(base, 2))
    json.dump({"kp": kp.tolist(), "kv": kv.tolist(), "delay_frames": int(delay), "rmse_fit_deg": fit.tolist(), "rmse_val_deg": val.tolist(), "rmse_before_deg": base.tolist(),
               "joints": NAMES, "source": "color_cube_stack_unstack2 (176 demos, P16), action replay"}, open(OUT, "w"), indent=2)
    print("저장:", OUT)
