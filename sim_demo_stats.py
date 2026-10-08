#!/usr/bin/env python3
"""시뮬 생성 시연과 실물 시연(쌓기 124회)의 통계 비교표. 사용: python ~/robotics/sim_demo_stats.py moonbeards/sim_stack_v1"""
import os, sys, glob, json, numpy as np, pandas as pd
repo = sys.argv[1] if len(sys.argv) > 1 else "moonbeards/sim_stack_v1"
def load(root, task_filter=None):
    df = pd.concat([pd.read_parquet(x) for x in sorted(glob.glob(root + "/data/*/*.parquet"))]); rows = []
    for e in sorted(df.episode_index.unique()):
        sub = df[df.episode_index == e]
        if task_filter is not None and int(sub.task_index.iloc[0]) != task_filter: continue
        A = np.stack(sub["action"].values).astype(float); S = np.stack(sub["observation.state"].values).astype(float); n = len(A); g = A[:, 5]
        sp = (np.abs(np.diff(S[:, :5], axis=0)) * 30).max(1); cl = np.where(g < 20)[0]; c0 = int(cl[0]) if len(cl) else n - 1
        lag = np.abs(A[:, :5] - S[:, :5]).mean()
        rows.append(dict(len_s=n / 30, v90=float(np.percentile(sp, 90)), vmax=float(sp.max()), grip_open=float(g[:c0].max()) if c0 > 0 else 0, grip_close=float(g[c0:].min()), close_s=c0 / 30, cmd_state_gap=lag, start=S[0], end=S[-1]))
    return rows
real = load(os.path.expanduser("~/.cache/huggingface/lerobot/moonbeards/color_cube_stack_unstack2"), 0)
sim = load(os.path.expanduser("~/.cache/huggingface/lerobot/" + repo))
def q(rows, k): x = np.array([r[k] for r in rows]); return f"{np.median(x):.1f} ({np.percentile(x,10):.1f}~{np.percentile(x,90):.1f})"
print(f"| 항목 | 실물 시연 (n={len(real)}) | 시뮬 시연 (n={len(sim)}) |\n|---|---|---|")
for k, lab in [("len_s", "회차 길이 s"), ("close_s", "닫기 시각 s"), ("v90", "관절 속도 90분위 °/s"), ("vmax", "관절 속도 최대 °/s"), ("grip_open", "접근 중 그리퍼 명령"), ("grip_close", "문 상태 그리퍼 명령"), ("cmd_state_gap", "명령−상태 평균 차 °")]:
    print(f"| {lab} | {q(real, k)} | {q(sim, k)} |")
rs, ss = np.array([r["start"] for r in real]), np.array([r["start"] for r in sim])
print(f"| 시작 자세 중앙값 | {np.round(np.median(rs,0),0).tolist()} | {np.round(np.median(ss,0),0).tolist()} |")
print(f"| 총 프레임 | {int(sum(r['len_s'] for r in real)*30)} | {int(sum(r['len_s'] for r in sim)*30)} |")
