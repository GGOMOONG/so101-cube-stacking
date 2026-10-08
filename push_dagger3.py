#!/usr/bin/env python3
"""DAgger 3차 데이터셋 2개를 허브에 비공개로 올린다 (로컬 ~/.cache/huggingface/lerobot/moonbeards/… 기준).
사용: python ~/robotics/push_dagger3.py     (먼저 hf auth login 되어 있어야 함)
"""
from lerobot.datasets.lerobot_dataset import LeRobotDataset
for rid in ["moonbeards/dagger_stack3_corrections3_h264", "moonbeards/color_cube_stack_unstack2_dagger3"]:
    ds = LeRobotDataset(rid)
    print(f"{rid}: {ds.meta.total_episodes} eps, {ds.meta.total_frames} frames -> pushing (private)…", flush=True)
    ds.push_to_hub(private=True)
    print("done", rid, flush=True)
