#!/bin/zsh
# 분리: 큐브 맞춤 카메라만(기판·조명 없이) / 기판만 / 조명만
source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; cd ~/robotics; LOG=logs/sim_ablation.log
V5=moonbeards/smolvla_color_cube_stack_unstack2_v5
run() { tag=$1; shift; echo "== $tag $V5 $@" >> $LOG; python sim_policy_eval.py --model $V5 --cond A --n 8 --dur 45 --stiff-cube --kp-joints 16,,,,16 --kv-joints 0.5,,,,0.5 --tag $tag "$@" 2>&1 | grep -v "Warning\|objc\|Loading\|torch_dtype\|Reducing" >> $LOG; }
run camonly_v5_A --wrist-cam calib/wrist_cam_v3.json
run pcbonly_v5_A --pcb
run lightonly_v5_A --appearance calib/sim_appearance_v3.json
echo "$(date '+%m-%d %H:%M') v3 요소 분리 완료" >> $LOG
