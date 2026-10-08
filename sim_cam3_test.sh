#!/bin/zsh
# 큐브 맞춤 카메라(v3)·기판·조명으로 실물 정책 재평가: 놓기 단계가 달라지는지
source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; cd ~/robotics; LOG=logs/sim_ablation.log
run() { m=$1; tag=$2; shift 2; echo "== $tag $m $@" >> $LOG; python sim_policy_eval.py --model $m --n 8 --dur 45 --stiff-cube --wrist-cam calib/wrist_cam_v3.json --pcb --appearance calib/sim_appearance_v3.json --tag $tag "$@" 2>&1 | grep -v "Warning\|objc\|Loading\|torch_dtype\|Reducing" >> $LOG; }
V4=moonbeards/smolvla_color_cube_stack_unstack2_v4; V5=moonbeards/smolvla_color_cube_stack_unstack2_v5
run $V5 cam3_v5_A --cond A --kp-joints 16,,,,16 --kv-joints 0.5,,,,0.5 --video 2
run $V4 cam3_v4_A --cond A --kp-joints 12,,,,12 --kv-joints 0.5,,,,0.5 --video 1
run $V5 cam3_v5_B --cond B --kp-joints 16,,,,16 --kv-joints 0.5,,,,0.5
echo "$(date '+%m-%d %H:%M') v3 카메라 재평가 완료" >> $LOG
