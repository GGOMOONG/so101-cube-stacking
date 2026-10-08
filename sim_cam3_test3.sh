#!/bin/zsh
# 보정 카메라에서 정책이 큐브 22 mm 앞에서 멈춤 → 실물의 숨은 처짐(인코더에 안 보임)이 그 22 mm를 메우는지: 숨은 컴플라이언스 K를 넣어 재시험
source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; cd ~/robotics; LOG=logs/sim_ablation.log
V5=moonbeards/smolvla_color_cube_stack_unstack2_v5; V4=moonbeards/smolvla_color_cube_stack_unstack2_v4
run() { m=$1; tag=$2; shift 2; echo "== $tag $m $@" >> $LOG; python sim_policy_eval.py --model $m --cond A --n 8 --dur 45 --stiff-cube --wrist-cam calib/wrist_cam_v3.json --tag $tag "$@" 2>&1 | grep -v "Warning\|objc\|Loading\|torch_dtype\|Reducing" >> $LOG; }
run $V5 cam3_hid25_v5_A --kp-joints 16,,,,16 --kv-joints 0.5,,,,0.5 --hidden 25
run $V5 cam3_hid12_v5_A --kp-joints 16,,,,16 --kv-joints 0.5,,,,0.5 --hidden 12 --video 2
run $V4 cam3_hid12_v4_A --kp-joints 12,,,,12 --kv-joints 0.5,,,,0.5 --hidden 12
echo "$(date '+%m-%d %H:%M') 보정 카메라+숨은 처짐 시험 완료" >> $LOG
