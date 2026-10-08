#!/bin/zsh
# 1) v2 10k를 B 배치에서 (v2 생성 조건: 옛 카메라) 2) v5 보정 카메라 + 화면 질감(노이즈/흐림/노출) 흔들림
source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; cd ~/robotics; LOG=logs/sim_ablation.log
V2=moonbeards/smolvla_sim_stack_v2; V5=moonbeards/smolvla_color_cube_stack_unstack2_v5
run() { m=$1; tag=$2; shift 2; echo "== $tag $m $@" >> $LOG; python sim_policy_eval.py --model $m --n 8 --dur 45 --stiff-cube --kp-joints 16,,,,16 --kv-joints 0.5,,,,0.5 --tag $tag "$@" 2>&1 | grep -v "Warning\|objc\|Loading\|torch_dtype\|Reducing" >> $LOG; }
run $V2 ckpt2_10k_B --cond B --n 10 --video 2
run $V5 aug_v5_A_n6b3   --cond A --wrist-cam calib/wrist_cam_v3.json --img-aug 6,3,0.15
run $V5 aug_v5_A_n12b5  --cond A --wrist-cam calib/wrist_cam_v3.json --img-aug 12,5,0.25
run $V5 aug_v5_A_blur7  --cond A --wrist-cam calib/wrist_cam_v3.json --img-aug 0,7,0
echo "$(date '+%m-%d %H:%M') 밤 시험 완료" >> $LOG
