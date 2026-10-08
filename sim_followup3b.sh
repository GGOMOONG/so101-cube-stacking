#!/bin/zsh
# pan/roll 게인 후보(sysid로는 구분 안 됨: kp16·kv0.5가 맞춤값과 같은 RMSE) + 단단한 큐브에서 v4·v5 비교 → sim_stack_v2 생성 동역학 결정
source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; cd ~/robotics; LOG=logs/sim_ablation.log
run() { m=$1; tag=$2; shift 2; echo "== $tag $m $@" >> $LOG; python sim_policy_eval.py --model $m --n 8 --dur 45 --stiff-cube --tag $tag "$@" 2>&1 | grep -v "Warning\|objc\|Loading\|torch_dtype\|Reducing" >> $LOG; }
V4=moonbeards/smolvla_color_cube_stack_unstack2_v4; V5=moonbeards/smolvla_color_cube_stack_unstack2_v5
run $V4 pr12_v4_A --cond A --kp-joints 12,,,,12 --kv-joints 0.5,,,,0.5
run $V5 pr16_v5_A --cond A --kp-joints 16,,,,16 --kv-joints 0.5,,,,0.5 --video 1
run $V5 pr12_v5_A --cond A --kp-joints 12,,,,12 --kv-joints 0.5,,,,0.5
run $V5 pr16_v5_B --cond B --kp-joints 16,,,,16 --kv-joints 0.5,,,,0.5
echo "$(date '+%m-%d %H:%M') pan/roll 게인 비교 완료" >> $LOG
# 숨은 처짐(인코더에 안 보이는 스프링 관절) 재시험: 시뮬 영상에서 정책이 손끝 z 22~31 mm에서 닫음(실물은 z 2) → 실물의 숨은 처짐 7~12 mm를 전제로 배운 정책
run $V5 hid25_v5_A --cond A --hidden 25 --video 2
run $V5 hid40_v5_A --cond A --hidden 40
run $V4 hid25_v4_A --cond A --hidden 25
run $V5 hid25_pr16_v5_A --cond A --hidden 25 --kp-joints 16,,,,16 --kv-joints 0.5,,,,0.5
echo "$(date '+%m-%d %H:%M') 숨은 처짐 재시험 완료" >> $LOG
