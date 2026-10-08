#!/bin/zsh
# 단단한 큐브(--stiff-cube)로 실물 정책 재평가: 쥔 상태 그리퍼 읽기값이 실물(13.6)과 같아짐 → 운반·놓기 단계 상태 분포 일치
source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; cd ~/robotics; LOG=logs/sim_ablation.log
run() { m=$1; tag=$2; shift 2; echo "== $tag $m $@" >> $LOG; python sim_policy_eval.py --model $m --cond A --n 8 --tag $tag "$@" 2>&1 | grep -v "Warning\|objc\|Loading\|torch_dtype\|Reducing" >> $LOG; }
run moonbeards/smolvla_color_cube_stack_unstack2_v5 stiff_v5_A     --stiff-cube --dur 45 --video 2
run moonbeards/smolvla_color_cube_stack_unstack2_v4 stiff_v4_A     --stiff-cube --dur 45 --video 1
run moonbeards/smolvla_color_cube_stack_unstack2_v4 stiff_v4_A_d0  --stiff-cube --dur 45 --cmd-delay 0
run moonbeards/smolvla_color_cube_stack_unstack2_v5 stiff_v5_B     --stiff-cube --dur 45 --cond B
echo "$(date '+%m-%d %H:%M') 단단한 큐브 재평가 완료" >> $LOG
