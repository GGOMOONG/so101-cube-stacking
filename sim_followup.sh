#!/bin/zsh
# 요인 분리 뒤 후속: v5 인코더식 상태·긴 시간(놓기 단계 관찰), v4 인코더식
source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; cd ~/robotics; LOG=logs/sim_ablation.log
run() { m=$1; tag=$2; shift 2; echo "== $tag $m $@" >> $LOG; python sim_policy_eval.py --model $m --cond A --n 8 --tag $tag "$@" 2>&1 | grep -v "Warning\|objc\|Loading\|torch_dtype\|Reducing" >> $LOG; }
run moonbeards/smolvla_color_cube_stack_unstack2_v5 v5_A_enc45  --state encoder --delay 4 --dur 45 --video 2
run moonbeards/smolvla_color_cube_stack_unstack2_v5 v5_A_fit45  --dur 45
run moonbeards/smolvla_color_cube_stack_unstack2_v4 v4_A_enc45  --state encoder --delay 4 --dur 45
echo "$(date '+%m-%d %H:%M') 후속 완료" >> $LOG
# 관절별 게인 분리: 균일 kp 12가 되는 요인이 pan·roll(45→12)인지 lift(32→12)인지
run moonbeards/smolvla_color_cube_stack_unstack2_v4 abl_panroll12_d0 --kp-joints 12,,,,12 --cmd-delay 0
run moonbeards/smolvla_color_cube_stack_unstack2_v4 abl_lift12_d0    --kp-joints ,12,,, --cmd-delay 0
run moonbeards/smolvla_color_cube_stack_unstack2_v4 abl_panroll12_d3 --kp-joints 12,,,,12
echo "$(date '+%m-%d %H:%M') 관절별 분리 완료" >> $LOG
# 그리퍼 게인이 핵심 요인으로 보임(grip07: 잡기 4/8) → 더 약하게·지연 0 조합
run moonbeards/smolvla_color_cube_stack_unstack2_v4 abl_grip05       --kp-grip-scale 0.5
run moonbeards/smolvla_color_cube_stack_unstack2_v4 abl_grip07_d0    --kp-grip-scale 0.7 --cmd-delay 0
run moonbeards/smolvla_color_cube_stack_unstack2_v4 abl_grip05_d0_45 --kp-grip-scale 0.5 --cmd-delay 0 --dur 45 --video 2
echo "$(date '+%m-%d %H:%M') 그리퍼 분리 완료" >> $LOG
# 기준선 재측정: v4 맞춤 동역학·지연 3(외관 B5·손목 카메라 B6 보정 이후 값이 없음 — 02시 기준선은 보정 전)
run moonbeards/smolvla_color_cube_stack_unstack2_v4 abl_base45 --dur 45
run moonbeards/smolvla_color_cube_stack_unstack2_v4 abl_d0_45b --cmd-delay 0 --dur 45
echo "$(date '+%m-%d %H:%M') 기준선 재측정 완료" >> $LOG
