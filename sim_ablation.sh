#!/bin/zsh
# v4가 시뮬에서 "균일 kp 12 + 지연 0"에서만 잡히는 이유를 요인별로 분리: 맞춤 동역학에서 한 요인씩 바꿔 A 8회.
# 로그 logs/sim_ablation.log, 결과 calib/sim_policy_runs.csv (tag abl_*)
source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot
cd ~/robotics; mkdir -p logs; LOG=logs/sim_ablation.log
M=moonbeards/smolvla_color_cube_stack_unstack2_v4
run() { tag=$1; shift; echo "== $tag $@" >> $LOG; python sim_policy_eval.py --model $M --cond A --n 8 --tag abl_$tag "$@" 2>&1 | grep -v "Warning\|objc\|Loading\|torch_dtype\|Reducing" >> $LOG; }
run kps05        --kp-scale 0.5                       # 팔 게인만 절반(지연 3 유지)
run kps05_d0     --kp-scale 0.5 --cmd-delay 0         # 팔 게인 절반 + 지연 0
run kps035_d0    --kp-scale 0.35 --cmd-delay 0        # 팔 게인 ≈ kp 12 수준 + 지연 0
run grip07       --kp-grip-scale 0.7                  # 그리퍼 게인만 약하게
run fric0        --friction-scale 0                   # 마찰 사구간 제거
run fric0_d0     --friction-scale 0 --cmd-delay 0     # 마찰 제거 + 지연 0
run d0_dur45     --cmd-delay 0 --dur 45               # 지연 0, 45 s(느려서 못 끝내는지)
echo "$(date '+%m-%d %H:%M') 요인 분리 완료" >> $LOG
