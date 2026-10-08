#!/bin/zsh
# 런포드 학습이 끝나 허브에 모델이 올라오면 자동으로 시뮬 평가(A/B 10회씩)를 돌린다.
# 로그: ~/robotics/logs/watch_sim_policy.log
source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot
cd ~/robotics; mkdir -p logs
MODEL=moonbeards/smolvla_sim_stack_v2
LOG=logs/watch_sim_policy_v2.log
echo "$(date '+%m-%d %H:%M') 감시 시작 $MODEL" >> $LOG
while true; do
  if python -c "from huggingface_hub import HfApi; import sys; sys.exit(0 if HfApi().file_exists('$MODEL','README.md') else 1)" 2>/dev/null; then
    echo "$(date '+%m-%d %H:%M') 모델 도착 → 시뮬 평가 시작" >> $LOG; break
  fi
  sleep 600
done
for cond in A B; do
  python sim_policy_eval.py --model $MODEL --cond $cond --n 10 --dur 45 --stiff-cube --kp-joints 16,,,,16 --kv-joints 0.5,,,,0.5 --tag sim2_${cond}_stiff --video 2 >> $LOG 2>&1
done
for cond in A B; do
  python sim_policy_eval.py --model $MODEL --cond $cond --n 10 --dur 45 --tag sim2_${cond}_fit --video 1 >> $LOG 2>&1
done
echo "$(date '+%m-%d %H:%M') 시뮬 평가 완료" >> $LOG
