#!/bin/zsh
# 런포드 학습이 끝나 허브에 모델이 올라오면 자동으로 시뮬 평가(A/B 10회씩)를 돌린다.
# 로그: ~/robotics/logs/watch_sim_policy.log
source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot
cd ~/robotics; mkdir -p logs
MODEL=moonbeards/smolvla_sim_stack_v1
LOG=logs/watch_sim_policy.log
echo "$(date '+%m-%d %H:%M') 감시 시작 $MODEL" >> $LOG
while true; do
  if python -c "from huggingface_hub import HfApi; import sys; sys.exit(0 if HfApi().file_exists('$MODEL','model.safetensors') else 1)" 2>/dev/null; then
    echo "$(date '+%m-%d %H:%M') 모델 도착 → 시뮬 평가 시작" >> $LOG; break
  fi
  sleep 600
done
for cond in A B; do
  python sim_policy_eval.py --model $MODEL --cond $cond --n 10 --tag sim_${cond}_fit --video 2 >> $LOG 2>&1
done
for cond in A B; do
  python sim_policy_eval.py --model $MODEL --cond $cond --n 10 --kp 12 --cmd-delay 0 --hidden 0 --tag sim_${cond}_kp12 --video 1 >> $LOG 2>&1
done
echo "$(date '+%m-%d %H:%M') 시뮬 평가 완료" >> $LOG
