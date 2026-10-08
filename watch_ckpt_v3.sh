#!/bin/zsh
# A. v2 학습 곡선: 팟이 30분마다 허브 루트에 올리는 중간 체크포인트가 바뀔 때마다 시뮬 A 10회 평가(단단한 큐브·kp16 = v2 생성 조건).
# 로그 logs/watch_ckpt_v3.log, 결과 calib/sim_policy_runs.csv 태그 ckpt3_<시각>_A. 최종(README.md) 도착 후엔 종료(최종 평가는 watch_sim_policy_v2.sh가 담당).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot
cd ~/robotics; mkdir -p logs; LOG=logs/watch_ckpt_v3.log; MODEL=moonbeards/smolvla_sim_stack_v3; last=""
echo "$(date '+%m-%d %H:%M') 체크포인트 감시 시작" >> $LOG
while true; do
  sha=$(python -c "
from huggingface_hub import HfApi
try:
    i = HfApi().get_paths_info('$MODEL', ['model.safetensors'])
    print(i[0].lfs.sha256 if i and i[0].lfs else '')
except Exception: print('')" 2>/dev/null)
  if [[ -n "$sha" && "$sha" != "$last" ]]; then
    last=$sha; tag=ckpt3_$(date '+%H%M')
    echo "$(date '+%m-%d %H:%M') 새 체크포인트(${sha:0:8}) → 시뮬 A 10회 ($tag)" >> $LOG
    mkdir -p outputs/sim_v3_ckpts/$tag && python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', local_dir='outputs/sim_v3_ckpts/$tag', allow_patterns=['*.json','*.safetensors','tokenizer/*','README.md'])" >/dev/null 2>&1
    python sim_policy_eval.py --model outputs/sim_v3_ckpts/$tag --cond A --n 10 --dur 45 --stiff-cube --kp-joints 16,,,,16 --kv-joints 0.5,,,,0.5 --wrist-cam calib/wrist_cam_v3.json --pcb --appearance calib/sim_appearance_v3.json --tag ${tag}_A --video 1 2>&1 | grep -E "^결과|^\[" | sed 's/추론.*//' | cut -c1-160 >> $LOG
  fi
  if python -c "from huggingface_hub import HfApi; import sys; sys.exit(0 if HfApi().file_exists('$MODEL','README.md') else 1)" 2>/dev/null; then
    tag=ckpt3_final; mkdir -p outputs/sim_v3_ckpts/$tag; python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', local_dir='outputs/sim_v3_ckpts/$tag', allow_patterns=['*.json','*.safetensors','tokenizer/*','README.md'])" >/dev/null 2>&1
    for c in A B; do python sim_policy_eval.py --model outputs/sim_v3_ckpts/$tag --cond $c --n 10 --dur 45 --stiff-cube --kp-joints 16,,,,16 --kv-joints 0.5,,,,0.5 --wrist-cam calib/wrist_cam_v3.json --pcb --appearance calib/sim_appearance_v3.json --tag ${tag}_$c --video 2 2>&1 | grep -E "^결과|^\[" | sed 's/추론.*//' | cut -c1-160 >> $LOG; done
    echo "$(date '+%m-%d %H:%M') 최종 모델 평가 완료 → 감시 종료" >> $LOG; break
  fi
  sleep 600
done
