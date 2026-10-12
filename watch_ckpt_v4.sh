#!/bin/zsh
# v4 학습 곡선: 시연 다양화 데이터(sim_stack_v4, v2 조건)로 학습한 체크포인트를 2.5k마다 시뮬 A 10회 평가, 최종은 A/B. 새 lerobot의 config.json dtype 키는 저장 직후 제거.
# 로그 logs/watch_ckpt_v4.log, 결과 calib/sim_policy_runs.csv 태그 ckpt4_<시각>_A. 최종(README.md) 도착 후엔 종료(최종 평가는 watch_sim_policy_v2.sh가 담당).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot
cd ~/robotics; mkdir -p logs; LOG=logs/watch_ckpt_v4.log; MODEL=moonbeards/smolvla_sim_stack_v4; last=""
echo "$(date '+%m-%d %H:%M') 체크포인트 감시 시작" >> $LOG
while true; do
  sha=$(python -c "
from huggingface_hub import HfApi
try:
    i = HfApi().get_paths_info('$MODEL', ['model.safetensors'])
    print(i[0].lfs.sha256 if i and i[0].lfs else '')
except Exception: print('')" 2>/dev/null)
  if [[ -n "$sha" && "$sha" != "$last" ]]; then
    last=$sha; tag=ckpt4_$(date '+%H%M')
    echo "$(date '+%m-%d %H:%M') 새 체크포인트(${sha:0:8}) → 시뮬 A 10회 ($tag)" >> $LOG
    mkdir -p outputs/sim_v4_ckpts/$tag && python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', local_dir='outputs/sim_v4_ckpts/$tag', allow_patterns=['*.json','*.safetensors','tokenizer/*','README.md'])" >/dev/null 2>&1; python -c "import json,sys; p='outputs/sim_v4_ckpts/$tag/config.json'; j=json.load(open(p)); j.pop('dtype',None); json.dump(j,open(p,'w'),indent=1)" 2>/dev/null
    python sim_policy_eval.py --model outputs/sim_v4_ckpts/$tag --cond A --n 10 --dur 45 --stiff-cube --kp-joints 16,,,,16 --kv-joints 0.5,,,,0.5 --tag ${tag}_A --video 1 2>&1 | grep -E "^결과|^\[" | sed 's/추론.*//' | cut -c1-160 >> $LOG
  fi
  if python -c "from huggingface_hub import HfApi; import sys; sys.exit(0 if HfApi().file_exists('$MODEL','README.md') else 1)" 2>/dev/null; then
    tag=ckpt4_final; mkdir -p outputs/sim_v4_ckpts/$tag; python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', local_dir='outputs/sim_v4_ckpts/$tag', allow_patterns=['*.json','*.safetensors','tokenizer/*','README.md'])" >/dev/null 2>&1; python -c "import json,sys; p='outputs/sim_v4_ckpts/$tag/config.json'; j=json.load(open(p)); j.pop('dtype',None); json.dump(j,open(p,'w'),indent=1)" 2>/dev/null
    for c in A B; do python sim_policy_eval.py --model outputs/sim_v4_ckpts/$tag --cond $c --n 10 --dur 45 --stiff-cube --kp-joints 16,,,,16 --kv-joints 0.5,,,,0.5 --tag ${tag}_$c --video 2 2>&1 | grep -E "^결과|^\[" | sed 's/추론.*//' | cut -c1-160 >> $LOG; done
    echo "$(date '+%m-%d %H:%M') 최종 모델 평가 완료 → 감시 종료" >> $LOG; break
  fi
  sleep 600
done
