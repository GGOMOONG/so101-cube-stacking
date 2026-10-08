# RunPod: 시뮬 정책 3b — sim_stack_v3b(기존 손목 카메라·기판·조명·DART 2.0°). v3는 틀린 카메라로 무효. 10k까지, 2.5k마다 저장, 20분마다 허브 백업. 3090 32 vCPU 권장, 약 2.5~3.5시간.
# 1) 설치
cd /workspace && curl -L -o lerobot.zip https://github.com/huggingface/lerobot/archive/refs/heads/main.zip && unzip -q lerobot.zip && mv lerobot-main lerobot && cd lerobot && pip install -e ".[smolvla,dataset,training]"
# 2) 로그인
hf auth login
# 3) tmux
tmux new -s simv3b
# 4) 학습
cd /workspace/lerobot && lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=moonbeards/sim_stack_v3b --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' --batch_size=64 --steps=10000 --num_workers=28 --save_freq=2500 --policy.device=cuda --output_dir=outputs/train/smolvla_sim_stack_v3b --job_name=smolvla_sim_stack_v3b --policy.repo_id=moonbeards/smolvla_sim_stack_v3b --policy.private=true --policy.push_to_hub=true --wandb.enable=false && runpodctl remove pod $RUNPOD_POD_ID
# 5) (학습이 돌기 시작하면, 새 웹 터미널에서) 20분마다 체크포인트 백업
HF=$(which hf); cd /workspace/lerobot && nohup bash -c "while true; do sleep 1200; $HF upload moonbeards/smolvla_sim_stack_v3b outputs/train/smolvla_sim_stack_v3b/checkpoints/last/pretrained_model . --private >> /workspace/backup.log 2>&1; echo \"\$(date +%H:%M) 백업 끝\" >> /workspace/backup.log; done" > /dev/null 2>&1 &
