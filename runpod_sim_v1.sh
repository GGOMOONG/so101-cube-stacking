# RunPod: 시뮬 시연만으로 학습한 정책 (sim_stack_v1, 310회·145,849프레임). 실물 통합 모델 1차(176회, 30k)와 같은 예산으로 smolvla_base에서 학습.
# 1) 설치
cd /workspace && curl -L -o lerobot.zip https://github.com/huggingface/lerobot/archive/refs/heads/main.zip && unzip -q lerobot.zip && mv lerobot-main lerobot && cd lerobot && pip install -e ".[smolvla,dataset,training]"
# 2) 로그인 (토큰 직접 입력)
hf auth login
# 3) tmux
tmux new -s simv1
# 4) 학습 (30k 스텝, 배치 64, 4090 기준 약 6~7시간) → moonbeards/smolvla_sim_stack_v1 비공개, 끝나면 팟 삭제
cd /workspace/lerobot && lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=moonbeards/sim_stack_v1 --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' --batch_size=64 --steps=30000 --num_workers=12 --save_freq=10000 --policy.device=cuda --output_dir=outputs/train/smolvla_sim_stack_v1 --job_name=smolvla_sim_stack_v1 --policy.repo_id=moonbeards/smolvla_sim_stack_v1 --policy.private=true --policy.push_to_hub=true --wandb.enable=false && runpodctl remove pod $RUNPOD_POD_ID
