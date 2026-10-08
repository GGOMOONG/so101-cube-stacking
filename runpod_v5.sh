# RunPod에서 v5 학습 (v4 + DAgger 3차 20회 = color_cube_stack_unstack2_dagger3, 236회). 팟(4090 또는 6000 Ada, PyTorch 템플릿) 터미널에 순서대로 붙여넣기.
# 1) 코드 설치 (git clone은 느릴 수 있어 zip 사용)
cd /workspace && curl -L -o lerobot.zip https://github.com/huggingface/lerobot/archive/refs/heads/main.zip && unzip -q lerobot.zip && mv lerobot-main lerobot && cd lerobot && pip install -e ".[smolvla]"
# 2) 허브 로그인 (토큰은 본인이 직접 입력)
hf auth login
# 3) tmux 세션 열기 (브라우저 터미널을 닫아도 학습이 계속됨. 다시 보려면: tmux attach -t v5)
tmux new -s v5
# 4) tmux 안에서 학습 실행. 끝나면 모델 업로드 후 팟 자동 삭제 (v4와 같은 설정: 배치 64, 1만 스텝, 비공개)
cd /workspace/lerobot && lerobot-train --policy.path=moonbeards/smolvla_color_cube_stack_unstack2_v4 --dataset.repo_id=moonbeards/color_cube_stack_unstack2_dagger3 --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' --batch_size=64 --steps=10000 --num_workers=12 --save_freq=5000 --policy.device=cuda --output_dir=outputs/train/smolvla_stack_unstack2_v5 --job_name=smolvla_stack_unstack2_v5 --policy.repo_id=moonbeards/smolvla_color_cube_stack_unstack2_v5 --policy.private=true --policy.push_to_hub=true --wandb.enable=false && runpodctl remove pod $RUNPOD_POD_ID
