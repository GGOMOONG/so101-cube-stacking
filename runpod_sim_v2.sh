# RunPod: 시뮬 전용 정책 2차 — sim_stack_v2(단단한 큐브·pan/roll kp16, 296회·150,130프레임), smolvla_base에서 30k 스텝.
# 팟 조건: vCPU 24 이상(영상 디코딩 병목), 5090/4090. 10-01 1차는 호스트 GPU 오류로 17k에서 죽었음 → 체크포인트 5k마다 저장 + 30분마다 허브에 자동 백업.
# 1) 설치
cd /workspace && curl -L -o lerobot.zip https://github.com/huggingface/lerobot/archive/refs/heads/main.zip && unzip -q lerobot.zip && mv lerobot-main lerobot && cd lerobot && pip install -e ".[smolvla,dataset,training]"
# 2) 로그인 (토큰 직접 입력)
hf auth login
# 3) 배정 CPU 확인 (워커 수 = 배정 코어 − 1)
cat /sys/fs/cgroup/cpu.max
# 4) tmux
tmux new -s simv2
# 5) 학습 (W = 배정 코어 − 1 자동 계산) → moonbeards/smolvla_sim_stack_v2 비공개, 끝나면 팟 삭제
cd /workspace/lerobot && W=$(( $(cut -d' ' -f1 /sys/fs/cgroup/cpu.max) / 100000 - 1 )) && echo "workers=$W" && lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=moonbeards/sim_stack_v2 --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' --batch_size=64 --steps=30000 --num_workers=$W --save_freq=5000 --policy.device=cuda --output_dir=outputs/train/smolvla_sim_stack_v2 --job_name=smolvla_sim_stack_v2 --policy.repo_id=moonbeards/smolvla_sim_stack_v2 --policy.private=true --policy.push_to_hub=true --wandb.enable=false && runpodctl remove pod $RUNPOD_POD_ID
# 6) (학습 시작 후) Ctrl-b c 로 새 창을 열고 30분마다 최신 체크포인트를 허브에 백업 — 팟이 죽어도 건질 수 있게
cd /workspace/lerobot && while true; do sleep 1800; hf upload moonbeards/smolvla_sim_stack_v2 outputs/train/smolvla_sim_stack_v2/checkpoints/last/pretrained_model . --private >/dev/null 2>&1; echo "$(date +%H:%M) 백업"; done
