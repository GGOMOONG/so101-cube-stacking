#!/bin/zsh
# DAgger 교정 3차 수집(새 자리, 격자 없음, P 16): v4 정책이 돌다가 막히면 리더 암으로 이어받아 마무리. 교정 구간만 에피소드로 저장.
# 사용: zsh ~/robotics/dagger_collect.sh          (기본 20회)   N=10 zsh ~/robotics/dagger_collect.sh
N=${N:-20}; REPO=${REPO:-moonbeards/rollout_dagger_stack_unstack2_stack3}
MODEL=${MODEL:-moonbeards/smolvla_color_cube_stack_unstack2_v4}; TOP=${TOP:-0}; WRIST=${WRIST:-1}
export LEROBOT_CAMERA_MAX_AGE_MS=1500 LEROBOT_GRIPPER_MAX_TORQUE=300 LEROBOT_CAMERA_MAX_STALE_S=6
lerobot-rollout --strategy.type=dagger --strategy.num_episodes=$N --strategy.record_autonomous=false --strategy.upload_every_n_episodes=5 \
  --policy.path=$MODEL --policy.device=mps --inference.type=rtc --inference.rtc.execution_horizon=10 --inference.rtc.max_guidance_weight=10.0 \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5B790156431 --robot.id=my_follower_arm \
  --robot.cameras="{top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, warmup_s: 3}}" \
  --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem5B790180261 --teleop.id=my_leader_arm \
  --dataset.repo_id=$REPO --dataset.single_task="Pick up the orange cube and place it on top of the red cube" --dataset.fps=30 --dataset.episode_time_s=60 --dataset.reset_time_s=5 --dataset.push_to_hub=true --dataset.private=true \
  --display_data=false
