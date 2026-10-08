#!/bin/zsh
# 같은 SmolVLA 모델(v4)에 문장만 바꿔 쌓기 → 허물기를 N회 번갈아 실행하는 시연 스크립트.
# 사용: zsh ~/robotics/loop_stack_unstack.sh 3      (3회 반복, 기본 3회)
# 각 단계는 정해진 시간(STACK_S / UNSTACK_S) 동안 정책을 돌린 뒤 팔을 초기 자세로 되돌립니다.
# 시작 배치: 주황 왼쪽, 빨강 오른쪽 (떨어져 있음). 카메라 인덱스는 실행 전에 lerobot-find-cameras로 확인.

N=${1:-3}
STACK_S=${STACK_S:-25}
UNSTACK_S=${UNSTACK_S:-25}
MODEL=${MODEL:-moonbeards/smolvla_color_cube_stack_unstack2_v4}
TOP=${TOP:-0}
WRIST=${WRIST:-1}

export LEROBOT_CAMERA_MAX_AGE_MS=1500 LEROBOT_GRIPPER_MAX_TORQUE=300 LEROBOT_CAMERA_MAX_STALE_S=6
COMMON=(--strategy.type=base --policy.path=$MODEL --policy.device=mps
        --inference.type=rtc --inference.rtc.execution_horizon=10 --inference.rtc.max_guidance_weight=10.0
        --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
        --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5B790156431 --robot.id=my_follower_arm
        --robot.cameras="{top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, warmup_s: 3}}"
        --display_data=false)

for i in $(seq 1 $N); do
  echo "===== [$i/$N] STACK ====="
  lerobot-rollout --duration=$STACK_S "${COMMON[@]}" --task="Pick up the orange cube and place it on top of the red cube" 2>&1 | grep -E "Rollout (finished|running)|ERROR|Error|stale|read failed"
  sleep 2
  echo "===== [$i/$N] UNSTACK ====="
  lerobot-rollout --duration=$UNSTACK_S "${COMMON[@]}" --task="Pick up the orange cube from the top of the red cube and place it on the table" 2>&1 | grep -E "Rollout (finished|running)|ERROR|Error|stale|read failed"
  sleep 2
done
echo "===== loop done ====="
