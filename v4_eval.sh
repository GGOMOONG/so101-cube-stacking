#!/bin/zsh
# v4(SmolVLA+RTC) 쌓기 1회 실행 후 결과 입력 → calib/v4_runs.csv 기록. 고전 방식과 같은 조건(P 48, 같은 배치, 같은 성공 기준).
# 사용: zsh ~/robotics/v4_eval.sh A        (태그 A/B, 기본 30초)      DUR=40 zsh ~/robotics/v4_eval.sh B
TAG=${1:-A}; DUR=${DUR:-30}; REC=${REC:-0}      # REC=1 이면 회차를 데이터셋(rollout_eval_<TAG>_<시각>, 로컬만)으로 녹화 → 실물 실패 장면을 시뮬과 비교
MODEL=${MODEL:-moonbeards/smolvla_color_cube_stack_unstack2_v4}; TOP=${TOP:-0}; WRIST=${WRIST:-1}; P=${P:-48}
export LEROBOT_CAMERA_MAX_AGE_MS=1500 LEROBOT_GRIPPER_MAX_TORQUE=300 LEROBOT_CAMERA_MAX_STALE_S=6
TS=$(date +%m%d_%H%M%S)
REC_ARGS=""; [[ "$REC" == "1" ]] && REC_ARGS="--dataset.repo_id=moonbeards/rollout_eval_${TAG}_${TS} --dataset.single_task=stack --dataset.num_episodes=1 --dataset.episode_time_s=$DUR --dataset.reset_time_s=1 --dataset.push_to_hub=false --dataset.rgb_encoder.vcodec=h264"
lerobot-rollout --duration=$DUR --strategy.type=base --policy.path=$MODEL --policy.device=mps \
  --inference.type=rtc --inference.rtc.execution_horizon=10 --inference.rtc.max_guidance_weight=10.0 \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5B790156431 --robot.id=my_follower_arm --robot.position_p_coefficient=$P \
  --robot.cameras="{top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, warmup_s: 3}}" \
  --display_data=false --task="Pick up the orange cube and place it on top of the red cube" ${REC_ARGS} 2>&1 | grep -E "Rollout (finished|running)|ERROR|Error|stale|read failed|Traceback"
echo -n "결과 입력: s=성공(주황이 빨강 위에 3초 유지) / f=실패 + 이유 / 빈칸=기록 안 함 > "; read ANS
if [[ -n "$ANS" ]]; then
  F=~/robotics/calib/v4_runs.csv; [[ -f $F ]] || echo "time,tag,P,duration_s,success,reason" > $F
  OK=0; [[ "${ANS:0:1}" == "s" ]] && OK=1; REASON="${ANS:1}"; REASON="${REASON## }"
  echo "$TS,$TAG,$P,$DUR,$OK,\"$REASON\"" >> $F
  N=$(awk -F, -v t=$TAG 'NR>1 && $2==t {n++; s+=$5} END{print s"/"n}' $F); echo "기록: $F (조건 $TAG: $N 성공)"
fi
