#!/bin/zsh
# v3b 체크포인트(2.5k·5k·7.5k·10k=final) 시뮬 A 10회 + 최종 B 10회 (기본 카메라·기판·v3 조명)
source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; cd ~/robotics; LOG=logs/watch_ckpt_v3b.log
run() { d=$1; tag=$2; c=$3; echo "== $tag $d" >> $LOG; python sim_policy_eval.py --model $d --cond $c --n 10 --dur 45 --stiff-cube --kp-joints 16,,,,16 --kv-joints 0.5,,,,0.5 --pcb --appearance calib/sim_appearance_v3.json --tag $tag --video 1 2>&1 | grep -E "^결과|^\[|Error" | sed 's/추론.*//' | cut -c1-160 >> $LOG; }
run outputs/sim_v3b_ckpts/ckpt3b_1134 ckpt3b_2500_A A
run outputs/sim_v3b_ckpts/ckpt3b_1234 ckpt3b_5000_A A
run outputs/sim_v3b_ckpts/ckpt3b_1355 ckpt3b_7500_A A
run outputs/sim_v3b_ckpts/ckpt3b_final ckpt3b_10k_A A
run outputs/sim_v3b_ckpts/ckpt3b_final ckpt3b_10k_B B
echo "$(date '+%m-%d %H:%M') v3b 재평가 완료" >> $LOG
