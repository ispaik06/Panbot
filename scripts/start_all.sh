#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Panbot/scripts/start_all.sh
# -----------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${CONFIG_PATH:-Panbot/config/runtime.yaml}"

echo "[start_all] ROOT_DIR=$ROOT_DIR"
echo "[start_all] CONFIG_PATH=$CONFIG_PATH"

# ---- optional env overrides (examples) ----
# export PANBOT_ROBOT_PORT=/dev/ttyACM0
# export PANBOT_ROBOT_ID=my_awesome_follower_arm
# export PANBOT_VISION_CAM=0
# export PANBOT_VISION_BACKEND=v4l2
# export PANBOT_SHOW=1
# export PANBOT_POLICY1_DURATION=10
# export PANBOT_POLICY2_DURATION=10
# export PANBOT_WAIT_23=30

# (권장) CUDA/torch 관련이 필요하면 여기서 설정
# export CUDA_VISIBLE_DEVICES=0

# pythonpath: Panbot 패키지 import 가능하게
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

echo "[start_all] PYTHONPATH=$PYTHONPATH"

# ---- run ----
python -u Panbot/control/main_runtime.py --config "$CONFIG_PATH"
