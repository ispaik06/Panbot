#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------
# Panbot start script
# - Works from anywhere (uses script location)
# - Writes ALL output (stdout/stderr) to a timestamped log file
# - Prints config summary before starting main_runtime
# --------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PANBOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"          # .../Panbot
REPO_ROOT="$(cd "${PANBOT_DIR}/.." && pwd)"           # project root

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# --------- defaults ----------
CAM_INDEX="${CAM_INDEX:-0}"
BACKEND="${BACKEND:-v4l2}"

WIDTH="${WIDTH:-3840}"
HEIGHT="${HEIGHT:-2160}"
FPS="${FPS:-30}"
MJPG="${MJPG:-1}"

YOLO_PREVIEW_SCALE="${YOLO_PREVIEW_SCALE:-0.55}"
GRU_PREVIEW_SCALE="${GRU_PREVIEW_SCALE:-0.30}"
BASE_PREVIEW_SCALE="${BASE_PREVIEW_SCALE:-0.30}"

CORNERS="${CORNERS:-Panbot/vision/calibration/corners.json}"
YOLO_MODEL="${YOLO_MODEL:-Panbot/vision/models/runs/batter_seg_local_v1/weights/best.pt}"
GRU_CKPT="${GRU_CKPT:-Panbot/vision/models/runs/resnet18_gru16_cls/best.pt}"

ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-so101_follower_1}"
ROBOT_CALIB_DIR="${ROBOT_CALIB_DIR:-}"

TASK2_DURATION="${TASK2_DURATION:-10}"
TASK3_DURATION="${TASK3_DURATION:-10}"
WAIT_TASK2_TO_TASK3_S="${WAIT_TASK2_TO_TASK3_S:-30}"

SHOW="${SHOW:-1}"

# --------- log file ----------
LOG_DIR="${LOG_DIR:-${PANBOT_DIR}/logs}"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_${TS}.log"

# Route ALL output to both terminal + log file (append mode)
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================"
echo "[start_all.sh] Panbot runtime starting..."
echo "TIME        : $(date)"
echo "REPO_ROOT   : ${REPO_ROOT}"
echo "PANBOT_DIR  : ${PANBOT_DIR}"
echo "PYTHON      : $(command -v python || true)"
echo "PYTHON_VER  : $(python --version 2>/dev/null || true)"
echo "PWD         : $(pwd)"
echo "LOG_FILE    : ${LOG_FILE}"
echo "--------------------------------------------"
echo "[CAM]"
echo "  CAM_INDEX : ${CAM_INDEX}"
echo "  BACKEND   : ${BACKEND}"
echo "  SIZE      : ${WIDTH}x${HEIGHT}"
echo "  FPS       : ${FPS}"
echo "  MJPG      : ${MJPG}"
echo "[PREVIEW]"
echo "  YOLO      : ${YOLO_PREVIEW_SCALE}"
echo "  GRU       : ${GRU_PREVIEW_SCALE}"
echo "  BASE      : ${BASE_PREVIEW_SCALE}"
echo "[PATHS]"
echo "  CORNERS   : ${CORNERS}"
echo "  YOLO_MODEL: ${YOLO_MODEL}"
echo "  GRU_CKPT  : ${GRU_CKPT}"
echo "[ROBOT]"
echo "  PORT      : ${ROBOT_PORT}"
echo "  ID        : ${ROBOT_ID}"
echo "  CALIB_DIR : ${ROBOT_CALIB_DIR:-<empty>}"
echo "[TASK]"
echo "  TASK2_DURATION       : ${TASK2_DURATION}"
echo "  TASK3_DURATION       : ${TASK3_DURATION}"
echo "  WAIT_TASK2_TO_TASK3_S: ${WAIT_TASK2_TO_TASK3_S}"
echo "[UI]"
echo "  SHOW      : ${SHOW}"
echo "============================================"
echo

# --------- preflight checks ----------
fail=0
check_file () {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "[ERROR] missing file: $p"
    fail=1
  else
    echo "[OK] file exists: $p"
  fi
}
check_dev () {
  local p="$1"
  if [[ ! -e "$p" ]]; then
    echo "[WARN] device not found: $p"
  else
    echo "[OK] device exists: $p"
  fi
}

check_file "${CORNERS}"
check_file "${YOLO_MODEL}"
check_file "${GRU_CKPT}"
check_dev  "${ROBOT_PORT}"

if [[ $fail -ne 0 ]]; then
  echo
  echo "[FATAL] preflight failed. Fix missing files and retry."
  exit 1
fi

# --------- build args ----------
ARGS=(
  "--cam" "${CAM_INDEX}"
  "--backend" "${BACKEND}"
  "--width" "${WIDTH}"
  "--height" "${HEIGHT}"
  "--fps" "${FPS}"
  "--yolo_preview_scale" "${YOLO_PREVIEW_SCALE}"
  "--gru_preview_scale" "${GRU_PREVIEW_SCALE}"
  "--base_preview_scale" "${BASE_PREVIEW_SCALE}"
  "--corners" "${CORNERS}"
  "--yolo_model" "${YOLO_MODEL}"
  "--gru_ckpt" "${GRU_CKPT}"
  "--robot_port" "${ROBOT_PORT}"
  "--robot_id" "${ROBOT_ID}"
  "--task2_duration" "${TASK2_DURATION}"
  "--task3_duration" "${TASK3_DURATION}"
  "--wait_task2_to_task3_s" "${WAIT_TASK2_TO_TASK3_S}"
)

if [[ "${MJPG}" == "1" ]]; then
  ARGS+=("--mjpg")
fi
if [[ -n "${ROBOT_CALIB_DIR}" ]]; then
  ARGS+=("--robot_calib_dir" "${ROBOT_CALIB_DIR}")
fi
if [[ "${SHOW}" == "1" ]]; then
  ARGS+=("--show")
fi

echo
echo "[CMD] python Panbot/control/main_runtime.py ${ARGS[*]}"
echo "--------------------------------------------"
echo

# --------- run ----------
python Panbot/control/main_runtime.py "${ARGS[@]}"
