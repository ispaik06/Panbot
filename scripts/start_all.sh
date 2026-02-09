#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Panbot start script (runtime.yaml 기반)
# - 어디서 실행해도 동작 (script 위치 기준)
# - PYTHONPATH 설정
# - 환경변수로 runtime.yaml 값을 override 가능 (원본 yaml은 유지)
# ------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PANBOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"      # .../Panbot
REPO_ROOT="$(cd "${PANBOT_DIR}/.." && pwd)"       # repo root (parent of Panbot)

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ---------- config path ----------
CONFIG="${CONFIG:-Panbot/config/runtime.yaml}"

# ---------- optional overrides (env) ----------
# 로봇
ROBOT_PORT="${ROBOT_PORT:-}"              # ex) /dev/ttyACM0
ROBOT_ID="${ROBOT_ID:-}"                  # ex) so101_follower_1
ROBOT_CALIB_DIR="${ROBOT_CALIB_DIR:-}"    # ex) Panbot/control/calib

# task duration
TASK2_DURATION_S="${TASK2_DURATION_S:-}"  # ex) 10
TASK3_DURATION_S="${TASK3_DURATION_S:-}"  # ex) 10
WAIT_TASK2_TO_TASK3_S="${WAIT_TASK2_TO_TASK3_S:-}" # ex) 30

# vision show on/off (true/false)
SHOW="${SHOW:-}"                          # ex) true / false

# vision camera index
VISION_CAM_INDEX="${VISION_CAM_INDEX:-}"  # ex) 0

# log: 추가로 시작 시 정보 크게 출력
PRINT_BANNER="${PRINT_BANNER:-1}"

# ---------- helper: yaml patch (python) ----------
PATCHED_CONFIG=""
if [[ -n "${ROBOT_PORT}${ROBOT_ID}${ROBOT_CALIB_DIR}${TASK2_DURATION_S}${TASK3_DURATION_S}${WAIT_TASK2_TO_TASK3_S}${SHOW}${VISION_CAM_INDEX}" ]]; then
  PATCHED_CONFIG="$(mktemp -t panbot_runtime_patched_XXXX.yaml)"

  python - <<'PY'
import os, yaml, copy

cfg_path = os.environ["CONFIG"]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

def set_if(env, path, cast=None):
    v = os.environ.get(env, "")
    if not v:
        return
    keys = path.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    vv = v
    if cast:
        vv = cast(v)
    cur[keys[-1]] = vv

def to_bool(x: str) -> bool:
    x = x.strip().lower()
    return x in ("1","true","t","yes","y","on")

def to_float(x: str) -> float:
    return float(x)

def to_int(x: str) -> int:
    return int(float(x))

# robot overrides
set_if("ROBOT_PORT", "robot.port", str)
set_if("ROBOT_ID", "robot.id", str)
set_if("ROBOT_CALIB_DIR", "robot.calibration_dir", str)

# task overrides
set_if("TASK2_DURATION_S", "task.task2_duration_s", to_float)
set_if("TASK3_DURATION_S", "task.task3_duration_s", to_float)
set_if("WAIT_TASK2_TO_TASK3_S", "task.wait_task2_to_task3_s", to_float)

# vision overrides
set_if("SHOW", "vision.show", to_bool)
set_if("VISION_CAM_INDEX", "vision.cam_index", to_int)

patched = os.environ["PATCHED_CONFIG"]
with open(patched, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print(patched)
PY
  # 위 python에서 patched 경로를 출력하므로 받기
  PATCHED_CONFIG="$(python - <<'PY'
import os
print(os.environ["PATCHED_CONFIG"])
PY
)"

  # ↑ 위 방식이 환경에 따라 꼬일 수 있어서 아래로 안전하게 재지정
  # (실제로는 위 python 실행에서 PATCHED_CONFIG 파일을 이미 만들어 둠)
fi

# 위 patch 파이썬에서 PATCHED_CONFIG를 env로 넣어야 해서 재구성
if [[ -n "${PATCHED_CONFIG}" ]]; then
  export PATCHED_CONFIG
fi

# 다시 patch를 "확실히" 수행 (위에서 파일 만들었다고 가정하지 않고 여기서 확정)
if [[ -n "${ROBOT_PORT}${ROBOT_ID}${ROBOT_CALIB_DIR}${TASK2_DURATION_S}${TASK3_DURATION_S}${WAIT_TASK2_TO_TASK3_S}${SHOW}${VISION_CAM_INDEX}" ]]; then
  PATCHED_CONFIG="$(mktemp -t panbot_runtime_patched_XXXX.yaml)"
  export PATCHED_CONFIG
  python - <<'PY'
import os, yaml

cfg_path = os.environ["CONFIG"]
patched = os.environ["PATCHED_CONFIG"]

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

def to_bool(x: str) -> bool:
    x = x.strip().lower()
    return x in ("1","true","t","yes","y","on")

def set_path(d, path, value):
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

# robot overrides
if os.environ.get("ROBOT_PORT",""):
    set_path(cfg, "robot.port", os.environ["ROBOT_PORT"])
if os.environ.get("ROBOT_ID",""):
    set_path(cfg, "robot.id", os.environ["ROBOT_ID"])
if os.environ.get("ROBOT_CALIB_DIR",""):
    set_path(cfg, "robot.calibration_dir", os.environ["ROBOT_CALIB_DIR"])

# task overrides
if os.environ.get("TASK2_DURATION_S",""):
    set_path(cfg, "task.task2_duration_s", float(os.environ["TASK2_DURATION_S"]))
if os.environ.get("TASK3_DURATION_S",""):
    set_path(cfg, "task.task3_duration_s", float(os.environ["TASK3_DURATION_S"]))
if os.environ.get("WAIT_TASK2_TO_TASK3_S",""):
    set_path(cfg, "task.wait_task2_to_task3_s", float(os.environ["WAIT_TASK2_TO_TASK3_S"]))

# vision overrides
if os.environ.get("SHOW",""):
    set_path(cfg, "vision.show", to_bool(os.environ["SHOW"]))
if os.environ.get("VISION_CAM_INDEX",""):
    set_path(cfg, "vision.cam_index", int(float(os.environ["VISION_CAM_INDEX"])))

with open(patched, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print(patched)
PY
fi

FINAL_CONFIG="${PATCHED_CONFIG:-$CONFIG}"

if [[ "${PRINT_BANNER}" == "1" ]]; then
  echo "============================================"
  echo "[start_all.sh]"
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "PYTHONPATH=${PYTHONPATH}"
  echo "CONFIG(original)=${CONFIG}"
  if [[ "${FINAL_CONFIG}" != "${CONFIG}" ]]; then
    echo "CONFIG(patched)=${FINAL_CONFIG}"
    echo "OVERRIDES:"
    [[ -n "${ROBOT_PORT}" ]] && echo "  ROBOT_PORT=${ROBOT_PORT}"
    [[ -n "${ROBOT_ID}" ]] && echo "  ROBOT_ID=${ROBOT_ID}"
    [[ -n "${ROBOT_CALIB_DIR}" ]] && echo "  ROBOT_CALIB_DIR=${ROBOT_CALIB_DIR}"
    [[ -n "${TASK2_DURATION_S}" ]] && echo "  TASK2_DURATION_S=${TASK2_DURATION_S}"
    [[ -n "${TASK3_DURATION_S}" ]] && echo "  TASK3_DURATION_S=${TASK3_DURATION_S}"
    [[ -n "${WAIT_TASK2_TO_TASK3_S}" ]] && echo "  WAIT_TASK2_TO_TASK3_S=${WAIT_TASK2_TO_TASK3_S}"
    [[ -n "${SHOW}" ]] && echo "  SHOW=${SHOW}"
    [[ -n "${VISION_CAM_INDEX}" ]] && echo "  VISION_CAM_INDEX=${VISION_CAM_INDEX}"
  fi
  echo "============================================"
  echo
fi

# Run main runtime
python Panbot/control/main_runtime.py --config "${FINAL_CONFIG}"

# Cleanup patched file
if [[ -n "${PATCHED_CONFIG:-}" && -f "${PATCHED_CONFIG}" ]]; then
  rm -f "${PATCHED_CONFIG}" || true
fi
