#!/usr/bin/env bash
set -euo pipefail

# 프로젝트 루트에서 실행한다고 가정
# (예: repo root)
CONFIG="Panbot/config/runtime.yaml"

echo "[start_all] using config: ${CONFIG}"
python -m Panbot.control.main_runtime --config "${CONFIG}"
