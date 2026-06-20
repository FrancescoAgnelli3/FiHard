#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/experiment.yaml}"
PYTHON_BIN="${DIFFUSION_HANDS_PYTHON:-$ROOT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

DIFFUSION_HANDS_PYTHON="$PYTHON_BIN" "$PYTHON_BIN" "$ROOT_DIR/tools/run_all_models.py" --config "$CONFIG_PATH"
