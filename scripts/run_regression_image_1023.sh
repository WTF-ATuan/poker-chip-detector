#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv311/bin/python"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/eval_overlap_single_image.py" \
  --gt "${ROOT_DIR}/data/regression/image_1023_gt.json" \
  --model "${ROOT_DIR}/runs/detect/color_train/weights/best.pt" \
  --output "${ROOT_DIR}/runs/regression/image_1023_eval.json"
