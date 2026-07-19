#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$WORK"
DATASET="${1:-$WORK/datasets/bug_suite.yaml}"
ROUNDS="${2:-1}"
GPU_ID="${3:-${GPU_ID:-0}}"

cd "$ROOT"
export GPU_ID
source "$WORK/scripts/activate_env.sh"
python -m dms.runner \
  --method dms_hierarchical_memory \
  --config "$WORK/configs/eval_baselines.yaml" \
  --dataset "$DATASET" \
  --rounds "$ROUNDS"
