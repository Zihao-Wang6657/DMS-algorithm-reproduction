#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${DMS_REMOTE_ROOT:-/root/autodl-tmp/dms_remote}"
ENV_PREFIX="${DMS_VLLM_ENV_PREFIX:-$RUNTIME_ROOT/conda_env}"
MODEL_ID="${DMS_MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
HF_HOME="${HF_HOME:-$RUNTIME_ROOT/huggingface}"
export HF_HOME
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

find_conda() {
  local candidate
  for candidate in \
    /root/miniconda3/etc/profile.d/conda.sh \
    /root/anaconda3/etc/profile.d/conda.sh \
    /opt/conda/etc/profile.d/conda.sh; do
    if [[ -f "$candidate" ]]; then
      # shellcheck disable=SC1090
      source "$candidate"
      return 0
    fi
  done
  eval "$(conda shell.bash hook)"
}
find_conda

exec conda run --no-capture-output -p "$ENV_PREFIX" vllm serve "$MODEL_ID" \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"min_pixels":200704,"max_pixels":1003520}' \
  --trust-remote-code
