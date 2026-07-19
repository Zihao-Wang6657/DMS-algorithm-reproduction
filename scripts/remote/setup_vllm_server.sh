#!/usr/bin/env bash
set -euo pipefail

VLLM_VERSION="${DMS_VLLM_VERSION:-0.10.2}"
RUNTIME_ROOT="${DMS_REMOTE_ROOT:-/root/autodl-tmp/dms_remote}"
ENV_PREFIX="${DMS_VLLM_ENV_PREFIX:-$RUNTIME_ROOT/conda_env}"

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
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    return 0
  fi
  return 1
}

if ! find_conda; then
  echo "Conda was not found. Run this script from the AutoDL interactive base shell." >&2
  exit 2
fi

mkdir -p "$RUNTIME_ROOT/logs" "$RUNTIME_ROOT/huggingface"

available_kib="$(df -Pk "$RUNTIME_ROOT" | awk 'NR==2 {print $4}')"
if (( available_kib < 30 * 1024 * 1024 )); then
  echo "At least 30 GiB free is required under $RUNTIME_ROOT." >&2
  df -h "$RUNTIME_ROOT" >&2
  exit 2
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  conda create -y -p "$ENV_PREFIX" python=3.11
fi

conda run -p "$ENV_PREFIX" python -m pip install --no-cache-dir --upgrade pip
conda run -p "$ENV_PREFIX" python -m pip install --no-cache-dir "vllm==$VLLM_VERSION"
# vLLM 0.10.2 declares only a lower Transformers bound. Newer releases
# normalize Qwen MRoPE fields in a way that conflicts with vLLM 0.10.2's
# legacy validator, so keep the contemporaneous tested floor explicitly.
conda run -p "$ENV_PREFIX" python -m pip install --no-cache-dir \
  "transformers==4.55.2"

if ! command -v tmux >/dev/null 2>&1; then
  apt-get update
  apt-get install -y tmux curl
fi

echo "vllm_environment_ready=1"
echo "environment=$ENV_PREFIX"
echo "runtime_root=$RUNTIME_ROOT"
echo "next=bash scripts/remote/start_vllm_server.sh"
