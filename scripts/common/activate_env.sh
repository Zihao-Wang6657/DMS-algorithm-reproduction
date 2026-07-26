#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export DMS_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
CONDA_ENV_PREFIX="$DMS_ROOT/conda_envs/dms_py310"

_conda_exe="${CONDA_EXE:-}"
if [[ -z "$_conda_exe" ]]; then
  if command -v conda >/dev/null 2>&1; then
    _conda_exe="$(command -v conda)"
  elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    _conda_exe="$HOME/miniconda3/bin/conda"
  elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
    _conda_exe="$HOME/anaconda3/bin/conda"
  fi
fi
if [[ -n "$_conda_exe" ]]; then
  export CONDA_EXE="$_conda_exe"
fi

_gpu_id="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
export GPU_ID="$_gpu_id"
export CUDA_VISIBLE_DEVICES="$_gpu_id"
export MODEL_REQUIRE_CUDA_VISIBLE_DEVICES="${MODEL_REQUIRE_CUDA_VISIBLE_DEVICES:-$_gpu_id}"
export HF_HOME="${HF_HOME:-$DMS_ROOT/model_cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export JAVA_HOME="$DMS_ROOT/jdks/jdk17"
export ANDROID_HOME="$DMS_ROOT/android_sdk"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export ANDROID_AVD_HOME="$DMS_ROOT/android_avd"
export PYTHONPATH="$DMS_ROOT/third_party/android_world:$DMS_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$JAVA_HOME/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator:$ANDROID_HOME/cmdline-tools/latest/bin:$CONDA_ENV_PREFIX/bin:$PATH"

if [[ -n "${CONDA_EXE:-}" ]]; then
  source "$("$CONDA_EXE" info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_PREFIX"
else
  export CONDA_PREFIX="$CONDA_ENV_PREFIX"
  export CONDA_DEFAULT_ENV="$CONDA_ENV_PREFIX"
fi
