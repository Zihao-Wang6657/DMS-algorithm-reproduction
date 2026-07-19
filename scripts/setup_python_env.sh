#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_PREFIX="$WORK/conda_envs/dms_py310"
REQ_FILE="$WORK/requirements.txt"
PATCH_ANDROID_ENV_SCRIPT="$SCRIPT_DIR/patch_android_env_py310.py"

discover_conda() {
  if [[ -n "${CONDA_EXE:-}" ]]; then
    printf '%s\n' "$CONDA_EXE"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi
  if [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    printf '%s\n' "$HOME/miniconda3/bin/conda"
    return 0
  fi
  if [[ -x "$HOME/anaconda3/bin/conda" ]]; then
    printf '%s\n' "$HOME/anaconda3/bin/conda"
    return 0
  fi
  return 1
}

if ! CONDA_BIN="$(discover_conda 2>/dev/null)"; then
  echo "conda is required for setup_python_env.sh" >&2
  echo "please install Miniconda or Anaconda first" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$("$CONDA_BIN" info --base)/etc/profile.d/conda.sh"
mkdir -p "$(dirname "$ENV_PREFIX")"
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

if [[ ! -d "$ENV_PREFIX/conda-meta" ]]; then
  conda create -y -p "$ENV_PREFIX" python=3.10
fi

conda activate "$ENV_PREFIX"
python -s -m pip install --upgrade pip setuptools wheel
python -s -m pip install -r "$REQ_FILE"
python -s "$PATCH_ANDROID_ENV_SCRIPT"
python -s - <<'PY'
from __future__ import annotations

import android_env.components.app_screen_checker as app_screen_checker

print(f"android_env_import_check={app_screen_checker.__file__}")
PY

echo "python_env=$ENV_PREFIX"
echo "activate_cmd=source \"$("$CONDA_BIN" info --base)/etc/profile.d/conda.sh\" && conda activate \"$ENV_PREFIX\""
python --version
