#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK="$(cd "$SCRIPTS_DIR/.." && pwd)"
ENV_PREFIX="$WORK/conda_envs/dms_py310"

if [[ ! -d "$ENV_PREFIX/conda-meta" ]]; then
  echo "conda environment is missing: $ENV_PREFIX" >&2
  echo "run scripts/setup/setup_python_env.sh first" >&2
  exit 2
fi

cd "$WORK"

"$SCRIPT_DIR/setup_androidworld_runtime.sh"

# shellcheck disable=SC1091
source "$SCRIPTS_DIR/common/activate_env.sh"

"$SCRIPT_DIR/download_models.sh"
"$SCRIPTS_DIR/run/start_androidworld_emulator.sh"
python "$SCRIPT_DIR/setup_androidworld_apps.py"
python "$SCRIPTS_DIR/monitor/check_androidworld_env.py"

echo "runtime_assets_ready=1"
echo "qwen_model_cache=$WORK/model_cache/huggingface"
echo "embedding_model_dir=$WORK/model_cache/modelscope/AI-ModelScope/all-MiniLM-L6-v2"
echo "android_sdk=$WORK/android_sdk"
echo "android_avd=$WORK/android_avd"
echo "java_home=$WORK/jdks/jdk17"
echo "accessibility_forwarder_apk=$WORK/downloads/accessibility_forwarder.apk"
echo "emulator_tmux_session=dms_androidworld"
echo "emulator_log=$WORK/logs/androidworld_emulator.log"
echo "androidworld_setup_log=$WORK/logs/androidworld_setup.json"
echo "androidworld_env_check_log=$WORK/logs/androidworld_env_check.json"
