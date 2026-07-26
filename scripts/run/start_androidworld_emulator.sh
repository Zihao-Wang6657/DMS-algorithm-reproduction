#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK="$(cd "$SCRIPTS_DIR/.." && pwd)"
SESSION="dms_androidworld"
LOG="$WORK/logs/androidworld_emulator.log"
APK_PATH="$WORK/downloads/accessibility_forwarder.apk"
AVD_PATH="$WORK/android_avd/AndroidWorldAvd.ini"

cd "$WORK"
source "$SCRIPTS_DIR/common/activate_env.sh"

for command in adb emulator tmux; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "missing required command: $command" >&2
    echo "run scripts/setup/setup_androidworld_runtime.sh first" >&2
    exit 2
  fi
done

if [[ ! -f "$APK_PATH" ]]; then
  echo "missing accessibility forwarder apk: $APK_PATH" >&2
  echo "run scripts/setup/setup_androidworld_runtime.sh first" >&2
  exit 2
fi

if [[ ! -f "$AVD_PATH" ]]; then
  echo "missing AndroidWorld AVD: $AVD_PATH" >&2
  echo "run scripts/setup/setup_androidworld_runtime.sh first" >&2
  exit 2
fi

mkdir -p "$WORK/logs"
: > "$LOG"

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "session=$SESSION"
  echo "avd=AndroidWorldAvd"
  echo "grpc_port=8554"
  echo "console_port=5554"
  echo
  echo "[emulator accel check]"
  emulator -accel-check || true
  echo
} | tee -a "$LOG"

tmux kill-session -t "$SESSION" 2>/dev/null || true
adb kill-server >/dev/null 2>&1 || true
rm -f "$WORK/android_avd/AndroidWorldAvd.avd/"*.lock 2>/dev/null || true

gpu_flag="swiftshader_indirect"
if [[ "$OSTYPE" == *linux* ]]; then
  gpu_flag="off"
fi

tmux new-session -d -s "$SESSION" \
  "cd '$WORK' && source '$SCRIPTS_DIR/common/activate_env.sh' && unset CUDA_VISIBLE_DEVICES && export LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe && emulator -avd AndroidWorldAvd -no-window -show-kernel -no-audio -no-boot-anim -no-snapshot -no-metrics -memory 2048 -gpu '$gpu_flag' -feature -Vulkan -accel on -grpc 8554 -feature -Wifi >> '$LOG' 2>&1"

echo "$SESSION" > "$WORK/logs/androidworld_emulator.tmux_session"

for attempt in $(seq 1 180); do
  state="$(adb get-state 2>/dev/null || true)"
  boot="$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r' || true)"
  anim="$(adb shell getprop init.svc.bootanim 2>/dev/null | tr -d '\r' || true)"
  echo "attempt=$attempt state=$state boot=$boot bootanim=$anim" | tee -a "$LOG"
  if [[ "$boot" == "1" ]]; then
    break
  fi
  sleep 2
done

if ! adb shell pm list packages 2>/dev/null | grep -q '^package:com.google.androidenv.accessibilityforwarder$'; then
  adb install -r "$APK_PATH" | tee -a "$LOG"
fi

{
  echo
  echo "[final adb devices]"
  adb devices -l
  echo
  echo "[boot props]"
  adb shell 'echo sys.boot_completed=$(getprop sys.boot_completed); echo dev.bootcomplete=$(getprop dev.bootcomplete); echo bootanim=$(getprop init.svc.bootanim); uptime'
} | tee -a "$LOG"
