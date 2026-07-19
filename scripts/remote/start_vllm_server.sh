#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${DMS_REMOTE_ROOT:-/root/autodl-tmp/dms_remote}"
SESSION_NAME="${DMS_VLLM_SESSION:-dms_vllm}"
LOG_FILE="$RUNTIME_ROOT/logs/vllm_server.log"
mkdir -p "$(dirname "$LOG_FILE")"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "vllm_session_already_running=1"
  echo "session=$SESSION_NAME"
  echo "log=$LOG_FILE"
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME" \
  "bash '$SCRIPT_DIR/serve_vllm_foreground.sh' >>'$LOG_FILE' 2>&1"

echo "vllm_server_started=1"
echo "session=$SESSION_NAME"
echo "log=$LOG_FILE"
echo "watch=tail -f $LOG_FILE"
echo "check=bash $SCRIPT_DIR/check_vllm_server.sh"
