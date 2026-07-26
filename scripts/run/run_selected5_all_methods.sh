#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$(cd "$SCRIPTS_DIR/.." && pwd)"

DATASET="$WORKSPACE/datasets/mini_benchmark_probe5.yaml"
CONFIG="$WORKSPACE/configs/eval_baselines.yaml"
ROUNDS=1
RUN_ROOT=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Run the selected five AndroidWorld tasks with Baseline A, Baseline B, and DMS.

Usage:
  bash scripts/run/run_selected5_all_methods.sh [options]

Options:
  --rounds N         Number of rounds for each method (default: 1).
  --dataset PATH     YAML dataset containing exactly five tasks.
  --config PATH      Evaluation config (default: configs/eval_baselines.yaml).
  --run-root PATH    New output directory. It must not already contain files.
  --dry-run          Validate inputs and print the commands without running tasks.
  -h, --help         Show this help message.

With no options, the script runs:
  datasets/mini_benchmark_probe5.yaml
  Baseline A -> Baseline B -> DMS
  one round per method
EOF
}

while (($# > 0)); do
  case "$1" in
    --rounds)
      [[ $# -ge 2 ]] || { echo "ERROR: --rounds requires a value." >&2; exit 2; }
      ROUNDS="$2"
      shift 2
      ;;
    --dataset)
      [[ $# -ge 2 ]] || { echo "ERROR: --dataset requires a path." >&2; exit 2; }
      DATASET="$2"
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || { echo "ERROR: --config requires a path." >&2; exit 2; }
      CONFIG="$2"
      shift 2
      ;;
    --run-root)
      [[ $# -ge 2 ]] || { echo "ERROR: --run-root requires a path." >&2; exit 2; }
      RUN_ROOT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "$ROUNDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --rounds must be a positive integer." >&2
  exit 2
}

DATASET="$(realpath -m "$DATASET")"
CONFIG="$(realpath -m "$CONFIG")"
[[ -f "$DATASET" ]] || { echo "ERROR: dataset not found: $DATASET" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 2; }

if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="$WORKSPACE/runs/selected5_3methods_${ROUNDS}rounds_$(date +%Y%m%d_%H%M%S)"
fi
RUN_ROOT="$(realpath -m "$RUN_ROOT")"

if [[ -d "$RUN_ROOT" ]] && [[ -n "$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "ERROR: run root is not empty; refusing to resume or overwrite: $RUN_ROOT" >&2
  exit 2
fi

METHOD_SPECS=(
  "baseline_a_zero_shot:baseline_a"
  "baseline_b_static_memory:baseline_b"
  "dms_hierarchical_memory:dms"
)

VALIDATION_PYTHON="$WORKSPACE/conda_envs/dms_py310/bin/python"
if [[ ! -x "$VALIDATION_PYTHON" ]]; then
  VALIDATION_PYTHON="$(command -v python3 || command -v python || true)"
fi
[[ -n "$VALIDATION_PYTHON" ]] || {
  echo "ERROR: no Python interpreter is available for dataset validation." >&2
  exit 2
}

if ((DRY_RUN)); then
  "$VALIDATION_PYTHON" - "$DATASET" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
payload = yaml.safe_load(path.read_text(encoding="utf-8"))
tasks = payload.get("tasks", [])
if len(tasks) != 5:
    raise SystemExit(
        f"ERROR: selected-task dataset must contain exactly 5 tasks; found {len(tasks)}."
    )
print("Validated five tasks:")
for index, task in enumerate(tasks, start=1):
    print(f"  {index}. {task['name']} (seed={task.get('seed')})")
PY
  echo
  echo "Dry run only. Commands that would be executed:"
  for spec in "${METHOD_SPECS[@]}"; do
    method="${spec%%:*}"
    short_name="${spec##*:}"
    printf 'python -u -m dms.runner --method %q --config %q --dataset %q --rounds %q --run-dir %q\n' \
      "$method" "$CONFIG" "$DATASET" "$ROUNDS" "$RUN_ROOT/$short_name"
  done
  exit 0
fi

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/launcher.stdout.log") 2>&1

CURRENT_METHOD_FILE="$RUN_ROOT/current_method.txt"
STREAM_PIDS=()
RUNNER_PID=""

cleanup() {
  if [[ -n "$RUNNER_PID" ]] && kill -0 "$RUNNER_PID" 2>/dev/null; then
    kill "$RUNNER_PID" 2>/dev/null || true
  fi
  for pid in "${STREAM_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}

on_error() {
  status=$?
  printf 'failed\n' > "$CURRENT_METHOD_FILE"
  echo "ERROR: selected-five run failed with exit code $status."
  echo "RunRoot: $RUN_ROOT"
  cleanup
  exit "$status"
}

trap on_error ERR
trap 'cleanup; exit 130' INT TERM

export GPU_ID="${GPU_ID:-0}"
source "$SCRIPTS_DIR/common/activate_env.sh"

python - "$DATASET" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
payload = yaml.safe_load(path.read_text(encoding="utf-8"))
tasks = payload.get("tasks", [])
if len(tasks) != 5:
    raise SystemExit(
        f"ERROR: selected-task dataset must contain exactly 5 tasks; found {len(tasks)}."
    )
print("Selected tasks:")
for index, task in enumerate(tasks, start=1):
    print(f"  {index}. {task['name']} (seed={task.get('seed')})")
PY

echo
echo "Preflight: model endpoint"
curl --fail --silent --show-error \
  --max-time 10 \
  "http://127.0.0.1:8000/v1/models" >/dev/null
echo "  model endpoint: ready"

echo "Preflight: AndroidWorld and accessibility"
python -u "$SCRIPTS_DIR/monitor/check_androidworld_env.py"

STEP_STREAM_PY="$(cat <<'PY'
import json
import sys

method = sys.argv[1]
for raw in sys.stdin:
    try:
        item = json.loads(raw)
        action = item.get("executed_action") or item.get("action") or {}
        action_name = action.get("type", "-") if isinstance(action, dict) else "-"
        print(
            f"[step][{method}] step={item.get('step')} "
            f"result={item.get('result')} action={action_name} "
            f"subtask={item.get('subtask')}",
            flush=True,
        )
    except Exception as exc:
        print(f"[step][{method}] unreadable log line: {exc}", flush=True)
PY
)"

RESULT_STREAM_PY="$(cat <<'PY'
import json
import sys

method = sys.argv[1]
for raw in sys.stdin:
    try:
        item = json.loads(raw)
        tokens = int(item.get("input_tokens", 0)) + int(item.get("output_tokens", 0))
        print(
            f"[task][{method}] {item.get('task_id')} "
            f"success={item.get('success')} steps={item.get('steps')} "
            f"tokens={tokens} memory={item.get('memory_size_after')}",
            flush=True,
        )
    except Exception as exc:
        print(f"[task][{method}] unreadable log line: {exc}", flush=True)
PY
)"

stream_steps() {
  local runner_pid="$1"
  local path="$2"
  local method="$3"
  tail --pid="$runner_pid" --sleep-interval=0.5 -n 0 -F "$path" 2>/dev/null |
    python -u -c "$STEP_STREAM_PY" "$method"
}

stream_results() {
  local runner_pid="$1"
  local path="$2"
  local method="$3"
  tail --pid="$runner_pid" --sleep-interval=0.5 -n 0 -F "$path" 2>/dev/null |
    python -u -c "$RESULT_STREAM_PY" "$method"
}

for spec in "${METHOD_SPECS[@]}"; do
  method="${spec%%:*}"
  short_name="${spec##*:}"
  method_dir="$RUN_ROOT/$short_name"
  method_log="$RUN_ROOT/${short_name}.stdout.log"
  mkdir -p "$method_dir"
  : > "$method_dir/steps.jsonl"
  : > "$method_dir/task_results.jsonl"
  printf '%s\n' "$short_name" > "$CURRENT_METHOD_FILE"

  echo
  echo "================================================================"
  echo "Starting $short_name"
  echo "Method:  $method"
  echo "Rounds:  $ROUNDS"
  echo "Dataset: $DATASET"
  echo "Output:  $method_dir"
  echo "================================================================"

  python -u -m dms.runner \
    --method "$method" \
    --config "$CONFIG" \
    --dataset "$DATASET" \
    --rounds "$ROUNDS" \
    --run-dir "$method_dir" \
    > >(tee -a "$method_log") 2>&1 &
  RUNNER_PID=$!

  stream_steps "$RUNNER_PID" "$method_dir/steps.jsonl" "$short_name" &
  STREAM_PIDS+=("$!")
  stream_results "$RUNNER_PID" "$method_dir/task_results.jsonl" "$short_name" &
  STREAM_PIDS+=("$!")

  if wait "$RUNNER_PID"; then
    runner_status=0
  else
    runner_status=$?
  fi
  RUNNER_PID=""

  for pid in "${STREAM_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  STREAM_PIDS=()

  if ((runner_status != 0)); then
    echo "ERROR: $short_name exited with code $runner_status."
    printf 'failed\n' > "$CURRENT_METHOD_FILE"
    exit "$runner_status"
  fi

  python - "$method_dir/metrics.json" "$short_name" <<'PY'
import json
from pathlib import Path
import sys

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
name = sys.argv[2]
tasks = int(metrics.get("tasks", 0))
successes = int(metrics.get("successful_tasks", 0))
rate = 100.0 * float(metrics.get("success_rate", 0.0))
print(
    f"Completed {name}: {successes}/{tasks} successful "
    f"({rate:.1f}%), steps={metrics.get('total_steps')}, "
    f"memory={metrics.get('memory_size')}"
)
PY
done

printf 'complete\n' > "$CURRENT_METHOD_FILE"

echo
echo "================================================================"
echo "All three methods completed."
echo "RunRoot: $RUN_ROOT"
echo "================================================================"

python - "$RUN_ROOT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
print(f"{'method':<12} {'success':>10} {'rate':>9} {'steps':>9} {'memory':>9}")
for name in ("baseline_a", "baseline_b", "dms"):
    metrics = json.loads((root / name / "metrics.json").read_text(encoding="utf-8"))
    tasks = int(metrics.get("tasks", 0))
    successes = int(metrics.get("successful_tasks", 0))
    rate = 100.0 * float(metrics.get("success_rate", 0.0))
    print(
        f"{name:<12} {successes:>4}/{tasks:<5} "
        f"{rate:>8.1f}% {int(metrics.get('total_steps', 0)):>9} "
        f"{int(metrics.get('memory_size', 0)):>9}"
    )
PY
