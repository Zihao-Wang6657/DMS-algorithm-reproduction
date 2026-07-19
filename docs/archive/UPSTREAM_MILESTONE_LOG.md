# Milestone Log

All timestamps are Asia/Shanghai local time. DMS core mechanisms are not
implemented in this milestone batch.

## 2026-06-14 Milestones 0-5

### Milestone 0: Workspace Classification

Status: completed.

Project files are organized under:

- `working_space/src/dms`: reproduction runner, PA-Lite agent, task
  helpers, action mapping, observations, static memory.
- `working_space/src/model_client`: Qwen2.5-VL local inference client.
- `working_space/src/android_world`: local AndroidWorld source.
- `working_space/configs`: runtime/model/eval configs.
- `working_space/datasets`: selected task suites.
- `working_space/scripts`: current reproduction scripts.
- `working_space/logs`: environment and validation evidence.
- `working_space/runs`: baseline run artifacts.

### Milestone 1: AndroidWorld Dynamic Benchmark Environment

Status: completed.

Evidence:

- Emulator AVD: `AndroidWorldAvd`
- Console device: `emulator-5554`
- gRPC port: `8554`
- Background session: `dms_androidworld`
- Latest env check: `working_space/logs/androidworld_env_check.json`
- Env check result: `androidworld_env_connected=true`, `boot_completed=1`,
  screen size `1080x2400`.
- Final boot probe: `adb shell getprop sys.boot_completed` returned `1`.

Note: the emulator process was already alive in a historical tmux session named
`dms_emulator`; the session was renamed to `dms_androidworld` without restarting
the emulator or changing AVD state.

### Milestone 2: Test Suites

Status: completed.

Evidence:

- Bug suite: `working_space/datasets/bug_suite.yaml`
  - `OpenAppTaskEval`
  - `SystemWifiTurnOnVerify`
  - `ClockStopWatchPausedVerify`
- Mini-benchmark: `working_space/datasets/mini_benchmark_20apps.yaml`
  - 20 tasks covering 20 AndroidWorld app scenarios.
- Validation log: `working_space/logs/dataset_validation.json`
  - `bug_suite`: 3 tasks.
  - `mini_benchmark_20apps`: 20 tasks, 20 apps.

Implementation note: dataset YAML stores AndroidWorld task class names, seeds,
and overrides. Runtime task parameters are generated through AndroidWorld task
APIs rather than inventing synthetic tasks.

### Milestone 3: Qwen2.5-VL-7B-Instruct Skeleton

Status: completed.

Evidence:

- Model config: `working_space/configs/model_qwen25vl_7b.yaml`
- Smoke log: `working_space/logs/qwen_vl_smoke.json`
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Local/offline loading: enabled.
- Device: `CUDA_VISIBLE_DEVICES=5`
- Dtype: `bfloat16`
- Smoke result: `structured_output_valid=true`, peak GPU memory about
  `15.61 GiB`, latency about `9.71 s`.

### Milestone 4: Baseline A, Memory-Free PA-Lite

Status: completed as runnable baseline implementation.

Code evidence:

- `working_space/src/dms/agent.py`
- `working_space/src/dms/actions.py`
- `working_space/src/dms/prompts.py`
- `working_space/src/dms/runner.py`
- `working_space/scripts/run_baseline_a_zero_shot.sh`

Run evidence:

- Run dir: `working_space/runs/baseline_a_zero_shot/20260614_174021`
- Metrics: `working_space/runs/baseline_a_zero_shot/20260614_174021/metrics.json`
- Steps: `working_space/runs/baseline_a_zero_shot/20260614_174021/steps.jsonl`
- Observation screenshots/UI trees:
  `working_space/runs/baseline_a_zero_shot/20260614_174021/observations`

Metrics summary:

- method: `baseline_a_zero_shot`
- tasks: `1`
- successful_tasks: `0`
- total_steps: `2`
- memory_size: `0`

Known behavior from smoke: the runner executed end-to-end, but Qwen2.5-VL-7B
grounded the visible Settings icon incorrectly. The UI tree showed Settings at
index `5`, while the model selected index `14`. This is recorded as model/action
grounding failure, not an environment or runner crash.

### Milestone 5: Baseline B, Static Memory

Status: completed as runnable baseline implementation.

Code evidence:

- `working_space/src/dms/static_memory.py`
- `working_space/src/dms/agent.py`
- `working_space/src/dms/runner.py`
- `working_space/scripts/run_baseline_b_static_memory.sh`

Run evidence:

- Run dir: `working_space/runs/baseline_b_static_memory/20260614_174220`
- Metrics:
  `working_space/runs/baseline_b_static_memory/20260614_174220/metrics.json`
- Static memory:
  `working_space/runs/baseline_b_static_memory/20260614_174220/static_memory.jsonl`
- Steps:
  `working_space/runs/baseline_b_static_memory/20260614_174220/steps.jsonl`

Metrics summary:

- method: `baseline_b_static_memory`
- tasks: `1`
- successful_tasks: `0`
- total_steps: `4`
- memory_size: `1`
- `static_memory.jsonl` line count: `1`

Static memory behavior:

- Historical trajectory is appended chronologically after task completion.
- No pruning, scoring, mutation, dual-factor retrieval, Bayesian risk, Survival
  Value, or DMS replacement is implemented.
- The persisted memory remains append-only; `max_context_entries` is only a
  prompt-context guard.

Known behavior from smoke: same model/action grounding issue as Baseline A. The
baseline is runnable and records append-only memory, but this smoke task did not
succeed.

## Paper-Alignment Notes

- PA-Lite uses Planner-Actor structure with Planner emitting 1-5 sub-tasks.
- Sub-tasks use precondition and goal fields.
- Baseline A has no cross-task memory.
- Baseline B appends historical trajectories chronologically and does not prune.
- DMS parameters are only recorded in `working_space/docs/PAPER_ALIGNMENT.md`
  and are not used by Baseline A/B.
- Model-visible Baseline A/B actions are restricted to paper-compatible
  executable actions: `swipe`, `input_text`, `press_key`, `tap`, `start_app`,
  `complete`.
- Paper `remember` is reserved for the later DMS memory-writing stage.
