# Darwinian Memory System Reproduction TODO

This TODO follows the requested order and keeps DMS implementation out of scope
until AndroidWorld, Qwen2.5-VL, and baselines A/B are verified.

## Milestone 0: Workspace Classification

Status: completed

- Keep third-party AndroidWorld source under `working_space/src/android_world`.
- Keep reproduction code under `working_space/src/dms`.
- Keep model client code under `working_space/src/model_client`.
- Keep reproducible configs under `working_space/configs`.
- Keep selected task suites under `working_space/datasets`.
- Keep run logs and milestone evidence under `working_space/logs` and
  `working_space/runs`.
- Replace old residual scripts in `working_space/scripts` with the current
  reproduction scripts.

Evidence:

- `working_space/docs/TODO.md`
- `working_space/docs/PAPER_ALIGNMENT.md`
- `working_space/docs/MILESTONE_LOG.md`
- `working_space/src/dms`

## Milestone 1: AndroidWorld Dynamic Benchmark Environment

Status: completed

- Start or reuse a live Android emulator named `AndroidWorldAvd` with gRPC port
  `8554`.
- Keep it detached in tmux session `dms_androidworld`.
- Verify boot status via adb.
- Verify AndroidWorld can connect via `env_launcher.load_and_setup_env`.
- Install/confirm the AndroidWorld accessibility forwarder.

Evidence:

- `working_space/logs/androidworld_emulator.log`
- `working_space/logs/androidworld_env_check.json`
- `tmux list-sessions` shows `dms_androidworld`.
- `adb shell getprop sys.boot_completed` returns `1`.

## Milestone 2: Test Suites

Status: completed

- Create a small bug suite with 2-3 simple tasks for debugging.
- Create a mini-benchmark suite that covers all 20 AndroidWorld app scenarios.
- Use AndroidWorld task classes and generated parameters; do not invent tasks.
- Record selected task names, app coverage, seeds, parameters, and goals.

Evidence:

- `working_space/datasets/bug_suite.yaml`
- `working_space/datasets/mini_benchmark_20apps.yaml`
- `working_space/logs/dataset_validation.json`

## Milestone 3: Qwen2.5-VL-7B-Instruct Minimal Skeleton

Status: completed

- Load Qwen2.5-VL-7B-Instruct from the local Hugging Face cache.
- Use `CUDA_VISIBLE_DEVICES=5`.
- Run one image-plus-text structured generation smoke test.
- Record model path, dtype, token counts, latency, and output JSON validity.

Evidence:

- `working_space/configs/model_qwen25vl_7b.yaml`
- `working_space/logs/qwen_vl_smoke.json`

## Milestone 4: Baseline A

Status: completed

- Implement memory-free PA-Lite / zero-shot VLM baseline.
- Use Planner-Actor loop with Planner 1-5 functional sub-tasks.
- Store only per-task execution history needed for current task control.
- Do not carry experience across tasks.

Evidence:

- `working_space/src/dms/agent.py`
- `working_space/scripts/run_baseline_a_zero_shot.sh`
- `working_space/runs/baseline_a_zero_shot/20260614_174021/metrics.json`

## Milestone 5: Baseline B

Status: completed

- Implement static memory baseline.
- Append historical task trajectories chronologically.
- Do not prune, score, mutate, retrieve by dual factors, or apply DMS risk.
- Feed static history context back into PA-Lite prompts.

Evidence:

- `working_space/src/dms/static_memory.py`
- `working_space/scripts/run_baseline_b_static_memory.sh`
- `working_space/runs/baseline_b_static_memory/20260614_174220/metrics.json`
- `working_space/runs/baseline_b_static_memory/20260614_174220/static_memory.jsonl`

## Next Milestone

Status: pending user confirmation

- Do not start DMS until explicitly requested.
- Recommended next step is to run the bug suite after deciding whether to keep
  pure PA-Lite behavior or add a documented execution-safety validator for
  invalid UI indices.

## Explicitly Deferred

Status: not started

- DMS hierarchical memory.
- Dual-factor retrieval.
- Survival Value.
- Adaptive pruning via Elbow Method.
- Bayesian risk/feedback regulation.
- Epsilon mutation and evolutionary replacement.
