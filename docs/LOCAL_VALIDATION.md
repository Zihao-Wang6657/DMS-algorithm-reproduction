# Local Validation Report

Validation date: 2026-07-18 (Asia/Shanghai)

This report records the local project snapshot used for environment validation.

## Completed checks

The reorganized repository was validated with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_quick_checks.ps1
```

Results:

- Unit and integration tests: `84 passed in 1.11s`.
- AndroidWorld `bug_suite.yaml`: 3 tasks across 5 referenced apps; every task
  resolved through the local task registry.
- AndroidWorld `mini_benchmark_20apps.yaml`: 20 tasks across 20 apps; every
  task resolved, instantiated, and produced a deterministic goal and step
  budget.
- Bash entrypoints: every `scripts/*.sh` file passed `bash -n` syntax checking
  under WSL2.
- Resolved runtime paths: repository root and
  `third_party/android_world` both resolve correctly after the directory move.

The pre- and post-move test result was identical (`81 passed`). The three new
tests cover the OpenAI-compatible image request, retry behavior, and mandatory
token-usage fields. DMS memory algorithms, agent prompts, action parsing and
AndroidWorld evaluators were not changed for the remote deployment.

The detailed dataset-validation record is generated locally at
`logs/dataset_validation.json`.

## Windows AndroidWorld status

The native Windows Android environment was validated after the remote transport
was added:

- Android Emulator acceleration: `WHPX(10.0.26200) is installed and usable`.
- AVD: `AndroidWorldAvd`, Pixel 6, API 33, x86_64.
- Device: `emulator-5554`, boot completed, gRPC port 8554 reachable.
- AndroidWorld controller: connected successfully; logical screen 1080x2400;
  21 UI elements returned from the launcher.
- Required benchmark apps including Markor, Tasks, Joplin, VLC, OsmAnd,
  Broccoli, Calendar, Gallery and Retro Music are installed.
- The local `all-MiniLM-L6-v2` DMS embedder loaded on CPU and returned
  384-dimensional embeddings.

The AndroidWorld check command is:

```powershell
$env:PYTHONPATH="$PWD\src;$PWD\third_party\android_world"
conda run -n android_world --no-capture-output python `
  scripts/check_androidworld_env.py `
  --runtime-config configs/runtime_windows.yaml
```

## Remote Qwen status

The detected GPU is an NVIDIA GeForce RTX 4060 with 8 GiB VRAM. The unchanged
model client loads Qwen2.5-VL-7B entirely on one CUDA device in bfloat16
(`device_map={"": 0}`). Seven billion bfloat16 parameters require roughly 14 GB
for weights alone, so the formal VLM smoke and AndroidWorld task run cannot fit
this GPU without changing the upstream model-loading logic. No quantization,
CPU offload or a smaller model was added. Instead, the same model and generation
settings are served on the AutoDL RTX 4090 through a loopback-only vLLM endpoint
and an encrypted SSH local forward. The real remote model smoke remains pending
until the password-authenticated server setup commands are run by the user.

The complete server and Windows procedure is documented in
`docs/REMOTE_MODEL_SETUP.md`.

## Command after the remote service is ready

With the emulator and SSH tunnel running, execute:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_remote_model_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_experiment_windows.ps1 `
  -Method baseline_a_zero_shot `
  -Dataset datasets/smoke_open_settings.yaml -Rounds 1
```

After that succeeds, the three 5-round commands in the remote setup guide use
the same algorithm configuration, task list, seed policy, and AndroidWorld
runtime.
