# Windows AndroidWorld + AutoDL Qwen API

This deployment keeps AndroidWorld and the DMS runner on Windows and serves the
unchanged `Qwen/Qwen2.5-VL-7B-Instruct` model on the AutoDL RTX 4090. The API is
bound to the server loopback interface and is reachable only through an SSH
local forward.

The DMS algorithms, prompts, action parsing, task generation, evaluators and
memory parameters still come from `configs/eval_baselines.yaml`. Only the model
transport and OS-specific runtime paths are overridden.

## 1. Upload and prepare the AutoDL model service

From Windows PowerShell:

```powershell
scp -o PubkeyAuthentication=no -o PreferredAuthentications=password `
  -P 42258 C:\Users\Administrator\Desktop\DMS-remote-api-20260718.tar.gz `
  root@connect.nmb1.seetacloud.com:/root/autodl-tmp/
```

Log in and extract into a new directory so existing server data is preserved:

```powershell
ssh -o PubkeyAuthentication=no -o PreferredAuthentications=password `
  -p 42258 root@connect.nmb1.seetacloud.com
```

Then run in the remote shell:

```bash
mkdir -p /root/autodl-tmp/dms-hybrid
tar -xzf /root/autodl-tmp/DMS-remote-api-20260718.tar.gz \
  -C /root/autodl-tmp/dms-hybrid
cd /root/autodl-tmp/dms-hybrid
bash scripts/remote/setup_vllm_server.sh
bash scripts/remote/start_vllm_server.sh
tail -f /root/autodl-tmp/dms_remote/logs/vllm_server.log
```

The first start downloads the model and can take several minutes. Once the log
reports that the API server is listening, press `Ctrl+C` to leave `tail`; the
model continues running inside tmux. Verify it with:

The server launcher defaults to AutoDL's documented Hugging Face acceleration
endpoint (`https://hf-mirror.com`) for the one-time model download. Once cached
under `/root/autodl-tmp/dms_remote/huggingface`, model inference does not depend
on that endpoint.

```bash
cd /root/autodl-tmp/dms-hybrid
bash scripts/remote/check_vllm_server.sh
```

The vLLM launch uses the same model, BF16 dtype, image processor limits
(`200704..1003520` pixels), greedy decoding and single-image request limit as
the local reproduction configuration. The server environment pins vLLM 0.10.2
with Transformers 4.55.2 to avoid later Transformers releases rewriting Qwen's
legacy MRoPE fields incompatibly with that vLLM release.

## 2. Start the Windows AndroidWorld emulator

From the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_local_androidworld.ps1
```

The script reuses a healthy `emulator-5554`; otherwise it starts
`AndroidWorldAvd` with `-no-snapshot -grpc 8554`. It never wipes AVD data.

## 3. Open the encrypted model tunnel

Open a second PowerShell window and keep it open:

```powershell
cd C:\Users\Administrator\Desktop\DMS
powershell -ExecutionPolicy Bypass -File scripts/open_model_tunnel.ps1
```

Enter the AutoDL password when prompted. The model becomes available only at
`http://127.0.0.1:8000/v1` on this PC; no public vLLM port is exposed.

## 4. Validate the model and AndroidWorld

In a third PowerShell window:

```powershell
cd C:\Users\Administrator\Desktop\DMS
powershell -ExecutionPolicy Bypass -File scripts/run_remote_model_smoke.ps1

$env:PYTHONPATH="$PWD\src;$PWD\third_party\android_world"
conda run -n android_world --no-capture-output python `
  scripts/check_androidworld_env.py `
  --runtime-config configs/runtime_windows.yaml
```

Before the DMS method is used for the first time, download its small local text
embedding model:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_local_embedder.ps1
```

## 5. Run the experiments

One-task Baseline A smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_experiment_windows.ps1 `
  -Method baseline_a_zero_shot `
  -Dataset datasets/smoke_open_settings.yaml `
  -Rounds 1
```

Formal 20-app, 5-round comparison:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_experiment_windows.ps1 `
  -Method baseline_a_zero_shot `
  -Dataset datasets/mini_benchmark_20apps.yaml -Rounds 5

powershell -ExecutionPolicy Bypass -File scripts/run_experiment_windows.ps1 `
  -Method baseline_b_static_memory `
  -Dataset datasets/mini_benchmark_20apps.yaml -Rounds 5

powershell -ExecutionPolicy Bypass -File scripts/run_experiment_windows.ps1 `
  -Method dms_hierarchical_memory `
  -Dataset datasets/mini_benchmark_20apps.yaml -Rounds 5
```

The remote API must return `usage.prompt_tokens` and
`usage.completion_tokens`. The client fails closed if these fields are absent,
so token comparisons cannot silently become invalid. API transport latency is
included in wall-clock time; success rate, steps, tokens and memory statistics
retain their original definitions.
