#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import yaml

from dms.android_tasks import TaskSpec, instantiate_task, step_budget
from dms.io_utils import now_iso


PROTOCOL_ID = "formal_mini_5tasks_balanced_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--preflight-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol-id", default=PROTOCOL_ID)
    parser.add_argument("--formal-run-root")
    parser.add_argument("--device-image-manifest")
    parser.add_argument("--restore-script")
    parser.add_argument("--orchestrator-script")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    dataset_path = Path(args.dataset).resolve()
    config_path = Path(args.config).resolve()
    model_path = Path(args.model_config).resolve()
    runtime_path = Path(args.runtime_config).resolve()
    preflight_path = Path(args.preflight_summary).resolve()
    output_dir = Path(args.output_dir).resolve()
    protocol_id = str(args.protocol_id)
    formal_run_root = (
        Path(args.formal_run_root).resolve()
        if args.formal_run_root
        else (repo_root / "runs/formal_mini_5tasks_balanced_v1_20260719").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not preflight.get("all_passed"):
        raise RuntimeError("Cannot freeze protocol until every manifest task passes preflight.")
    if int(preflight.get("token_usage", -1)) != 0 or int(
        preflight.get("scored_steps", -1)
    ) != 0:
        raise RuntimeError("Preflight must be non-scoring and consume zero model tokens.")
    if not preflight.get("model_inference_probe_ok"):
        raise RuntimeError("Preflight must include one successful real non-scoring model inference.")

    dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    specs = [TaskSpec.from_mapping(item) for item in dataset["tasks"]]
    if not specs:
        raise ValueError("Formal dataset must contain at least one task.")
    if int(preflight.get("passed", 0)) != len(specs):
        raise RuntimeError(
            f"Preflight passed {preflight.get('passed')} tasks but the manifest has {len(specs)}."
        )

    tasks: list[dict[str, Any]] = []
    evaluator_files: set[Path] = set()
    for index, spec in enumerate(specs):
        first = instantiate_task(spec)
        second = instantiate_task(spec)
        first_signature = (
            first.name,
            first.goal,
            tuple(first.app_names),
            float(first.complexity),
            step_budget(first),
        )
        second_signature = (
            second.name,
            second.goal,
            tuple(second.app_names),
            float(second.complexity),
            step_budget(second),
        )
        if first_signature != second_signature:
            raise RuntimeError(
                f"Task {spec.name} is not deterministic under seed {spec.seed}: "
                f"{first_signature!r} != {second_signature!r}"
            )
        # Some AndroidWorld information-retrieval tasks are dynamically created
        # with ``abc`` as their reported module. Hash every repository-backed
        # class in the MRO so their real evaluator implementation is covered,
        # while deliberately excluding Python standard-library sources.
        for task_class in inspect.getmro(type(first)):
            try:
                source = inspect.getsourcefile(task_class)
            except TypeError:
                source = None
            if not source:
                continue
            source_path = Path(source).resolve()
            try:
                source_path.relative_to(repo_root)
            except ValueError:
                continue
            evaluator_files.add(source_path)
        tasks.append(
            {
                "order": index + 1,
                "name": first.name,
                "seed": spec.seed,
                "param_overrides": spec.param_overrides or {},
                "goal": first.goal,
                "app_names": list(first.app_names),
                "complexity": float(first.complexity),
                "official_step_budget": step_budget(first),
                "budget_source": "AndroidWorld int(10 * task.complexity)",
            }
        )

    fixed_paths = {
        dataset_path,
        config_path,
        model_path,
        runtime_path,
        preflight_path,
        repo_root / "src/dms/formal_runner.py",
        repo_root / "src/dms/runner.py",
        repo_root / "src/dms/config.py",
        repo_root / "src/dms/io_utils.py",
        repo_root / "src/dms/agent.py",
        repo_root / "src/dms/prompts.py",
        repo_root / "src/dms/actions.py",
        repo_root / "src/dms/paper_tools.py",
        repo_root / "src/dms/darwinian_memory.py",
        repo_root / "src/dms/static_memory.py",
        repo_root / "src/dms/android_tasks.py",
        repo_root / "src/env/androidworld_env.py",
        repo_root / "src/model_client/remote_qwen_vl.py",
        repo_root / "src/model_client/contracts.py",
        repo_root / "scripts/preflight_formal_20apps.py",
        repo_root / "scripts/freeze_formal_protocol.py",
        repo_root / "scripts/analyze_main_experiment.py",
        repo_root / "scripts/run_formal_main_experiment_windows.ps1",
        repo_root / "scripts/remote/start_vllm_server.sh",
        repo_root / "scripts/remote/serve_vllm_foreground.sh",
        repo_root / "scripts/remote/check_vllm_server.sh",
        repo_root / "third_party/android_world/android_world/suite_utils.py",
        repo_root / "third_party/android_world/android_world/task_evals/task_eval.py",
        repo_root / "third_party/android_world/android_world/task_evals/utils/sqlite_utils.py",
        repo_root / "third_party/android_world/android_world/task_evals/utils/user_data_generation.py",
        repo_root / "third_party/android_world/android_world/env/setup_device/apps.py",
        repo_root / "third_party/android_world/android_world/utils/app_snapshot.py",
    } | evaluator_files
    device_policy: dict[str, Any] | None = None
    if args.device_image_manifest:
        device_manifest_path = Path(args.device_image_manifest).resolve()
        restore_script_path = Path(args.restore_script).resolve()
        orchestrator_script_path = Path(args.orchestrator_script).resolve()
        fixed_paths |= {
            device_manifest_path,
            restore_script_path,
            orchestrator_script_path,
            repo_root / "scripts/device_image_manifest.py",
            repo_root / "scripts/setup_androidworld_apps.py",
        }
        device_manifest = json.loads(device_manifest_path.read_text(encoding="utf-8"))
        device_policy = {
            "round_boundary_reset": "restore validated golden AVD and official app snapshots",
            "device_image_manifest": _relative(repo_root, device_manifest_path),
            "device_tree_sha256": device_manifest["tree_sha256"],
            "device_asset_file_count": device_manifest["file_count"],
            "device_asset_total_bytes": device_manifest["total_bytes"],
            "restore_script": _relative(repo_root, restore_script_path),
            "orchestrator_script": _relative(repo_root, orchestrator_script_path),
            "host_experiment_state_preserved": True,
            "host_state_fingerprint_checked_before_and_after_restore": True,
            "memory_is_never_stored_inside_the_avd": True,
            "official_app_snapshots_reinjected_after_cold_boot": 24,
        }
    hashes = {
        _relative(repo_root, path): _sha256(path)
        for path in sorted(fixed_paths, key=lambda item: str(item).lower())
    }

    algorithm_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    runtime_config = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    manifest = {
        "protocol_id": protocol_id,
        "frozen_at": now_iso(),
        "workspace_root": str(repo_root),
        "formal_run_root": str(formal_run_root),
        "frozen_inputs": {
            "dataset": _relative(repo_root, dataset_path),
            "config": _relative(repo_root, config_path),
            "model_config": _relative(repo_root, model_path),
            "runtime_config": _relative(repo_root, runtime_path),
        },
        "excluded_run_roots": [
            str((repo_root / "runs/formal_main_20apps_5rounds_v1_20260718").resolve()),
            str((repo_root / "runs/formal_main_device_separated_v4_20260719").resolve()),
        ],
        "rounds": 5,
        "tasks_per_round": len(specs),
        "method_order": [
            "baseline_a_zero_shot",
            "baseline_b_static_memory",
            "dms_hierarchical_memory",
        ],
        "task_order_repeated_identically_each_round": True,
        "tasks": tasks,
        "algorithm_config": algorithm_config,
        "model_config": model_config,
        "runtime_config": runtime_config,
        "prompt_source": "src/dms/prompts.py",
        "remote_inference_server": {
            "software": "vLLM 0.10.2",
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "gpu": "NVIDIA GeForce RTX 4090 24 GiB",
            "host": "127.0.0.1",
            "port": 8000,
            "dtype": "bfloat16",
            "max_model_len": 32768,
            "max_num_seqs": 1,
            "gpu_memory_utilization": 0.92,
            "limit_mm_per_prompt": {"image": 1, "video": 0},
            "min_pixels": 200704,
            "max_pixels": 1003520,
            "transport": "Windows localhost SSH forward to AutoDL localhost",
        },
        "androidworld_runtime": {
            "mode": "Windows local Android emulator",
            "avd": "AndroidWorldAvd",
            "device": "emulator-5554",
            "observation": "UIAutomator plus screenshot",
            "grpc_port": 8554,
            "official_app_snapshot_count": 24,
            "official_app_snapshot_path": "/data/data/android_world/snapshots",
        },
        "device_state_policy": device_policy,
        "open_app_shortcut": {
            "enabled": True,
            "source": "src/dms/agent.py::_shortcut_open_app_action",
            "effect": (
                "Uses task app metadata to inject a legal start_app/open_app action, "
                "bypassing Actor model generation for that launch step and recording "
                "zero model tokens for the injected action."
            ),
            "disclosure": (
                "Retained from the selected upstream implementation; it lowers app-launch "
                "difficulty and can inflate success on app-opening-heavy tasks."
            ),
        },
        "success_rule": {
            "androidworld_reward": "success requires reward >= 1.0",
            "step_budget": "scoring_attempt_steps < int(10 * task.complexity)",
            "step_semantics": "only calls passed to env.execute_action consume the AndroidWorld action budget",
            "control_turns": "planner and non-environment control outputs are separately counted and capped",
            "normal_failures_are_not_retried": True,
            "normal_model_failure_examples": [
                "wrong page",
                "wrong click or text input",
                "invalid action JSON",
                "model cannot complete or declares failure",
                "scoring attempt reaches or exceeds official step budget",
                "AndroidWorld returns success=0 without an evaluator exception",
            ],
        },
        "infrastructure_error_examples": [
            "ADB or emulator disconnect",
            "SSH tunnel or remote model API interruption",
            "AndroidWorld evaluator exception",
            "forest=None or otherwise unavailable success judgment",
            "task initialization or cleanup exception",
            "computer restart or formal runner process termination",
        ],
        "infrastructure_retry_rule": {
            "max_retries_per_task": 1,
            "model_client_hidden_request_retries": 0,
            "same_task_seed_prompt_parameters": True,
            "attempt_1_is_audited_but_not_scored": True,
            "attempt_1_memory_state_is_rolled_back": True,
            "attempt_2_is_final": True,
            "attempt_2_infrastructure_error_category": (
                "infrastructure_failure_after_retry"
            ),
            "resource_accounting": "sum tokens and steps across both attempts",
            "per_attempt_artifacts_retained": True,
        },
        "task_selection_disclosure": {
            "balanced_difficulty": "two easy, two medium, one hard",
            "selection_bias": (
                "Task selection was informed by abandoned diagnostic runs; no task, seed, "
                "or order may change after freeze."
            ),
        },
        "post_start_changes_forbidden": [
            "task replacement",
            "seed change",
            "success evaluator change",
            "step budget change",
            "Prompt or model parameter change",
            "task-specific action rule",
            "unbounded retry",
            "deletion of a normal failure",
        ],
        "preflight_summary": _relative(repo_root, preflight_path),
        "sha256": hashes,
    }
    manifest_path = output_dir / "protocol_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "sha256sums.txt").write_text(
        "\n".join(f"{digest}  {path}" for path, digest in hashes.items()) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Formal Balanced Mini Experiment Protocol v1",
        "",
        f"- Protocol ID: `{protocol_id}`",
        f"- Frozen at: `{manifest['frozen_at']}`",
        f"- Scale: {len(specs)} tasks × 5 rounds × 3 methods = {len(specs) * 15} scored records",
        "- Method order: Baseline A → Baseline B → DMS",
        "- Budget success rule: scoring attempt steps must be strictly below the official budget",
        "- Infrastructure retry: at most once with the identical task, seed, Prompt and parameters",
        "- Resource accounting: cumulative across the infrastructure attempt and final attempt",
        "- Pilot data: explicitly excluded",
        "- Device state: restored from a validated golden AVD before every round",
        "- Experiment state: host-side results, audit logs, memory and RNG state persist across rounds",
        "",
        "## Frozen task order",
        "",
        "| # | Task | Seed | App | Complexity | Official budget |",
        "| ---: | --- | ---: | --- | ---: | ---: |",
    ]
    for item in tasks:
        lines.append(
            f"| {item['order']} | `{item['name']}` | {item['seed']} | "
            f"{', '.join(item['app_names'])} | {item['complexity']} | "
            f"{item['official_step_budget']} |"
        )
    lines.extend(
        [
            "",
            "## Task selection disclosure",
            "",
            "The fixed list contains two easy, two medium, and one hard task.",
            "Selection was informed by abandoned diagnostic runs, so conclusions are",
            "limited to this mini benchmark and the selection bias is explicit.",
            "",
            "## Integrity",
            "",
            "Every formal runner process verifies the SHA256 mapping in",
            "`protocol_manifest.json` before execution and before each task.",
        ]
    )
    (output_dir / "PROTOCOL.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), "sha256": hashes}, indent=2))


if __name__ == "__main__":
    main()
