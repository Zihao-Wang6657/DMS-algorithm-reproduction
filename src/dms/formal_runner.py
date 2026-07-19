from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pickle
import shutil
import traceback
from pathlib import Path
from typing import Any

from dms.agent import DMSAgent, PALiteAgent
from dms.android_tasks import TaskSpec, instantiate_task, step_budget
from dms.config import apply_runtime_environment, load_yaml, resolve_path
from dms.darwinian_memory import DarwinianMemorySystem
from dms.io_utils import append_jsonl, now_iso, write_json
from dms.runner import _instantiate_dms_memory
from dms.static_memory import StaticMemory


MEMORY_ARTIFACTS = (
    "static_memory.jsonl",
    "memory_bank.json",
    "memory_trajectories",
    "dms_retrievals.jsonl",
    "dms_pruning.jsonl",
    "dms_mutations.jsonl",
    "dms_events.jsonl",
    "dms_summary.json",
    "dms_random_state.json",
)


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_hashes(manifest_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workspace_root = Path(manifest["workspace_root"]).resolve()
    mismatches: list[str] = []
    for relative_path, expected in manifest["sha256"].items():
        path = (workspace_root / relative_path).resolve()
        actual = _sha256(path) if path.is_file() else "MISSING"
        if actual != expected:
            mismatches.append(
                f"{relative_path}: expected={expected} actual={actual}"
            )
    if mismatches:
        raise RuntimeError(
            "Formal protocol hash verification failed; this RunRoot must not "
            "continue as a formal experiment.\n" + "\n".join(mismatches)
        )
    return str(manifest["protocol_id"])


class AttemptMeteredModel:
    def __init__(self, base: Any, log_path: Path) -> None:
        self.base = base
        self.log_path = log_path
        self.call_index = 0

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        self.call_index += 1
        started_at = now_iso()
        try:
            result = self.base.generate(*args, **kwargs)
        except Exception as exc:
            append_jsonl(
                self.log_path,
                {
                    "call_index": self.call_index,
                    "started_at": started_at,
                    "finished_at": now_iso(),
                    "status": "infrastructure_error",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "error": "".join(traceback.format_exception(exc)),
                },
            )
            raise
        append_jsonl(
            self.log_path,
            {
                "call_index": self.call_index,
                "started_at": started_at,
                "finished_at": now_iso(),
                "status": "complete",
                "input_tokens": int(result.input_tokens),
                "output_tokens": int(result.output_tokens),
                "latency_seconds": float(result.latency_seconds),
            },
        )
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_task_specs(path: Path) -> list[TaskSpec]:
    payload = load_yaml(path)
    return [TaskSpec.from_mapping(item) for item in payload["tasks"]]


def _build_memory(
    *,
    method: str,
    config: dict[str, Any],
    run_dir: Path,
) -> Any | None:
    if method == "baseline_a_zero_shot":
        return None
    if method == "baseline_b_static_memory":
        memory_cfg = config.get("baseline_b_static_memory", {})
        return StaticMemory(
            run_dir / "static_memory.jsonl",
            max_context_entries=int(memory_cfg.get("max_context_entries", 12)),
        )
    memory = _instantiate_dms_memory(config, run_dir=run_dir)
    _load_dms_random_state(memory, run_dir / "dms_random_state.json")
    return memory


def _build_attempt_agent(
    *,
    method: str,
    config: dict[str, Any],
    model: Any,
    memory: Any | None,
    artifact_dir: Path,
) -> PALiteAgent:
    pa_config = config["pa_lite"]
    common = {
        "model": model,
        "run_dir": artifact_dir,
        "max_subtasks": int(pa_config["planner_max_subtasks"]),
        "actor_local_step_guard": int(pa_config["actor_local_step_guard"]),
        "post_action_wait_seconds": float(
            pa_config.get("post_action_wait_seconds", 3.0)
        ),
        "engineering_optimization": bool(
            pa_config.get("engineering_optimization", False)
        ),
        "planner_max_cycles": (
            int(pa_config["planner_max_cycles"])
            if pa_config.get("planner_max_cycles") is not None
            else None
        ),
        "control_turn_limit": (
            int(pa_config["control_turn_limit"])
            if pa_config.get("control_turn_limit") is not None
            else None
        ),
    }
    if method == "dms_hierarchical_memory":
        return DMSAgent(**common, dms_memory=memory)
    return PALiteAgent(
        **common,
        static_memory=memory if method == "baseline_b_static_memory" else None,
    )


def _save_dms_random_state(memory: Any | None, path: Path) -> None:
    if not isinstance(memory, DarwinianMemorySystem):
        return
    encoded = base64.b64encode(pickle.dumps(memory.random.getstate())).decode("ascii")
    _atomic_write_json(path, {"encoding": "pickle-base64", "state": encoded})


def _load_dms_random_state(memory: Any | None, path: Path) -> None:
    if not isinstance(memory, DarwinianMemorySystem) or not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    memory.random.setstate(pickle.loads(base64.b64decode(payload["state"])))


def _snapshot_memory(
    *,
    memory: Any | None,
    run_dir: Path,
    checkpoint_dir: Path,
) -> None:
    _save_dms_random_state(memory, run_dir / "dms_random_state.json")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True)
    present: list[str] = []
    for name in MEMORY_ARTIFACTS:
        source = run_dir / name
        if not source.exists():
            continue
        present.append(name)
        destination = checkpoint_dir / name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    _atomic_write_json(checkpoint_dir / "checkpoint_manifest.json", {"present": present})


def _restore_memory(*, run_dir: Path, checkpoint_dir: Path) -> None:
    manifest = json.loads(
        (checkpoint_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    present = set(manifest["present"])
    for name in MEMORY_ARTIFACTS:
        target = run_dir / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        if name not in present:
            continue
        source = checkpoint_dir / name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def _attempt_usage(artifact_dir: Path) -> tuple[int, int]:
    calls = _load_jsonl(artifact_dir / "model_calls.jsonl")
    return (
        sum(int(item.get("input_tokens", 0)) for item in calls),
        sum(int(item.get("output_tokens", 0)) for item in calls),
    )


def _attempt_steps(artifact_dir: Path) -> int:
    return len(_load_jsonl(artifact_dir / "steps.jsonl"))


def _interrupted_attempt_record(
    *,
    task_id: str,
    task: Any,
    attempt_number: int,
    artifact_dir: Path,
    started_at: str,
) -> dict[str, Any]:
    input_tokens, output_tokens = _attempt_usage(artifact_dir)
    return {
        "task_id": task_id,
        "task_name": task.name,
        "goal": task.goal,
        "attempt_number": attempt_number,
        "status": (
            "infrastructure_error_attempt_1"
            if attempt_number == 1
            else "infrastructure_error_attempt_2"
        ),
        "success": False,
        "reward": 0.0,
        "steps": _attempt_steps(artifact_dir),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "started_at": started_at,
        "finished_at": now_iso(),
        "artifact_dir": str(artifact_dir.resolve()),
        "trajectory": _load_jsonl(artifact_dir / "steps.jsonl"),
        "error": (
            "[infrastructure_phase=process_lifecycle]\n"
            "The previous formal-runner process terminated before this attempt "
            "returned a TaskRunResult (for example, computer restart or process crash)."
        ),
    }


def _attempt_record_from_result(
    *,
    result: dict[str, Any],
    attempt_number: int,
    artifact_dir: Path,
    started_at: str,
) -> dict[str, Any]:
    infrastructure_error = bool(result.get("error"))
    return {
        **result,
        "attempt_number": attempt_number,
        "status": (
            f"infrastructure_error_attempt_{attempt_number}"
            if infrastructure_error
            else "normal_scored_attempt"
        ),
        "started_at": started_at,
        "finished_at": now_iso(),
        "artifact_dir": str(artifact_dir.resolve()),
    }


def _append_attempt_once(path: Path, record: dict[str, Any]) -> None:
    key = (str(record["task_id"]), int(record["attempt_number"]))
    existing = {
        (str(item["task_id"]), int(item["attempt_number"]))
        for item in _load_jsonl(path)
    }
    if key not in existing:
        append_jsonl(path, record)


def _recovery_action(
    *, attempt_number: int, returned_record: dict[str, Any] | None
) -> str:
    """Chooses a protocol-safe action for a durable or interrupted attempt."""
    if returned_record is not None and (
        not returned_record.get("error") or attempt_number >= 2
    ):
        return "finalize_returned"
    if attempt_number >= 2:
        return "finalize_interrupted_after_restore"
    return "retry_after_restore"


def _open_env(runtime_config: dict[str, Any]) -> Any:
    from android_world.env import env_launcher

    return env_launcher.load_and_setup_env(
        console_port=int(runtime_config["android"]["console_port"]),
        emulator_setup=False,
        adb_path=runtime_config["android"]["adb_path"],
        grpc_port=int(runtime_config["android"]["grpc_port"]),
    )


def _recover_env(env: Any, runtime_config: dict[str, Any]) -> Any:
    try:
        env.close()
    except Exception:
        pass
    return _open_env(runtime_config)


def _run_attempt(
    *,
    canonical_task_id: str,
    spec: TaskSpec,
    attempt_number: int,
    prior_attempts: list[dict[str, Any]],
    method: str,
    config: dict[str, Any],
    base_model: Any,
    memory: Any | None,
    env: Any,
    run_dir: Path,
    journal_path: Path,
    active_result_path: Path,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    artifact_dir = (
        run_dir
        / "attempt_artifacts"
        / canonical_task_id
        / f"attempt_{attempt_number}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started_at = now_iso()
    journal = {
        "task_id": canonical_task_id,
        "task_spec": spec.to_dict(),
        "attempt_number": attempt_number,
        "prior_attempts": prior_attempts,
        "phase": "running",
        "started_at": started_at,
        "artifact_dir": str(artifact_dir.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
    }
    _atomic_write_json(journal_path, journal)
    if active_result_path.exists():
        active_result_path.unlink()

    task = instantiate_task(spec)
    metered_model = AttemptMeteredModel(
        base_model,
        artifact_dir / "model_calls.jsonl",
    )
    agent = _build_attempt_agent(
        method=method,
        config=config,
        model=metered_model,
        memory=memory,
        artifact_dir=artifact_dir,
    )
    try:
        result = agent.run_task(
            env=env,
            task=task,
            task_id=canonical_task_id,
        ).to_dict()
        record = _attempt_record_from_result(
            result=result,
            attempt_number=attempt_number,
            artifact_dir=artifact_dir,
            started_at=started_at,
        )
    except Exception as exc:
        input_tokens, output_tokens = _attempt_usage(artifact_dir)
        record = {
            "task_id": canonical_task_id,
            "task_name": task.name,
            "goal": task.goal,
            "success": False,
            "reward": 0.0,
            "steps": _attempt_steps(artifact_dir),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "memory_size_after": int(getattr(memory, "size", 0) or 0),
            "memory_stats": {},
            "trajectory": _load_jsonl(artifact_dir / "steps.jsonl"),
            "error": "".join(traceback.format_exception(exc)),
            "attempt_number": attempt_number,
            "status": f"infrastructure_error_attempt_{attempt_number}",
            "started_at": started_at,
            "finished_at": now_iso(),
            "artifact_dir": str(artifact_dir.resolve()),
        }
    # Make the post-attempt DMS state durable before marking the attempt returned.
    # A process restart may then finalize this exact attempt without replaying it.
    _save_dms_random_state(memory, run_dir / "dms_random_state.json")
    _atomic_write_json(active_result_path, record)
    journal["phase"] = "attempt_returned"
    _atomic_write_json(journal_path, journal)
    return record


def _attempt_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key)
        for key in (
            "attempt_number",
            "status",
            "success",
            "reward",
            "steps",
            "control_turns",
            "event_count",
            "input_tokens",
            "output_tokens",
            "started_at",
            "finished_at",
            "artifact_dir",
            "error",
        )
    }


def _aggregate_final_result(
    *,
    canonical_task_id: str,
    final_attempt: dict[str, Any],
    attempts: list[dict[str, Any]],
    round_id: int,
    budget: int,
    protocol_id: str,
) -> dict[str, Any]:
    final_infrastructure_error = bool(final_attempt.get("error"))
    scoring_steps = int(final_attempt.get("steps", 0))
    within_budget = scoring_steps < budget
    raw_success = bool(final_attempt.get("success"))
    success = raw_success and within_budget and not final_infrastructure_error
    if final_infrastructure_error and len(attempts) >= 2:
        failure_category = "infrastructure_failure_after_retry"
    elif success:
        failure_category = "success"
    else:
        failure_category = "model_failure"
    total_input = sum(int(item.get("input_tokens", 0)) for item in attempts)
    total_output = sum(int(item.get("output_tokens", 0)) for item in attempts)
    total_steps = sum(int(item.get("steps", 0)) for item in attempts)
    total_control_turns = sum(int(item.get("control_turns", 0)) for item in attempts)
    total_events = sum(int(item.get("event_count", item.get("steps", 0))) for item in attempts)
    retry_attempts = attempts[:-1]
    return {
        "task_id": canonical_task_id,
        "task_name": final_attempt["task_name"],
        "goal": final_attempt["goal"],
        "success": success,
        "raw_androidworld_success": raw_success,
        "reward": float(final_attempt.get("reward", 0.0)),
        "official_step_budget": budget,
        "budget_rule": "scoring_attempt_steps < official_step_budget",
        "scoring_attempt_steps": scoring_steps,
        "steps": total_steps,
        "control_turns": total_control_turns,
        "event_count": total_events,
        "scoring_attempt_control_turns": int(final_attempt.get("control_turns", 0)),
        "control_turn_limit_reached": bool(
            final_attempt.get("control_turn_limit_reached", False)
        ),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "memory_size_after": int(final_attempt.get("memory_size_after", 0)),
        "memory_stats": final_attempt.get("memory_stats", {}),
        "failure_category": failure_category,
        "attempt_count": len(attempts),
        "infrastructure_retry_count": max(0, len(attempts) - 1),
        "infrastructure_retry_overhead_steps": sum(
            int(item.get("steps", 0)) for item in retry_attempts
        ),
        "infrastructure_retry_overhead_control_turns": sum(
            int(item.get("control_turns", 0)) for item in retry_attempts
        ),
        "infrastructure_retry_overhead_tokens": sum(
            int(item.get("input_tokens", 0))
            + int(item.get("output_tokens", 0))
            for item in retry_attempts
        ),
        "attempts": [_attempt_summary(item) for item in attempts],
        "trajectory": final_attempt.get("trajectory", []),
        "error": final_attempt.get("error") if final_infrastructure_error else None,
        "round": round_id,
        "protocol_id": protocol_id,
    }


def _clear_transaction(
    *,
    journal_path: Path,
    active_result_path: Path,
    checkpoint_dir: Path,
) -> None:
    for path in (journal_path, active_result_path):
        if path.exists():
            path.unlink()
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)


def _metrics(
    *,
    method: str,
    rounds: int,
    results: list[dict[str, Any]],
    memory: Any | None,
    protocol_id: str,
) -> dict[str, Any]:
    successes = sum(bool(item["success"]) for item in results)
    infrastructure_failures = sum(
        item.get("failure_category") == "infrastructure_failure_after_retry"
        for item in results
    )
    return {
        "timestamp": now_iso(),
        "method": method,
        "rounds": rounds,
        "tasks": len(results),
        "successful_tasks": successes,
        "success_rate": successes / len(results) if results else 0.0,
        "total_steps": sum(int(item["steps"]) for item in results),
        "total_control_turns": sum(
            int(item.get("control_turns", 0)) for item in results
        ),
        "input_tokens": sum(int(item["input_tokens"]) for item in results),
        "output_tokens": sum(int(item["output_tokens"]) for item in results),
        "infrastructure_retry_count": sum(
            int(item["infrastructure_retry_count"]) for item in results
        ),
        "infrastructure_failure_after_retry": infrastructure_failures,
        "memory_size": int(getattr(memory, "size", 0) or 0),
        "protocol_id": protocol_id,
        "results": results,
    }


def run_formal(args: argparse.Namespace) -> dict[str, Any]:
    from env.androidworld_env import configure_windows_adb_stability
    from model_client import QwenVLClient

    manifest_path = Path(args.manifest).resolve()
    protocol_id = verify_frozen_hashes(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workspace_root = Path(manifest["workspace_root"]).resolve()
    config_path = Path(args.config).resolve()
    model_config_path = Path(args.model_config).resolve()
    runtime_config_path = Path(args.runtime_config).resolve()
    dataset_path = Path(args.dataset).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    supplied_inputs = {
        "dataset": dataset_path,
        "config": config_path,
        "model_config": model_config_path,
        "runtime_config": runtime_config_path,
    }
    for name, supplied_path in supplied_inputs.items():
        expected_path = (
            workspace_root / manifest["frozen_inputs"][name]
        ).resolve()
        if supplied_path != expected_path:
            raise RuntimeError(
                f"Formal {name} path differs from the frozen protocol: "
                f"expected={expected_path} supplied={supplied_path}"
            )
    expected_method_dirs = {
        "baseline_a_zero_shot": "baseline_a",
        "baseline_b_static_memory": "baseline_b",
        "dms_hierarchical_memory": "dms",
    }
    expected_run_root = Path(manifest["formal_run_root"]).resolve()
    if run_dir.parent != expected_run_root or run_dir.name != expected_method_dirs[args.method]:
        raise RuntimeError(
            "Formal method output directory differs from the frozen RunRoot: "
            f"expected={expected_run_root / expected_method_dirs[args.method]} "
            f"supplied={run_dir}"
        )

    config = load_yaml(config_path)
    runtime_config = load_yaml(runtime_config_path)
    apply_runtime_environment(runtime_config)
    configure_windows_adb_stability(
        a11y_method=runtime_config.get("android", {}).get("a11y_method")
    )
    os.environ["DMS_STRICT_INFRA_PROTOCOL"] = "1"

    task_specs = _load_task_specs(dataset_path)
    expected_task_count = int(manifest["tasks_per_round"])
    expected_rounds = int(manifest["rounds"])
    if len(task_specs) != expected_task_count or args.rounds != expected_rounds:
        raise ValueError(
            "Formal inputs differ from the frozen scale: "
            f"tasks={len(task_specs)}/{expected_task_count}, "
            f"rounds={args.rounds}/{expected_rounds}."
        )
    round_limit = args.round_limit if args.round_limit is not None else args.rounds
    if round_limit < 1 or round_limit > args.rounds:
        raise ValueError(f"Formal round limit must be between 1 and {args.rounds}.")

    configured_base_url = str(
        load_yaml(model_config_path)["runtime"]["base_url"]
    ).rstrip("/")
    effective_base_url = os.environ.get(
        "DMS_MODEL_BASE_URL", configured_base_url
    ).rstrip("/")
    if effective_base_url != configured_base_url:
        raise RuntimeError(
            "Formal model endpoint differs from the frozen protocol: "
            f"expected={configured_base_url} supplied={effective_base_url}"
        )
    results_path = run_dir / "task_results.jsonl"
    attempts_path = run_dir / "attempt_results.jsonl"
    journal_path = run_dir / "active_task_transaction.json"
    active_result_path = run_dir / "active_attempt_result.json"
    checkpoint_dir = run_dir / ".active_memory_checkpoint"
    if not results_path.exists() and not journal_path.exists():
        unexpected_memory = [
            name for name in MEMORY_ARTIFACTS if (run_dir / name).exists()
        ]
        if unexpected_memory:
            raise RuntimeError(
                "Formal memory must start empty; unexpected artifacts found: "
                + ", ".join(unexpected_memory)
            )

    base_model = QwenVLClient(model_config_path)
    health = base_model.healthcheck()
    expected_model = str(load_yaml(model_config_path)["model"]["name"])
    served_models = {
        str(item.get("id"))
        for item in health.get("data", [])
        if isinstance(item, dict)
    }
    if expected_model not in served_models:
        raise RuntimeError(
            f"Frozen model {expected_model!r} is not served by the endpoint: "
            f"{sorted(served_models)}"
        )
    memory = _build_memory(
        method=args.method,
        config=config,
        run_dir=run_dir,
    )
    env = _open_env(runtime_config)
    results = _load_jsonl(results_path)
    completed = {str(item["task_id"]) for item in results}

    write_json(
        run_dir / "run_config.json",
        {
            "timestamp": now_iso(),
            "method": args.method,
            "config": str(config_path),
            "model_config": str(model_config_path),
            "runtime_config": str(runtime_config_path),
            "dataset": str(dataset_path),
            "manifest": str(manifest_path),
            "protocol_id": protocol_id,
            "rounds": args.rounds,
            "round_limit": round_limit,
            "task_specs": [spec.to_dict() for spec in task_specs],
        },
    )

    try:
        # The optional limit lets the infrastructure orchestrator isolate formal
        # rounds in separate processes. Canonical task ids, memory files, frozen
        # task order, and the declared five-round protocol remain unchanged.
        for round_index in range(round_limit):
            for task_index, spec in enumerate(task_specs):
                verify_frozen_hashes(manifest_path)
                task = instantiate_task(spec)
                canonical_task_id = (
                    f"r{round_index + 1:02d}_{task_index:03d}_{task.name}"
                )
                if canonical_task_id in completed:
                    if journal_path.is_file():
                        stale = json.loads(journal_path.read_text(encoding="utf-8"))
                        if stale.get("task_id") == canonical_task_id:
                            _clear_transaction(
                                journal_path=journal_path,
                                active_result_path=active_result_path,
                                checkpoint_dir=checkpoint_dir,
                            )
                    continue

                prior_attempts: list[dict[str, Any]] = []
                starting_attempt = 1
                if journal_path.is_file():
                    journal = json.loads(journal_path.read_text(encoding="utf-8"))
                    if journal.get("task_id") != canonical_task_id:
                        raise RuntimeError(
                            "Active transaction does not match the next incomplete task: "
                            f"active={journal.get('task_id')} next={canonical_task_id}"
                        )
                    artifact_dir = Path(journal["artifact_dir"])
                    attempt_number = int(journal["attempt_number"])
                    returned: dict[str, Any] | None = None
                    if active_result_path.is_file():
                        candidate = json.loads(
                            active_result_path.read_text(encoding="utf-8")
                        )
                        if (
                            str(candidate.get("task_id")) != canonical_task_id
                            or int(candidate.get("attempt_number", -1))
                            != attempt_number
                        ):
                            raise RuntimeError(
                                "Active attempt result does not match its transaction."
                            )
                        returned = candidate
                    recovered_attempt = returned
                    if recovered_attempt is None:
                        recovered_attempt = _interrupted_attempt_record(
                            task_id=canonical_task_id,
                            task=task,
                            attempt_number=attempt_number,
                            artifact_dir=artifact_dir,
                            started_at=str(journal.get("started_at", now_iso())),
                        )
                    _append_attempt_once(attempts_path, recovered_attempt)
                    prior_attempts = [
                        *list(journal.get("prior_attempts", [])),
                        recovered_attempt,
                    ]
                    recovery_action = _recovery_action(
                        attempt_number=attempt_number,
                        returned_record=returned,
                    )
                    if recovery_action == "finalize_returned":
                        final = _aggregate_final_result(
                            canonical_task_id=canonical_task_id,
                            final_attempt=returned,
                            attempts=prior_attempts,
                            round_id=round_index + 1,
                            budget=step_budget(task),
                            protocol_id=protocol_id,
                        )
                        append_jsonl(results_path, final)
                        write_json(run_dir / "latest_result.json", final)
                        results.append(final)
                        completed.add(canonical_task_id)
                        _clear_transaction(
                            journal_path=journal_path,
                            active_result_path=active_result_path,
                            checkpoint_dir=checkpoint_dir,
                        )
                        continue
                    _restore_memory(run_dir=run_dir, checkpoint_dir=checkpoint_dir)
                    memory = _build_memory(
                        method=args.method,
                        config=config,
                        run_dir=run_dir,
                    )
                    if recovery_action == "finalize_interrupted_after_restore":
                        final = _aggregate_final_result(
                            canonical_task_id=canonical_task_id,
                            final_attempt=recovered_attempt,
                            attempts=prior_attempts,
                            round_id=round_index + 1,
                            budget=step_budget(task),
                            protocol_id=protocol_id,
                        )
                        final["memory_size_after"] = int(
                            getattr(memory, "size", 0) or 0
                        )
                        append_jsonl(results_path, final)
                        write_json(run_dir / "latest_result.json", final)
                        results.append(final)
                        completed.add(canonical_task_id)
                        _clear_transaction(
                            journal_path=journal_path,
                            active_result_path=active_result_path,
                            checkpoint_dir=checkpoint_dir,
                        )
                        continue
                    starting_attempt = 2
                    env = _recover_env(env, runtime_config)

                if starting_attempt == 1:
                    _snapshot_memory(
                        memory=memory,
                        run_dir=run_dir,
                        checkpoint_dir=checkpoint_dir,
                    )

                attempt = _run_attempt(
                    canonical_task_id=canonical_task_id,
                    spec=spec,
                    attempt_number=starting_attempt,
                    prior_attempts=prior_attempts,
                    method=args.method,
                    config=config,
                    base_model=base_model,
                    memory=memory,
                    env=env,
                    run_dir=run_dir,
                    journal_path=journal_path,
                    active_result_path=active_result_path,
                    checkpoint_dir=checkpoint_dir,
                )
                _append_attempt_once(attempts_path, attempt)
                attempts = [*prior_attempts, attempt]

                if attempt.get("error") and starting_attempt == 1:
                    _restore_memory(run_dir=run_dir, checkpoint_dir=checkpoint_dir)
                    memory = _build_memory(
                        method=args.method,
                        config=config,
                        run_dir=run_dir,
                    )
                    try:
                        env = _recover_env(env, runtime_config)
                        retry = _run_attempt(
                            canonical_task_id=canonical_task_id,
                            spec=spec,
                            attempt_number=2,
                            prior_attempts=attempts,
                            method=args.method,
                            config=config,
                            base_model=base_model,
                            memory=memory,
                            env=env,
                            run_dir=run_dir,
                            journal_path=journal_path,
                            active_result_path=active_result_path,
                            checkpoint_dir=checkpoint_dir,
                        )
                    except Exception as exc:
                        retry_artifact = (
                            run_dir
                            / "attempt_artifacts"
                            / canonical_task_id
                            / "attempt_2"
                        )
                        retry_artifact.mkdir(parents=True, exist_ok=True)
                        retry = {
                            "task_id": canonical_task_id,
                            "task_name": task.name,
                            "goal": task.goal,
                            "success": False,
                            "reward": 0.0,
                            "steps": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "memory_size_after": int(
                                getattr(memory, "size", 0) or 0
                            ),
                            "memory_stats": {},
                            "trajectory": [],
                            "error": "".join(traceback.format_exception(exc)),
                            "attempt_number": 2,
                            "status": "infrastructure_error_attempt_2",
                            "started_at": now_iso(),
                            "finished_at": now_iso(),
                            "artifact_dir": str(retry_artifact.resolve()),
                        }
                    _append_attempt_once(attempts_path, retry)
                    attempts.append(retry)
                    attempt = retry

                _save_dms_random_state(memory, run_dir / "dms_random_state.json")
                final = _aggregate_final_result(
                    canonical_task_id=canonical_task_id,
                    final_attempt=attempt,
                    attempts=attempts,
                    round_id=round_index + 1,
                    budget=step_budget(task),
                    protocol_id=protocol_id,
                )
                append_jsonl(results_path, final)
                write_json(run_dir / "latest_result.json", final)
                results.append(final)
                completed.add(canonical_task_id)
                _clear_transaction(
                    journal_path=journal_path,
                    active_result_path=active_result_path,
                    checkpoint_dir=checkpoint_dir,
                )
    finally:
        try:
            env.close()
        except Exception:
            pass

    metrics = _metrics(
        method=args.method,
        rounds=args.rounds,
        results=results,
        memory=memory,
        protocol_id=protocol_id,
    )
    write_json(run_dir / "metrics.json", metrics)
    return {"run_dir": str(run_dir), **metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=(
            "baseline_a_zero_shot",
            "baseline_b_static_memory",
            "dms_hierarchical_memory",
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--round-limit", type=int)
    args = parser.parse_args()
    print(json.dumps(run_formal(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
