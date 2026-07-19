from __future__ import annotations

import argparse
from datetime import datetime
import importlib
from pathlib import Path
from typing import Any

from dms.agent import DMSAgent, PALiteAgent
from dms.config import apply_runtime_environment, load_yaml, resolve_path
from dms.io_utils import append_jsonl, now_iso, write_json
from dms.paths import workspace_path
from dms.static_memory import StaticMemory


def _method_memory_path(method: str) -> str:
    return "static_memory.jsonl" if method == "baseline_b_static_memory" else "dms_memory.jsonl"


def _resolve_object(import_path: str) -> Any:
    module_name, _, attribute = import_path.partition(":")
    if not module_name or not attribute:
        raise ValueError(
            f"Invalid import path {import_path!r}. Expected 'module.path:Attribute'."
        )
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ValueError(
            f"Import path {import_path!r} does not define attribute {attribute!r}."
        ) from exc


def _instantiate_dms_memory(
    config: dict[str, Any],
    *,
    run_dir: Path,
) -> Any:
    config_base_dir = Path(str(config.get("__config_path__", ""))).resolve().parent
    method_cfg = dict(config.get("dms", {}))
    backend_path = method_cfg.get("memory_backend")
    if not backend_path:
        raise ValueError(
            "DMS method requires config['dms']['memory_backend'] with "
            "a 'module.path:Attribute' import path."
        )

    backend_cls = _resolve_object(str(backend_path))
    init_kwargs = dict(method_cfg.get("memory_kwargs", {}))
    embedding_model_path = init_kwargs.get("embedding_model_path")
    if embedding_model_path:
        init_kwargs["embedding_model_path"] = str(
            resolve_path(embedding_model_path, base_dir=config_base_dir)
        )
    init_kwargs.setdefault("path", str(run_dir / _method_memory_path("dms_hierarchical_memory")))
    init_kwargs.setdefault("run_dir", str(run_dir))
    init_kwargs.setdefault("workspace_root", str(workspace_path()))
    return backend_cls(**init_kwargs)


def _build_agent(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    model: Any,
    run_dir: Path,
) -> tuple[PALiteAgent, Any | None]:
    post_action_wait_seconds = float(
        config.get("pa_lite", {}).get("post_action_wait_seconds", 3.0)
    )
    pa_config = config.get("pa_lite", {})
    common_agent_kwargs = {
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
    if args.method == "baseline_a_zero_shot":
        return (
            PALiteAgent(
                model=model,
                run_dir=run_dir,
                max_subtasks=int(config["pa_lite"]["planner_max_subtasks"]),
                actor_local_step_guard=int(config["pa_lite"]["actor_local_step_guard"]),
                post_action_wait_seconds=post_action_wait_seconds,
                **common_agent_kwargs,
            ),
            None,
        )

    if args.method == "baseline_b_static_memory":
        memory_cfg = config.get("baseline_b_static_memory", {})
        memory = StaticMemory(
            run_dir / _method_memory_path(args.method),
            max_context_entries=int(memory_cfg.get("max_context_entries", 12)),
        )
        return (
            PALiteAgent(
                model=model,
                run_dir=run_dir,
                max_subtasks=int(config["pa_lite"]["planner_max_subtasks"]),
                actor_local_step_guard=int(config["pa_lite"]["actor_local_step_guard"]),
                static_memory=memory,
                post_action_wait_seconds=post_action_wait_seconds,
                **common_agent_kwargs,
            ),
            memory,
        )

    if args.method == "dms_hierarchical_memory":
        memory = _instantiate_dms_memory(config, run_dir=run_dir)
        return (
            DMSAgent(
                model=model,
                run_dir=run_dir,
                dms_memory=memory,
                max_subtasks=int(config["pa_lite"]["planner_max_subtasks"]),
                actor_local_step_guard=int(config["pa_lite"]["actor_local_step_guard"]),
                post_action_wait_seconds=post_action_wait_seconds,
                **common_agent_kwargs,
            ),
            memory,
        )

    raise ValueError(f"Unsupported method for this stage: {args.method}")


def _load_task_specs(path: str | Path) -> list[TaskSpec]:
    from dms.android_tasks import TaskSpec

    data = load_yaml(path)
    return [TaskSpec.from_mapping(item) for item in data["tasks"]]


def _make_run_dir(method: str, run_dir: str | None) -> Path:
    if run_dir:
        return Path(run_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return workspace_path("runs", method, timestamp)


def _load_incremental_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    results: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        results.append(loads_json_line(stripped))
    return results


def loads_json_line(line: str) -> dict[str, Any]:
    import json

    return json.loads(line)


def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    from android_world.env import env_launcher
    from dms.android_tasks import instantiate_task
    from env.androidworld_env import configure_windows_adb_stability
    from model_client import QwenVLClient

    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    config_base_dir = config_path.parent
    model_config_override = getattr(args, "model_config", None)
    runtime_config_override = getattr(args, "runtime_config", None)
    model_config_path = (
        resolve_path(model_config_override)
        if model_config_override
        else resolve_path(config["model_config"], base_dir=config_base_dir)
    )
    runtime_config_path = (
        resolve_path(runtime_config_override)
        if runtime_config_override
        else resolve_path(config["runtime_config"], base_dir=config_base_dir)
    )

    runtime_config = load_yaml(runtime_config_path)
    apply_runtime_environment(runtime_config)
    configure_windows_adb_stability(
        a11y_method=runtime_config.get("android", {}).get("a11y_method")
    )
    run_dir = _make_run_dir(args.method, args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    task_specs = _load_task_specs(args.dataset)
    model = QwenVLClient(model_config_path)
    agent, memory = _build_agent(
        args=args,
        config=config,
        model=model,
        run_dir=run_dir,
    )

    env = env_launcher.load_and_setup_env(
        console_port=int(runtime_config["android"]["console_port"]),
        emulator_setup=False,
        adb_path=runtime_config["android"]["adb_path"],
        grpc_port=int(runtime_config["android"]["grpc_port"]),
    )
    run_config = {
        "timestamp": now_iso(),
        "method": args.method,
        "config": str(Path(args.config).resolve()),
        "model_config": str(model_config_path),
        "runtime_config": str(runtime_config_path),
        "dataset": str(Path(args.dataset).resolve()),
        "run_dir": str(run_dir.resolve()),
        "task_specs": [spec.to_dict() for spec in task_specs],
    }
    write_json(run_dir / "run_config.json", run_config)
    incremental_results_path = run_dir / "task_results.jsonl"
    results = _load_incremental_results(incremental_results_path)
    completed_task_ids = {
        str(item.get("task_id"))
        for item in results
        if str(item.get("task_id", "")).strip()
    }

    try:
        for round_index in range(args.rounds):
            for task_index, spec in enumerate(task_specs):
                task = instantiate_task(spec)
                task_id = (
                    f"r{round_index + 1:02d}_"
                    f"{task_index:03d}_{task.name}"
                )
                if task_id in completed_task_ids:
                    continue
                result = agent.run_task(env=env, task=task, task_id=task_id)
                record = result.to_dict()
                record["round"] = round_index + 1
                results.append(record)
                completed_task_ids.add(task_id)
                append_jsonl(incremental_results_path, record)
                write_json(run_dir / "latest_result.json", record)
    finally:
        env.close()

    successful = sum(1 for item in results if item["success"])
    metrics = {
        "timestamp": now_iso(),
        "method": args.method,
        "rounds": args.rounds,
        "tasks": len(results),
        "successful_tasks": successful,
        "success_rate": successful / len(results) if results else 0.0,
        "total_steps": sum(item["steps"] for item in results),
        "input_tokens": sum(item["input_tokens"] for item in results),
        "output_tokens": sum(item["output_tokens"] for item in results),
        "memory_size": (
            int(getattr(memory, "size"))
            if memory is not None and getattr(memory, "size", None) is not None
            else max((int(item.get("memory_size_after", 0)) for item in results), default=0)
        ),
        "memory_stats": results[-1].get("memory_stats", {}) if results else {},
        "results": results,
    }
    write_json(run_dir / "metrics.json", metrics)
    return {"run_dir": str(run_dir.resolve()), **metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=(
            "baseline_a_zero_shot",
            "baseline_b_static_memory",
            "dms_hierarchical_memory",
        ),
        required=True,
    )
    parser.add_argument(
        "--config",
        default=str(workspace_path("configs", "eval_baselines.yaml")),
    )
    parser.add_argument(
        "--model-config",
        help="Optional infrastructure override for the model transport config.",
    )
    parser.add_argument(
        "--runtime-config",
        help="Optional infrastructure override for the Android/runtime config.",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    result = run_baseline(args)
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
