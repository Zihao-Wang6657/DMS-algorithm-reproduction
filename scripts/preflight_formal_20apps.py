#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from pathlib import Path
from typing import Any

from android_world.env import env_launcher

from dms.android_tasks import TaskSpec, instantiate_task, step_budget
from dms.config import apply_runtime_environment, load_yaml
from dms.io_utils import append_jsonl, now_iso, write_json
from env.androidworld_env import (
    AndroidWorldObservationStore,
    configure_windows_adb_stability,
    reset_task_environment,
)
from model_client import QwenVLClient


def _specs(path: Path) -> list[TaskSpec]:
    payload = load_yaml(path)
    return [TaskSpec.from_mapping(item) for item in payload["tasks"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    runtime_path = Path(args.runtime_config).resolve()
    model_path = Path(args.model_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "preflight_results.jsonl"
    if results_path.exists():
        raise FileExistsError(
            f"Preflight output already exists; use a new output directory: {results_path}"
        )

    runtime = load_yaml(runtime_path)
    apply_runtime_environment(runtime)
    configure_windows_adb_stability(
        a11y_method=runtime.get("android", {}).get("a11y_method")
    )
    os.environ["DMS_STRICT_INFRA_PROTOCOL"] = "1"

    model = QwenVLClient(model_path)
    model_health_before = model.healthcheck()
    specs = _specs(dataset_path)
    if not specs:
        raise ValueError("Preflight dataset must contain at least one task.")

    env = env_launcher.load_and_setup_env(
        console_port=int(runtime["android"]["console_port"]),
        emulator_setup=False,
        adb_path=runtime["android"]["adb_path"],
        grpc_port=int(runtime["android"]["grpc_port"]),
    )
    observations = AndroidWorldObservationStore(output_dir / "observations")
    results: list[dict[str, Any]] = []
    model_probe: dict[str, Any] | None = None
    try:
        for index, spec in enumerate(specs):
            task = instantiate_task(spec)
            task_id = f"preflight_{index:03d}_{task.name}"
            phase = "initialize"
            initialized = False
            cleanup_completed = False
            started_at = now_iso()
            try:
                task.initialize_task(env)
                initialized = True
                phase = "observation"
                state = reset_task_environment(
                    env,
                    go_home=task.start_on_home_screen,
                )
                observation = observations.capture(
                    state,
                    env,
                    task_id,
                    0,
                )
                if observation.ui_element_count <= 0:
                    raise RuntimeError("Preflight UI observation is empty.")
                if model_probe is None:
                    probe = model.generate(
                        image_path=observation.screenshot_path,
                        system_prompt=(
                            "You are performing a transport-only GUI compatibility probe."
                        ),
                        prompt=(
                            "Inspect the screenshot and reply with one short sentence "
                            "confirming that an Android UI is visible. Do not propose or "
                            "execute any action."
                        ),
                    )
                    if not str(probe.text).strip():
                        raise RuntimeError("The real model probe returned empty text.")
                    model_probe = {
                        "ok": True,
                        "input_tokens": int(probe.input_tokens),
                        "output_tokens": int(probe.output_tokens),
                        "latency_seconds": float(probe.latency_seconds),
                        "response_nonempty": True,
                    }
                phase = "success_evaluator"
                reward = float(task.is_successful(env))
                if not math.isfinite(reward):
                    raise ValueError(f"Evaluator returned non-finite reward: {reward}")
                phase = "cleanup"
                task.tear_down(env)
                cleanup_completed = True
                record = {
                    "index": index,
                    "task_name": task.name,
                    "seed": spec.seed,
                    "goal": task.goal,
                    "app_names": list(task.app_names),
                    "complexity": task.complexity,
                    "official_step_budget": step_budget(task),
                    "initialized": True,
                    "screenshot_ok": Path(observation.screenshot_path).is_file(),
                    "ui_observation_ok": observation.ui_element_count > 0,
                    "ui_element_count": observation.ui_element_count,
                    "evaluator_ok": True,
                    "evaluator_value": reward,
                    "cleanup_ok": True,
                    "success": True,
                    "phase": "complete",
                    "error": None,
                    "started_at": started_at,
                    "finished_at": now_iso(),
                }
            except Exception as exc:
                cleanup_error = None
                if initialized and not cleanup_completed:
                    try:
                        task.tear_down(env)
                    except Exception as cleanup_exc:
                        cleanup_error = "".join(
                            traceback.format_exception(cleanup_exc)
                        )
                rendered = "".join(traceback.format_exception(exc))
                if cleanup_error:
                    rendered += "\n[infrastructure_phase=cleanup_after_failure]\n"
                    rendered += cleanup_error
                record = {
                    "index": index,
                    "task_name": task.name,
                    "seed": spec.seed,
                    "goal": task.goal,
                    "app_names": list(task.app_names),
                    "complexity": task.complexity,
                    "official_step_budget": step_budget(task),
                    "initialized": initialized,
                    "screenshot_ok": False,
                    "ui_observation_ok": False,
                    "evaluator_ok": False,
                    "cleanup_ok": cleanup_completed and cleanup_error is None,
                    "success": False,
                    "phase": phase,
                    "error": rendered,
                    "started_at": started_at,
                    "finished_at": now_iso(),
                }
            results.append(record)
            append_jsonl(results_path, record)
            write_json(output_dir / "latest_preflight.json", record)
    finally:
        env.close()

    model_health_after = model.healthcheck()
    failed = [item for item in results if not item["success"]]
    summary = {
        "timestamp": now_iso(),
        "dataset": str(dataset_path),
        "runtime_config": str(runtime_path),
        "model_config": str(model_path),
        "task_count": len(results),
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "all_passed": not failed,
        "failed_tasks": [item["task_name"] for item in failed],
        "model_api_before_ok": bool(model_health_before),
        "model_api_after_ok": bool(model_health_after),
        "model_inference_probe": model_probe,
        "model_inference_probe_ok": bool(model_probe and model_probe.get("ok")),
        "model_probe_token_usage": (
            int(model_probe.get("input_tokens", 0))
            + int(model_probe.get("output_tokens", 0))
            if model_probe
            else 0
        ),
        "formal_results_written": False,
        "memory_written": False,
        "token_usage": 0,
        "scored_steps": 0,
        "results": results,
    }
    write_json(output_dir / "preflight_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
