from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any

from android_world import registry
from android_world import suite_utils
from android_world.task_evals import task_eval


@dataclass(frozen=True)
class TaskSpec:
    name: str
    seed: int
    params: dict[str, Any] | None = None
    param_overrides: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, item: dict[str, Any]) -> "TaskSpec":
        return cls(
            name=str(item["name"]),
            seed=int(item.get("seed", 0)),
            params=dict(item["params"]) if "params" in item else None,
            param_overrides=dict(item.get("param_overrides", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def android_world_registry() -> dict[str, type[task_eval.TaskEval]]:
    task_registry = registry.TaskRegistry()
    return task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)


def stable_seed(base_seed: int, name: str, index: int) -> int:
    raw = f"{base_seed}_{name}_{index}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest(), 16) % (2**32)


def instantiate_task(spec: TaskSpec) -> task_eval.TaskEval:
    available = android_world_registry()
    if spec.name not in available:
        raise ValueError(f"Unknown AndroidWorld task: {spec.name}")
    return available[spec.name](resolve_params(spec))


def generate_task_spec(name: str, seed: int) -> TaskSpec:
    available = android_world_registry()
    if name not in available:
        raise ValueError(f"Unknown AndroidWorld task: {name}")
    task_cls = available[name]
    random.seed(seed)
    params = task_cls.generate_random_params()
    params["seed"] = seed
    return TaskSpec(name=name, params=params, seed=seed)


def resolve_params(spec: TaskSpec) -> dict[str, Any]:
    available = android_world_registry()
    if spec.name not in available:
        raise ValueError(f"Unknown AndroidWorld task: {spec.name}")
    if spec.params is not None:
        params = dict(spec.params)
    else:
        task_cls = available[spec.name]
        random.seed(spec.seed)
        params = task_cls.generate_random_params()
    params.update(spec.param_overrides or {})
    params["seed"] = spec.seed
    return params


def sanitize_for_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {
            "_type": type(value).__name__,
            "repr": repr(value),
        }
    return {
        "_type": type(value).__name__,
        "repr": repr(value),
    }


def step_budget(task: task_eval.TaskEval) -> int:
    return int(10 * task.complexity)


def task_summary(spec: TaskSpec) -> dict[str, Any]:
    task = instantiate_task(spec)
    return {
        "name": task.name,
        "params": sanitize_for_json(resolve_params(spec)),
        "seed": spec.seed,
        "goal": task.goal,
        "app_names": list(task.app_names),
        "complexity": task.complexity,
        "step_budget": step_budget(task),
        "step_budget_source": "AndroidWorld suite_utils._allocate_step_budget: int(10 * task.complexity)",
    }


def validate_specs(specs: list[TaskSpec]) -> list[dict[str, Any]]:
    return [task_summary(spec) for spec in specs]


def all_androidworld_task_names() -> list[str]:
    return sorted(android_world_registry())


def create_suite_from_specs(specs: list[TaskSpec]) -> suite_utils.Suite:
    suite = suite_utils.Suite()
    for spec in specs:
        task = instantiate_task(spec)
        suite.setdefault(task.name, []).append(task)
    suite.suite_family = registry.TaskRegistry.ANDROID_WORLD_FAMILY
    return suite
