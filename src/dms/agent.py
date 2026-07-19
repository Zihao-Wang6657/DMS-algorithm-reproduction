from __future__ import annotations

import inspect
import hashlib
import json
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from dms.actions import to_json_action
from dms.io_utils import append_jsonl
from dms.darwinian_memory import DarwinianMemorySystem
from dms.paper_tools import (
    CodeActExecutionResult,
    PLANNER_TOOL_SPECS,
    execute_codeact,
    normalize_planner_tool_call,
)
from dms.prompts import ACTOR_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT
from dms.prompts import actor_prompt, planner_prompt
from env import (
    AndroidWorldObservationStore,
    get_state_with_a11y_retries,
    reset_task_environment,
)


def _strict_infrastructure_protocol() -> bool:
    return os.environ.get("DMS_STRICT_INFRA_PROTOCOL", "").strip() == "1"


def _append_infrastructure_error(
    existing: str | None,
    *,
    phase: str,
    error: BaseException,
) -> str:
    rendered = "".join(traceback.format_exception(error))
    entry = f"[infrastructure_phase={phase}]\n{rendered}"
    return f"{existing.rstrip()}\n\n{entry}" if existing else entry


def _history_for_prompt(trajectory: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    compact = []
    history = trajectory if limit is None else trajectory[-limit:]
    for item in history:
        compact.append(
            {
                "step": item.get("step"),
                "subtask": item.get("subtask"),
                "action": item.get("action"),
                "result": item.get("result"),
                "error": item.get("error"),
            }
        )
    return compact


def _try_refresh_prompt_state(
    *,
    env: Any,
    state: Any,
    observation: dict[str, Any],
    prompt_elements: list[dict[str, Any]],
    image_path: str,
    capture_fn: Any,
    task_id: str,
    step_id: int,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], str]:
    """Best-effort observation refresh that preserves the last good frame on failure."""
    try:
        next_state = get_state_with_a11y_retries(
            env,
            wait_to_stabilize=False,
        )
        next_observation, next_prompt_elements, next_image_path = capture_fn(
            env=env,
            state=next_state,
            task_id=task_id,
            step_id=step_id,
        )
        return next_state, next_observation, next_prompt_elements, next_image_path
    except Exception:
        if _strict_infrastructure_protocol():
            raise
        return state, observation, prompt_elements, image_path


@dataclass
class StepRecord:
    step: int
    subtask: str
    precondition: str
    observation: dict[str, Any]
    planner_output: dict[str, Any] | None
    actor_output: dict[str, Any] | None
    action: dict[str, Any] | None
    result: str
    executed_action: dict[str, Any] | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskRunResult:
    task_id: str
    task_name: str
    goal: str
    success: bool
    reward: float
    steps: int
    input_tokens: int
    output_tokens: int
    memory_size_after: int
    planner_cycles: int = 0
    unsupported_completions: int = 0
    control_turns: int = 0
    event_count: int = 0
    control_turn_limit_reached: bool = False
    memory_stats: dict[str, Any] = field(default_factory=dict)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DMSMemoryAdapter:
    """Thin compatibility layer for an external DMS memory backend."""

    _PLANNER_CONTEXT_METHODS = (
        "planner_context",
        "build_planner_context",
        "get_planner_context",
    )
    _ACTOR_CONTEXT_METHODS = (
        "actor_context",
        "build_actor_context",
        "get_actor_context",
    )
    _RETRIEVE_METHODS = (
        "retrieve_for_planner",
        "retrieve",
        "retrieve_entries",
        "query",
    )
    _REPLAY_METHODS = (
        "replay_for_actor",
        "replay",
        "replay_entries",
    )
    _MUTATION_METHODS = (
        "mutation_fallback",
        "mutate_fallback",
        "fallback_action",
        "fallback_actions",
    )
    _REMEMBER_METHODS = (
        "remember",
        "write_memory",
        "store_memory",
        "append_memory",
        "add_memory",
    )
    _STEP_METHODS = (
        "record_step",
        "on_step",
        "observe_step",
    )
    _FINALIZE_METHODS = (
        "finalize_task",
        "record_task",
        "append_task",
        "on_task_complete",
    )
    _STATS_METHODS = (
        "stats",
        "get_stats",
    )
    _PRUNING_STATS_METHODS = (
        "pruning_stats",
        "get_pruning_stats",
    )
    _RISK_STATS_METHODS = (
        "risk_stats",
        "get_risk_stats",
    )

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    @property
    def backend_name(self) -> str:
        return type(self.backend).__name__

    def _has_any_method(self, names: tuple[str, ...]) -> bool:
        return any(callable(getattr(self.backend, name, None)) for name in names)

    @staticmethod
    def _callable_payload(**kwargs: Any) -> dict[str, Any]:
        aliases = dict(kwargs)
        payload = dict(kwargs)
        payload.setdefault("request", aliases)
        payload.setdefault("context", aliases)
        payload.setdefault("event", aliases)
        return payload

    @staticmethod
    def _invoke(method: Any, payload: dict[str, Any]) -> Any:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return method(**payload)

        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_var_kwargs:
            return method(**payload)

        kwargs = {
            key: value
            for key, value in payload.items()
            if key in signature.parameters
        }
        if kwargs:
            return method(**kwargs)

        parameters = list(signature.parameters.values())
        if len(parameters) == 1 and parameters[0].kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return method(payload.get(parameters[0].name, payload))
        return method()

    def _call_first_available(
        self,
        names: tuple[str, ...],
        **payload: Any,
    ) -> Any:
        event = self._callable_payload(**payload)
        for name in names:
            method = getattr(self.backend, name, None)
            if callable(method):
                return self._invoke(method, event)
        return None

    @staticmethod
    def _render_value(title: str, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            rendered = value.strip()
            if not rendered:
                return None
        else:
            rendered = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        return f"{title}:\n{rendered}"

    def _render_sections(self, sections: list[tuple[str, Any]]) -> str | None:
        rendered = [
            block
            for title, value in sections
            for block in [self._render_value(title, value)]
            if block
        ]
        if not rendered:
            return None
        return "\n\n".join(rendered)

    def planner_context(
        self,
        *,
        task: Any,
        task_id: str,
        task_history: list[dict[str, Any]],
        observation: dict[str, Any],
    ) -> str | None:
        payload = {
            "task": task,
            "task_id": task_id,
            "goal": getattr(task, "goal", ""),
            "task_name": getattr(task, "name", ""),
            "task_history": task_history,
            "observation": observation,
        }
        direct = self._call_first_available(
            self._PLANNER_CONTEXT_METHODS,
            **payload,
        )
        if direct is not None:
            return self._render_sections([("DMS planner context", direct)])
        retrieved = self._call_first_available(
            self._RETRIEVE_METHODS,
            **payload,
        )
        pruning_stats = self._call_first_available(
            self._PRUNING_STATS_METHODS,
            **payload,
        )
        risk_stats = self._call_first_available(
            self._RISK_STATS_METHODS,
            **payload,
        )
        return self._render_sections(
            [
                ("Retrieved hierarchical entries", retrieved),
                ("Pruning diagnostics", pruning_stats),
                ("Risk diagnostics", risk_stats),
            ]
        )

    def actor_context(
        self,
        *,
        task: Any,
        task_id: str,
        subtask: dict[str, str],
        step_history: list[dict[str, Any]],
        observation: dict[str, Any],
    ) -> str | None:
        payload = {
            "task": task,
            "task_id": task_id,
            "task_name": getattr(task, "name", ""),
            "goal": getattr(task, "goal", ""),
            "subtask": subtask,
            "step_history": step_history,
            "observation": observation,
        }
        direct = self._call_first_available(
            self._ACTOR_CONTEXT_METHODS,
            **payload,
        )
        if direct is not None:
            return self._render_sections([("DMS actor context", direct)])
        replay = self._call_first_available(
            self._REPLAY_METHODS,
            **payload,
        )
        mutation_fallback = self._call_first_available(
            self._MUTATION_METHODS,
            **payload,
        )
        risk_stats = self._call_first_available(
            self._RISK_STATS_METHODS,
            **payload,
        )
        return self._render_sections(
            [
                ("Replay candidates", replay),
                ("Mutation fallback guidance", mutation_fallback),
                ("Risk diagnostics", risk_stats),
            ]
        )

    def remember(
        self,
        *,
        information: str,
        task: Any,
        task_id: str,
        step_id: int,
        subtask: dict[str, str],
        observation: dict[str, Any],
        trajectory: list[dict[str, Any]],
    ) -> Any:
        if not self._has_any_method(self._REMEMBER_METHODS):
            raise ValueError(
                "DMS memory backend does not expose a remember-compatible hook."
            )
        return self._call_first_available(
            self._REMEMBER_METHODS,
            information=information,
            text=information,
            memory_text=information,
            task=task,
            task_id=task_id,
            task_name=getattr(task, "name", ""),
            goal=getattr(task, "goal", ""),
            step=step_id,
            step_id=step_id,
            subtask=subtask,
            observation=observation,
            trajectory=trajectory,
        )

    def record_step(
        self,
        *,
        task: Any,
        task_id: str,
        step_record: dict[str, Any],
    ) -> Any:
        if not self._has_any_method(self._STEP_METHODS):
            return None
        return self._call_first_available(
            self._STEP_METHODS,
            task=task,
            task_id=task_id,
            task_name=getattr(task, "name", ""),
            goal=getattr(task, "goal", ""),
            step_record=step_record,
            record=step_record,
            trajectory_step=step_record,
        )

    def finalize_task(
        self,
        *,
        task: Any,
        task_id: str,
        success: bool,
        steps: int,
        trajectory: list[dict[str, Any]],
    ) -> Any:
        if not self._has_any_method(self._FINALIZE_METHODS):
            return None
        return self._call_first_available(
            self._FINALIZE_METHODS,
            task=task,
            task_id=task_id,
            task_name=getattr(task, "name", ""),
            goal=getattr(task, "goal", ""),
            success=success,
            steps=steps,
            trajectory=trajectory,
        )

    def stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        root_stats = self._call_first_available(self._STATS_METHODS)
        if isinstance(root_stats, dict):
            stats.update(root_stats)
        elif root_stats is not None:
            stats["stats"] = root_stats

        pruning_stats = self._call_first_available(self._PRUNING_STATS_METHODS)
        if pruning_stats is not None and "pruning_stats" not in stats:
            stats["pruning_stats"] = pruning_stats

        risk_stats = self._call_first_available(self._RISK_STATS_METHODS)
        if risk_stats is not None and "risk_stats" not in stats:
            stats["risk_stats"] = risk_stats

        if "memory_size" not in stats:
            size = getattr(self.backend, "size", None)
            if size is not None:
                stats["memory_size"] = int(size)
            elif "size" in stats:
                stats["memory_size"] = stats["size"]
        return stats

    @property
    def size(self) -> int:
        size = getattr(self.backend, "size", None)
        if size is not None:
            return int(size)
        stats = self.stats()
        for key in ("memory_size", "size", "entry_count", "entries"):
            value = stats.get(key)
            if isinstance(value, (int, float)):
                return int(value)
        return 0


def _normalize_subtask_dicts(sub_tasks: list[dict[str, Any]], default_agent: str = "CodeActAgent") -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in sub_tasks:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "precondition": str(item.get("precondition", "None")),
                "goal": str(item.get("goal", "")),
                "agent": str(item.get("agent", default_agent)),
            }
        )
    return normalized


def _normalize_app_name(app_name: str) -> str:
    normalized = str(app_name).strip().lower()
    if "/" in normalized:
        normalized = normalized.split("/", maxsplit=1)[0]
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


_START_APP_NAME_ALIASES = {
    "file manager": "files",
    "files app": "files",
    "file_manager": "files",
    "simple calendar": "simple calendar pro",
    "simple.calendar.pro": "simple calendar pro",
    "com.simple.calendar.pro": "simple calendar pro",
    "simple gallery": "simple gallery pro",
    "simple.gallery.pro": "simple gallery pro",
    "com.simple.gallery.pro": "simple gallery pro",
    "simple draw": "simple draw pro",
    "simple.draw.pro": "simple draw pro",
    "com.simple.draw.pro": "simple draw pro",
    "simple sms": "simple sms messenger",
    "simple.sms.messenger": "simple sms messenger",
    "com.simplemobiletools.smsmessenger": "simple sms messenger",
    "retromusic": "retro music",
}


def _package_from_activity(activity: str) -> str:
    return str(activity).split("/", maxsplit=1)[0].strip()


@lru_cache(maxsize=128)
def _expected_package_for_app_name(app_name: str) -> str | None:
    normalized = str(app_name).strip()
    if not normalized:
        return None
    if "." in normalized and " " not in normalized and "/" not in normalized:
        return normalized
    try:
        from android_world.env import adb_utils
    except Exception:
        return None
    try:
        activity = adb_utils.get_adb_activity(normalized)
    except Exception:
        return None
    if not activity:
        return None
    try:
        return adb_utils.extract_package_name(activity)
    except Exception:
        return _package_from_activity(str(activity))


def _expected_packages_for_task_scope(task_app_names: list[str]) -> set[str]:
    packages: set[str] = set()
    for app_name in task_app_names:
        package = _expected_package_for_app_name(app_name)
        if package:
            packages.add(package)
    return packages


def _foreground_matches_task_scope(
    foreground_activity: str,
    task_app_names: list[str],
) -> bool:
    foreground_package = _package_from_activity(foreground_activity)
    if not foreground_package:
        return False
    expected_packages = _expected_packages_for_task_scope(task_app_names)
    if expected_packages:
        return foreground_package in expected_packages
    normalized_foreground = _normalize_app_name(foreground_package)
    return any(
        _normalize_app_name(app_name) == normalized_foreground
        for app_name in task_app_names
    )


def _task_app_scope(task: Any) -> list[str]:
    params = getattr(task, "params", {}) or {}
    scoped = []
    for key in ("app_name", "app_names"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            scoped.append(value.strip())
        elif isinstance(value, (list, tuple)):
            scoped.extend(str(item).strip() for item in value if str(item).strip())
    if scoped:
        return scoped
    app_names = getattr(task, "app_names", ()) or ()
    return [str(app_name).strip() for app_name in app_names if str(app_name).strip()]


def _looks_like_open_app_request(
    subtask_goal: str,
    task_app_names: list[str] | None = None,
) -> bool:
    goal = _normalize_app_name(subtask_goal)
    if "open" in goal and "app" in goal:
        return True
    if any(verb in goal.split() for verb in ("open", "launch", "start")):
        scoped_apps = task_app_names or []
        return any(
            _normalize_app_name(app_name) in goal
            for app_name in scoped_apps
        )
    return False


def _fallback_open_app_action(
    *,
    action: dict[str, Any] | None,
    subtask_goal: str,
    task_app_names: list[str],
) -> dict[str, Any] | None:
    if not action or action.get("type") != "tap":
        return None
    if not _looks_like_open_app_request(subtask_goal, task_app_names):
        return None
    if len(task_app_names) != 1:
        return None
    return {"type": "start_app", "package": task_app_names[0]}


def _shortcut_open_app_action(
    *,
    subtask_goal: str,
    task_app_names: list[str],
    foreground_activity: str,
    guard_system_dialogs: bool = False,
) -> dict[str, Any] | None:
    if guard_system_dialogs and "permissioncontroller" in str(
        foreground_activity
    ).lower():
        # Let the Actor handle the visible generic permission UI. Repeatedly
        # injecting start_app cannot dismiss a system-owned dialog.
        return None
    if not _looks_like_open_app_request(subtask_goal, task_app_names):
        return None

    chosen_app: str | None = None
    normalized_goal = _normalize_app_name(subtask_goal)
    for app_name in task_app_names:
        if _normalize_app_name(app_name) in normalized_goal:
            chosen_app = app_name
            break
    if chosen_app is None and len(task_app_names) == 1:
        chosen_app = task_app_names[0]
    if not chosen_app:
        return None

    if _foreground_matches_task_scope(foreground_activity, [chosen_app]):
        return {
            "type": "complete",
            "success": True,
            "reason": f"{chosen_app} is already open.",
        }
    return {"type": "start_app", "package": chosen_app}


def _completion_fingerprint(
    *,
    subtask_goal: str,
    foreground_activity: str,
    observation: dict[str, Any],
) -> str:
    """Stable fingerprint for detecting completion claims on unchanged state."""
    payload = {
        "subtask_goal": str(subtask_goal).strip().lower(),
        "foreground_activity": str(foreground_activity).strip().lower(),
        "compact_ui_elements": observation.get("compact_ui_elements", []),
    }
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _task_requires_answer(goal: str) -> bool:
    goal_lower = str(goal).lower()
    return any(
        marker in goal_lower
        for marker in (
            "answer with",
            "express your answer",
            "how many",
            "what ",
            "which ",
            "when ",
            "do i have",
        )
    )


def _fallback_answer_from_complete(
    *,
    action: dict[str, Any] | None,
    task_goal: str,
) -> dict[str, Any] | None:
    if not action or action.get("type") != "complete":
        return None
    if not action.get("success", True):
        return None
    if not _task_requires_answer(task_goal):
        return None
    reason = str(action.get("reason") or "").strip()
    if not reason:
        return None
    return {"type": "answer", "text": reason}


def _normalize_start_app_action(
    *,
    action: dict[str, Any] | None,
    subtask_goal: str,
    task_app_names: list[str],
) -> dict[str, Any] | None:
    if not action or action.get("type") != "start_app":
        return action
    if not _looks_like_open_app_request(subtask_goal, task_app_names):
        return action
    if len(task_app_names) != 1:
        return action

    requested = str(action.get("package") or action.get("app_name") or "").strip()
    scoped = str(task_app_names[0]).strip()
    if not scoped:
        return action
    if not requested:
        return {"type": "start_app", "package": scoped}

    normalized_requested = _normalize_app_name(requested)
    aliased_requested = _START_APP_NAME_ALIASES.get(
        normalized_requested,
        normalized_requested,
    )
    normalized_scoped = _normalize_app_name(scoped)
    if aliased_requested == normalized_scoped:
        return {"type": "start_app", "package": scoped}
    if "." in requested or requested.lower().startswith("com "):
        requested_package = requested.replace(" ", "").strip().lower()
        expected_package = _expected_package_for_app_name(scoped)
        if expected_package and requested_package == expected_package.lower():
            return {"type": "start_app", "package": scoped}
        return {"type": "start_app", "package": scoped}
    if aliased_requested != normalized_requested:
        return {"type": "start_app", "package": aliased_requested}
    return action


class PALiteAgent:
    """Paper-faithful Planner-Actor baseline for AndroidWorld.

    The planner follows Appendix H's set_tasks_with_agents / complete_goal
    semantics. The actor follows Appendix H's CodeAct prompt and returns a
    single Python tool call that is translated to AndroidWorld JSONAction.
    DMS mechanisms remain absent in Baseline A/B.
    """

    def __init__(
        self,
        *,
        model: Any,
        run_dir: str | Path,
        max_subtasks: int = 5,
        actor_local_step_guard: int = 8,
        static_memory: Any | None = None,
        post_action_wait_seconds: float = 3.0,
        engineering_optimization: bool = False,
        planner_max_cycles: int | None = None,
        control_turn_limit: int | None = None,
    ) -> None:
        if max_subtasks > 5:
            raise ValueError("Planner max_subtasks must stay <= 5 per paper.")
        if post_action_wait_seconds < 0:
            raise ValueError("post_action_wait_seconds must be non-negative.")
        if planner_max_cycles is not None and planner_max_cycles < 1:
            raise ValueError("planner_max_cycles must be positive when provided.")
        if control_turn_limit is not None and control_turn_limit < 1:
            raise ValueError("control_turn_limit must be positive when provided.")
        self.model = model
        self.run_dir = Path(run_dir)
        self.max_subtasks = max_subtasks
        self.actor_local_step_guard = actor_local_step_guard
        self.static_memory = static_memory
        self.post_action_wait_seconds = post_action_wait_seconds
        self.engineering_optimization = engineering_optimization
        self.planner_max_cycles = planner_max_cycles
        self.control_turn_limit = control_turn_limit
        self.observations = AndroidWorldObservationStore(
            self.run_dir / "observations"
        )
        self.step_log_path = self.run_dir / "steps.jsonl"

    def _planner_memory_context(
        self,
        *,
        task: Any | None,
        task_id: str,
        task_history: list[dict[str, Any]],
        observation: dict[str, Any],
    ) -> str | None:
        del task, task_id, task_history, observation
        return self.static_memory.context() if self.static_memory else None

    def _planner_memory_context_title(self) -> str:
        return "Cross-task Memory Context"

    def _dms_mode(self) -> bool:
        return False

    def _actor_memory_context(
        self,
        *,
        task: Any,
        task_id: str,
        subtask: dict[str, str],
        step_history: list[dict[str, Any]],
        observation: dict[str, Any],
    ) -> str | None:
        del task, task_id, subtask, step_history, observation
        return None

    def _allow_remember(self) -> bool:
        return False

    def _remember_action(
        self,
        *,
        task: Any,
        task_id: str,
        step_id: int,
        subtask: dict[str, str],
        action: dict[str, Any],
        observation: dict[str, Any],
        trajectory: list[dict[str, Any]],
    ) -> None:
        del task, task_id, step_id, subtask, action, observation, trajectory
        raise ValueError("remember is disabled for Baseline A/B.")

    def _on_step_record(
        self,
        *,
        task: Any,
        task_id: str,
        step_record: dict[str, Any],
    ) -> None:
        del task, task_id, step_record

    def _append_step_record(
        self,
        *,
        trajectory: list[dict[str, Any]],
        record: StepRecord,
        task: Any,
        task_id: str,
    ) -> None:
        entry = record.to_dict()
        trajectory.append(entry)
        append_jsonl(self.step_log_path, entry)
        self._on_step_record(
            task=task,
            task_id=task_id,
            step_record=entry,
        )

    def _finalize_task_memory(
        self,
        *,
        task: Any,
        task_id: str,
        success: bool,
        steps: int,
        trajectory: list[dict[str, Any]],
    ) -> None:
        if self.static_memory is None:
            return
        self.static_memory.append_task(
            task_id=task_id,
            task_name=task.name,
            goal=task.goal,
            success=success,
            steps=steps,
            trajectory=trajectory,
        )

    def _memory_size_after(self) -> int:
        return self.static_memory.size if self.static_memory else 0

    def _memory_stats(self) -> dict[str, Any]:
        return {}

    def _capture_for_prompt(
        self,
        *,
        env: Any,
        state: Any,
        task_id: str,
        step_id: int,
    ) -> tuple[
        dict[str, Any],
        list[dict[str, Any]],
        str,
    ]:
        record = self.observations.capture(
            state,
            env,
            task_id,
            step_id,
        )
        ui_elements_path = Path(record.ui_elements_path)
        import json

        elements = json.loads(ui_elements_path.read_text(encoding="utf-8"))
        observation = record.to_dict()
        observation["compact_ui_elements"] = [
            {
                "index": element["index"],
                "text": element["text"],
                "content_description": element["content_description"],
                "class_name": element["class_name"],
                "bounds": element["bounds"],
                "clickable": element["is_clickable"],
                "editable": element["is_editable"],
                "scrollable": element["is_scrollable"],
            }
            for element in elements
            if element["is_visible"]
            and (
                element["text"]
                or element["content_description"]
                or element["is_clickable"]
                or element["is_editable"]
                or element["is_scrollable"]
            )
        ][:80]
        return observation, elements, record.screenshot_path

    @staticmethod
    def _bind_action_to_state(
        action: dict[str, Any],
        prompt_elements: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if action.get("type") != "tap" or action.get("index") is None:
            return dict(action)

        index = action["index"]
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"tap index must be an integer, got {index!r}.")
        if index < 0 or index >= len(prompt_elements):
            raise ValueError(
                f"tap index {index} is out of range for "
                f"{len(prompt_elements)} UI elements."
            )

        element = prompt_elements[index]
        if not element.get("is_visible", False):
            raise ValueError(f"tap index {index} targets an invisible UI element.")
        bbox = element.get("bounds")
        if bbox is None:
            raise ValueError(f"tap index {index} has no bounding box.")
        if len(bbox) != 4:
            raise ValueError(f"tap index {index} has malformed bounds {bbox!r}.")

        bound_action = dict(action)
        bound_action.pop("index", None)
        bound_action["x"] = int((bbox[0] + bbox[2]) / 2)
        bound_action["y"] = int((bbox[1] + bbox[3]) / 2)
        return bound_action

    def _plan(
        self,
        *,
        image_path: str,
        task: Any | None = None,
        task_id: str = "",
        goal: str,
        task_app_names: list[str],
        foreground_activity: str,
        compact_ui: list[dict[str, Any]],
        task_history: list[dict[str, Any]],
        observation: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        memory_context = self._planner_memory_context(
            task=task,
            task_id=task_id,
            task_history=task_history,
            observation=observation or {},
        )
        result = self.model.generate(
            image_path=image_path,
            prompt=planner_prompt(
                goal=goal,
                task_app_names=task_app_names,
                foreground_activity=foreground_activity,
                compact_ui=compact_ui,
                task_history=task_history,
                memory_context=memory_context,
                max_subtasks=self.max_subtasks,
                memory_context_title=self._planner_memory_context_title(),
                dms_mode=self._dms_mode(),
                engineering_optimization=self.engineering_optimization,
            ),
            system_prompt=PLANNER_SYSTEM_PROMPT,
            tools=PLANNER_TOOL_SPECS,
        )
        parsed = normalize_planner_tool_call(result.parsed_json, self.max_subtasks)
        parsed["raw_text"] = result.text
        parsed["parsed_json"] = result.parsed_json
        return parsed, result.input_tokens, result.output_tokens

    def _act(
        self,
        *,
        image_path: str,
        task: Any,
        task_id: str,
        global_goal: str,
        task_app_names: list[str],
        subtask: dict[str, str],
        foreground_activity: str,
        compact_ui: list[dict[str, Any]],
        step_history: list[dict[str, Any]],
        observation: dict[str, Any],
    ) -> tuple[CodeActExecutionResult, int, int]:
        memory_context = self._actor_memory_context(
            task=task,
            task_id=task_id,
            subtask=subtask,
            step_history=step_history,
            observation=observation,
        )
        result = self.model.generate(
            image_path=image_path,
            prompt=actor_prompt(
                global_goal=global_goal,
                task_app_names=task_app_names,
                subtask=subtask,
                foreground_activity=foreground_activity,
                compact_ui=compact_ui,
                step_history=step_history,
                memory_context=memory_context,
                allow_remember=self._allow_remember(),
                engineering_optimization=self.engineering_optimization,
            ),
            system_prompt=ACTOR_SYSTEM_PROMPT,
        )
        execution = execute_codeact(
            result.text,
            allow_remember=self._allow_remember(),
        )
        return execution, result.input_tokens, result.output_tokens

    def run_task(self, *, env: Any, task: Any, task_id: str) -> TaskRunResult:
        trajectory: list[dict[str, Any]] = []
        input_tokens = 0
        output_tokens = 0
        task_error = None
        reward = 0.0
        success = False
        step_id = 0
        env_steps = 0
        control_turns = 0
        planner_cycles = 0
        unsupported_completions = 0
        action_epoch = 0
        last_completion_fingerprint: str | None = None
        last_completion_action_epoch = -1
        task_app_names = _task_app_scope(task)

        try:
            task.initialize_task(env)
            state = reset_task_environment(
                env,
                go_home=task.start_on_home_screen,
            )
            max_steps = int(10 * task.complexity)

            def budget_steps() -> int:
                return env_steps if self.engineering_optimization else step_id

            def control_available() -> bool:
                return (
                    self.control_turn_limit is None
                    or control_turns < self.control_turn_limit
                )
            observation, prompt_elements, image_path = self._capture_for_prompt(
                env=env,
                state=state,
                task_id=task_id,
                step_id=step_id,
            )

            def refresh_prompt_state() -> None:
                nonlocal state, observation, prompt_elements, image_path
                if budget_steps() >= max_steps:
                    return
                state, observation, prompt_elements, image_path = _try_refresh_prompt_state(
                    env=env,
                    state=state,
                    observation=observation,
                    prompt_elements=prompt_elements,
                    image_path=image_path,
                    capture_fn=self._capture_for_prompt,
                    task_id=task_id,
                    step_id=step_id,
                )

            while budget_steps() < max_steps and control_available() and (
                self.planner_max_cycles is None
                or planner_cycles < self.planner_max_cycles
            ):
                planner_cycles += 1
                control_turns += 1
                plan, plan_in, plan_out = self._plan(
                    image_path=image_path,
                    task=task,
                    task_id=task_id,
                    goal=task.goal,
                    task_app_names=task_app_names,
                    foreground_activity=observation["foreground_activity"],
                    compact_ui=observation["compact_ui_elements"],
                    task_history=_history_for_prompt(trajectory),
                    observation=observation,
                )
                input_tokens += plan_in
                output_tokens += plan_out
                if plan.get("complete"):
                    break
                sub_tasks = plan.get("sub_tasks") or [
                    {"precondition": "None", "goal": task.goal, "agent": "CodeActAgent"}
                ]
                sub_tasks = _normalize_subtask_dicts(sub_tasks)[: self.max_subtasks]

                plan_failed = False
                for subtask in sub_tasks:
                    if budget_steps() >= max_steps or not control_available():
                        break
                    local_steps = 0
                    while (
                        local_steps < self.actor_local_step_guard
                        and budget_steps() < max_steps
                        and control_available()
                    ):
                        shortcut_action = _shortcut_open_app_action(
                            subtask_goal=str(subtask.get("goal", "")),
                            task_app_names=task_app_names,
                            foreground_activity=observation["foreground_activity"],
                            guard_system_dialogs=self.engineering_optimization,
                        )
                        if shortcut_action is not None:
                            actor_output = CodeActExecutionResult(
                                code="",
                                tool_name=shortcut_action["type"],
                                action=shortcut_action,
                                error=None,
                            )
                            act_in = 0
                            act_out = 0
                        else:
                            actor_output, act_in, act_out = self._act(
                                image_path=image_path,
                                task=task,
                                task_id=task_id,
                                global_goal=task.goal,
                                task_app_names=task_app_names,
                                subtask=subtask,
                                foreground_activity=observation["foreground_activity"],
                                compact_ui=observation["compact_ui_elements"],
                                step_history=_history_for_prompt(trajectory),
                                observation=observation,
                            )
                            input_tokens += act_in
                            output_tokens += act_out

                        action = actor_output.action
                        if action and action.get("type") == "complete" and _task_requires_answer(task.goal):
                            reason = str(action.get("reason") or "").strip()
                            if reason:
                                action = {"type": "answer", "text": reason}
                        action = _normalize_start_app_action(
                            action=action,
                            subtask_goal=str(subtask.get("goal", "")),
                            task_app_names=task_app_names,
                        )
                        record = StepRecord(
                            step=step_id,
                            subtask=str(subtask.get("goal", "")),
                            precondition=str(subtask.get("precondition", "")),
                            observation=observation,
                            planner_output=plan,
                            actor_output={
                                "tool_name": actor_output.tool_name,
                                "code": actor_output.code,
                                "action": action,
                                "error": actor_output.error,
                            },
                            action=action,
                            result="pending",
                            input_tokens=act_in,
                            output_tokens=act_out,
                        )

                        if actor_output.error:
                            control_turns += 1
                            record.result = "invalid_action"
                            record.error = actor_output.error
                            self._append_step_record(
                                trajectory=trajectory,
                                record=record,
                                task=task,
                                task_id=task_id,
                            )
                            step_id += 1
                            local_steps += 1
                            refresh_prompt_state()
                            plan_failed = True
                            break

                        if action is None:
                            control_turns += 1
                            record.result = "invalid_action"
                            record.error = "Actor did not produce a valid paper tool call."
                            self._append_step_record(
                                trajectory=trajectory,
                                record=record,
                                task=task,
                                task_id=task_id,
                            )
                            step_id += 1
                            local_steps += 1
                            refresh_prompt_state()
                            plan_failed = True
                            break

                        if action.get("type") == "remember":
                            control_turns += 1
                            try:
                                self._remember_action(
                                    task=task,
                                    task_id=task_id,
                                    step_id=step_id,
                                    subtask=subtask,
                                    action=action,
                                    observation=observation,
                                    trajectory=trajectory,
                                )
                                record.result = "remembered"
                                record.executed_action = dict(action)
                            except Exception as exc:
                                record.result = "memory_error"
                                record.error = str(exc)
                                plan_failed = True
                            self._append_step_record(
                                trajectory=trajectory,
                                record=record,
                                task=task,
                                task_id=task_id,
                            )
                            step_id += 1
                            local_steps += 1
                            if plan_failed:
                                refresh_prompt_state()
                            break

                        if action.get("type") == "complete":
                            control_turns += 1
                            completion_success = bool(action.get("success", True))
                            completion_fingerprint = _completion_fingerprint(
                                subtask_goal=str(subtask.get("goal", "")),
                                foreground_activity=observation["foreground_activity"],
                                observation=observation,
                            )
                            repeated_unsupported_completion = (
                                self.engineering_optimization
                                and completion_success
                                and completion_fingerprint
                                == last_completion_fingerprint
                                and action_epoch == last_completion_action_epoch
                            )
                            if repeated_unsupported_completion:
                                completion_success = False
                                unsupported_completions += 1
                                record.result = "unsupported_completion"
                                record.error = (
                                    "Repeated completion claim on unchanged device state; "
                                    "the next plan must use a visibly different recovery."
                                )
                                plan_failed = True
                            else:
                                last_completion_fingerprint = completion_fingerprint
                                last_completion_action_epoch = action_epoch
                            if completion_success and _looks_like_open_app_request(
                                str(subtask.get("goal", ""))
                            ):
                                if not _foreground_matches_task_scope(
                                    observation["foreground_activity"],
                                    task_app_names,
                                ):
                                    completion_success = False
                                    expected_packages = sorted(
                                        _expected_packages_for_task_scope(
                                            task_app_names
                                        )
                                    )
                                    record.error = (
                                        "Subtask claimed target app was open, "
                                        "but foreground activity did not match "
                                        f"the task scope. foreground="
                                        f"{observation['foreground_activity']!r} "
                                        f"expected_packages={expected_packages!r}"
                                    )
                                    plan_failed = True
                            if record.result != "unsupported_completion":
                                record.result = (
                                    "subtask_complete"
                                    if completion_success
                                    else "subtask_failed"
                                )
                            self._append_step_record(
                                trajectory=trajectory,
                                record=record,
                                task=task,
                                task_id=task_id,
                            )
                            if not completion_success:
                                plan_failed = True
                            step_id += 1
                            break

                        try:
                            executed_action = self._bind_action_to_state(
                                action,
                                prompt_elements,
                            )
                            json_action = to_json_action(executed_action)
                        except (TypeError, ValueError) as exc:
                            fallback_action = _fallback_open_app_action(
                                action=action,
                                subtask_goal=str(subtask.get("goal", "")),
                                task_app_names=task_app_names,
                            )
                            if fallback_action is None:
                                control_turns += 1
                                record.result = "invalid_action"
                                record.error = str(exc)
                                self._append_step_record(
                                    trajectory=trajectory,
                                    record=record,
                                    task=task,
                                    task_id=task_id,
                                )
                                step_id += 1
                                local_steps += 1
                                refresh_prompt_state()
                                plan_failed = True
                                break
                            executed_action = fallback_action
                            json_action = to_json_action(executed_action)
                        if json_action is None:
                            control_turns += 1
                            record.result = "invalid_action"
                            record.error = f"Unsupported paper action: {action}"
                            self._append_step_record(
                                trajectory=trajectory,
                                record=record,
                                task=task,
                                task_id=task_id,
                            )
                            step_id += 1
                            local_steps += 1
                            refresh_prompt_state()
                            plan_failed = True
                            break

                        record.executed_action = executed_action
                        try:
                            env_steps += 1
                            env.execute_action(json_action)
                            action_epoch += 1
                            if self.post_action_wait_seconds:
                                time.sleep(self.post_action_wait_seconds)
                            record.result = "executed"
                            reward = float(task.is_successful(env))
                            if reward >= 1.0:
                                success = True
                                self._append_step_record(
                                    trajectory=trajectory,
                                    record=record,
                                    task=task,
                                    task_id=task_id,
                                )
                                step_id += 1
                                local_steps += 1
                                break
                            state = get_state_with_a11y_retries(
                                env,
                                wait_to_stabilize=False,
                            )
                        except Exception as exc:  # Environment dependent.
                            if _strict_infrastructure_protocol():
                                raise
                            record.result = "execution_error"
                            record.error = str(exc)
                            plan_failed = True
                        self._append_step_record(
                            trajectory=trajectory,
                            record=record,
                            task=task,
                            task_id=task_id,
                        )
                        step_id += 1
                        local_steps += 1
                        if plan_failed:
                            break
                        if budget_steps() < max_steps and control_available():
                            observation, prompt_elements, image_path = self._capture_for_prompt(
                                env=env,
                                state=state,
                                task_id=task_id,
                                step_id=step_id,
                            )
                    if plan_failed:
                        break

                reward = float(task.is_successful(env))
                if reward >= 1.0:
                    success = True
                    break
        except Exception as exc:  # Environment dependent.
            task_error = "".join(traceback.format_exception(exc))
        finally:
            try:
                reward = float(task.is_successful(env))
                success = reward >= 1.0
            except Exception as exc:
                if _strict_infrastructure_protocol():
                    task_error = _append_infrastructure_error(
                        task_error,
                        phase="final_success_evaluation",
                        error=exc,
                    )
            try:
                task.tear_down(env)
            except Exception as exc:
                if _strict_infrastructure_protocol():
                    task_error = _append_infrastructure_error(
                        task_error,
                        phase="task_cleanup",
                        error=exc,
                    )

        self._finalize_task_memory(
            task=task,
            task_id=task_id,
            success=success,
            steps=budget_steps() if "budget_steps" in locals() else step_id,
            trajectory=trajectory,
        )
        return TaskRunResult(
            task_id=task_id,
            task_name=task.name,
            goal=task.goal,
            success=success,
            reward=reward,
            steps=budget_steps() if "budget_steps" in locals() else step_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            memory_size_after=self._memory_size_after(),
            planner_cycles=planner_cycles,
            unsupported_completions=unsupported_completions,
            control_turns=control_turns,
            event_count=step_id,
            control_turn_limit_reached=(
                self.control_turn_limit is not None
                and control_turns >= self.control_turn_limit
            ),
            memory_stats=self._memory_stats(),
            trajectory=trajectory,
            error=task_error,
        )


class DMSAgent(PALiteAgent):
    """PA-Lite control loop with an external DMS memory subsystem."""

    def __init__(
        self,
        *,
        model: Any,
        run_dir: str | Path,
        dms_memory: Any,
        max_subtasks: int = 5,
        actor_local_step_guard: int = 8,
        post_action_wait_seconds: float = 3.0,
        engineering_optimization: bool = False,
        planner_max_cycles: int | None = None,
        control_turn_limit: int | None = None,
    ) -> None:
        super().__init__(
            model=model,
            run_dir=run_dir,
            max_subtasks=max_subtasks,
            actor_local_step_guard=actor_local_step_guard,
            static_memory=None,
            post_action_wait_seconds=post_action_wait_seconds,
            engineering_optimization=engineering_optimization,
            planner_max_cycles=planner_max_cycles,
            control_turn_limit=control_turn_limit,
        )
        self.dms_memory = (
            dms_memory
            if isinstance(dms_memory, DMSMemoryAdapter)
            else DMSMemoryAdapter(dms_memory)
        )

    def _planner_memory_context(
        self,
        *,
        task: Any | None,
        task_id: str,
        task_history: list[dict[str, Any]],
        observation: dict[str, Any],
    ) -> str | None:
        if task is None:
            return None
        return self.dms_memory.planner_context(
            task=task,
            task_id=task_id,
            task_history=task_history,
            observation=observation,
        )

    def _planner_memory_context_title(self) -> str:
        return "DMS Memory Context"

    def _dms_mode(self) -> bool:
        return True

    def _actor_memory_context(
        self,
        *,
        task: Any,
        task_id: str,
        subtask: dict[str, str],
        step_history: list[dict[str, Any]],
        observation: dict[str, Any],
    ) -> str | None:
        return self.dms_memory.actor_context(
            task=task,
            task_id=task_id,
            subtask=subtask,
            step_history=step_history,
            observation=observation,
        )

    def _allow_remember(self) -> bool:
        return True

    def _remember_action(
        self,
        *,
        task: Any,
        task_id: str,
        step_id: int,
        subtask: dict[str, str],
        action: dict[str, Any],
        observation: dict[str, Any],
        trajectory: list[dict[str, Any]],
    ) -> None:
        information = str(action.get("information") or "").strip()
        if not information:
            raise ValueError("remember requires information.")
        self.dms_memory.remember(
            information=information,
            task=task,
            task_id=task_id,
            step_id=step_id,
            subtask=subtask,
            observation=observation,
            trajectory=trajectory,
        )

    def _on_step_record(
        self,
        *,
        task: Any,
        task_id: str,
        step_record: dict[str, Any],
    ) -> None:
        self.dms_memory.record_step(
            task=task,
            task_id=task_id,
            step_record=step_record,
        )

    def _finalize_task_memory(
        self,
        *,
        task: Any,
        task_id: str,
        success: bool,
        steps: int,
        trajectory: list[dict[str, Any]],
    ) -> None:
        self.dms_memory.finalize_task(
            task=task,
            task_id=task_id,
            success=success,
            steps=steps,
            trajectory=trajectory,
        )

    def _memory_size_after(self) -> int:
        return self.dms_memory.size

    def _memory_stats(self) -> dict[str, Any]:
        stats = self.dms_memory.stats()
        if "backend" not in stats:
            stats["backend"] = self.dms_memory.backend_name
        return stats

    def run_task(self, *, env: Any, task: Any, task_id: str) -> TaskRunResult:
        if isinstance(self.dms_memory.backend, DarwinianMemorySystem):
            return self._run_native_dms_task(env=env, task=task, task_id=task_id)
        return super().run_task(env=env, task=task, task_id=task_id)

    def _run_native_dms_task(self, *, env: Any, task: Any, task_id: str) -> TaskRunResult:
        trajectory: list[dict[str, Any]] = []
        input_tokens = 0
        output_tokens = 0
        task_error = None
        reward = 0.0
        success = False
        step_id = 0
        env_steps = 0
        control_turns = 0
        task_app_names = _task_app_scope(task)
        active_memory_ids: list[str] = []
        created_memory_ids: list[str] = []
        planner_cycles = 0
        unsupported_completions = 0
        action_epoch = 0
        last_completion_fingerprint: str | None = None
        last_completion_action_epoch = -1

        try:
            task.initialize_task(env)
            state = reset_task_environment(
                env,
                go_home=task.start_on_home_screen,
            )
            max_steps = int(10 * task.complexity)

            def budget_steps() -> int:
                return env_steps if self.engineering_optimization else step_id

            def control_available() -> bool:
                return (
                    self.control_turn_limit is None
                    or control_turns < self.control_turn_limit
                )
            observation, prompt_elements, image_path = self._capture_for_prompt(
                env=env,
                state=state,
                task_id=task_id,
                step_id=step_id,
            )

            def refresh_prompt_state() -> None:
                nonlocal state, observation, prompt_elements, image_path
                if budget_steps() >= max_steps:
                    return
                state, observation, prompt_elements, image_path = _try_refresh_prompt_state(
                    env=env,
                    state=state,
                    observation=observation,
                    prompt_elements=prompt_elements,
                    image_path=image_path,
                    capture_fn=self._capture_for_prompt,
                    task_id=task_id,
                    step_id=step_id,
                )

            while budget_steps() < max_steps and control_available() and (
                self.planner_max_cycles is None
                or planner_cycles < self.planner_max_cycles
            ):
                planner_cycles += 1
                control_turns += 1
                plan, plan_in, plan_out = self._plan(
                    image_path=image_path,
                    task=task,
                    task_id=task_id,
                    goal=task.goal,
                    task_app_names=task_app_names,
                    foreground_activity=observation["foreground_activity"],
                    compact_ui=observation["compact_ui_elements"],
                    task_history=_history_for_prompt(trajectory),
                    observation=observation,
                )
                input_tokens += plan_in
                output_tokens += plan_out
                if plan.get("complete"):
                    break
                sub_tasks = plan.get("sub_tasks") or [
                    {"precondition": "None", "goal": task.goal, "agent": "CodeActAgent"}
                ]
                sub_tasks = _normalize_subtask_dicts(sub_tasks)[: self.max_subtasks]

                plan_failed = False
                for subtask in sub_tasks:
                    if budget_steps() >= max_steps or not control_available():
                        break
                    retrieval = self.dms_memory.backend.retrieve(
                        subtask=subtask,
                        task_id=task_id,
                        task_name=task.name,
                        task_goal=task.goal,
                        task_app_names=task_app_names,
                        foreground_activity=observation["foreground_activity"],
                    )
                    if getattr(retrieval, "mode", "miss") == "replay":
                        selected_memory_id = str(
                            getattr(retrieval, "selected_memory_id", "") or ""
                        )
                        if selected_memory_id:
                            active_memory_ids.append(selected_memory_id)
                        replay_trajectory = list(getattr(retrieval, "selected_trajectory", []))
                        replay_failed = False
                        replay_subtask_completed = False
                        for replay_step in replay_trajectory:
                            if budget_steps() >= max_steps or not control_available():
                                replay_failed = True
                                break
                            replay_action = dict(
                                replay_step.get("action")
                                or replay_step.get("executed_action")
                                or {}
                            )
                            replay_record = StepRecord(
                                step=step_id,
                                subtask=str(subtask.get("goal", "")),
                                precondition=str(subtask.get("precondition", "")),
                                observation=observation,
                                planner_output=plan,
                                actor_output={
                                    "tool_name": "replay",
                                    "code": None,
                                    "action": replay_action,
                                    "error": None,
                                },
                                action=replay_action,
                                result="pending",
                                input_tokens=0,
                                output_tokens=0,
                            )
                            if replay_action.get("type") == "complete":
                                control_turns += 1
                                completion_success = bool(
                                    replay_action.get("success", True)
                                )
                                completion_fingerprint = _completion_fingerprint(
                                    subtask_goal=str(subtask.get("goal", "")),
                                    foreground_activity=observation["foreground_activity"],
                                    observation=observation,
                                )
                                if (
                                    self.engineering_optimization
                                    and completion_success
                                    and completion_fingerprint
                                    == last_completion_fingerprint
                                    and action_epoch == last_completion_action_epoch
                                ):
                                    completion_success = False
                                    unsupported_completions += 1
                                    replay_record.result = "unsupported_completion"
                                    replay_record.error = (
                                        "Repeated replay completion on unchanged device state."
                                    )
                                else:
                                    last_completion_fingerprint = completion_fingerprint
                                    last_completion_action_epoch = action_epoch
                                if completion_success and _looks_like_open_app_request(
                                    str(subtask.get("goal", "")),
                                    task_app_names,
                                ):
                                    if not _foreground_matches_task_scope(
                                        observation["foreground_activity"],
                                        task_app_names,
                                    ):
                                        completion_success = False
                                        replay_record.error = (
                                            "Replayed open-app memory claimed success, "
                                            "but the foreground activity did not match "
                                            f"the task scope: {observation['foreground_activity']!r}"
                                        )
                                if replay_record.result != "unsupported_completion":
                                    replay_record.result = (
                                        "subtask_complete"
                                        if completion_success
                                        else "subtask_failed"
                                    )
                                replay_subtask_completed = completion_success
                                replay_failed = not completion_success
                            elif replay_action.get("type") == "remember":
                                control_turns += 1
                                replay_record.result = "remembered"
                                replay_record.executed_action = dict(replay_action)
                            else:
                                execute_attempted = False
                                try:
                                    executed_action = self._bind_action_to_state(
                                        replay_action,
                                        prompt_elements,
                                    )
                                except Exception:
                                    executed_action = dict(
                                        replay_step.get("executed_action")
                                        or replay_action
                                        or {}
                                    )
                                try:
                                    json_action = to_json_action(executed_action)
                                    if json_action is None:
                                        raise ValueError(
                                            f"Unsupported replay action: {replay_action}"
                                        )
                                    replay_record.executed_action = executed_action
                                    env_steps += 1
                                    execute_attempted = True
                                    env.execute_action(json_action)
                                    action_epoch += 1
                                    if self.post_action_wait_seconds:
                                        time.sleep(self.post_action_wait_seconds)
                                    replay_record.result = "replayed"
                                    reward = float(task.is_successful(env))
                                    if reward >= 1.0:
                                        success = True
                                        replay_record.result = "subtask_complete"
                                        replay_subtask_completed = True
                                    else:
                                        state = get_state_with_a11y_retries(
                                            env,
                                            wait_to_stabilize=False,
                                        )
                                except Exception as exc:
                                    if execute_attempted and _strict_infrastructure_protocol():
                                        raise
                                    if not execute_attempted:
                                        control_turns += 1
                                    replay_record.result = "replay_failed"
                                    replay_record.error = str(exc)
                                    replay_failed = True
                            record = StepRecord(
                                step=step_id,
                                subtask=str(subtask.get("goal", "")),
                                precondition=str(subtask.get("precondition", "")),
                                observation=observation,
                                planner_output=plan,
                                actor_output=replay_record.actor_output,
                                action=replay_record.action,
                                result=replay_record.result,
                                executed_action=replay_record.executed_action,
                                input_tokens=0,
                                output_tokens=0,
                                error=replay_record.error,
                            )
                            self._append_step_record(
                                trajectory=trajectory,
                                record=record,
                                task=task,
                                task_id=task_id,
                            )
                            step_id += 1
                            if success:
                                break
                            if replay_failed:
                                break
                            if replay_subtask_completed:
                                break
                            if budget_steps() < max_steps and control_available():
                                observation, prompt_elements, image_path = self._capture_for_prompt(
                                    env=env,
                                    state=state,
                                    task_id=task_id,
                                    step_id=step_id,
                                )
                        if replay_subtask_completed:
                            self.dms_memory.backend.mark_reuse_success(selected_memory_id)
                            if success:
                                break
                            continue
                        if replay_failed:
                            self.dms_memory.backend.mark_reuse_failure(selected_memory_id)
                            plan_failed = True
                            break
                        self.dms_memory.backend.mark_reuse_failure(selected_memory_id)
                        plan_failed = True
                        break

                    local_steps = 0
                    local_subtask_trajectory_start = len(trajectory)
                    selected_memory_id = str(
                        getattr(retrieval, "selected_memory_id", "") or ""
                    )
                    subtask_completed = False
                    while (
                        local_steps < self.actor_local_step_guard
                        and budget_steps() < max_steps
                        and control_available()
                    ):
                        shortcut_action = _shortcut_open_app_action(
                            subtask_goal=str(subtask.get("goal", "")),
                            task_app_names=task_app_names,
                            foreground_activity=observation["foreground_activity"],
                            guard_system_dialogs=self.engineering_optimization,
                        )
                        if shortcut_action is not None:
                            actor_output = CodeActExecutionResult(
                                code="",
                                tool_name=shortcut_action["type"],
                                action=shortcut_action,
                                error=None,
                            )
                            act_in = 0
                            act_out = 0
                        else:
                            actor_output, act_in, act_out = self._act(
                                image_path=image_path,
                                task=task,
                                task_id=task_id,
                                global_goal=task.goal,
                                task_app_names=task_app_names,
                                subtask=subtask,
                                foreground_activity=observation["foreground_activity"],
                                compact_ui=observation["compact_ui_elements"],
                                step_history=_history_for_prompt(trajectory),
                                observation=observation,
                            )
                            input_tokens += act_in
                            output_tokens += act_out

                        action = actor_output.action
                        if action and action.get("type") == "complete" and _task_requires_answer(task.goal):
                            reason = str(action.get("reason") or "").strip()
                            if reason:
                                action = {"type": "answer", "text": reason}
                        action = _normalize_start_app_action(
                            action=action,
                            subtask_goal=str(subtask.get("goal", "")),
                            task_app_names=task_app_names,
                        )
                        record = StepRecord(
                            step=step_id,
                            subtask=str(subtask.get("goal", "")),
                            precondition=str(subtask.get("precondition", "")),
                            observation=observation,
                            planner_output=plan,
                            actor_output={
                                "tool_name": actor_output.tool_name,
                                "code": actor_output.code,
                                "action": action,
                                "error": actor_output.error,
                            },
                            action=action,
                            result="pending",
                            input_tokens=act_in,
                            output_tokens=act_out,
                        )

                        if actor_output.error:
                            control_turns += 1
                            record.result = "invalid_action"
                            record.error = actor_output.error
                            self._append_step_record(
                                trajectory=trajectory,
                                record=record,
                                task=task,
                                task_id=task_id,
                            )
                            step_id += 1
                            local_steps += 1
                            refresh_prompt_state()
                            plan_failed = True
                            break

                        if action is None:
                            control_turns += 1
                            record.result = "invalid_action"
                            record.error = "Actor did not produce a valid paper tool call."
                            self._append_step_record(
                                trajectory=trajectory,
                                record=record,
                                task=task,
                                task_id=task_id,
                            )
                            step_id += 1
                            local_steps += 1
                            refresh_prompt_state()
                            plan_failed = True
                            break

                        if action.get("type") == "remember":
                            control_turns += 1
                            self.dms_memory.backend.remember(
                                information=str(action.get("information") or ""),
                                task_id=task_id,
                                step_id=step_id,
                                subtask=subtask,
                                observation=observation,
                                trajectory=trajectory,
                            )
                            record.result = "remembered"
                            record.executed_action = dict(action)
                            self._append_step_record(
                                trajectory=trajectory,
                                record=record,
                                task=task,
                                task_id=task_id,
                            )
                            step_id += 1
                            local_steps += 1
                            break

                        if action.get("type") == "complete":
                            control_turns += 1
                            completion_success = bool(action.get("success", True))
                            completion_fingerprint = _completion_fingerprint(
                                subtask_goal=str(subtask.get("goal", "")),
                                foreground_activity=observation["foreground_activity"],
                                observation=observation,
                            )
                            repeated_unsupported_completion = (
                                self.engineering_optimization
                                and completion_success
                                and completion_fingerprint
                                == last_completion_fingerprint
                                and action_epoch == last_completion_action_epoch
                            )
                            if repeated_unsupported_completion:
                                completion_success = False
                                unsupported_completions += 1
                                record.result = "unsupported_completion"
                                record.error = (
                                    "Repeated completion claim on unchanged device state; "
                                    "the next plan must use a visibly different recovery."
                                )
                                plan_failed = True
                            else:
                                last_completion_fingerprint = completion_fingerprint
                                last_completion_action_epoch = action_epoch
                            if completion_success and _looks_like_open_app_request(
                                str(subtask.get("goal", "")),
                                task_app_names,
                            ):
                                if not _foreground_matches_task_scope(
                                    observation["foreground_activity"],
                                    task_app_names,
                                ):
                                    completion_success = False
                                    expected_packages = sorted(
                                        _expected_packages_for_task_scope(
                                            task_app_names
                                        )
                                    )
                                    record.error = (
                                        "Subtask claimed target app was open, "
                                        "but foreground activity did not match "
                                        f"the task scope. foreground="
                                        f"{observation['foreground_activity']!r} "
                                        f"expected_packages={expected_packages!r}"
                                    )
                                    plan_failed = True
                            if record.result != "unsupported_completion":
                                record.result = (
                                    "subtask_complete"
                                    if completion_success
                                    else "subtask_failed"
                                )
                            self._append_step_record(
                                trajectory=trajectory,
                                record=record,
                                task=task,
                                task_id=task_id,
                            )
                            if not completion_success:
                                plan_failed = True
                            else:
                                subtask_completed = True
                            step_id += 1
                            break

                        try:
                            executed_action = self._bind_action_to_state(
                                action,
                                prompt_elements,
                            )
                            json_action = to_json_action(executed_action)
                        except (TypeError, ValueError) as exc:
                            control_turns += 1
                            record.result = "invalid_action"
                            record.error = str(exc)
                            self._append_step_record(
                                trajectory=trajectory,
                                record=record,
                                task=task,
                                task_id=task_id,
                            )
                            step_id += 1
                            local_steps += 1
                            refresh_prompt_state()
                            plan_failed = True
                            break
                        if json_action is None:
                            control_turns += 1
                            record.result = "invalid_action"
                            record.error = f"Unsupported paper action: {action}"
                            self._append_step_record(
                                trajectory=trajectory,
                                record=record,
                                task=task,
                                task_id=task_id,
                            )
                            step_id += 1
                            local_steps += 1
                            refresh_prompt_state()
                            plan_failed = True
                            break

                        record.executed_action = executed_action
                        try:
                            if retrieval.mode == "mutate":
                                if hasattr(self.dms_memory.backend, "mutation_fallback"):
                                    self.dms_memory.backend.mutation_fallback(
                                        task_id=task_id,
                                        subtask=subtask,
                                    )
                                executed_action = executed_action
                            env_steps += 1
                            env.execute_action(json_action)
                            action_epoch += 1
                            if self.post_action_wait_seconds:
                                time.sleep(self.post_action_wait_seconds)
                            record.result = "executed"
                            reward = float(task.is_successful(env))
                            if reward >= 1.0:
                                success = True
                                subtask_completed = True
                                record.result = "subtask_complete"
                                self._append_step_record(
                                    trajectory=trajectory,
                                    record=record,
                                    task=task,
                                    task_id=task_id,
                                )
                                step_id += 1
                                local_steps += 1
                                break
                            state = get_state_with_a11y_retries(
                                env,
                                wait_to_stabilize=False,
                            )
                        except Exception as exc:
                            if _strict_infrastructure_protocol():
                                raise
                            record.result = "execution_error"
                            record.error = str(exc)
                            plan_failed = True
                        self._append_step_record(
                            trajectory=trajectory,
                            record=record,
                            task=task,
                            task_id=task_id,
                        )
                        step_id += 1
                        local_steps += 1
                        if plan_failed:
                            break
                        if budget_steps() < max_steps and control_available():
                            observation, prompt_elements, image_path = self._capture_for_prompt(
                                env=env,
                                state=state,
                                task_id=task_id,
                                step_id=step_id,
                            )
                    local_subtask_trajectory = trajectory[local_subtask_trajectory_start:]
                    if subtask_completed and local_subtask_trajectory:
                        if retrieval.mode == "mutate" and selected_memory_id:
                            replaced = self.dms_memory.backend.replace_with_mutation(
                                memory_id=selected_memory_id,
                                subtask=subtask,
                                trajectory=local_subtask_trajectory,
                                task_id=task_id,
                                task_name=task.name,
                                task_goal=task.goal,
                                app_names=task_app_names,
                            )
                            if replaced:
                                active_memory_ids.append(selected_memory_id)
                        elif retrieval.mode != "replay":
                            memory_id = self.dms_memory.backend.create_memory(
                                subtask=subtask,
                                trajectory=local_subtask_trajectory,
                                task_id=task_id,
                                task_name=task.name,
                                task_goal=task.goal,
                                app_names=task_app_names,
                            )
                            if memory_id:
                                active_memory_ids.append(memory_id)
                                created_memory_ids.append(memory_id)
                    if plan_failed:
                        break

                reward = float(task.is_successful(env))
                if reward >= 1.0:
                    success = True
                    break
        except Exception as exc:
            task_error = "".join(traceback.format_exception(exc))
        finally:
            try:
                reward = float(task.is_successful(env))
                success = reward >= 1.0
            except Exception as exc:
                if _strict_infrastructure_protocol():
                    task_error = _append_infrastructure_error(
                        task_error,
                        phase="final_success_evaluation",
                        error=exc,
                    )
            try:
                task.tear_down(env)
            except Exception as exc:
                if _strict_infrastructure_protocol():
                    task_error = _append_infrastructure_error(
                        task_error,
                        phase="task_cleanup",
                        error=exc,
                    )

        if hasattr(self.dms_memory.backend, "finalize_task"):
            self.dms_memory.backend.finalize_task(
                task=task,
                task_id=task_id,
                task_name=task.name,
                goal=task.goal,
                success=success,
                steps=budget_steps() if "budget_steps" in locals() else step_id,
                trajectory=trajectory,
                active_memory_ids=active_memory_ids,
            )
        return TaskRunResult(
            task_id=task_id,
            task_name=task.name,
            goal=task.goal,
            success=success,
            reward=reward,
            steps=budget_steps() if "budget_steps" in locals() else step_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            memory_size_after=self._memory_size_after(),
            planner_cycles=planner_cycles,
            unsupported_completions=unsupported_completions,
            control_turns=control_turns,
            event_count=step_id,
            control_turn_limit_reached=(
                self.control_turn_limit is not None
                and control_turns >= self.control_turn_limit
            ),
            memory_stats=self._memory_stats(),
            trajectory=trajectory,
            error=task_error,
        )
