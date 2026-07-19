from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass
from typing import Any


PLANNER_TOOL_SPECS = [
    {
        "name": "set_tasks_with_agents",
        "description": (
            "Provide the next 1-5 functional steps and assign each step to the "
            "most appropriate specialized agent."
        ),
        "arguments": {
            "task_assignments": [
                {
                    "task": "Precondition: None. Goal: ...",
                    "agent": "CodeActAgent",
                }
            ]
        },
    },
    {
        "name": "complete_goal",
        "description": "Call this when the overall user goal has been achieved.",
        "arguments": {"message": "The overall task is complete."},
    },
]


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_ALIAS_KEY_RE = re.compile(r"[^a-z0-9]+")

_TOOL_NAME_ALIASES = {
    "tap": "tap",
    "swipe": "swipe",
    "inputtext": "input_text",
    "presskey": "press_key",
    "presskeyboard": "press_key",
    "startapp": "start_app",
    "answer": "answer",
    "print": "answer",
    "complete": "complete",
    "remember": "remember",
}

_PLANNER_TOOL_NAME_ALIASES = {
    "settaskswithagents": "set_tasks_with_agents",
    "completegoal": "complete_goal",
}

_ARGUMENT_NAME_ALIASES = {
    "index": "index",
    "x": "x",
    "y": "y",
    "text": "text",
    "clear": "clear",
    "cleartext": "clear_text",
    "durationms": "duration_ms",
    "startx": "start_x",
    "starty": "start_y",
    "endx": "end_x",
    "endy": "end_y",
    "keycode": "keycode",
    "presskey": "keycode",
    "presskeyboard": "keycode",
    "package": "package",
    "appname": "app_name",
    "information": "information",
    "success": "success",
    "reason": "reason",
    "answer": "text",
}


@dataclass(frozen=True)
class PlannerToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class CodeActExecutionResult:
    code: str
    tool_name: str | None
    action: dict[str, Any] | None
    error: str | None = None


def extract_code_block(text: str) -> str:
    """Returns the first fenced Python code block, or the stripped text."""
    candidate = text.strip()
    match = _CODE_FENCE_RE.search(candidate)
    if match:
        candidate = match.group(1)
    return textwrap.dedent(candidate).strip()


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception as exc:  # pragma: no cover - defensive.
        raise ValueError("Only literal Python arguments are allowed.") from exc


def _alias_key(name: str) -> str:
    return _ALIAS_KEY_RE.sub("", str(name).strip().lower())


def _canonical_tool_name(name: str) -> str:
    raw_name = str(name).strip()
    return _TOOL_NAME_ALIASES.get(
        _alias_key(raw_name),
        raw_name.replace("-", "_"),
    )


def _canonical_planner_tool_name(name: str) -> str:
    raw_name = str(name).strip()
    return _PLANNER_TOOL_NAME_ALIASES.get(
        _alias_key(raw_name),
        raw_name.replace("-", "_"),
    )


def _normalize_name(name: str) -> str:
    raw_name = str(name).strip()
    return _ARGUMENT_NAME_ALIASES.get(
        _alias_key(raw_name),
        raw_name.replace("-", "_"),
    )


def _normalize_keycode(keycode: Any) -> str:
    if keycode is None:
        raise ValueError("press_key requires keycode.")
    normalized = str(keycode).strip()
    if not normalized:
        raise ValueError("press_key requires keycode.")
    normalized = normalized.upper()
    if normalized.startswith("KEYCODE_"):
        return normalized
    return f"KEYCODE_{normalized}"


def _pick(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def split_precondition_goal(task_text: str) -> tuple[str, str]:
    marker_pre = "Precondition:"
    marker_goal = "Goal:"
    text = str(task_text).strip()
    if marker_pre in text and marker_goal in text:
        before_goal, goal = text.split(marker_goal, 1)
        precondition = before_goal.split(marker_pre, 1)[1].strip(" .")
        return precondition or "None", goal.strip()
    if marker_goal in text:
        precondition, goal = text.split(marker_goal, 1)
        precondition = precondition.replace(marker_pre, "").strip(" .")
        return precondition or "None", goal.strip()
    return "None", text


def normalize_task_assignments(
    task_assignments: Any,
    max_subtasks: int,
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if not isinstance(task_assignments, list):
        return normalized

    for item in task_assignments:
        if len(normalized) >= max_subtasks:
            break

        if isinstance(item, str):
            precondition, goal = split_precondition_goal(item)
            normalized.append(
                {
                    "precondition": precondition,
                    "goal": goal,
                    "agent": "CodeActAgent",
                }
            )
            continue
        if not isinstance(item, dict):
            continue

        task_text = str(
            _pick(item, "task", "subtask") or ""
        ).strip()
        if task_text:
            precondition, goal = split_precondition_goal(task_text)
        else:
            precondition = str(_pick(item, "precondition") or "None").strip()
            if "Goal:" in precondition:
                precondition, _ = split_precondition_goal(precondition)
            goal = str(_pick(item, "goal") or "").strip()
            if "Precondition:" in goal or "Goal:" in goal:
                parsed_precondition, goal = split_precondition_goal(goal)
                if precondition == "None":
                    precondition = parsed_precondition

        normalized.append(
            {
                "precondition": precondition or "None",
                "goal": goal,
                "agent": str(_pick(item, "agent") or "CodeActAgent"),
            }
        )
    return normalized


def normalize_planner_tool_call(
    parsed: Any,
    max_subtasks: int,
) -> dict[str, Any]:
    """Normalizes planner output into the local planner contract."""
    if isinstance(parsed, list):
        return {
            "complete": False,
            "message": "",
            "sub_tasks": normalize_task_assignments(parsed, max_subtasks),
        }

    if not isinstance(parsed, dict):
        return {
            "complete": False,
            "message": "Planner did not return a valid tool call.",
            "sub_tasks": [],
        }

    if "set_tasks_with_agents" in parsed:
        return {
            "complete": False,
            "message": str(_pick(parsed, "message", "reason") or ""),
            "sub_tasks": normalize_task_assignments(
                parsed["set_tasks_with_agents"],
                max_subtasks,
            ),
        }

    if "complete_goal" in parsed:
        payload = parsed["complete_goal"]
        if isinstance(payload, dict):
            message = str(_pick(payload, "message") or "")
        else:
            message = str(payload or "")
        return {"complete": True, "message": message, "sub_tasks": []}

    name = _canonical_planner_tool_name(
        _pick(parsed, "name", "tool", "type", "action") or ""
    )
    arguments = parsed.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {
            key: value
            for key, value in parsed.items()
            if key not in {"name", "tool", "type", "action"}
        }

    if name == "complete_goal":
        message = str(_pick(arguments, "message") or _pick(parsed, "message") or "")
        return {"complete": True, "message": message, "sub_tasks": []}

    if name == "set_tasks_with_agents":
        task_assignments = _pick(arguments, "task_assignments")
        if task_assignments is None:
            task_assignments = _pick(parsed, "task_assignments", "sub_tasks", "tasks")
        return {
            "complete": False,
            "message": str(_pick(parsed, "message") or ""),
            "sub_tasks": normalize_task_assignments(task_assignments, max_subtasks),
        }

    if "task_assignments" in parsed or "sub_tasks" in parsed or "tasks" in parsed:
        task_assignments = _pick(parsed, "task_assignments", "sub_tasks", "tasks")
        return {
            "complete": bool(_pick(parsed, "complete") or False),
            "message": str(_pick(parsed, "message", "reason") or ""),
            "sub_tasks": normalize_task_assignments(task_assignments, max_subtasks),
        }

    if bool(_pick(parsed, "complete") or False):
        return {
            "complete": True,
            "message": str(_pick(parsed, "message", "reason") or ""),
            "sub_tasks": [],
        }

    return {
        "complete": False,
        "message": str(_pick(parsed, "message", "reason") or ""),
        "sub_tasks": [],
    }


def _tool_call_to_action(
    tool_name: str,
    args: list[Any],
    kwargs: dict[str, Any],
    *,
    allow_remember: bool,
) -> dict[str, Any]:
    tool_name = _canonical_tool_name(tool_name)

    if tool_name == "remember":
        if not allow_remember:
            raise ValueError("remember is disabled for Baseline A/B.")
        information = _pick(kwargs, "information", "text")
        if information is None and args:
            information = args[0]
        if information is None:
            raise ValueError("remember requires information.")
        return {"type": "remember", "information": information}

    if tool_name == "tap":
        if args and kwargs:
            raise ValueError("tap should use either positional or keyword arguments, not both.")
        if args:
            if len(args) == 1:
                kwargs = {"index": args[0]}
            elif len(args) == 2:
                kwargs = {"x": args[0], "y": args[1]}
            else:
                raise ValueError("tap requires index or x/y coordinates.")
        index = _pick(kwargs, "index")
        x = _pick(kwargs, "x")
        y = _pick(kwargs, "y")
        duration_ms = _pick(kwargs, "duration_ms", "durationms")
        if index is None and (x is None or y is None):
            raise ValueError("tap requires index or both x/y coordinates.")
        if index is not None and (x is not None or y is not None):
            raise ValueError("tap cannot mix index with x/y coordinates.")
        action: dict[str, Any] = {"type": "tap"}
        if index is not None:
            action["index"] = int(index)
        else:
            action["x"] = int(x)
            action["y"] = int(y)
        if duration_ms is not None:
            action["duration_ms"] = int(duration_ms)
        return action

    if tool_name == "input_text":
        if args and kwargs:
            if len(args) == 1 and "text" not in kwargs:
                kwargs = {"text": args[0], **kwargs}
            elif len(args) == 1 and "text" in kwargs and isinstance(args[0], str):
                kwargs = dict(kwargs)
            elif len(args) == 2 and "text" not in kwargs and "clear" not in kwargs and "clear_text" not in kwargs:
                kwargs = {"text": args[0], "clear": args[1], **kwargs}
            else:
                raise ValueError("input_text should use either positional or keyword arguments, not both.")
        if args:
            if len(args) == 1 and "text" not in kwargs:
                kwargs = {"text": args[0]}
            elif len(args) == 2 and "text" not in kwargs and "clear" not in kwargs and "clear_text" not in kwargs:
                kwargs = {"text": args[0], "clear": args[1]}
            else:
                if len(args) > 2:
                    raise ValueError("input_text accepts text and optional clear.")
        text = _pick(kwargs, "text")
        if text is None:
            raise ValueError("input_text requires text.")
        action = {"type": "input_text", "text": str(text)}
        clear = _pick(kwargs, "clear", "clear_text")
        if clear is not None:
            action["clear"] = bool(clear)
        index = _pick(kwargs, "index")
        x = _pick(kwargs, "x")
        y = _pick(kwargs, "y")
        if index is not None:
            action["index"] = int(index)
        if x is not None:
            action["x"] = int(x)
        if y is not None:
            action["y"] = int(y)
        return action

    if tool_name == "press_key":
        if args and kwargs:
            raise ValueError("press_key should use either positional or keyword arguments, not both.")
        if args:
            if len(args) != 1:
                raise ValueError("press_key accepts exactly one keycode.")
            kwargs = {"keycode": args[0]}
        keycode = _pick(kwargs, "keycode")
        return {"type": "press_key", "keycode": _normalize_keycode(keycode)}

    if tool_name == "swipe":
        if args and kwargs:
            raise ValueError("swipe should use either positional or keyword arguments, not both.")
        if args:
            if len(args) not in (4, 5):
                raise ValueError("swipe requires start_x, start_y, end_x, end_y, and optional duration_ms.")
            kwargs = {
                "start_x": args[0],
                "start_y": args[1],
                "end_x": args[2],
                "end_y": args[3],
            }
            if len(args) == 5:
                kwargs["duration_ms"] = args[4]
        start_x = _pick(kwargs, "start_x", "startx")
        start_y = _pick(kwargs, "start_y", "starty")
        end_x = _pick(kwargs, "end_x", "endx")
        end_y = _pick(kwargs, "end_y", "endy")
        if None in (start_x, start_y, end_x, end_y):
            raise ValueError("swipe requires start_x, start_y, end_x, and end_y.")
        action = {
            "type": "swipe",
            "start_x": int(start_x),
            "start_y": int(start_y),
            "end_x": int(end_x),
            "end_y": int(end_y),
        }
        duration_ms = _pick(kwargs, "duration_ms", "durationms")
        if duration_ms is not None:
            action["duration_ms"] = int(duration_ms)
        return action

    if tool_name == "start_app":
        if args and kwargs:
            raise ValueError("start_app should use either positional or keyword arguments, not both.")
        if args:
            if len(args) != 1:
                raise ValueError("start_app accepts exactly one package name.")
            kwargs = {"package": args[0]}
        package = _pick(kwargs, "package", "app_name")
        if package is None:
            raise ValueError("start_app requires package.")
        return {"type": "start_app", "package": str(package)}

    if tool_name == "complete":
        if args and kwargs:
            raise ValueError("complete should use either positional or keyword arguments, not both.")
        if args:
            if len(args) != 2:
                raise ValueError("complete accepts success and reason.")
            kwargs = {"success": args[0], "reason": args[1]}
        success = _pick(kwargs, "success")
        reason = _pick(kwargs, "reason")
        if success is None:
            raise ValueError("complete requires success.")
        return {
            "type": "complete",
            "success": bool(success),
            "reason": str(reason or ""),
        }

    if tool_name == "answer":
        if args and kwargs:
            if len(args) == 1 and "text" not in kwargs:
                kwargs = {"text": args[0], **kwargs}
            else:
                raise ValueError("answer should use either positional or keyword arguments, not both.")
        if args:
            if len(args) != 1:
                raise ValueError("answer accepts exactly one text argument.")
            kwargs = {"text": args[0]}
        text = _pick(kwargs, "text", "answer")
        if text is None:
            raise ValueError("answer requires text.")
        return {"type": "answer", "text": str(text)}

    raise ValueError(f"Unsupported paper tool: {tool_name}")


def execute_codeact(
    code_text: str,
    *,
    allow_remember: bool = False,
) -> CodeActExecutionResult:
    """Parses and executes a single CodeAct-style Python tool call."""
    code = extract_code_block(code_text)
    if not code:
        return CodeActExecutionResult(
            code="",
            tool_name=None,
            action=None,
            error="Actor did not return a Python code block.",
        )

    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return CodeActExecutionResult(
            code=code,
            tool_name=None,
            action=None,
            error=f"Invalid Python code: {exc.msg}",
        )

    statements: list[ast.stmt] = []
    for statement in module.body:
        if isinstance(statement, ast.Expr) and isinstance(
            statement.value, ast.Constant
        ) and isinstance(statement.value.value, str):
            continue
        if isinstance(statement, ast.Pass):
            continue
        statements.append(statement)

    if len(statements) != 1 or not isinstance(statements[0], ast.Expr):
        return CodeActExecutionResult(
            code=code,
            tool_name=None,
            action=None,
            error="CodeAct output must contain exactly one tool call.",
        )

    call = statements[0].value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return CodeActExecutionResult(
            code=code,
            tool_name=None,
            action=None,
            error="CodeAct output must call a single paper tool.",
        )

    tool_name = _canonical_tool_name(call.func.id)
    try:
        args = [_literal(arg) for arg in call.args]
        kwargs: dict[str, Any] = {}
        for keyword in call.keywords:
            if keyword.arg is None:
                return CodeActExecutionResult(
                    code=code,
                    tool_name=tool_name,
                    action=None,
                    error="Keyword unpacking is not allowed in CodeAct output.",
                )
            kwargs[_normalize_name(keyword.arg)] = _literal(keyword.value)
    except ValueError as exc:
        return CodeActExecutionResult(
            code=code,
            tool_name=tool_name,
            action=None,
            error=str(exc),
        )

    try:
        action = _tool_call_to_action(
            tool_name,
            args,
            kwargs,
            allow_remember=allow_remember,
        )
    except Exception as exc:
        return CodeActExecutionResult(
            code=code,
            tool_name=tool_name,
            action=None,
            error=str(exc),
        )

    return CodeActExecutionResult(
        code=code,
        tool_name=tool_name,
        action=action,
        error=None,
    )
