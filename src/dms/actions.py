from __future__ import annotations

import re
from typing import Any

from android_world.env import json_action


_ALIAS_KEY_RE = re.compile(r"[^a-z0-9]+")

_ACTION_TYPE_ALIASES = {
    "complete": "complete",
    "tap": "tap",
    "swipe": "swipe",
    "inputtext": "input_text",
    "presskey": "press_key",
    "presskeyboard": "press_key",
    "startapp": "start_app",
    "remember": "remember",
}

_ACTION_KEY_ALIASES = {
    "actiontype": "action_type",
    "name": "name",
    "type": "type",
    "success": "success",
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
    "direction": "direction",
    "keycode": "keycode",
    "presskey": "keycode",
    "presskeyboard": "keycode",
    "package": "package",
    "appname": "app_name",
}


def _alias_key(name: str) -> str:
    return _ALIAS_KEY_RE.sub("", str(name).strip().lower())


def _normalize_action_key(name: str) -> str:
    raw_name = str(name).strip()
    return _ACTION_KEY_ALIASES.get(
        _alias_key(raw_name),
        raw_name.replace("-", "_"),
    )


def _normalize_action(action: dict[str, Any]) -> dict[str, Any]:
    return {
        _normalize_action_key(key): value
        for key, value in action.items()
    }


def _normalize_action_type(action_type: Any) -> str | None:
    if action_type is None:
        return None
    raw_type = str(action_type).strip()
    return _ACTION_TYPE_ALIASES.get(
        _alias_key(raw_type),
        raw_type.replace("-", "_"),
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


def _action_value(action: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in action:
            return action[key]
    return None


def to_json_action(action: dict[str, Any]) -> json_action.JSONAction | None:
    action = _normalize_action(action)
    action_type = _normalize_action_type(
        action.get("type") or action.get("action_type") or action.get("name")
    )
    if action_type == "complete":
        return json_action.JSONAction(
            action_type=json_action.STATUS,
            goal_status="complete" if action.get("success", True) else "infeasible",
        )
    if action_type == "tap":
        duration = int(_action_value(action, "duration_ms", "durationms") or 0)
        aw_type = json_action.LONG_PRESS if duration >= 800 else json_action.CLICK
        index = _action_value(action, "index")
        x = _action_value(action, "x")
        y = _action_value(action, "y")
        kwargs: dict[str, Any] = {"action_type": aw_type}
        if index is not None:
            kwargs["index"] = index
        if x is not None:
            kwargs["x"] = x
        if y is not None:
            kwargs["y"] = y
        return json_action.JSONAction(**kwargs)
    if action_type == "input_text":
        return json_action.JSONAction(
            action_type=json_action.INPUT_TEXT,
            index=_action_value(action, "index"),
            x=_action_value(action, "x"),
            y=_action_value(action, "y"),
            text=str(action.get("text", "")),
            clear_text=bool(_action_value(action, "clear", "clear_text") or False),
        )
    if action_type == "press_key":
        return json_action.JSONAction(
            action_type=json_action.PRESS_KEYBOARD,
            keycode=_normalize_keycode(
                _action_value(action, "keycode", "press_key")
            ),
        )
    if action_type == "swipe":
        return json_action.JSONAction(
            action_type=json_action.SWIPE,
            start_x=_action_value(action, "start_x", "startx"),
            start_y=_action_value(action, "start_y", "starty"),
            end_x=_action_value(action, "end_x", "endx"),
            end_y=_action_value(action, "end_y", "endy"),
            duration_ms=_action_value(action, "duration_ms", "durationms"),
            direction=action.get("direction"),
        )
    if action_type == "start_app":
        return json_action.JSONAction(
            action_type=json_action.OPEN_APP,
            app_name=action.get("package") or action.get("app_name"),
        )
    if action_type == "answer":
        return json_action.JSONAction(
            action_type=json_action.ANSWER,
            text=str(action.get("text", "")),
        )
    if action_type == "remember":
        return None
    return None
