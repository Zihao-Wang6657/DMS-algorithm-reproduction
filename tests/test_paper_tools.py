from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


WORKING_SPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKING_SPACE / "src"))
sys.path.insert(0, str(WORKING_SPACE / "third_party" / "android_world"))

from android_world.env import json_action  # noqa: E402
from dms.agent import PALiteAgent, StepRecord  # noqa: E402
from dms.actions import to_json_action  # noqa: E402
from dms.paper_tools import (  # noqa: E402
    execute_codeact,
    normalize_planner_tool_call,
)


def test_planner_normalization_limits_subtasks_and_parses_preconditions():
    parsed = {
        "name": "set_tasks_with_agents",
        "arguments": {
            "task_assignments": [
                {
                    "task": "Precondition: Home screen. Goal: Open Settings.",
                    "agent": "CodeActAgent",
                },
                "Precondition: Settings is open. Goal: Search for Wi-Fi.",
                {"precondition": "Wi-Fi screen", "goal": "Toggle Wi-Fi"},
            ]
        },
    }

    normalized = normalize_planner_tool_call(parsed, max_subtasks=2)

    assert normalized == {
        "complete": False,
        "message": "",
        "sub_tasks": [
            {
                "precondition": "Home screen",
                "goal": "Open Settings.",
                "agent": "CodeActAgent",
            },
            {
                "precondition": "Settings is open",
                "goal": "Search for Wi-Fi.",
                "agent": "CodeActAgent",
            },
        ],
    }


def test_planner_normalization_accepts_bare_task_assignment_list():
    normalized = normalize_planner_tool_call(
        [
            {"task": "Goal: Open the settings app.", "agent": "CodeActAgent"},
            {"task": "Goal: Search for Wi-Fi.", "agent": "CodeActAgent"},
        ],
        max_subtasks=1,
    )

    assert normalized == {
        "complete": False,
        "message": "",
        "sub_tasks": [
            {
                "precondition": "None",
                "goal": "Open the settings app.",
                "agent": "CodeActAgent",
            },
        ],
    }


def test_planner_normalization_accepts_direct_set_tasks_mapping():
    normalized = normalize_planner_tool_call(
        {
            "set_tasks_with_agents": [
                {
                    "task": "Precondition: None. Goal: Open Settings.",
                    "agent": "CodeActAgent",
                }
            ]
        },
        max_subtasks=5,
    )

    assert normalized == {
        "complete": False,
        "message": "",
        "sub_tasks": [
            {
                "precondition": "None",
                "goal": "Open Settings.",
                "agent": "CodeActAgent",
            }
        ],
    }


def test_planner_normalization_accepts_direct_complete_goal_mapping():
    normalized = normalize_planner_tool_call(
        {"complete_goal": {"message": "done"}},
        max_subtasks=5,
    )

    assert normalized == {
        "complete": True,
        "message": "done",
        "sub_tasks": [],
    }


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        (
            "setTasksWithAgents",
            {
                "complete": False,
                "message": "",
                "sub_tasks": [
                    {
                        "precondition": "None",
                        "goal": "Open Settings.",
                        "agent": "CodeActAgent",
                    }
                ],
            },
        ),
        (
            "set-tasks-with-agents",
            {
                "complete": False,
                "message": "",
                "sub_tasks": [
                    {
                        "precondition": "None",
                        "goal": "Open Settings.",
                        "agent": "CodeActAgent",
                    }
                ],
            },
        ),
        (
            "completeGoal",
            {"complete": True, "message": "done", "sub_tasks": []},
        ),
    ],
)
def test_planner_normalization_accepts_tool_name_aliases(tool_name, expected):
    if "set" in tool_name.lower():
        parsed = {
            "name": tool_name,
            "arguments": {
                "task_assignments": [
                    {"task": "Goal: Open Settings.", "agent": "CodeActAgent"}
                ]
            },
        }
    else:
        parsed = {
            "name": tool_name,
            "arguments": {"message": "done"},
        }

    assert normalize_planner_tool_call(parsed, max_subtasks=5) == expected


@pytest.mark.parametrize("parsed", [None, "not-json", 42])
def test_planner_normalization_rejects_invalid_non_list_non_dict(parsed):
    assert normalize_planner_tool_call(parsed, max_subtasks=5) == {
        "complete": False,
        "message": "Planner did not return a valid tool call.",
        "sub_tasks": [],
    }


def test_pa_lite_plan_records_raw_text_and_parsed_json(tmp_path):
    parsed_json = [
        {"task": "Goal: Open the settings app.", "agent": "CodeActAgent"}
    ]

    class FakeModel:
        def generate(self, **kwargs):
            return SimpleNamespace(
                text="raw planner text",
                parsed_json=parsed_json,
                input_tokens=11,
                output_tokens=7,
            )

    agent = PALiteAgent(model=FakeModel(), run_dir=tmp_path)

    plan, input_tokens, output_tokens = agent._plan(
        image_path="/tmp/screenshot.png",
        goal="Open settings",
        task_app_names=["settings"],
        foreground_activity="Launcher",
        compact_ui=[],
        task_history=[],
    )

    assert input_tokens == 11
    assert output_tokens == 7
    assert plan["raw_text"] == "raw planner text"
    assert plan["parsed_json"] == parsed_json
    assert plan["sub_tasks"] == [
        {
            "precondition": "None",
            "goal": "Open the settings app.",
            "agent": "CodeActAgent",
        }
    ]

    record = StepRecord(
        step=0,
        subtask="Open the settings app.",
        precondition="None",
        observation={},
        planner_output=plan,
        actor_output=None,
        action=None,
        result="pending",
    )
    json.dumps(record.to_dict())


@pytest.mark.parametrize(
    ("code", "expected_tool_name", "expected_action"),
    [
        (
            "tap(index=3, durationMs=900)",
            "tap",
            {"type": "tap", "index": 3, "duration_ms": 900},
        ),
        (
            "input_text('hello', True)",
            "input_text",
            {"type": "input_text", "text": "hello", "clear": True},
        ),
        (
            "inputText(text='hello', clearText=True)",
            "input_text",
            {"type": "input_text", "text": "hello", "clear": True},
        ),
        (
            "input_text('hello', clear=True)",
            "input_text",
            {"type": "input_text", "text": "hello", "clear": True},
        ),
        (
            "input_text('hello', index=3)",
            "input_text",
            {"type": "input_text", "text": "hello", "index": 3},
        ),
        (
            "input_text('Title', text='Sausage and Peppers Skillet')",
            "input_text",
            {"type": "input_text", "text": "Sausage and Peppers Skillet"},
        ),
        (
            "input_text('Description', text='hello', clear=True)",
            "input_text",
            {"type": "input_text", "text": "hello", "clear": True},
        ),
        (
            "pressKeyboard('ENTER')",
            "press_key",
            {"type": "press_key", "keycode": "KEYCODE_ENTER"},
        ),
        (
            "pressKey(keycode='KEYCODE_BACK')",
            "press_key",
            {"type": "press_key", "keycode": "KEYCODE_BACK"},
        ),
        (
            "swipe(startX=1, startY=2, endX=3, endY=4, durationMs=5)",
            "swipe",
            {
                "type": "swipe",
                "start_x": 1,
                "start_y": 2,
                "end_x": 3,
                "end_y": 4,
                "duration_ms": 5,
            },
        ),
        (
            "startApp(appName='Settings')",
            "start_app",
            {"type": "start_app", "package": "Settings"},
        ),
        (
            "complete(success=True, reason='done')",
            "complete",
            {"type": "complete", "success": True, "reason": "done"},
        ),
        (
            "answer('42')",
            "answer",
            {"type": "answer", "text": "42"},
        ),
    ],
)
def test_execute_codeact_valid_actions_and_aliases(
    code,
    expected_tool_name,
    expected_action,
):
    result = execute_codeact(f"```python\n{code}\n```")

    assert result.error is None
    assert result.tool_name == expected_tool_name
    assert result.action == expected_action


def test_execute_codeact_disables_remember_for_baselines():
    result = execute_codeact("remember('persist this')", allow_remember=False)

    assert result.action is None
    assert result.error == "remember is disabled for Baseline A/B."


def test_execute_codeact_enables_remember_for_dms():
    result = execute_codeact("remember('persist this')", allow_remember=True)

    assert result.error is None
    assert result.tool_name == "remember"
    assert result.action == {"type": "remember", "information": "persist this"}


def test_execute_codeact_rejects_multiple_statements():
    result = execute_codeact("tap(index=1)\npress_key('BACK')")

    assert result.action is None
    assert result.error == "CodeAct output must contain exactly one tool call."


def test_execute_codeact_rejects_non_literal_arguments():
    result = execute_codeact("tap(index=button_index)")

    assert result.action is None
    assert result.error == "Only literal Python arguments are allowed."


@pytest.mark.parametrize(
    ("paper_action", "expected"),
    [
        (
            {"type": "tap", "index": 7, "durationms": 0},
            {"action_type": json_action.CLICK, "index": 7},
        ),
        (
            {"type": "tap", "x": 10, "y": 20, "duration_ms": 800},
            {"action_type": json_action.LONG_PRESS, "x": 10, "y": 20},
        ),
        (
            {"type": "input text", "text": "abc", "clear text": True},
            {
                "action_type": json_action.INPUT_TEXT,
                "text": "abc",
                "clear_text": True,
            },
        ),
        (
            {"type": "swipe", "startx": 1, "starty": 2, "endx": 3, "endy": 4},
            {
                "action_type": json_action.SWIPE,
                "start_x": 1,
                "start_y": 2,
                "end_x": 3,
                "end_y": 4,
            },
        ),
        (
            {"type": "start app", "app name": "Settings"},
            {"action_type": json_action.OPEN_APP, "app_name": "Settings"},
        ),
        (
            {"type": "complete", "success": False},
            {"action_type": json_action.STATUS, "goal_status": "infeasible"},
        ),
        (
            {"type": "answer", "text": "42"},
            {"action_type": json_action.ANSWER, "text": "42"},
        ),
        (
            {"type": "press key", "press key": "ENTER"},
            {
                "action_type": json_action.PRESS_KEYBOARD,
                "keycode": "KEYCODE_ENTER",
            },
        ),
        (
            {"type": "pressKeyboard", "keycode": "KEYCODE_BACK"},
            {
                "action_type": json_action.PRESS_KEYBOARD,
                "keycode": "KEYCODE_BACK",
            },
        ),
    ],
)
def test_to_json_action_maps_paper_actions_and_normalizes_aliases(
    paper_action,
    expected,
):
    action = to_json_action(paper_action)

    assert action is not None
    for key, value in expected.items():
        assert getattr(action, key) == value


@pytest.mark.parametrize("paper_action", [
    {"type": "press_key", "keycode": ""},
    {"type": "press_key"},
])
def test_to_json_action_rejects_empty_press_key(paper_action):
    with pytest.raises(ValueError, match="press_key requires keycode"):
        to_json_action(paper_action)
