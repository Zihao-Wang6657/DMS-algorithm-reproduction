from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


WORKING_SPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKING_SPACE / "src"))
sys.path.insert(0, str(WORKING_SPACE / "third_party" / "android_world"))

from android_world.env import json_action  # noqa: E402
from dms.agent import (  # noqa: E402
    PALiteAgent,
    _history_for_prompt,
    _fallback_answer_from_complete,
    _normalize_start_app_action,
    _task_app_scope,
)
from env.androidworld_env import AndroidWorldObservationStore  # noqa: E402
from android_world.utils import file_utils  # noqa: E402


def _element(bounds=(10, 20, 30, 60)):
    bbox = None
    if bounds is not None:
        bbox = SimpleNamespace(
            x_min=bounds[0],
            y_min=bounds[1],
            x_max=bounds[2],
            y_max=bounds[3],
        )
    return SimpleNamespace(
        text="button",
        content_description=None,
        class_name="Button",
        resource_name=None,
        package_name="example",
        bbox_pixels=bbox,
        is_clickable=True,
        is_editable=False,
        is_enabled=True,
        is_scrollable=False,
        is_visible=True,
    )


def _non_interactable_element(bounds=(10, 20, 30, 60)):
    element = _element(bounds)
    element.is_clickable = False
    element.is_editable = False
    element.is_scrollable = False
    return element


def _state(*elements):
    return SimpleNamespace(
        pixels=np.zeros((4, 4, 3), dtype=np.uint8),
        ui_elements=list(elements),
    )


class FakeEnv:
    def __init__(self, initial_state, next_state=None):
        self.initial_state = initial_state
        self.next_state = next_state
        self.executed_actions = []
        self.get_state_calls = []
        self.activity_reads = 0

    @property
    def foreground_activity_name(self):
        self.activity_reads += 1
        return f"example/Activity{self.activity_reads}"

    @property
    def logical_screen_size(self):
        return (1080, 2400)

    def reset(self, go_home=False):
        return self.initial_state

    def execute_action(self, action):
        self.executed_actions.append(action)

    def get_state(self, wait_to_stabilize=False):
        self.get_state_calls.append(wait_to_stabilize)
        return self.next_state


class FakeTask:
    name = "fake_task"
    goal = "Tap the button"
    start_on_home_screen = True
    app_names = ("settings",)

    def __init__(self, complexity=1):
        self.complexity = complexity

    def initialize_task(self, env):
        pass

    def is_successful(self, env):
        return bool(env.executed_actions)

    def tear_down(self, env):
        pass


class SuccessAfterTwoActionsTask(FakeTask):
    def is_successful(self, env):
        return len(env.executed_actions) >= 2


class ScriptedModel:
    def __init__(self, actor_outputs):
        self.actor_outputs = iter(actor_outputs)

    def generate(self, *, tools=None, **kwargs):
        if tools is not None:
            return SimpleNamespace(
                text="plan",
                parsed_json=[
                    {"task": "Goal: Tap the button.", "agent": "CodeActAgent"}
                ],
                input_tokens=0,
                output_tokens=0,
            )
        return SimpleNamespace(
            text=next(self.actor_outputs),
            parsed_json=None,
            input_tokens=0,
            output_tokens=0,
        )


def test_capture_reads_activity_from_env(tmp_path):
    class EnvWithActivityAccess:
        logical_screen_size = (100, 200)

        @property
        def foreground_activity_name(self):
            return "example/EnvActivity"

    record = AndroidWorldObservationStore(tmp_path).capture(
        _state(_element()),
        EnvWithActivityAccess(),
        "task",
        0,
    )

    assert record.foreground_activity == "example/EnvActivity"


def test_index_tap_binds_to_observed_bbox_and_refreshes_state_once(
    tmp_path, monkeypatch
):
    initial_state = _state(_element((10, 20, 30, 60)))
    next_state = _state(_element((100, 200, 300, 600)))
    env = FakeEnv(initial_state, next_state)
    sleep_calls = []
    monkeypatch.setattr(
        "dms.agent.time.sleep",
        lambda seconds: sleep_calls.append(seconds),
    )
    agent = PALiteAgent(
        model=ScriptedModel(["tap(index=0)", "complete(success=True)"]),
        run_dir=tmp_path,
        post_action_wait_seconds=1.0,
    )

    result = agent.run_task(env=env, task=SuccessAfterTwoActionsTask(), task_id="task")

    assert sleep_calls == [1.0]
    assert env.get_state_calls == [False]
    assert env.activity_reads == 2
    assert env.executed_actions == [
        json_action.JSONAction(action_type=json_action.CLICK, x=20, y=40)
    ]
    assert result.trajectory[0]["action"] == {"type": "tap", "index": 0}
    assert result.trajectory[0]["executed_action"] == {
        "type": "tap",
        "x": 20,
        "y": 40,
    }
    assert result.trajectory[1]["observation"]["foreground_activity"] == "example/Activity2"
    assert result.trajectory[1]["observation"]["step_id"] == 1
    refreshed_elements = json.loads(
        Path(
            result.trajectory[1]["observation"]["ui_elements_path"]
        ).read_text(encoding="utf-8")
    )
    assert refreshed_elements[0]["bounds"] == [100, 200, 300, 600]


def test_post_action_wait_defaults_to_three_seconds(tmp_path):
    agent = PALiteAgent(model=object(), run_dir=tmp_path)

    assert agent.post_action_wait_seconds == 3.0


def test_history_for_prompt_keeps_full_history_by_default():
    trajectory = [
        {
            "step": idx,
            "subtask": f"task-{idx}",
            "action": {"type": "tap", "index": idx},
            "result": "executed",
            "error": None,
        }
        for idx in range(10)
    ]

    history = _history_for_prompt(trajectory)

    assert len(history) == 10
    assert history[0]["step"] == 0
    assert history[-1]["step"] == 9


def test_task_app_scope_prefers_explicit_task_param_app_name():
    task = SimpleNamespace(
        app_names=("camera", "clock", "contacts", "settings"),
        params={"app_name": "settings"},
    )

    assert _task_app_scope(task) == ["settings"]


def test_normalize_start_app_action_prefers_single_task_app_scope():
    action = _normalize_start_app_action(
        action={"type": "start_app", "package": "com.example.audio_recorder"},
        subtask_goal="Open the Audio Recorder app.",
        task_app_names=["audio recorder"],
    )

    assert action == {"type": "start_app", "package": "audio recorder"}


def test_normalize_start_app_action_maps_file_manager_alias():
    action = _normalize_start_app_action(
        action={"type": "start_app", "package": "file_manager"},
        subtask_goal="Open the file manager app.",
        task_app_names=["chrome"],
    )

    assert action == {"type": "start_app", "package": "files"}


def test_normalize_start_app_action_maps_simple_calendar_package_alias():
    action = _normalize_start_app_action(
        action={"type": "start_app", "package": "com.simple.calendar.pro"},
        subtask_goal="Open the Simple Calendar Pro app.",
        task_app_names=["simple calendar pro"],
    )

    assert action == {"type": "start_app", "package": "simple calendar pro"}


def test_normalize_start_app_action_handles_open_goal_without_app_suffix():
    action = _normalize_start_app_action(
        action={"type": "start_app", "package": "simple.calendar.pro"},
        subtask_goal="Open Simple Calendar Pro",
        task_app_names=["simple calendar pro"],
    )

    assert action == {"type": "start_app", "package": "simple calendar pro"}


def test_complete_reason_falls_back_to_answer_for_information_goal():
    action = _fallback_answer_from_complete(
        action={"type": "complete", "success": True, "reason": "42"},
        task_goal="How many tasks do I have due October 19 2023 in Tasks app? Express your answer as a single integer.",
    )

    assert action == {"type": "answer", "text": "42"}


def test_open_app_subtask_falls_back_to_start_app_on_invalid_tap(
    tmp_path, monkeypatch
):
    env = FakeEnv(_state(_element(None)))
    monkeypatch.setattr("dms.agent.time.sleep", lambda seconds: None)

    class OpenAppModel(ScriptedModel):
        def generate(self, *, tools=None, **kwargs):
            if tools is not None:
                return SimpleNamespace(
                    text="plan",
                    parsed_json=[
                        {"task": "Goal: Open the settings app.", "agent": "CodeActAgent"}
                    ],
                    input_tokens=0,
                    output_tokens=0,
                )
            return SimpleNamespace(
                text="tap(index=0)",
                parsed_json=None,
                input_tokens=0,
                output_tokens=0,
            )

    class OpenAppTask(FakeTask):
        goal = "Open the settings app."

    agent = PALiteAgent(
        model=OpenAppModel([]),
        run_dir=tmp_path,
        post_action_wait_seconds=0,
    )

    result = agent.run_task(env=env, task=OpenAppTask(complexity=0.1), task_id="task")

    assert env.executed_actions == [
        json_action.JSONAction(action_type=json_action.OPEN_APP, app_name="settings")
    ]
    assert result.trajectory[0]["executed_action"] == {
        "type": "start_app",
        "package": "settings",
    }


def test_clear_directory_ignores_missing_files_for_empty_directory(monkeypatch):
    calls = []

    def fake_check_directory_exists(directory_path, env):
        calls.append(("exists", directory_path, env))
        return True

    def fake_issue_generic_request(command, env):
        calls.append(("cmd", command, env))
        class Response:
            class generic:
                output = b""
        return Response()

    def fake_check_ok(response, message):
        calls.append(("check_ok", message))

    monkeypatch.setattr(file_utils, "check_directory_exists", fake_check_directory_exists)
    monkeypatch.setattr(file_utils.adb_utils, "issue_generic_request", fake_issue_generic_request)
    monkeypatch.setattr(file_utils.adb_utils, "check_ok", fake_check_ok)

    file_utils.clear_directory("/data/data/example.app", env="ENV")

    assert calls == [
        ("exists", "/data/data/example.app", "ENV"),
        ("cmd", ["shell", "ls", "-1", "/data/data/example.app"], "ENV"),
    ]


@pytest.mark.parametrize(
    ("element", "expected_error"),
    [
        (_element(), "tap index 2 is out of range for 1 UI elements."),
        (_element(None), "tap index 0 has no bounding box."),
    ],
)
def test_invalid_index_tap_fails_without_executing(
    tmp_path, element, expected_error
):
    index = 2 if expected_error.startswith("tap index 2") else 0
    env = FakeEnv(_state(element))
    agent = PALiteAgent(
        model=ScriptedModel([f"tap(index={index})"]),
        run_dir=tmp_path,
        post_action_wait_seconds=0,
    )

    result = agent.run_task(
        env=env,
        task=FakeTask(complexity=0.1),
        task_id="task",
    )

    assert env.executed_actions == []
    assert env.get_state_calls == []
    assert result.trajectory[0]["result"] == "invalid_action"
    assert result.trajectory[0]["error"] == expected_error
    assert result.trajectory[0]["executed_action"] is None


def test_invalid_actor_output_refreshes_observation_for_next_step(
    tmp_path,
    monkeypatch,
):
    initial_state = _state(_element((10, 20, 30, 60)))
    next_state = _state(_element((100, 200, 300, 600)))
    env = FakeEnv(initial_state, next_state)
    monkeypatch.setattr("dms.agent.time.sleep", lambda seconds: None)
    agent = PALiteAgent(
        model=ScriptedModel(["# wait for UI", "tap(index=0)"]),
        run_dir=tmp_path,
        post_action_wait_seconds=0,
    )

    result = agent.run_task(
        env=env,
        task=SuccessAfterTwoActionsTask(complexity=0.2),
        task_id="task",
    )

    assert env.get_state_calls == [False, False]
    assert result.trajectory[0]["result"] == "invalid_action"
    assert result.trajectory[1]["observation"]["step_id"] == 1
    assert result.trajectory[1]["executed_action"] == {
        "type": "tap",
        "x": 200,
        "y": 400,
    }


def test_index_tap_allows_visible_non_clickable_elements(tmp_path, monkeypatch):
    env = FakeEnv(_state(_non_interactable_element((100, 200, 300, 600))))
    monkeypatch.setattr("dms.agent.time.sleep", lambda seconds: None)
    agent = PALiteAgent(
        model=ScriptedModel(["tap(index=0)"]),
        run_dir=tmp_path,
        post_action_wait_seconds=0,
    )

    result = agent.run_task(
        env=env,
        task=FakeTask(complexity=0.1),
        task_id="task",
    )

    assert env.executed_actions == [
        json_action.JSONAction(action_type=json_action.CLICK, x=200, y=400)
    ]
    assert result.trajectory[0]["result"] == "executed"
    assert result.trajectory[0]["executed_action"] == {
        "type": "tap",
        "x": 200,
        "y": 400,
    }


def test_engineering_variant_separates_control_turns_from_env_steps(tmp_path):
    env = FakeEnv(_state(_element()))
    agent = PALiteAgent(
        model=ScriptedModel(["# invalid", "# still invalid"]),
        run_dir=tmp_path,
        post_action_wait_seconds=0,
        engineering_optimization=True,
        planner_max_cycles=2,
        control_turn_limit=4,
    )

    result = agent.run_task(
        env=env,
        task=SuccessAfterTwoActionsTask(complexity=0.1),
        task_id="task",
    )

    assert result.steps == 0
    assert result.event_count == 2
    assert result.control_turns == 4
    assert result.control_turn_limit_reached is True
    assert env.executed_actions == []


def test_engineering_variant_counts_only_execute_action_against_budget(
    tmp_path, monkeypatch
):
    env = FakeEnv(_state(_element()))
    monkeypatch.setattr("dms.agent.time.sleep", lambda seconds: None)
    agent = PALiteAgent(
        model=ScriptedModel(["tap(index=0)"]),
        run_dir=tmp_path,
        post_action_wait_seconds=0,
        engineering_optimization=True,
        planner_max_cycles=2,
        control_turn_limit=4,
    )

    result = agent.run_task(
        env=env,
        task=FakeTask(complexity=0.1),
        task_id="task",
    )

    assert result.steps == 1
    assert result.event_count == 1
    assert result.control_turns == 1
    assert result.success is True
