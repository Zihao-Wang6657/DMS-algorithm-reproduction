from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


WORKING_SPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKING_SPACE / "src"))
sys.path.insert(0, str(WORKING_SPACE / "third_party" / "android_world"))

from dms.agent import DMSAgent, DMSMemoryAdapter, PALiteAgent  # noqa: E402
from dms.runner import _build_agent  # noqa: E402


class FakeTask:
    name = "fake_task"
    goal = "Remember the value"
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


class FakeEnv:
    def __init__(self):
        self.executed_actions = []
        self.state = SimpleNamespace(
            pixels=np.zeros((4, 4, 3), dtype=np.uint8),
            ui_elements=[],
        )

    @property
    def foreground_activity_name(self):
        return "settings/SettingsActivity"

    @property
    def logical_screen_size(self):
        return (1080, 2400)

    def reset(self, go_home=False):
        return self.state

    def execute_action(self, action):
        self.executed_actions.append(action)

    def get_state(self, wait_to_stabilize=False):
        return self.state


class MemoryBackend:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.remember_calls = []
        self.record_calls = []
        self.finalize_calls = []
        self.entry_count = 2

    @property
    def size(self):
        return self.entry_count + len(self.remember_calls)

    def planner_context(self, **kwargs):
        return {
            "hierarchical_entries": [{"goal": kwargs["goal"]}],
            "pruning_stats": {"pruned_entries": 0},
            "risk_stats": {"risk_score": 0.25},
        }

    def actor_context(self, **kwargs):
        return {
            "replay_candidates": [{"goal": kwargs["subtask"]["goal"]}],
            "mutation_fallback": {"fallback": "reuse_recent_success"},
            "risk_stats": {"risk_score": 0.25},
        }

    def remember(self, **kwargs):
        self.remember_calls.append(kwargs)
        return kwargs

    def record_step(self, **kwargs):
        self.record_calls.append(kwargs)
        return kwargs

    def finalize_task(self, **kwargs):
        self.finalize_calls.append(kwargs)
        return kwargs

    def stats(self):
        return {
            "memory_size": self.size,
            "pruning_stats": {"pruned_entries": 0},
            "risk_stats": {"risk_score": 0.25},
        }


class ScriptedModel:
    def __init__(self, actor_outputs):
        self.actor_outputs = iter(actor_outputs)
        self.prompts = []

    def generate(self, *, tools=None, prompt=None, **kwargs):
        self.prompts.append(prompt)
        if tools is not None:
            return SimpleNamespace(
                text="planner",
                parsed_json=[
                    {"task": "Precondition: None. Goal: Remember the value.", "agent": "CodeActAgent"}
                ],
                input_tokens=1,
                output_tokens=1,
            )
        return SimpleNamespace(
            text=next(self.actor_outputs),
            parsed_json=None,
            input_tokens=1,
            output_tokens=1,
        )


def test_dms_memory_adapter_renders_external_context():
    backend = MemoryBackend()
    adapter = DMSMemoryAdapter(backend)

    planner_context = adapter.planner_context(
        task=SimpleNamespace(name="task", goal="Goal"),
        task_id="task-1",
        task_history=[],
        observation={},
    )
    actor_context = adapter.actor_context(
        task=SimpleNamespace(name="task", goal="Goal"),
        task_id="task-1",
        subtask={"goal": "remember"},
        step_history=[],
        observation={},
    )

    assert "DMS planner context" in planner_context
    assert "hierarchical_entries" in planner_context
    assert "DMS actor context" in actor_context
    assert "mutation_fallback" in actor_context
    assert adapter.size == 2


def test_dms_agent_supports_remember_and_collects_memory_stats(tmp_path, monkeypatch):
    env = FakeEnv()
    memory = MemoryBackend()
    monkeypatch.setattr("dms.agent.time.sleep", lambda seconds: None)
    agent = DMSAgent(
        model=ScriptedModel(["remember('persist this')", "tap(x=10, y=20)"]),
        run_dir=tmp_path,
        dms_memory=memory,
        post_action_wait_seconds=0,
    )

    result = agent.run_task(env=env, task=FakeTask(complexity=0.2), task_id="task")

    assert memory.remember_calls
    assert memory.record_calls
    assert memory.finalize_calls
    assert result.memory_size_after == memory.size
    assert result.memory_stats["backend"] == type(memory).__name__
    assert result.trajectory[0]["result"] == "remembered"
    assert result.trajectory[0]["executed_action"] == {
        "type": "remember",
        "information": "persist this",
    }


def test_baseline_agent_keeps_remember_disabled(tmp_path):
    agent = PALiteAgent(model=object(), run_dir=tmp_path)

    assert agent._allow_remember() is False


def test_runner_builds_dms_agent_from_config(tmp_path):
    config = {
        "pa_lite": {"planner_max_subtasks": 5, "actor_local_step_guard": 8},
        "dms": {
            "memory_backend": "dms.memory_backends:HierarchicalMemoryStub",
            "memory_kwargs": {"entry_limit": 4},
        },
    }
    args = argparse.Namespace(method="dms_hierarchical_memory")

    agent, memory = _build_agent(args=args, config=config, model=object(), run_dir=tmp_path)

    assert isinstance(agent, DMSAgent)
    assert memory.entry_limit == 4
    assert memory.path.endswith("dms_memory.jsonl")


def test_runner_rejects_dms_without_memory_backend(tmp_path):
    config = {"pa_lite": {"planner_max_subtasks": 5, "actor_local_step_guard": 8}, "dms": {}}
    args = argparse.Namespace(method="dms_hierarchical_memory")

    with pytest.raises(ValueError, match="memory_backend"):
        _build_agent(args=args, config=config, model=object(), run_dir=tmp_path)
