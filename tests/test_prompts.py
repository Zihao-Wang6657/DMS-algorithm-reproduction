from __future__ import annotations

import sys
from pathlib import Path


WORKING_SPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKING_SPACE / "src"))

from dms.prompts import actor_prompt, planner_prompt  # noqa: E402


def _actor_prompt() -> str:
    return actor_prompt(
        global_goal="Open Settings.",
        task_app_names=["settings"],
        subtask={"precondition": "Home screen", "goal": "Open Settings."},
        foreground_activity="Launcher",
        compact_ui=[],
        step_history=[],
    )


def test_actor_prompt_requires_one_executable_tool_call_in_python_fence():
    prompt = _actor_prompt()

    assert "After completing your reasoning" in prompt
    assert "Exactly one Python fenced code block (```python ... ```)." in prompt
    assert (
        "The code block must contain exactly one executable tool call and no "
        "other executable code."
    ) in prompt
    assert "Python comments are allowed" in prompt
    assert "must not contain only comments" in prompt


def test_actor_prompt_disallows_batching_multiple_tool_calls():
    prompt = _actor_prompt()

    assert "Never batch or chain multiple tool calls" in prompt
    assert "Only batch independent actions" not in prompt
    assert "**BATCHING:**" not in prompt


def test_actor_prompt_preserves_index_constraints():
    prompt = _actor_prompt()

    assert "**INDEX BINDING:**" in prompt
    assert "**NO VISUAL INDEXING:**" in prompt
    assert "**INDEX DRIFT:**" in prompt
    assert "Never derive an index from screenshot position" in prompt
    assert "Never reuse an index from chat history" in prompt


def test_actor_prompt_prefers_start_app_by_androidworld_app_name():
    prompt = _actor_prompt()

    assert '**APP LAUNCHING RULE:**' in prompt
    assert 'prefer `start_app("<app name>")`' in prompt
    assert '"clock"' in prompt
    assert "or a full Android package name" in prompt
    assert "do not substitute a different visible app" in prompt


def test_planner_prompt_describes_direct_app_launch_capability():
    prompt = planner_prompt(
        goal="Pause the stopwatch.",
        task_app_names=["clock"],
        foreground_activity="Launcher",
        compact_ui=[],
        task_history=[],
        memory_context=None,
        max_subtasks=5,
    )

    assert "CodeActAgent: writes and executes Python tool calls" in prompt
    assert "directly launch an installed app by its common app name" in prompt
    assert "even when the app is not visible on the current screen" in prompt


def test_dms_planner_prompt_mentions_dms_memory_context():
    prompt = planner_prompt(
        goal="Open settings",
        task_app_names=["settings"],
        foreground_activity="Launcher",
        compact_ui=[],
        task_history=[],
        memory_context="hierarchical memory",
        max_subtasks=5,
        dms_mode=True,
        memory_context_title="DMS Memory Context",
    )

    assert "DMS Memory Context" in prompt
    assert "hierarchical memory" in prompt
    assert "retrieved hierarchical memory" in prompt.lower()
    assert "pruning/risk diagnostics" in prompt.lower()


def test_prompts_include_task_app_scope():
    actor = _actor_prompt()
    planner = planner_prompt(
        goal="Open settings",
        task_app_names=["settings"],
        foreground_activity="Launcher",
        compact_ui=[],
        task_history=[],
        memory_context=None,
        max_subtasks=5,
    )

    assert "**task_app_scope**: ['settings']" in actor
    assert "The AndroidWorld task is scoped to these app(s):" in planner
    assert "['settings']" in planner


def test_dms_actor_prompt_enables_remember_and_memory_context():
    prompt = actor_prompt(
        global_goal="Open settings",
        task_app_names=["settings"],
        subtask={"precondition": "Home", "goal": "Open Settings"},
        foreground_activity="Launcher",
        compact_ui=[],
        step_history=[],
        memory_context="replay guidance",
        allow_remember=True,
    )

    assert "**memory mode**: DMS" in prompt
    assert "remember(information)" in prompt
    assert "enabled only for this DMS run" in prompt
    assert "replay guidance" in prompt
