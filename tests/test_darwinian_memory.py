from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest


WORKING_SPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKING_SPACE / "src"))

from dms.darwinian_memory import DMSConfig, DarwinianMemorySystem  # noqa: E402


class FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = {
            key: np.asarray(value, dtype=np.float32) for key, value in mapping.items()
        }

    def encode(self, texts):
        return np.asarray([self.mapping[text] for text in texts], dtype=np.float32)


def _backend(tmp_path: Path, embedder: FakeEmbedder) -> DarwinianMemorySystem:
    config = DMSConfig(
        embedding_model_path="unused",
        min_capacity=3,
        max_capacity=5,
        capacity_step=2,
    )
    return DarwinianMemorySystem(tmp_path, config, embedder=embedder)


def _trajectory(n: int) -> list[dict[str, object]]:
    result = []
    for idx in range(n):
        result.append(
            {
                "step": idx,
                "action": {"type": "tap", "index": idx},
                "executed_action": {"type": "tap", "x": 10 + idx, "y": 20 + idx},
                "result": "executed" if idx < n - 1 else "subtask_complete",
            }
        )
    return result


def test_survival_value_matches_paper_formula(tmp_path):
    embedder = FakeEmbedder({"home": [1.0, 0.0], "open settings": [0.0, 1.0]})
    memory = _backend(tmp_path, embedder)
    memory_id = memory.create_memory(
        subtask={"precondition": "home", "goal": "open settings"},
        trajectory=_trajectory(2),
        task_id="task",
        task_name="task",
        task_goal="goal",
        app_names=["settings"],
    )
    assert memory_id is not None
    entry = memory.entries[memory_id]
    entry.reuse_count = 4
    entry.verification_failures = 2
    memory.logical_time = 50
    entry.last_retrieved_logical_time = 40
    memory._refresh_entry(entry)

    expected_utility = math.log1p(4) + memory.config.novelty_bonus
    expected_half_life = memory.config.base_retention + (
        memory.config.longevity_coefficient * math.log1p(4)
    )
    expected_decay = 1.0 / (
        1.0
        + math.exp(memory.config.decay_steepness * (10.0 - expected_half_life))
    )
    expected_reliability = 1.0 / (
        1.0 + memory.config.penalty_coefficient * 2.0
    )

    assert entry.survival_value == pytest.approx(
        expected_utility * expected_decay * expected_reliability
    )


def test_survival_value_penalizes_execution_noise_and_failures(tmp_path):
    embedder = FakeEmbedder({"home": [1.0, 0.0], "open settings": [0.0, 1.0]})
    memory = _backend(tmp_path, embedder)
    memory_id = memory.create_memory(
        subtask={"precondition": "home", "goal": "open settings"},
        trajectory=_trajectory(4),
        task_id="task",
        task_name="task",
        task_goal="goal",
        app_names=["settings"],
    )
    assert memory_id is not None
    entry = memory.entries[memory_id]
    baseline = entry.survival_value

    entry.total_actions = 4
    entry.invalid_action_count = 1
    entry.execution_error_count = 1
    entry.failure_count = 1
    entry.success_count = 1
    memory._refresh_entry(entry)

    assert entry.survival_value < baseline


def test_dual_factor_retrieval_requires_precondition_and_goal_alignment(tmp_path):
    embedder = FakeEmbedder(
        {
            "home": [1.0, 0.0, 0.0],
            "settings": [0.0, 1.0, 0.0],
            "wifi": [0.0, 0.0, 1.0],
            "wrong-pre": [0.0, 1.0, 0.0],
            "wrong-goal": [0.0, 1.0, 0.0],
        }
    )
    memory = _backend(tmp_path, embedder)
    good_id = memory.create_memory(
        subtask={"precondition": "home", "goal": "wifi"},
        trajectory=_trajectory(2),
        task_id="good",
        task_name="good",
        task_goal="good",
        app_names=["settings"],
    )
    bad_id = memory.create_memory(
        subtask={"precondition": "wrong-pre", "goal": "wrong-goal"},
        trajectory=_trajectory(2),
        task_id="bad",
        task_name="bad",
        task_goal="bad",
        app_names=["settings"],
    )
    assert good_id and bad_id

    decision = memory.retrieve(
        subtask={"precondition": "home", "goal": "wifi"},
        task_app_names=["settings"],
    )

    assert decision.mode == "replay"
    assert decision.selected_memory_id == good_id
    assert decision.candidates[0].score > decision.candidates[-1].score


def test_dual_factor_retrieval_rejects_low_goal_similarity_false_positives(tmp_path):
    embedder = FakeEmbedder(
        {
            "home": [1.0, 0.0],
            "open settings": [0.0, 1.0],
            "open storage manager": [0.0, 0.40],
        }
    )
    memory = _backend(tmp_path, embedder)
    memory_id = memory.create_memory(
        subtask={"precondition": "home", "goal": "open settings"},
        trajectory=_trajectory(2),
        task_id="task",
        task_name="task",
        task_goal="goal",
        app_names=["settings"],
    )
    assert memory_id is not None

    decision = memory.retrieve(
        subtask={"precondition": "home", "goal": "open storage manager"},
        task_app_names=["settings"],
    )

    assert decision.mode == "miss"
    assert "retrieval threshold" in decision.reason


def test_retrieve_supports_legacy_app_names_alias_for_scope_filter(tmp_path):
    embedder = FakeEmbedder({"home": [1.0, 0.0], "open settings": [0.0, 1.0]})
    memory = _backend(tmp_path, embedder)
    memory_id = memory.create_memory(
        subtask={"precondition": "home", "goal": "open settings"},
        trajectory=_trajectory(2),
        task_id="task",
        task_name="task",
        task_goal="goal",
        app_names=["settings"],
    )
    assert memory_id is not None

    decision = memory.retrieve(
        subtask={"precondition": "home", "goal": "open settings"},
        app_names=["clock"],
    )

    assert decision.mode == "miss"
    assert "app-scope filter" in decision.reason


def test_mutation_replacement_only_accepts_shorter_successful_trajectory(tmp_path):
    embedder = FakeEmbedder({"home": [1.0, 0.0], "settings": [0.0, 1.0]})
    memory = _backend(tmp_path, embedder)
    memory_id = memory.create_memory(
        subtask={"precondition": "home", "goal": "settings"},
        trajectory=_trajectory(3),
        task_id="task",
        task_name="task",
        task_goal="goal",
        app_names=["settings"],
    )
    assert memory_id is not None

    assert memory.replace_with_mutation(
        memory_id=memory_id,
        subtask={"precondition": "home", "goal": "settings"},
        trajectory=_trajectory(2),
        task_id="task",
        task_name="task",
        task_goal="goal",
        app_names=["settings"],
    )
    assert memory.entries[memory_id].steps == 2
    assert memory.entries[memory_id].version == 2

    assert not memory.replace_with_mutation(
        memory_id=memory_id,
        subtask={"precondition": "home", "goal": "settings"},
        trajectory=_trajectory(4),
        task_id="task",
        task_name="task",
        task_goal="goal",
        app_names=["settings"],
    )
    assert memory.entries[memory_id].steps == 2


def test_dynamic_risk_threshold_responds_to_global_task_feedback(tmp_path):
    embedder = FakeEmbedder({"home": [1.0, 0.0], "settings": [0.0, 1.0]})
    memory = _backend(tmp_path, embedder)

    baseline = memory._dynamic_risk_threshold()
    memory.global_task_successes = 20
    memory.global_task_failures = 0
    healthier = memory._dynamic_risk_threshold()
    memory.global_task_successes = 0
    memory.global_task_failures = 20
    riskier = memory._dynamic_risk_threshold()

    assert healthier > baseline
    assert riskier < baseline


def test_pruning_and_capacity_expansion_follow_elbow_logic(tmp_path, monkeypatch):
    embedder = FakeEmbedder(
        {
            "p0": [1.0, 0.0],
            "g0": [0.0, 1.0],
            "p1": [1.0, 0.0],
            "g1": [0.0, 1.0],
            "p2": [1.0, 0.0],
            "g2": [0.0, 1.0],
            "p3": [1.0, 0.0],
            "g3": [0.0, 1.0],
        }
    )
    memory = _backend(tmp_path, embedder)
    ids = []
    for idx in range(4):
        memory_id = memory.create_memory(
            subtask={"precondition": f"p{idx}", "goal": f"g{idx}"},
            trajectory=_trajectory(2),
            task_id=str(idx),
            task_name=str(idx),
            task_goal=str(idx),
            app_names=["settings"],
        )
        ids.append(memory_id)
    assert all(ids)

    values = [0.95, 0.90, 0.30, 0.10]
    for memory_id, value in zip(ids, values):
        memory.entries[memory_id].survival_value = value
    monkeypatch.setattr(memory, "_refresh_all_scores", lambda: None)
    memory.current_capacity = 3
    memory._prune_if_needed()

    assert memory.size <= 3

    expanded = _backend(tmp_path / "expand", embedder)
    expand_ids = []
    for idx in range(3):
        memory_id = expanded.create_memory(
            subtask={"precondition": f"p{idx}", "goal": f"g{idx}"},
            trajectory=_trajectory(2),
            task_id=str(idx),
            task_name=str(idx),
            task_goal=str(idx),
            app_names=["settings"],
        )
        expand_ids.append(memory_id)
    values = [0.82, 0.81, 0.80]
    for memory_id, value in zip(expand_ids, values):
        expanded.entries[memory_id].survival_value = value
    monkeypatch.setattr(expanded, "_refresh_all_scores", lambda: None)
    expanded.current_capacity = 3
    expanded._prune_if_needed()

    assert expanded.current_capacity == 5
