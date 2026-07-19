from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


WORKING_SPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKING_SPACE / "src"))

from dms.hierarchical_memory import HierarchicalMemory  # noqa: E402


class FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def embed(self, texts):
        return [self.mapping[text] for text in texts]


def test_bayesian_risk_and_survival_value_math():
    memory = HierarchicalMemory(
        embedder=FakeEmbedder({"memory": [1.0, 0.0]}),
        autosave=False,
        prior_success=2.0,
        prior_failure=3.0,
        recency_half_life_seconds=10.0,
        access_temperature=4.0,
    )
    entry = memory.ingest(
        "memory",
        importance=0.8,
        success_count=3.0,
        failure_count=1.0,
        cumulative_reward=2.0,
        access_count=6,
        created_at="2026-06-01T00:00:00+00:00",
    ).entry

    risk = memory.estimate_risk(entry)
    assert risk.alpha == pytest.approx(4.0)
    assert risk.beta == pytest.approx(5.0)
    assert risk.mean == pytest.approx(4.0 / 9.0)
    assert risk.variance == pytest.approx((4.0 * 5.0) / ((9.0**2) * 10.0))
    assert risk.upper_bound == pytest.approx(
        min(1.0, risk.mean + 1.2815515655446004 * math.sqrt(risk.variance))
    )

    survival = memory.survival_value(entry, now="2026-06-01T00:00:00+00:00")
    expected_usage = 1.0 - math.exp(-6.0 / 4.0)
    expected_utility = 0.75
    expected_score = 0.4 * expected_utility + 0.2 * expected_usage + 0.2 * 1.0 + 0.2 * 0.8

    assert survival.utility_signal == pytest.approx(expected_utility)
    assert survival.usage_signal == pytest.approx(expected_usage)
    assert survival.recency_signal == pytest.approx(1.0)
    assert survival.importance_signal == pytest.approx(0.8)
    assert survival.score == pytest.approx(expected_score)


def test_retrieval_combines_semantic_context_survival_and_risk():
    memory = HierarchicalMemory(
        embedder=FakeEmbedder(
            {
                "query": [1.0, 0.0],
                "match": [1.0, 0.0],
                "far": [0.0, 1.0],
            }
        ),
        autosave=False,
        semantic_weight=0.55,
        contextual_weight=0.25,
        survival_weight=0.15,
        risk_penalty_weight=0.05,
    )
    match = memory.ingest(
        "match",
        hierarchy_path=["home", "settings"],
        tags=["wifi"],
        level=1,
        importance=0.9,
        success_count=2.0,
        cumulative_reward=2.0,
        access_count=4,
        created_at="2026-06-01T00:00:00+00:00",
    ).entry
    memory.ingest(
        "far",
        hierarchy_path=["other"],
        tags=["music"],
        level=3,
        importance=0.1,
        failure_count=4.0,
        cumulative_reward=-4.0,
        access_count=0,
        created_at="2026-06-01T00:00:00+00:00",
    )

    results = memory.retrieve(
        "query",
        top_k=1,
        hierarchy_path=["home", "settings"],
        tags=["wifi"],
        level=1,
        now="2026-06-01T00:00:00+00:00",
        track_access=False,
    )

    assert len(results) == 1
    result = results[0]
    assert result.entry.memory_id == match.memory_id
    assert result.semantic_score == pytest.approx(1.0)
    assert result.contextual_score == pytest.approx(1.0)
    assert result.score == pytest.approx(
        memory.semantic_weight * result.semantic_score
        + memory.contextual_weight * result.contextual_score
        + memory.survival_weight * result.survival_score
        - memory.risk_penalty_weight * result.risk_score
    )


def test_ingest_replaces_the_weakest_entry_in_place(monkeypatch):
    memory = HierarchicalMemory(
        capacity=2,
        replacement_margin=0.01,
        embedder=FakeEmbedder(
            {
                "weak-one": [1.0, 0.0],
                "weak-two": [1.0, 0.0],
                "strong-new": [1.0, 0.0],
            }
        ),
        autosave=False,
    )
    scores = {
        "weak-one": 0.1,
        "weak-two": 0.3,
        "strong-new": 0.9,
    }
    monkeypatch.setattr(
        memory,
        "retention_score",
        lambda entry, now=None: scores[entry.text],
    )

    first = memory.ingest("weak-one").entry
    second = memory.ingest("weak-two").entry
    decision = memory.ingest("strong-new")

    assert decision.action == "replaced"
    assert decision.replaced_entry_id == first.memory_id
    assert decision.entry is not None
    assert decision.entry.slot == first.slot
    assert decision.entry.replaced_from == first.memory_id
    assert set(memory.memory_ids()) == {second.memory_id, decision.entry.memory_id}


def test_prune_uses_elbow_cutoff_when_forced(monkeypatch):
    memory = HierarchicalMemory(
        capacity=6,
        min_capacity=1,
        embedder=FakeEmbedder({f"item-{index}": [1.0, 0.0] for index in range(6)}),
        autosave=False,
    )
    created_ids = [memory.ingest(f"item-{index}").entry.memory_id for index in range(6)]
    scores = [0.96, 0.91, 0.86, 0.35, 0.33, 0.31]
    score_map = dict(zip(created_ids, scores))
    monkeypatch.setattr(
        memory,
        "retention_score",
        lambda entry, now=None: score_map[entry.memory_id],
    )

    decision = memory.prune(force=True)

    assert decision.cutoff_index == 4
    assert decision.kept_entry_ids == created_ids[:4]
    assert decision.pruned_entry_ids == created_ids[4:]
    assert memory.memory_ids() == created_ids[:4]


def test_prune_expands_capacity_when_scores_are_flat(monkeypatch):
    memory = HierarchicalMemory(
        capacity=4,
        min_capacity=1,
        allow_capacity_expansion=True,
        capacity_expansion_factor=1.5,
        max_capacity=8,
        embedder=FakeEmbedder({f"flat-{index}": [1.0, 0.0] for index in range(4)}),
        autosave=False,
    )
    created_ids = [memory.ingest(f"flat-{index}").entry.memory_id for index in range(4)]
    monkeypatch.setattr(
        memory,
        "retention_score",
        lambda entry, now=None: 0.5,
    )

    decision = memory.prune()

    assert decision.expanded is True
    assert decision.pruned_entry_ids == []
    assert decision.kept_entry_ids == created_ids
    assert memory.capacity == 6
