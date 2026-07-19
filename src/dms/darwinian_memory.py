from __future__ import annotations

import json
import math
import random
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence
import re

import numpy as np

from dms.io_utils import append_jsonl, now_iso, write_json


class TextEmbedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Returns L2-normalized embeddings for the given texts."""


class SentenceTransformerEmbedder:
    """Offline sentence embedder backed by a local SentenceTransformer model."""

    def __init__(self, model_path: str | Path) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_path = str(Path(model_path).resolve())
        self.model = SentenceTransformer(self.model_path, device="cpu")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return np.asarray(
            self.model.encode(
                list(texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


@dataclass(frozen=True)
class DMSConfig:
    embedding_model_path: str
    epsilon: float = 0.1
    novelty_bonus: float = 1.0
    base_retention: float = 30.0
    longevity_coefficient: float = 15.0
    decay_steepness: float = 0.5
    penalty_coefficient: float = 1.0
    threshold_sensitivity: float = 0.3
    verification_limit: int = 3
    risk_prior_failures: float = 1.0
    risk_prior_successes: float = 1.0
    risk_threshold_base: float = 0.5
    min_capacity: int = 24
    max_capacity: int = 96
    capacity_step: int = 8
    top_k: int = 3
    min_retrieval_score: float = 0.25
    min_component_similarity: float = 0.35
    min_goal_similarity: float = 0.55
    planner_context_entries: int = 6
    random_seed: int = 7

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "DMSConfig":
        return cls(
            embedding_model_path=str(mapping["embedding_model_path"]),
            epsilon=float(mapping.get("epsilon", 0.1)),
            novelty_bonus=float(mapping.get("novelty_bonus", 1.0)),
            base_retention=float(mapping.get("base_retention", 30.0)),
            longevity_coefficient=float(mapping.get("longevity_coefficient", 15.0)),
            decay_steepness=float(mapping.get("decay_steepness", 0.5)),
            penalty_coefficient=float(mapping.get("penalty_coefficient", 1.0)),
            threshold_sensitivity=float(mapping.get("threshold_sensitivity", 0.3)),
            verification_limit=int(mapping.get("verification_limit", 3)),
            risk_prior_failures=float(mapping.get("risk_prior_failures", 1.0)),
            risk_prior_successes=float(mapping.get("risk_prior_successes", 1.0)),
            risk_threshold_base=float(mapping.get("risk_threshold_base", 0.5)),
            min_capacity=int(mapping.get("min_capacity", 24)),
            max_capacity=int(mapping.get("max_capacity", 96)),
            capacity_step=int(mapping.get("capacity_step", 8)),
            top_k=int(mapping.get("top_k", 3)),
            min_retrieval_score=float(mapping.get("min_retrieval_score", 0.25)),
            min_component_similarity=float(
                mapping.get("min_component_similarity", 0.35)
            ),
            min_goal_similarity=float(mapping.get("min_goal_similarity", 0.55)),
            planner_context_entries=int(mapping.get("planner_context_entries", 6)),
            random_seed=int(mapping.get("random_seed", 7)),
        )


@dataclass
class MemoryEntry:
    id: str
    plan: dict[str, str]
    trajectory_path: str
    precondition_embedding: list[float]
    goal_embedding: list[float]
    created_at: str
    created_logical_time: int
    last_retrieved_logical_time: int
    last_updated: str
    steps: int
    reuse_count: int
    success_count: int
    failure_count: int
    verification_failures: int
    task_id: str
    task_name: str
    task_goal: str
    app_names: list[str]
    total_actions: int
    invalid_action_count: int
    execution_error_count: int
    terminal_result: str
    survival_value: float = 0.0
    posterior_failure_mean: float = 0.0
    posterior_failure_std: float = 0.0
    risk_score: float = 0.0
    version: int = 1
    deleted: bool = False
    deleted_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalCandidate:
    memory_id: str
    score: float
    precondition_similarity: float
    goal_similarity: float
    survival_value: float
    risk_score: float
    suppressed: bool
    steps: int
    plan: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalDecision:
    mode: str
    reason: str
    selected_memory_id: str | None
    selected_trajectory: list[dict[str, Any]]
    candidates: list[RetrievalCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "selected_memory_id": self.selected_memory_id,
            "selected_trajectory_length": len(self.selected_trajectory),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def _cosine_similarity(query: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.dot(query, candidate))


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _logical_plan_text(plan: dict[str, str]) -> str:
    return (
        f"Precondition: {plan.get('precondition', 'None')} "
        f"Goal: {plan.get('goal', '')}"
    ).strip()


def _normalize_app_name(app_name: str) -> str:
    normalized = str(app_name).strip().lower()
    if "/" in normalized:
        normalized = normalized.split("/", maxsplit=1)[0]
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def build_darwinian_memory_backend(
    *,
    run_dir: str,
    embedding_model_path: str,
    workspace_root: str | None = None,
    path: str | None = None,
    **config_kwargs: Any,
) -> "DarwinianMemorySystem":
    del workspace_root, path
    config = DMSConfig.from_mapping(
        {
            "embedding_model_path": embedding_model_path,
            **config_kwargs,
        }
    )
    return DarwinianMemorySystem(run_dir=run_dir, config=config)


class DarwinianMemorySystem:
    """Paper-aligned Darwinian memory backend.

    The implementation keeps intent-level sub-task keys separate from action-level
    trajectories, scores entries with the paper's survival value, performs
    Bayesian risk suppression, and prunes low-value memories with an elbow-based
    cutoff when the bank reaches its active capacity.
    """

    def __init__(
        self,
        run_dir: str | Path,
        config: DMSConfig,
        embedder: TextEmbedder | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.config = config
        self.embedder = embedder or SentenceTransformerEmbedder(
            config.embedding_model_path
        )
        self.random = random.Random(config.random_seed)
        self.state_path = self.run_dir / "memory_bank.json"
        self.trajectory_dir = self.run_dir / "memory_trajectories"
        self.retrieval_log_path = self.run_dir / "dms_retrievals.jsonl"
        self.pruning_log_path = self.run_dir / "dms_pruning.jsonl"
        self.mutation_log_path = self.run_dir / "dms_mutations.jsonl"
        self.events_log_path = self.run_dir / "dms_events.jsonl"
        self.summary_path = self.run_dir / "dms_summary.json"
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)

        self.entries: dict[str, MemoryEntry] = {}
        self.logical_time = 0
        self.current_capacity = self.config.min_capacity
        self.total_created = 0
        self.total_retrievals = 0
        self.total_replays = 0
        self.total_mutations = 0
        self.total_replacements = 0
        self.total_pruned = 0
        self.total_deleted_by_risk = 0
        self.global_task_successes = 0
        self.global_task_failures = 0
        self._load_state()
        self._refresh_all_scores()

    @property
    def size(self) -> int:
        return sum(0 if entry.deleted else 1 for entry in self.entries.values())

    def context(self) -> str:
        visible = self._sorted_active_entries()
        if not visible:
            return "No Darwinian memory entries are available yet."
        lines = [
            "Darwinian memory summaries. Entries are utility-ranked and dynamically pruned.",
        ]
        for index, entry in enumerate(
            visible[: self.config.planner_context_entries],
            start=1,
        ):
            lines.append(
                f"{index}. {_logical_plan_text(entry.plan)} "
                f"survival={entry.survival_value:.3f} "
                f"reuse={entry.reuse_count} "
                f"risk={entry.risk_score:.3f} "
                f"steps={entry.steps}"
            )
        return "\n".join(lines)

    def planner_context(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        visible = self._sorted_active_entries()[: self.config.planner_context_entries]
        return {
            "memory_size": self.size,
            "current_capacity": self.current_capacity,
            "dynamic_risk_threshold": self._dynamic_risk_threshold(),
            "entries": [
                {
                    "memory_id": entry.id,
                    "plan": entry.plan,
                    "survival_value": entry.survival_value,
                    "risk_score": entry.risk_score,
                    "reuse_count": entry.reuse_count,
                    "steps": entry.steps,
                }
                for entry in visible
            ],
        }

    def actor_context(self, **kwargs: Any) -> dict[str, Any]:
        subtask = kwargs.get("subtask") or {}
        return {
            "mode": "native_dms_controller",
            "subtask": subtask,
            "memory_size": self.size,
            "note": (
                "Trajectory replay, epsilon mutation, and risk suppression are "
                "handled in the DMSAgent control loop."
            ),
        }

    def remember(self, **kwargs: Any) -> dict[str, Any]:
        information = str(
            kwargs.get("information") or kwargs.get("text") or ""
        ).strip()
        payload = {
            "timestamp": now_iso(),
            "event": "remember_note",
            "information": information,
            "task_id": kwargs.get("task_id"),
            "step_id": kwargs.get("step_id"),
        }
        append_jsonl(self.events_log_path, payload)
        return payload

    def record_step(self, **kwargs: Any) -> dict[str, Any]:
        step_record = dict(kwargs.get("step_record") or kwargs.get("record") or {})
        payload = {
            "timestamp": now_iso(),
            "event": "step_record",
            "task_id": kwargs.get("task_id"),
            "subtask": step_record.get("subtask"),
            "result": step_record.get("result"),
            "action": step_record.get("action"),
            "executed_action": step_record.get("executed_action"),
        }
        append_jsonl(self.events_log_path, payload)
        return payload

    def retrieve(
        self,
        *,
        subtask: dict[str, str],
        task_app_names: list[str] | None = None,
        foreground_activity: str | None = None,
        **_: Any,
    ) -> RetrievalDecision:
        del foreground_activity
        self.logical_time += 1
        active_entries = self._sorted_active_entries()
        if not active_entries:
            decision = RetrievalDecision(
                mode="miss",
                reason="Memory bank is empty.",
                selected_memory_id=None,
                selected_trajectory=[],
            )
            self._log_retrieval(subtask=subtask, decision=decision)
            return decision

        query_pre = str(subtask.get("precondition", "None") or "None")
        query_goal = str(subtask.get("goal", "") or "")
        embeddings = self.embedder.encode([query_pre, query_goal])
        query_pre_embedding = embeddings[0]
        query_goal_embedding = embeddings[1]
        task_scope = task_app_names or _.get("app_names") or []
        if isinstance(task_scope, str):
            task_scope = [task_scope]
        task_scope = [
            _normalize_app_name(str(item))
            for item in task_scope
            if str(item).strip()
        ]

        candidates: list[RetrievalCandidate] = []
        for entry in active_entries:
            if task_scope and entry.app_names:
                entry_scope = {
                    _normalize_app_name(app_name)
                    for app_name in entry.app_names
                    if str(app_name).strip()
                }
                if not set(task_scope).intersection(entry_scope):
                    continue
            pre_sim = _cosine_similarity(
                query_pre_embedding,
                np.asarray(entry.precondition_embedding, dtype=np.float32),
            )
            goal_sim = _cosine_similarity(
                query_goal_embedding,
                np.asarray(entry.goal_embedding, dtype=np.float32),
            )
            score = pre_sim * goal_sim
            suppressed = entry.risk_score > self._dynamic_risk_threshold()
            candidates.append(
                RetrievalCandidate(
                    memory_id=entry.id,
                    score=score,
                    precondition_similarity=pre_sim,
                    goal_similarity=goal_sim,
                    survival_value=entry.survival_value,
                    risk_score=entry.risk_score,
                    suppressed=suppressed,
                    steps=entry.steps,
                    plan=dict(entry.plan),
                )
            )

        candidates.sort(
            key=lambda candidate: (
                candidate.score,
                candidate.survival_value,
                -candidate.risk_score,
            ),
            reverse=True,
        )
        candidates = candidates[: self.config.top_k]
        top = candidates[0] if candidates else None
        self.total_retrievals += 1

        if top is None:
            decision = RetrievalDecision(
                mode="miss",
                reason="No candidate survived the app-scope filter.",
                selected_memory_id=None,
                selected_trajectory=[],
                candidates=[],
            )
            self._log_retrieval(subtask=subtask, decision=decision)
            return decision

        if (
            top.score < self.config.min_retrieval_score
            or top.precondition_similarity < self.config.min_component_similarity
            or top.goal_similarity < self.config.min_goal_similarity
        ):
            decision = RetrievalDecision(
                mode="miss",
                reason=(
                    "Top dual-factor score fell below the retrieval threshold. "
                    f"score={top.score:.3f}"
                ),
                selected_memory_id=None,
                selected_trajectory=[],
                candidates=candidates,
            )
            self._log_retrieval(subtask=subtask, decision=decision)
            return decision

        if top.suppressed:
            decision = RetrievalDecision(
                mode="suppressed",
                reason=(
                    "Top memory exceeded the dynamic Bayesian risk threshold. "
                    f"risk={top.risk_score:.3f} "
                    f"threshold={self._dynamic_risk_threshold():.3f}"
                ),
                selected_memory_id=None,
                selected_trajectory=[],
                candidates=candidates,
            )
            self._log_retrieval(subtask=subtask, decision=decision)
            return decision

        if self.random.random() < self.config.epsilon:
            self.total_mutations += 1
            decision = RetrievalDecision(
                mode="mutate",
                reason=(
                    "Epsilon mutation triggered despite a safe retrieval hit. "
                    f"epsilon={self.config.epsilon:.3f}"
                ),
                selected_memory_id=top.memory_id,
                selected_trajectory=[],
                candidates=candidates,
            )
            self._log_retrieval(subtask=subtask, decision=decision)
            return decision

        self.total_replays += 1
        entry = self.entries[top.memory_id]
        entry.last_retrieved_logical_time = self.logical_time
        decision = RetrievalDecision(
            mode="replay",
            reason="Safe dual-factor retrieval hit.",
            selected_memory_id=top.memory_id,
            selected_trajectory=self.load_trajectory(top.memory_id),
            candidates=candidates,
        )
        self._persist_state()
        self._log_retrieval(subtask=subtask, decision=decision)
        return decision

    def load_trajectory(self, memory_id: str) -> list[dict[str, Any]]:
        entry = self.entries[memory_id]
        return json.loads(Path(entry.trajectory_path).read_text(encoding="utf-8"))

    def create_memory(
        self,
        *,
        subtask: dict[str, str],
        trajectory: list[dict[str, Any]],
        task_id: str,
        task_name: str,
        task_goal: str,
        app_names: list[str],
    ) -> str | None:
        if len(trajectory) <= 1:
            return None
        self.logical_time += 1
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        precondition = str(subtask.get("precondition", "None") or "None")
        goal = str(subtask.get("goal", "") or "")
        embeddings = self.embedder.encode([precondition, goal])
        total_actions = len(
            [step for step in trajectory if isinstance(step.get("action"), dict)]
        )
        invalid_action_count = sum(
            1 for step in trajectory if step.get("result") == "invalid_action"
        )
        execution_error_count = sum(
            1 for step in trajectory if step.get("result") == "execution_error"
        )
        terminal_result = str(trajectory[-1].get("result", "unknown"))

        trajectory_path = self.trajectory_dir / f"{memory_id}.json"
        write_json(trajectory_path, trajectory)
        entry = MemoryEntry(
            id=memory_id,
            plan={"precondition": precondition, "goal": goal},
            trajectory_path=str(trajectory_path.resolve()),
            precondition_embedding=embeddings[0].tolist(),
            goal_embedding=embeddings[1].tolist(),
            created_at=now_iso(),
            created_logical_time=self.logical_time,
            last_retrieved_logical_time=self.logical_time,
            last_updated=now_iso(),
            steps=len(trajectory),
            reuse_count=1,
            success_count=1,
            failure_count=0,
            verification_failures=0,
            task_id=task_id,
            task_name=task_name,
            task_goal=task_goal,
            app_names=[_normalize_app_name(app_name) for app_name in app_names],
            total_actions=total_actions,
            invalid_action_count=invalid_action_count,
            execution_error_count=execution_error_count,
            terminal_result=terminal_result,
        )
        self.entries[memory_id] = entry
        self.total_created += 1
        self._refresh_entry(entry)
        self._prune_if_needed()
        self._persist_state()
        append_jsonl(
            self.events_log_path,
            {
                "timestamp": now_iso(),
                "event": "create_memory",
                "memory_id": memory_id,
                "plan": entry.plan,
                "steps": entry.steps,
                "task_id": task_id,
            },
        )
        return memory_id

    def replace_with_mutation(
        self,
        *,
        memory_id: str,
        subtask: dict[str, str],
        trajectory: list[dict[str, Any]],
        task_id: str,
        task_name: str,
        task_goal: str,
        app_names: list[str],
    ) -> bool:
        if memory_id not in self.entries or len(trajectory) <= 1:
            return False
        incumbent = self.entries[memory_id]
        if len(trajectory) >= incumbent.steps:
            append_jsonl(
                self.mutation_log_path,
                {
                    "timestamp": now_iso(),
                    "memory_id": memory_id,
                    "accepted": False,
                    "reason": "Mutated trajectory was not more efficient.",
                    "old_steps": incumbent.steps,
                    "new_steps": len(trajectory),
                },
            )
            return False

        self.logical_time += 1
        old_steps = incumbent.steps
        precondition = str(subtask.get("precondition", "None") or "None")
        goal = str(subtask.get("goal", "") or "")
        embeddings = self.embedder.encode([precondition, goal])
        trajectory_path = Path(incumbent.trajectory_path)
        write_json(trajectory_path, trajectory)

        incumbent.plan = {"precondition": precondition, "goal": goal}
        incumbent.precondition_embedding = embeddings[0].tolist()
        incumbent.goal_embedding = embeddings[1].tolist()
        incumbent.last_updated = now_iso()
        incumbent.last_retrieved_logical_time = self.logical_time
        incumbent.steps = len(trajectory)
        incumbent.reuse_count = 1
        incumbent.success_count = 1
        incumbent.failure_count = 0
        incumbent.verification_failures = 0
        incumbent.task_id = task_id
        incumbent.task_name = task_name
        incumbent.task_goal = task_goal
        incumbent.app_names = [_normalize_app_name(app_name) for app_name in app_names]
        incumbent.total_actions = len(
            [step for step in trajectory if isinstance(step.get("action"), dict)]
        )
        incumbent.invalid_action_count = sum(
            1 for step in trajectory if step.get("result") == "invalid_action"
        )
        incumbent.execution_error_count = sum(
            1 for step in trajectory if step.get("result") == "execution_error"
        )
        incumbent.terminal_result = str(trajectory[-1].get("result", "unknown"))
        incumbent.version += 1
        incumbent.deleted = False
        incumbent.deleted_reason = None
        self.total_replacements += 1
        self._refresh_entry(incumbent)
        self._prune_if_needed()
        self._persist_state()
        append_jsonl(
            self.mutation_log_path,
            {
                "timestamp": now_iso(),
                "memory_id": memory_id,
                "accepted": True,
                "old_steps": old_steps,
                "new_steps": len(trajectory),
                "version": incumbent.version,
            },
        )
        return True

    def mark_reuse_success(self, memory_id: str) -> None:
        if memory_id not in self.entries:
            return
        self.logical_time += 1
        entry = self.entries[memory_id]
        entry.reuse_count += 1
        entry.success_count += 1
        entry.last_retrieved_logical_time = self.logical_time
        entry.last_updated = now_iso()
        self._refresh_entry(entry)
        self._persist_state()

    def mark_reuse_failure(self, memory_id: str) -> bool:
        if memory_id not in self.entries:
            return False
        self.logical_time += 1
        entry = self.entries[memory_id]
        entry.verification_failures += 1
        entry.last_updated = now_iso()
        delete_now = entry.verification_failures >= self.config.verification_limit
        if delete_now:
            entry.deleted = True
            entry.deleted_reason = "verification_limit"
            self.total_deleted_by_risk += 1
        self._refresh_entry(entry)
        self._persist_state()
        append_jsonl(
            self.events_log_path,
            {
                "timestamp": now_iso(),
                "event": "reuse_failure",
                "memory_id": memory_id,
                "verification_failures": entry.verification_failures,
                "deleted": delete_now,
            },
        )
        return delete_now

    def finalize_task(
        self,
        *,
        active_memory_ids: Sequence[str] | None = None,
        task_success: bool | None = None,
        success: bool | None = None,
        **_: Any,
    ) -> None:
        effective_success = bool(task_success if task_success is not None else success)
        unique_ids = [
            memory_id for memory_id in dict.fromkeys(active_memory_ids or [])
        ]
        if effective_success:
            self.global_task_successes += 1
        else:
            self.global_task_failures += 1
        if not effective_success:
            for memory_id in unique_ids:
                if memory_id not in self.entries:
                    continue
                entry = self.entries[memory_id]
                if entry.deleted:
                    continue
                entry.failure_count += 1
                entry.last_updated = now_iso()
                self._refresh_entry(entry)
        self._prune_if_needed()
        self._persist_state()

    def summary(self) -> dict[str, Any]:
        self._refresh_all_scores()
        summary = {
            "memory_size": self.size,
            "current_capacity": self.current_capacity,
            "total_created": self.total_created,
            "total_retrievals": self.total_retrievals,
            "total_replays": self.total_replays,
            "total_mutations": self.total_mutations,
            "total_replacements": self.total_replacements,
            "total_pruned": self.total_pruned,
            "total_deleted_by_risk": self.total_deleted_by_risk,
            "logical_time": self.logical_time,
            "global_task_successes": self.global_task_successes,
            "global_task_failures": self.global_task_failures,
            "global_failure_rate": self._global_failure_rate(),
            "dynamic_risk_threshold": self._dynamic_risk_threshold(),
        }
        write_json(self.summary_path, summary)
        return summary

    def stats(self) -> dict[str, Any]:
        return self.summary()

    def _load_state(self) -> None:
        if not self.state_path.is_file():
            return
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.logical_time = int(payload.get("logical_time", 0))
        self.current_capacity = int(
            payload.get("current_capacity", self.config.min_capacity)
        )
        self.total_created = int(payload.get("total_created", 0))
        self.total_retrievals = int(payload.get("total_retrievals", 0))
        self.total_replays = int(payload.get("total_replays", 0))
        self.total_mutations = int(payload.get("total_mutations", 0))
        self.total_replacements = int(payload.get("total_replacements", 0))
        self.total_pruned = int(payload.get("total_pruned", 0))
        self.total_deleted_by_risk = int(payload.get("total_deleted_by_risk", 0))
        self.global_task_successes = int(payload.get("global_task_successes", 0))
        self.global_task_failures = int(payload.get("global_task_failures", 0))
        self.entries = {
            item["id"]: MemoryEntry(**item)
            for item in payload.get("entries", [])
        }

    def _persist_state(self) -> None:
        write_json(
            self.state_path,
            {
                "logical_time": self.logical_time,
                "current_capacity": self.current_capacity,
                "total_created": self.total_created,
                "total_retrievals": self.total_retrievals,
                "total_replays": self.total_replays,
                "total_mutations": self.total_mutations,
                "total_replacements": self.total_replacements,
                "total_pruned": self.total_pruned,
                "total_deleted_by_risk": self.total_deleted_by_risk,
                "global_task_successes": self.global_task_successes,
                "global_task_failures": self.global_task_failures,
                "entries": [entry.to_dict() for entry in self.entries.values()],
            },
        )
        self.summary()

    def _refresh_all_scores(self) -> None:
        for entry in self.entries.values():
            self._refresh_entry(entry)

    def _refresh_entry(self, entry: MemoryEntry) -> None:
        if entry.deleted:
            entry.survival_value = 0.0
            entry.posterior_failure_mean = 1.0
            entry.posterior_failure_std = 0.0
            entry.risk_score = 1.0
            return
        entry.posterior_failure_mean = self._posterior_failure_mean(entry)
        entry.posterior_failure_std = self._posterior_failure_std(entry)
        entry.risk_score = entry.posterior_failure_mean - entry.posterior_failure_std
        entry.survival_value = self._survival_value(entry)

    def _posterior_failure_mean(self, entry: MemoryEntry) -> float:
        prior_strength = self._risk_prior_strength()
        global_failure_rate = self._global_failure_rate()
        return (
            entry.failure_count + prior_strength * global_failure_rate
        ) / (entry.failure_count + entry.success_count + prior_strength)

    def _posterior_failure_std(self, entry: MemoryEntry) -> float:
        numerator = entry.posterior_failure_mean * (1.0 - entry.posterior_failure_mean)
        denominator = (
            entry.failure_count
            + entry.success_count
            + self.config.risk_prior_failures
            + self.config.risk_prior_successes
            + 1.0
        )
        return math.sqrt(max(numerator / denominator, 0.0))

    def _dynamic_risk_threshold(self) -> float:
        if self._risk_prior_strength() <= 0:
            return self.config.risk_threshold_base
        return self.config.risk_threshold_base * (
            1.0 - self.config.threshold_sensitivity * self._global_failure_rate()
        )

    def _survival_value(self, entry: MemoryEntry) -> float:
        utility = math.log1p(entry.reuse_count) + self.config.novelty_bonus
        delta_t = max(self.logical_time - entry.last_retrieved_logical_time, 0)
        half_life = self.config.base_retention + (
            self.config.longevity_coefficient * math.log1p(entry.reuse_count)
        )
        adaptive_decay = 1.0 / (
            1.0
            + math.exp(
                self.config.decay_steepness * (float(delta_t) - float(half_life))
            )
        )
        success_total = entry.success_count + entry.failure_count
        success_ratio = entry.success_count / success_total if success_total > 0 else 1.0
        action_denominator = max(entry.total_actions, 1)
        invalid_rate = entry.invalid_action_count / action_denominator
        execution_error_rate = entry.execution_error_count / action_denominator
        feedback_penalty = (
            entry.verification_failures
            + (1.0 - success_ratio)
            + invalid_rate
            + execution_error_rate
        )
        reliability = 1.0 / (
            1.0 + self.config.penalty_coefficient * feedback_penalty
        )
        return utility * adaptive_decay * reliability

    def _risk_prior_strength(self) -> float:
        return self.config.risk_prior_failures + self.config.risk_prior_successes

    def _global_failure_rate(self) -> float:
        prior_strength = self._risk_prior_strength()
        total = self.global_task_successes + self.global_task_failures + prior_strength
        if total <= 0:
            return 0.5
        return (self.global_task_failures + self.config.risk_prior_failures) / total

    def _sorted_active_entries(self) -> list[MemoryEntry]:
        self._refresh_all_scores()
        return sorted(
            (
                entry
                for entry in self.entries.values()
                if not entry.deleted
            ),
            key=lambda entry: (
                entry.survival_value,
                entry.reuse_count,
                -entry.risk_score,
                -entry.steps,
            ),
            reverse=True,
        )

    def _prune_if_needed(self) -> None:
        active_entries = self._sorted_active_entries()
        if len(active_entries) < self.current_capacity:
            return

        scores = [entry.survival_value for entry in active_entries]
        if len(scores) < 3:
            self._trim_to_capacity(active_entries, self.current_capacity)
            return

        curvatures: list[float] = []
        for index in range(1, len(scores) - 1):
            curvature = scores[index - 1] - (2.0 * scores[index]) + scores[index + 1]
            curvatures.append(curvature)
        if not curvatures:
            self._trim_to_capacity(active_entries, self.current_capacity)
            return

        elbow_offset = int(np.argmax(np.asarray(curvatures, dtype=np.float32)))
        elbow_index = elbow_offset + 1
        elbow_score = scores[elbow_index]
        mean_score = _safe_mean(scores)

        if elbow_score >= mean_score and self.current_capacity < self.config.max_capacity:
            old_capacity = self.current_capacity
            self.current_capacity = min(
                self.current_capacity + self.config.capacity_step,
                self.config.max_capacity,
            )
            append_jsonl(
                self.pruning_log_path,
                {
                    "timestamp": now_iso(),
                    "action": "expand_capacity",
                    "old_capacity": old_capacity,
                    "new_capacity": self.current_capacity,
                    "elbow_index": elbow_index,
                    "elbow_score": elbow_score,
                    "mean_score": mean_score,
                },
            )
            return

        retain_count = min(max(elbow_index + 1, 1), self.current_capacity)
        to_delete = active_entries[retain_count:]
        if not to_delete:
            return
        for entry in to_delete:
            entry.deleted = True
            entry.deleted_reason = "dynamic_pruning"
            self._refresh_entry(entry)
        self.total_pruned += len(to_delete)
        append_jsonl(
            self.pruning_log_path,
            {
                "timestamp": now_iso(),
                "action": "prune",
                "retain_count": retain_count,
                "pruned_ids": [entry.id for entry in to_delete],
                "elbow_index": elbow_index,
                "elbow_score": elbow_score,
                "mean_score": mean_score,
            },
        )

    def _trim_to_capacity(
        self,
        active_entries: Sequence[MemoryEntry],
        capacity: int,
    ) -> None:
        if len(active_entries) <= capacity:
            return
        to_delete = list(active_entries[capacity:])
        for entry in to_delete:
            entry.deleted = True
            entry.deleted_reason = "capacity_trim"
            self._refresh_entry(entry)
        self.total_pruned += len(to_delete)
        append_jsonl(
            self.pruning_log_path,
            {
                "timestamp": now_iso(),
                "action": "trim_capacity",
                "retain_count": capacity,
                "pruned_ids": [entry.id for entry in to_delete],
            },
        )

    def _log_retrieval(
        self,
        *,
        subtask: dict[str, str],
        decision: RetrievalDecision,
    ) -> None:
        append_jsonl(
            self.retrieval_log_path,
            {
                "timestamp": now_iso(),
                "logical_time": self.logical_time,
                "subtask": dict(subtask),
                "decision": decision.to_dict(),
            },
        )
