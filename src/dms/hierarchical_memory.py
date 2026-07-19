from __future__ import annotations

import importlib.util
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from dms.io_utils import now_iso, write_json


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ONE_SIDED_Z90 = 1.2815515655446004


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def _stable_bucket(feature: str, dimensions: int) -> tuple[int, float]:
    total = 0
    for index, char in enumerate(feature.encode("utf-8"), start=1):
        total = (total * 131 + index * (char + 17)) % 2_147_483_647
    bucket = total % dimensions
    sign = -1.0 if (total // dimensions) % 2 else 1.0
    return bucket, sign


def _l2_normalize(values: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return [0.0 for _ in values]
    return [value / norm for value in values]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    return dot / (left_norm * right_norm)


def _normalize_tokens(text: str) -> list[str]:
    lowered = text.casefold()
    tokens = _TOKEN_RE.findall(lowered)
    if not tokens:
        return [lowered.strip()] if lowered.strip() else []
    return tokens


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    collapsed = re.sub(r"\s+", " ", text.casefold()).strip()
    if len(collapsed) < n:
        return [collapsed] if collapsed else []
    return [collapsed[index : index + n] for index in range(len(collapsed) - n + 1)]


def _normalize_path(path: Iterable[str] | None) -> tuple[str, ...]:
    if path is None:
        return ()
    return tuple(part.strip().casefold() for part in path if str(part).strip())


def _normalize_tags(tags: Iterable[str] | None) -> tuple[str, ...]:
    if tags is None:
        return ()
    return tuple(tag.strip().casefold() for tag in tags if str(tag).strip())


def _parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.fromisoformat(now_iso())
    return datetime.fromisoformat(value)


def _to_iso(value: str | datetime | None) -> str:
    if value is None:
        return now_iso()
    if isinstance(value, datetime):
        return value.astimezone().isoformat()
    return value


class TextEmbedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class HashingTextEmbedder:
    """Deterministic lexical embedder with no network or model dependency."""

    def __init__(self, dimensions: int = 256, char_ngram_size: int = 3) -> None:
        self.dimensions = dimensions
        self.char_ngram_size = char_ngram_size

    def _features(self, text: str) -> list[str]:
        features = [f"tok:{token}" for token in _normalize_tokens(text)]
        features.extend(
            f"chr:{gram}" for gram in _char_ngrams(text, n=self.char_ngram_size)
        )
        if not features:
            return ["empty"]
        return features

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        encoded: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for feature in self._features(text):
                bucket, sign = _stable_bucket(feature, self.dimensions)
                vector[bucket] += sign
            encoded.append(_l2_normalize(vector))
        return encoded


class CachedTransformerTextEmbedder:
    """Uses an already-cached local transformer model when available."""

    def __init__(self, model_name_or_path: str) -> None:
        if importlib.util.find_spec("transformers") is None:
            raise RuntimeError("transformers is not installed")
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("torch is not installed")

        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=True,
        )
        self._model = AutoModel.from_pretrained(
            model_name_or_path,
            local_files_only=True,
        )
        self._model.eval()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        with self._torch.no_grad():
            batch = self._tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            output = self._model(**batch)
            hidden = output.last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            masked = hidden * mask
            pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return [_l2_normalize(vector.tolist()) for vector in pooled]


def build_embedder(model_name_or_path: str | None = None) -> TextEmbedder:
    preferred = model_name_or_path or os.environ.get("DMS_REPRO_EMBED_MODEL")
    if preferred:
        try:
            return CachedTransformerTextEmbedder(preferred)
        except Exception:
            pass
    return HashingTextEmbedder()


@dataclass
class BayesianRiskEstimate:
    alpha: float
    beta: float
    mean: float
    variance: float
    upper_bound: float


@dataclass
class SurvivalValue:
    utility_signal: float
    usage_signal: float
    recency_signal: float
    importance_signal: float
    score: float


@dataclass
class RetrievalResult:
    entry: "MemoryEntry"
    score: float
    semantic_score: float
    contextual_score: float
    survival_score: float
    risk_score: float


@dataclass
class EvolutionDecision:
    action: str
    entry: "MemoryEntry | None"
    replaced_entry_id: str | None = None
    reason: str = ""


@dataclass
class PruneDecision:
    pruned_entry_ids: list[str]
    kept_entry_ids: list[str]
    cutoff_index: int
    cutoff_score: float | None
    old_capacity: int
    new_capacity: int
    expanded: bool


@dataclass
class MemoryEntry:
    memory_id: str
    slot: int
    level: int
    text: str
    summary: str
    hierarchy_path: tuple[str, ...]
    tags: tuple[str, ...]
    parent_id: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    last_accessed_at: str
    access_count: int = 0
    reinforcement_count: int = 0
    success_count: float = 0.0
    failure_count: float = 0.0
    cumulative_reward: float = 0.0
    importance: float = 0.5
    embedding: list[float] = field(default_factory=list)
    generation: int = 0
    replaced_from: str | None = None
    replacement_lineage: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "slot": self.slot,
            "level": self.level,
            "text": self.text,
            "summary": self.summary,
            "hierarchy_path": list(self.hierarchy_path),
            "tags": list(self.tags),
            "parent_id": self.parent_id,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "reinforcement_count": self.reinforcement_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "cumulative_reward": self.cumulative_reward,
            "importance": self.importance,
            "embedding": self.embedding,
            "generation": self.generation,
            "replaced_from": self.replaced_from,
            "replacement_lineage": list(self.replacement_lineage),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        return cls(
            memory_id=str(data["memory_id"]),
            slot=int(data["slot"]),
            level=int(data.get("level", 0)),
            text=str(data["text"]),
            summary=str(data.get("summary", "")),
            hierarchy_path=tuple(data.get("hierarchy_path", [])),
            tags=tuple(data.get("tags", [])),
            parent_id=data.get("parent_id"),
            metadata=dict(data.get("metadata", {})),
            created_at=str(data["created_at"]),
            updated_at=str(data.get("updated_at", data["created_at"])),
            last_accessed_at=str(
                data.get("last_accessed_at", data.get("updated_at", data["created_at"]))
            ),
            access_count=int(data.get("access_count", 0)),
            reinforcement_count=int(data.get("reinforcement_count", 0)),
            success_count=float(data.get("success_count", 0.0)),
            failure_count=float(data.get("failure_count", 0.0)),
            cumulative_reward=float(data.get("cumulative_reward", 0.0)),
            importance=float(data.get("importance", 0.5)),
            embedding=list(data.get("embedding", [])),
            generation=int(data.get("generation", 0)),
            replaced_from=data.get("replaced_from"),
            replacement_lineage=tuple(data.get("replacement_lineage", [])),
        )


class HierarchicalMemory:
    """Standalone DMS-style memory with offline embeddings and adaptive pruning."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        capacity: int = 64,
        min_capacity: int = 4,
        max_capacity: int | None = None,
        allow_capacity_expansion: bool = False,
        capacity_expansion_factor: float = 1.5,
        replacement_margin: float = 0.05,
        recency_half_life_seconds: float = 7 * 24 * 60 * 60,
        access_temperature: float = 4.0,
        prior_success: float = 1.0,
        prior_failure: float = 1.0,
        semantic_weight: float = 0.55,
        contextual_weight: float = 0.25,
        survival_weight: float = 0.15,
        risk_penalty_weight: float = 0.05,
        expansion_flatness_threshold: float = 0.08,
        embedder: TextEmbedder | None = None,
        autosave: bool = True,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.capacity = max(1, int(capacity))
        self.min_capacity = max(1, int(min_capacity))
        self.max_capacity = None if max_capacity is None else max(1, int(max_capacity))
        self.allow_capacity_expansion = allow_capacity_expansion
        self.capacity_expansion_factor = max(1.0, float(capacity_expansion_factor))
        self.replacement_margin = max(0.0, float(replacement_margin))
        self.recency_half_life_seconds = max(1.0, float(recency_half_life_seconds))
        self.access_temperature = max(1.0, float(access_temperature))
        self.prior_success = max(1e-6, float(prior_success))
        self.prior_failure = max(1e-6, float(prior_failure))
        self.semantic_weight = float(semantic_weight)
        self.contextual_weight = float(contextual_weight)
        self.survival_weight = float(survival_weight)
        self.risk_penalty_weight = float(risk_penalty_weight)
        self.expansion_flatness_threshold = max(0.0, float(expansion_flatness_threshold))
        self.embedder = embedder or build_embedder()
        self.autosave = autosave

        self.entries: dict[str, MemoryEntry] = {}
        self._slots: list[str | None] = []
        self._replacement_history: list[dict[str, Any]] = []
        self._sequence = 0

        if self.path and self.path.is_file():
            self._load()

    @property
    def size(self) -> int:
        return len(self.entries)

    def memory_ids(self) -> list[str]:
        return [memory_id for memory_id in self._slots if memory_id is not None]

    def get(self, memory_id: str) -> MemoryEntry:
        return self.entries[memory_id]

    def children_of(self, parent_id: str) -> list[MemoryEntry]:
        children = [entry for entry in self.entries.values() if entry.parent_id == parent_id]
        return sorted(children, key=lambda entry: (entry.level, entry.slot, entry.memory_id))

    def entries_at_level(self, level: int) -> list[MemoryEntry]:
        return sorted(
            [entry for entry in self.entries.values() if entry.level == level],
            key=lambda entry: (entry.slot, entry.memory_id),
        )

    def ingest(
        self,
        text: str,
        *,
        summary: str | None = None,
        level: int = 0,
        hierarchy_path: Iterable[str] | None = None,
        tags: Iterable[str] | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
        success_count: float = 0.0,
        failure_count: float = 0.0,
        cumulative_reward: float = 0.0,
        access_count: int = 0,
        created_at: str | datetime | None = None,
    ) -> EvolutionDecision:
        candidate = self._build_entry(
            text=text,
            summary=summary,
            level=level,
            hierarchy_path=hierarchy_path,
            tags=tags,
            parent_id=parent_id,
            metadata=metadata,
            importance=importance,
            success_count=success_count,
            failure_count=failure_count,
            cumulative_reward=cumulative_reward,
            access_count=access_count,
            created_at=created_at,
        )

        if self.size < self.capacity:
            self._insert_entry(candidate)
            self._save_if_needed()
            return EvolutionDecision(action="inserted", entry=candidate)

        if self.allow_capacity_expansion:
            probe = self.prune(force=False)
            if probe.expanded and self.size < self.capacity:
                self._insert_entry(candidate)
                self._save_if_needed()
                return EvolutionDecision(
                    action="inserted",
                    entry=candidate,
                    reason="capacity_expanded",
                )

        weakest_entry, weakest_score = self._weakest_retained_entry()
        candidate_score = self.retention_score(candidate)
        if weakest_entry is None:
            self._insert_entry(candidate)
            self._save_if_needed()
            return EvolutionDecision(action="inserted", entry=candidate)
        if candidate_score > weakest_score + self.replacement_margin:
            replaced_id = weakest_entry.memory_id
            self._replace_entry(weakest_entry, candidate)
            self._save_if_needed()
            return EvolutionDecision(
                action="replaced",
                entry=candidate,
                replaced_entry_id=replaced_id,
            )

        return EvolutionDecision(
            action="discarded",
            entry=None,
            reason="candidate_below_replacement_margin",
        )

    def record_feedback(
        self,
        memory_id: str,
        *,
        success: bool,
        reward: float | None = None,
        timestamp: str | datetime | None = None,
    ) -> MemoryEntry:
        entry = self.entries[memory_id]
        reward_value = 1.0 if reward is None and success else -1.0 if reward is None else reward
        entry.reinforcement_count += 1
        entry.success_count += 1.0 if success else 0.0
        entry.failure_count += 0.0 if success else 1.0
        entry.cumulative_reward += reward_value
        stamp = _to_iso(timestamp)
        entry.updated_at = stamp
        entry.last_accessed_at = stamp
        self._save_if_needed()
        return entry

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        hierarchy_path: Iterable[str] | None = None,
        tags: Iterable[str] | None = None,
        level: int | None = None,
        min_score: float | None = None,
        now: str | datetime | None = None,
        track_access: bool = True,
    ) -> list[RetrievalResult]:
        if not self.entries:
            return []

        query_path = _normalize_path(hierarchy_path)
        query_tags = _normalize_tags(tags)
        query_vector = self.embedder.embed([query])[0]
        results: list[RetrievalResult] = []

        for entry in self.entries.values():
            semantic_score = _cosine_similarity(query_vector, entry.embedding)
            contextual_score = self._contextual_affinity(
                query_path=query_path,
                query_tags=query_tags,
                query_level=level,
                entry=entry,
            )
            survival_score = self.survival_value(entry, now=now).score
            risk_score = self.estimate_risk(entry).upper_bound
            total_score = (
                self.semantic_weight * semantic_score
                + self.contextual_weight * contextual_score
                + self.survival_weight * survival_score
                - self.risk_penalty_weight * risk_score
            )
            if min_score is not None and total_score < min_score:
                continue
            results.append(
                RetrievalResult(
                    entry=entry,
                    score=total_score,
                    semantic_score=semantic_score,
                    contextual_score=contextual_score,
                    survival_score=survival_score,
                    risk_score=risk_score,
                )
            )

        results.sort(key=lambda result: result.score, reverse=True)
        selected = results[: max(0, top_k)]
        if track_access:
            stamp = _to_iso(now)
            for result in selected:
                result.entry.access_count += 1
                result.entry.last_accessed_at = stamp
            self._save_if_needed()
        return selected

    def estimate_risk(self, entry_or_id: str | MemoryEntry) -> BayesianRiskEstimate:
        entry = self._resolve_entry(entry_or_id)
        alpha = self.prior_failure + entry.failure_count
        beta = self.prior_success + entry.success_count
        mean = alpha / (alpha + beta)
        variance = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
        upper_bound = _clamp(mean + _ONE_SIDED_Z90 * math.sqrt(variance))
        return BayesianRiskEstimate(
            alpha=alpha,
            beta=beta,
            mean=mean,
            variance=variance,
            upper_bound=upper_bound,
        )

    def survival_value(
        self,
        entry_or_id: str | MemoryEntry,
        *,
        now: str | datetime | None = None,
    ) -> SurvivalValue:
        entry = self._resolve_entry(entry_or_id)
        current_time = _parse_timestamp(_to_iso(now))
        reference_time = _parse_timestamp(
            entry.last_accessed_at or entry.updated_at or entry.created_at
        )
        age_seconds = max(0.0, (current_time - reference_time).total_seconds())
        recency_signal = math.exp(-age_seconds / self.recency_half_life_seconds)
        usage_signal = 1.0 - math.exp(-entry.access_count / self.access_temperature)
        importance_signal = _clamp(entry.importance)
        if entry.reinforcement_count > 0:
            average_reward = entry.cumulative_reward / entry.reinforcement_count
            utility_signal = _clamp((average_reward + 1.0) / 2.0)
        else:
            utility_signal = 0.5
        score = _clamp(
            0.4 * utility_signal
            + 0.2 * usage_signal
            + 0.2 * recency_signal
            + 0.2 * importance_signal
        )
        return SurvivalValue(
            utility_signal=utility_signal,
            usage_signal=usage_signal,
            recency_signal=recency_signal,
            importance_signal=importance_signal,
            score=score,
        )

    def novelty_score(self, entry_or_id: str | MemoryEntry) -> float:
        entry = self._resolve_entry(entry_or_id)
        competitors = [
            other
            for other in self.entries.values()
            if other.memory_id != entry.memory_id
        ]
        if not competitors:
            return 1.0
        highest_similarity = max(
            _cosine_similarity(entry.embedding, other.embedding) for other in competitors
        )
        return _clamp(1.0 - max(0.0, highest_similarity))

    def retention_score(
        self,
        entry_or_id: str | MemoryEntry,
        *,
        now: str | datetime | None = None,
    ) -> float:
        entry = self._resolve_entry(entry_or_id)
        survival = self.survival_value(entry, now=now).score
        novelty = self.novelty_score(entry)
        risk = self.estimate_risk(entry).upper_bound
        return _clamp(0.65 * survival + 0.20 * novelty + 0.15 * (1.0 - risk))

    def prune(
        self,
        *,
        force: bool = False,
        now: str | datetime | None = None,
    ) -> PruneDecision:
        if not self.entries:
            return PruneDecision(
                pruned_entry_ids=[],
                kept_entry_ids=[],
                cutoff_index=0,
                cutoff_score=None,
                old_capacity=self.capacity,
                new_capacity=self.capacity,
                expanded=False,
            )

        scored = sorted(
            (
                (entry.memory_id, self.retention_score(entry, now=now))
                for entry in self.entries.values()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        scores = [score for _, score in scored]
        old_capacity = self.capacity

        if (
            self.allow_capacity_expansion
            and self.size >= self.capacity
            and self._is_flat_score_curve(scores)
            and (self.max_capacity is None or self.capacity < self.max_capacity)
        ):
            growth = max(1, math.ceil(self.capacity * (self.capacity_expansion_factor - 1.0)))
            self.capacity += growth
            if self.max_capacity is not None:
                self.capacity = min(self.capacity, self.max_capacity)
            self._save_if_needed()
            return PruneDecision(
                pruned_entry_ids=[],
                kept_entry_ids=[memory_id for memory_id, _ in scored],
                cutoff_index=len(scored),
                cutoff_score=scores[-1] if scores else None,
                old_capacity=old_capacity,
                new_capacity=self.capacity,
                expanded=True,
            )

        if not force and self.size <= self.capacity:
            return PruneDecision(
                pruned_entry_ids=[],
                kept_entry_ids=[memory_id for memory_id, _ in scored],
                cutoff_index=len(scored),
                cutoff_score=scores[-1] if scores else None,
                old_capacity=old_capacity,
                new_capacity=self.capacity,
                expanded=False,
            )

        desired_keep = self._elbow_cutoff(scores)
        if not force:
            desired_keep = min(desired_keep, self.capacity)
        effective_floor = min(self.min_capacity, len(scored), self.capacity if not force else len(scored))
        keep_count = max(effective_floor, desired_keep)
        if not force:
            keep_count = min(keep_count, self.capacity)
        keep_count = min(keep_count, len(scored))

        kept_ids = [memory_id for memory_id, _ in scored[:keep_count]]
        pruned_ids = [memory_id for memory_id, _ in scored[keep_count:]]
        for memory_id in pruned_ids:
            self._remove_entry(memory_id)
        self._save_if_needed()
        return PruneDecision(
            pruned_entry_ids=pruned_ids,
            kept_entry_ids=kept_ids,
            cutoff_index=keep_count,
            cutoff_score=scores[keep_count - 1] if keep_count > 0 else None,
            old_capacity=old_capacity,
            new_capacity=self.capacity,
            expanded=False,
        )

    def save(self) -> None:
        if self.path is None:
            return
        write_json(self.path, self._snapshot())

    def _save_if_needed(self) -> None:
        if self.autosave and self.path is not None:
            self.save()

    def _snapshot(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "min_capacity": self.min_capacity,
            "max_capacity": self.max_capacity,
            "allow_capacity_expansion": self.allow_capacity_expansion,
            "capacity_expansion_factor": self.capacity_expansion_factor,
            "replacement_margin": self.replacement_margin,
            "recency_half_life_seconds": self.recency_half_life_seconds,
            "access_temperature": self.access_temperature,
            "prior_success": self.prior_success,
            "prior_failure": self.prior_failure,
            "semantic_weight": self.semantic_weight,
            "contextual_weight": self.contextual_weight,
            "survival_weight": self.survival_weight,
            "risk_penalty_weight": self.risk_penalty_weight,
            "expansion_flatness_threshold": self.expansion_flatness_threshold,
            "sequence": self._sequence,
            "slots": self._slots,
            "entries": [entry.to_dict() for entry in self.entries.values()],
            "replacement_history": self._replacement_history,
        }

    def _load(self) -> None:
        assert self.path is not None
        import json

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.capacity = int(payload.get("capacity", self.capacity))
        self.min_capacity = int(payload.get("min_capacity", self.min_capacity))
        self.max_capacity = payload.get("max_capacity", self.max_capacity)
        if self.max_capacity is not None:
            self.max_capacity = int(self.max_capacity)
        self.allow_capacity_expansion = bool(
            payload.get("allow_capacity_expansion", self.allow_capacity_expansion)
        )
        self.capacity_expansion_factor = float(
            payload.get("capacity_expansion_factor", self.capacity_expansion_factor)
        )
        self.replacement_margin = float(
            payload.get("replacement_margin", self.replacement_margin)
        )
        self.recency_half_life_seconds = float(
            payload.get("recency_half_life_seconds", self.recency_half_life_seconds)
        )
        self.access_temperature = float(
            payload.get("access_temperature", self.access_temperature)
        )
        self.prior_success = float(payload.get("prior_success", self.prior_success))
        self.prior_failure = float(payload.get("prior_failure", self.prior_failure))
        self.semantic_weight = float(payload.get("semantic_weight", self.semantic_weight))
        self.contextual_weight = float(
            payload.get("contextual_weight", self.contextual_weight)
        )
        self.survival_weight = float(payload.get("survival_weight", self.survival_weight))
        self.risk_penalty_weight = float(
            payload.get("risk_penalty_weight", self.risk_penalty_weight)
        )
        self.expansion_flatness_threshold = float(
            payload.get("expansion_flatness_threshold", self.expansion_flatness_threshold)
        )
        self._sequence = int(payload.get("sequence", 0))
        self._slots = list(payload.get("slots", []))
        self.entries = {}
        for raw_entry in payload.get("entries", []):
            entry = MemoryEntry.from_dict(raw_entry)
            self.entries[entry.memory_id] = entry
        self._replacement_history = list(payload.get("replacement_history", []))

    def _build_entry(
        self,
        *,
        text: str,
        summary: str | None,
        level: int,
        hierarchy_path: Iterable[str] | None,
        tags: Iterable[str] | None,
        parent_id: str | None,
        metadata: dict[str, Any] | None,
        importance: float,
        success_count: float,
        failure_count: float,
        cumulative_reward: float,
        access_count: int,
        created_at: str | datetime | None,
    ) -> MemoryEntry:
        timestamp = _to_iso(created_at)
        path = _normalize_path(hierarchy_path)
        normalized_tags = _normalize_tags(tags)
        embedding = self.embedder.embed([text])[0]
        reinforcement_count = int(success_count + failure_count)
        if reinforcement_count > 0 and cumulative_reward == 0.0:
            cumulative_reward = success_count - failure_count
        return MemoryEntry(
            memory_id=self._next_memory_id(),
            slot=-1,
            level=int(level),
            text=text,
            summary=summary or text[:200],
            hierarchy_path=path,
            tags=normalized_tags,
            parent_id=parent_id,
            metadata=dict(metadata or {}),
            created_at=timestamp,
            updated_at=timestamp,
            last_accessed_at=timestamp,
            access_count=max(0, int(access_count)),
            reinforcement_count=max(0, reinforcement_count),
            success_count=max(0.0, float(success_count)),
            failure_count=max(0.0, float(failure_count)),
            cumulative_reward=float(cumulative_reward),
            importance=_clamp(float(importance)),
            embedding=embedding,
        )

    def _insert_entry(self, entry: MemoryEntry) -> None:
        try:
            slot = self._slots.index(None)
            self._slots[slot] = entry.memory_id
        except ValueError:
            slot = len(self._slots)
            self._slots.append(entry.memory_id)
        entry.slot = slot
        self.entries[entry.memory_id] = entry

    def _replace_entry(self, existing: MemoryEntry, candidate: MemoryEntry) -> None:
        slot = existing.slot
        candidate.slot = slot
        candidate.parent_id = candidate.parent_id or existing.parent_id
        candidate.generation = existing.generation + 1
        candidate.replaced_from = existing.memory_id
        candidate.replacement_lineage = existing.replacement_lineage + (existing.memory_id,)
        self._replacement_history.append(
            {
                "replaced_entry_id": existing.memory_id,
                "replacement_entry_id": candidate.memory_id,
                "slot": slot,
                "timestamp": now_iso(),
            }
        )
        del self.entries[existing.memory_id]
        self.entries[candidate.memory_id] = candidate
        self._slots[slot] = candidate.memory_id

    def _remove_entry(self, memory_id: str) -> None:
        entry = self.entries.pop(memory_id)
        if 0 <= entry.slot < len(self._slots):
            self._slots[entry.slot] = None

    def _next_memory_id(self) -> str:
        self._sequence += 1
        return f"mem-{self._sequence:06d}"

    def _resolve_entry(self, entry_or_id: str | MemoryEntry) -> MemoryEntry:
        if isinstance(entry_or_id, MemoryEntry):
            return entry_or_id
        return self.entries[entry_or_id]

    def _contextual_affinity(
        self,
        *,
        query_path: tuple[str, ...],
        query_tags: tuple[str, ...],
        query_level: int | None,
        entry: MemoryEntry,
    ) -> float:
        signals: list[float] = []
        if query_path:
            shared = 0
            for left, right in zip(query_path, entry.hierarchy_path):
                if left != right:
                    break
                shared += 1
            signals.append(shared / max(len(query_path), len(entry.hierarchy_path), 1))
        if query_tags:
            left = set(query_tags)
            right = set(entry.tags)
            if left or right:
                signals.append(len(left & right) / max(len(left | right), 1))
        if query_level is not None:
            level_gap = abs(int(query_level) - entry.level)
            signals.append(_clamp(1.0 - 0.25 * level_gap))
        if not signals:
            return 0.5
        return _clamp(_mean(signals))

    def _weakest_retained_entry(self) -> tuple[MemoryEntry | None, float]:
        weakest_entry: MemoryEntry | None = None
        weakest_score = math.inf
        for entry in self.entries.values():
            score = self.retention_score(entry)
            if score < weakest_score:
                weakest_entry = entry
                weakest_score = score
        return weakest_entry, 0.0 if weakest_entry is None else weakest_score

    def _is_flat_score_curve(self, scores: Sequence[float]) -> bool:
        if len(scores) < 2:
            return True
        span = max(scores) - min(scores)
        if span <= self.expansion_flatness_threshold:
            return True
        drops = [scores[index] - scores[index + 1] for index in range(len(scores) - 1)]
        return max(drops) <= self.expansion_flatness_threshold / 2.0

    def _elbow_cutoff(self, scores: Sequence[float]) -> int:
        if len(scores) <= 2:
            return len(scores)
        x0, y0 = 0.0, float(scores[0])
        x1, y1 = float(len(scores) - 1), float(scores[-1])
        denominator = math.hypot(x1 - x0, y1 - y0)
        if denominator == 0.0:
            return len(scores)
        distances: list[tuple[int, float]] = []
        for index, score in enumerate(scores[1:-1], start=1):
            numerator = abs((y1 - y0) * index - (x1 - x0) * score + x1 * y0 - y1 * x0)
            distances.append((index, numerator / denominator))
        if not distances:
            return len(scores)
        elbow_index, elbow_distance = max(distances, key=lambda item: item[1])
        if elbow_distance < 1e-6:
            return len(scores)
        return elbow_index + 1
