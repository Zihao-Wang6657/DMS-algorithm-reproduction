from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class HierarchicalMemoryStub:
    """Importable DMS memory shim for integration tests and local wiring.

    This stub deliberately does not implement the DMS math. It only exposes the
    hierarchical retrieval/replay/mutation/pruning/stat hooks that the
    integration layer expects from a dedicated memory subsystem.
    """

    path: str | None = None
    run_dir: str | None = None
    workspace_root: str | None = None
    entry_limit: int = 256
    entries: list[dict[str, Any]] = field(default_factory=list)
    remembers: list[dict[str, Any]] = field(default_factory=list)
    step_records: list[dict[str, Any]] = field(default_factory=list)
    finalizations: list[dict[str, Any]] = field(default_factory=list)

    def _payload(self, **kwargs: Any) -> dict[str, Any]:
        return dict(kwargs)

    @property
    def size(self) -> int:
        return len(self.entries) + len(self.remembers)

    def planner_context(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "hierarchical_entries": list(self.entries[-self.entry_limit :]),
            "retrieval": {"count": len(self.entries)},
            "replay": {"count": len(self.remembers)},
            "pruning_stats": self.pruning_stats(**kwargs),
            "risk_stats": self.risk_stats(**kwargs),
        }

    def actor_context(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "replay_candidates": list(self.remembers[-self.entry_limit :]),
            "mutation_fallback": self.mutation_fallback(**kwargs),
            "risk_stats": self.risk_stats(**kwargs),
        }

    def retrieve(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "hierarchical_entries": list(self.entries[-self.entry_limit :]),
            "query": self._payload(**kwargs),
        }

    def replay(self, **kwargs: Any) -> list[dict[str, Any]]:
        return list(self.remembers[-self.entry_limit :])

    def mutation_fallback(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "fallback": "reuse_recent_success",
            "entry_limit": self.entry_limit,
        }

    def remember(self, **kwargs: Any) -> dict[str, Any]:
        payload = self._payload(**kwargs)
        self.remembers.append(payload)
        self.entries.append({"kind": "remember", **payload})
        return payload

    def record_step(self, **kwargs: Any) -> dict[str, Any]:
        payload = self._payload(**kwargs)
        self.step_records.append(payload)
        self.entries.append({"kind": "step", **payload})
        return payload

    def finalize_task(self, **kwargs: Any) -> dict[str, Any]:
        payload = self._payload(**kwargs)
        self.finalizations.append(payload)
        self.entries.append({"kind": "finalize", **payload})
        return payload

    def stats(self) -> dict[str, Any]:
        return {
            "memory_size": self.size,
            "entry_limit": self.entry_limit,
            "remember_count": len(self.remembers),
            "step_record_count": len(self.step_records),
            "finalization_count": len(self.finalizations),
        }

    def pruning_stats(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {
            "pruned_entries": max(0, self.size - self.entry_limit),
            "entry_limit": self.entry_limit,
        }

    def risk_stats(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {
            "risk_score": 0.0,
            "mutation_risk": 0.0,
            "pruning_risk": 0.0,
        }

