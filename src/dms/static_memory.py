from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dms.io_utils import append_jsonl, now_iso


@dataclass(frozen=True)
class StaticMemoryEntry:
    timestamp: str
    task_id: str
    task_name: str
    goal: str
    success: bool
    steps: int
    trajectory: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StaticMemory:
    """Chronological append-only memory for Baseline B.

    This intentionally does not implement DMS retrieval, survival value,
    pruning, risk suppression, mutation, or replacement.
    """

    def __init__(self, path: str | Path, max_context_entries: int = 12) -> None:
        self.path = Path(path)
        self.max_context_entries = max_context_entries
        self.entries: list[dict[str, Any]] = []
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.entries.append(json.loads(line))

    @property
    def size(self) -> int:
        return len(self.entries)

    def append_task(
        self,
        *,
        task_id: str,
        task_name: str,
        goal: str,
        success: bool,
        steps: int,
        trajectory: list[dict[str, Any]],
    ) -> None:
        entry = StaticMemoryEntry(
            timestamp=now_iso(),
            task_id=task_id,
            task_name=task_name,
            goal=goal,
            success=success,
            steps=steps,
            trajectory=trajectory,
        ).to_dict()
        self.entries.append(entry)
        append_jsonl(self.path, entry)

    def context(self) -> str:
        if not self.entries:
            return "No previous task trajectories."
        selected = self.entries[-self.max_context_entries :]
        lines = [
            "Chronological static memory from previous tasks. This memory is append-only and unpruned."
        ]
        for index, entry in enumerate(selected, start=1):
            lines.append(
                f"{index}. task_id={entry['task_id']} task={entry['task_name']} "
                f"success={entry['success']} steps={entry['steps']} goal={entry['goal']}"
            )
            for step in entry.get("trajectory", [])[-6:]:
                lines.append(
                    "   - "
                    f"subtask={step.get('subtask', '')} "
                    f"action={step.get('action', {})} "
                    f"result={step.get('result', '')}"
                )
        return "\n".join(lines)

