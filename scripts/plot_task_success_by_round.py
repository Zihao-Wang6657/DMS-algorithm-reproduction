#!/usr/bin/env python
"""Render five readable task-level success timelines from the formal summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


METHODS = (
    ("baseline_a", "Baseline A", "#4C78A8"),
    ("baseline_b", "Baseline B", "#F58518"),
    ("dms", "DMS", "#54A24B"),
)


def _load_rows(summary_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return list(payload["individual_task_success_by_round"])


def _task_order(rows: list[dict[str, Any]]) -> list[str]:
    return list(
        dict.fromkeys(
            str(row["task_name"])
            for row in rows
            if row["method_key"] == "baseline_a"
        )
    )


def render(summary_path: Path, output_dir: Path) -> None:
    rows = _load_rows(summary_path)
    tasks = _task_order(rows)
    lookup = {
        (str(row["method_key"]), str(row["task_name"]), int(row["round"])): int(
            row["success"]
        )
        for row in rows
    }
    rounds = sorted({int(row["round"]) for row in rows})
    output_dir.mkdir(parents=True, exist_ok=True)

    for task_name in tasks:
        fig, ax = plt.subplots(figsize=(8.8, 3.6))
        y_positions = {method_key: 2 - index for index, (method_key, _, _) in enumerate(METHODS)}

        for method_key, _, color in METHODS:
            y = y_positions[method_key]
            ax.plot(
                rounds,
                [y] * len(rounds),
                color=color,
                linewidth=1.7,
                alpha=0.7,
                zorder=1,
            )
            for round_id in rounds:
                success = lookup[(method_key, task_name, round_id)]
                facecolor = "#2E7D32" if success else "#E0E0E0"
                text_color = "white" if success else "#333333"
                ax.scatter(
                    round_id,
                    y,
                    s=520,
                    marker="s",
                    facecolor=facecolor,
                    edgecolor=color,
                    linewidth=2.2,
                    zorder=2,
                )
                ax.text(
                    round_id,
                    y,
                    "S" if success else "F",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=11,
                    fontweight="bold",
                    zorder=3,
                )

        ax.set_title(f"{task_name}: Success by Round", fontsize=14)
        ax.set_xlabel("Round")
        ax.set_xticks(rounds)
        ax.set_xlim(min(rounds) - 0.45, max(rounds) + 0.45)
        ax.set_yticks(
            [y_positions[method_key] for method_key, _, _ in METHODS],
            labels=[label for _, label, _ in METHODS],
        )
        ax.set_ylim(-0.55, 2.55)
        ax.grid(True, axis="x", linestyle="--", alpha=0.28)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(
            handles=(
                Patch(facecolor="#2E7D32", label="Success (1)"),
                Patch(facecolor="#E0E0E0", edgecolor="#777777", label="Failure (0)"),
            ),
            frameon=False,
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{task_name}_success_by_round.png", dpi=220)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="fig/summary.json")
    parser.add_argument("--output-dir", default="fig/task_success_by_round")
    args = parser.parse_args()
    render(Path(args.summary).resolve(), Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
