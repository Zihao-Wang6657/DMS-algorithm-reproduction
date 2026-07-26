#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _group_by_round(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        grouped[int(item["round"])].append(item)

    summaries: list[dict[str, Any]] = []
    for round_id in sorted(grouped):
        round_items = grouped[round_id]
        task_count = len(round_items)
        successes = sum(1 for item in round_items if item.get("success"))
        input_tokens = sum(int(item.get("input_tokens", 0)) for item in round_items)
        output_tokens = sum(int(item.get("output_tokens", 0)) for item in round_items)
        total_tokens = input_tokens + output_tokens
        total_steps = sum(int(item.get("steps", 0)) for item in round_items)
        avg_memory_size = (
            sum(int(item.get("memory_size_after", 0)) for item in round_items) / task_count
            if task_count
            else 0.0
        )
        summaries.append(
            {
                "round": round_id,
                "tasks": task_count,
                "successful_tasks": successes,
                "success_rate": successes / task_count if task_count else 0.0,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "avg_total_tokens_per_task": total_tokens / task_count if task_count else 0.0,
                "total_steps": total_steps,
                "avg_steps_per_task": total_steps / task_count if task_count else 0.0,
                "avg_memory_size_after_task": avg_memory_size,
                "end_memory_size": int(round_items[-1].get("memory_size_after", 0))
                if round_items
                else 0,
            }
        )
    return summaries


def _memory_timeline(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for index, item in enumerate(results, start=1):
        timeline.append(
            {
                "index": index,
                "round": int(item["round"]),
                "task_id": str(item["task_id"]),
                "task_name": str(item["task_name"]),
                "memory_size_after": int(item.get("memory_size_after", 0)),
            }
        )
    return timeline


def _plot_round_metric(
    *,
    rounds: list[int],
    values: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
    color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rounds, values, marker="o", linewidth=2.2, color=color)
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_xticks(rounds)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_memory_timeline(
    timeline: list[dict[str, Any]],
    output_path: Path,
) -> None:
    xs = [item["index"] for item in timeline]
    ys = [item["memory_size_after"] for item in timeline]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(xs, ys, marker="o", markersize=3.2, linewidth=1.8, color="#2E8B57")
    for point in timeline:
        if point["index"] == 1 or (
            point["index"] > 1
            and point["round"] != timeline[point["index"] - 2]["round"]
        ):
            ax.axvline(point["index"], color="#C7C7C7", linestyle=":", linewidth=1.0)
    ax.set_title("DMS Memory Size Over Task Timeline")
    ax.set_xlabel("Task Attempt Index Across 5-Round Lifespan")
    ax.set_ylabel("Memory Size After Task")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_markdown_summary(
    *,
    output_path: Path,
    round_summaries: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
) -> None:
    lines = [
        "# DMS 5-Round Lifespan Analysis",
        "",
        "| Round | Tasks | Successes | Success Rate | Avg Tokens/Task | Avg Steps/Task | End Memory Size |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in round_summaries:
        lines.append(
            "| {round} | {tasks} | {successful_tasks} | {success_rate:.2%} | "
            "{avg_total_tokens_per_task:.1f} | {avg_steps_per_task:.2f} | "
            "{end_memory_size} |".format(**item)
        )
    lines.extend(
        [
            "",
            f"- Total task attempts: {len(timeline)}",
            f"- Final memory size: {timeline[-1]['memory_size_after'] if timeline else 0}",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_dir / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _load_json(run_dir / "metrics.json")
    results = list(metrics.get("results", []))
    if not results:
        raise ValueError(f"No task results found in {run_dir / 'metrics.json'}")

    round_summaries = _group_by_round(results)
    timeline = _memory_timeline(results)

    (output_dir / "round_summary.json").write_text(
        json.dumps(round_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "memory_timeline.json").write_text(
        json.dumps(timeline, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown_summary(
        output_path=output_dir / "summary.md",
        round_summaries=round_summaries,
        timeline=timeline,
    )

    rounds = [item["round"] for item in round_summaries]
    _plot_round_metric(
        rounds=rounds,
        values=[item["success_rate"] for item in round_summaries],
        title="DMS Success Rate Across 5 Rounds",
        ylabel="Success Rate",
        output_path=output_dir / "success_rate_by_round.png",
        color="#1F77B4",
    )
    _plot_round_metric(
        rounds=rounds,
        values=[item["avg_total_tokens_per_task"] for item in round_summaries],
        title="DMS Average Token Consumption Per Task",
        ylabel="Average Tokens / Task",
        output_path=output_dir / "tokens_by_round.png",
        color="#D62728",
    )
    _plot_round_metric(
        rounds=rounds,
        values=[item["avg_steps_per_task"] for item in round_summaries],
        title="DMS Average Execution Steps Per Task",
        ylabel="Average Steps / Task",
        output_path=output_dir / "steps_by_round.png",
        color="#FF7F0E",
    )
    _plot_memory_timeline(
        timeline=timeline,
        output_path=output_dir / "memory_size_timeline.png",
    )


if __name__ == "__main__":
    main()
