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


def _build_timeline(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "round": int(item["round"]),
            "task_id": str(item["task_id"]),
            "task_name": str(item["task_name"]),
            "memory_size_after": int(item.get("memory_size_after", 0)),
        }
        for index, item in enumerate(results, start=1)
    ]


def _overall_summary(
    metrics: dict[str, Any],
    round_summaries: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    tasks = int(metrics.get("tasks", 0))
    total_tokens = int(metrics.get("input_tokens", 0)) + int(metrics.get("output_tokens", 0))
    total_steps = int(metrics.get("total_steps", 0))
    return {
        "method": label,
        "tasks": tasks,
        "successful_tasks": int(metrics.get("successful_tasks", 0)),
        "success_rate": float(metrics.get("success_rate", 0.0)),
        "input_tokens": int(metrics.get("input_tokens", 0)),
        "output_tokens": int(metrics.get("output_tokens", 0)),
        "total_tokens": total_tokens,
        "avg_total_tokens_per_task": total_tokens / tasks if tasks else 0.0,
        "total_steps": total_steps,
        "avg_steps_per_task": total_steps / tasks if tasks else 0.0,
        "final_memory_size": int(metrics.get("memory_size", 0)),
        "rounds": len(round_summaries),
    }


def _line_plot(
    *,
    output_path: Path,
    title: str,
    ylabel: str,
    rounds: list[int],
    left_values: list[float],
    right_values: list[float],
    left_label: str,
    right_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(rounds, left_values, marker="o", linewidth=2.2, color="#1F77B4", label=left_label)
    ax.plot(rounds, right_values, marker="s", linewidth=2.2, color="#D62728", label=right_label)
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_xticks(rounds)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _memory_timeline_plot(
    *,
    output_path: Path,
    left_timeline: list[dict[str, Any]],
    right_timeline: list[dict[str, Any]],
    left_label: str,
    right_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(
        [item["index"] for item in left_timeline],
        [item["memory_size_after"] for item in left_timeline],
        marker="o",
        markersize=2.8,
        linewidth=1.8,
        color="#1F77B4",
        label=left_label,
    )
    ax.plot(
        [item["index"] for item in right_timeline],
        [item["memory_size_after"] for item in right_timeline],
        marker="s",
        markersize=2.8,
        linewidth=1.8,
        color="#D62728",
        label=right_label,
    )
    ax.set_title("Memory Size Timeline Across 5-Round Lifespan")
    ax.set_xlabel("Task Attempt Index")
    ax.set_ylabel("Memory Size After Task")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_markdown(
    *,
    output_path: Path,
    left_name: str,
    right_name: str,
    left_overall: dict[str, Any],
    right_overall: dict[str, Any],
    left_rounds: list[dict[str, Any]],
    right_rounds: list[dict[str, Any]],
) -> None:
    lines = [
        "# Run Comparison Summary",
        "",
        "## Overall",
        "",
        "| Method | Tasks | Successes | Success Rate | Avg Tokens/Task | Avg Steps/Task | Final Memory Size |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| {method} | {tasks} | {successful_tasks} | {success_rate:.2%} | {avg_total_tokens_per_task:.1f} | {avg_steps_per_task:.2f} | {final_memory_size} |".format(
            **left_overall
        ),
        "| {method} | {tasks} | {successful_tasks} | {success_rate:.2%} | {avg_total_tokens_per_task:.1f} | {avg_steps_per_task:.2f} | {final_memory_size} |".format(
            **right_overall
        ),
        "",
        "## By Round",
        "",
        "| Method | Round | Tasks | Successes | Success Rate | Avg Tokens/Task | Avg Steps/Task | End Memory Size |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for item in left_rounds:
        lines.append(
            f"| {left_name} | {item['round']} | {item['tasks']} | {item['successful_tasks']} | "
            f"{item['success_rate']:.2%} | {item['avg_total_tokens_per_task']:.1f} | "
            f"{item['avg_steps_per_task']:.2f} | {item['end_memory_size']} |"
        )
    for item in right_rounds:
        lines.append(
            f"| {right_name} | {item['round']} | {item['tasks']} | {item['successful_tasks']} | "
            f"{item['success_rate']:.2%} | {item['avg_total_tokens_per_task']:.1f} | "
            f"{item['avg_steps_per_task']:.2f} | {item['end_memory_size']} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-run-dir", required=True)
    parser.add_argument("--right-run-dir", required=True)
    parser.add_argument("--left-name", default="Baseline A")
    parser.add_argument("--right-name", default="DMS")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    left_run_dir = Path(args.left_run_dir).resolve()
    right_run_dir = Path(args.right_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    left_metrics = _load_json(left_run_dir / "metrics.json")
    right_metrics = _load_json(right_run_dir / "metrics.json")

    left_results = list(left_metrics.get("results", []))
    right_results = list(right_metrics.get("results", []))
    if not left_results or not right_results:
        raise ValueError("Both runs must contain non-empty metrics.json results.")

    left_rounds = _group_by_round(left_results)
    right_rounds = _group_by_round(right_results)
    left_timeline = _build_timeline(left_results)
    right_timeline = _build_timeline(right_results)
    left_overall = _overall_summary(left_metrics, left_rounds, label=args.left_name)
    right_overall = _overall_summary(right_metrics, right_rounds, label=args.right_name)

    rounds = sorted({item["round"] for item in left_rounds} | {item["round"] for item in right_rounds})

    def _series(items: list[dict[str, Any]], key: str) -> list[float]:
        by_round = {int(item["round"]): float(item[key]) for item in items}
        return [by_round.get(round_id, 0.0) for round_id in rounds]

    (output_dir / "left_round_summary.json").write_text(
        json.dumps(left_rounds, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "right_round_summary.json").write_text(
        json.dumps(right_rounds, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "overall_summary.json").write_text(
        json.dumps(
            {
                args.left_name: left_overall,
                args.right_name: right_overall,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_markdown(
        output_path=output_dir / "summary.md",
        left_name=args.left_name,
        right_name=args.right_name,
        left_overall=left_overall,
        right_overall=right_overall,
        left_rounds=left_rounds,
        right_rounds=right_rounds,
    )

    _line_plot(
        output_path=output_dir / "success_rate_by_round.png",
        title=f"{args.left_name} vs {args.right_name}: Success Rate by Round",
        ylabel="Success Rate",
        rounds=rounds,
        left_values=_series(left_rounds, "success_rate"),
        right_values=_series(right_rounds, "success_rate"),
        left_label=args.left_name,
        right_label=args.right_name,
    )
    _line_plot(
        output_path=output_dir / "avg_tokens_by_round.png",
        title=f"{args.left_name} vs {args.right_name}: Avg Tokens per Task",
        ylabel="Average Tokens / Task",
        rounds=rounds,
        left_values=_series(left_rounds, "avg_total_tokens_per_task"),
        right_values=_series(right_rounds, "avg_total_tokens_per_task"),
        left_label=args.left_name,
        right_label=args.right_name,
    )
    _line_plot(
        output_path=output_dir / "avg_steps_by_round.png",
        title=f"{args.left_name} vs {args.right_name}: Avg Steps per Task",
        ylabel="Average Steps / Task",
        rounds=rounds,
        left_values=_series(left_rounds, "avg_steps_per_task"),
        right_values=_series(right_rounds, "avg_steps_per_task"),
        left_label=args.left_name,
        right_label=args.right_name,
    )
    _memory_timeline_plot(
        output_path=output_dir / "memory_size_timeline.png",
        left_timeline=left_timeline,
        right_timeline=right_timeline,
        left_label=args.left_name,
        right_label=args.right_name,
    )


if __name__ == "__main__":
    main()
