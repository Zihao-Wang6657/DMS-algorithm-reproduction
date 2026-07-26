#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


METHODS: tuple[tuple[str, str, str, str], ...] = (
    ("baseline_a", "Baseline A", "#4C78A8", "o"),
    ("baseline_b", "Baseline B", "#F58518", "s"),
    ("dms", "DMS", "#54A24B", "^"),
)

ROUND_FIELDS = (
    "method",
    "round",
    "tasks",
    "strict_successes",
    "strict_success_rate",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "avg_tokens_per_task",
    "total_steps",
    "avg_steps_per_task",
    "runtime_errors",
    "end_memory_size",
)

TASK_FIELDS = (
    "method",
    "task_id",
    "task_name",
    "round",
    "success",
    "failure_category",
    "steps",
    "control_turns",
    "event_count",
    "input_tokens",
    "output_tokens",
    "memory_size_after",
    "error",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing task results: {path}")
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Expected a JSON object at {path}:{line_number}")
        records.append(item)
    return records


def _has_runtime_error(record: dict[str, Any]) -> bool:
    error = record.get("error")
    return error not in (None, "", {}, [])


def _render_error(error: Any) -> str:
    if error in (None, "", {}, []):
        return ""
    if isinstance(error, str):
        return error
    return json.dumps(error, ensure_ascii=False, sort_keys=True)


def _validate_run_root(
    *,
    run_root: Path,
    expected_rounds: int,
    expected_tasks_per_round: int,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    list[str],
]:
    expected_round_ids = list(range(1, expected_rounds + 1))
    expected_records = expected_rounds * expected_tasks_per_round
    results_by_method: dict[str, list[dict[str, Any]]] = {}
    metrics_by_method: dict[str, dict[str, Any]] = {}
    reference_task_names: list[str] | None = None

    for method_key, method_label, _, _ in METHODS:
        method_dir = run_root / method_key
        results = _load_jsonl(method_dir / "task_results.jsonl")
        if len(results) != expected_records:
            raise ValueError(
                f"{method_label} must contain exactly {expected_records} task results; "
                f"found {len(results)}."
            )

        task_ids = [str(item.get("task_id", "")).strip() for item in results]
        if any(not task_id for task_id in task_ids):
            raise ValueError(f"{method_label} contains a result without task_id.")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError(f"{method_label} contains duplicate task_id values.")

        by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in results:
            by_round[int(item["round"])].append(item)
        if sorted(by_round) != expected_round_ids:
            raise ValueError(
                f"{method_label} rounds are {sorted(by_round)}; "
                f"expected {expected_round_ids}."
            )

        for round_id in expected_round_ids:
            round_items = by_round[round_id]
            if len(round_items) != expected_tasks_per_round:
                raise ValueError(
                    f"{method_label} round {round_id} must contain "
                    f"{expected_tasks_per_round} tasks; found {len(round_items)}."
                )
            task_names = [str(item["task_name"]) for item in round_items]
            if len(set(task_names)) != expected_tasks_per_round:
                raise ValueError(
                    f"{method_label} round {round_id} contains duplicate task names."
                )
            if reference_task_names is None:
                reference_task_names = task_names
            elif task_names != reference_task_names:
                raise ValueError(
                    f"{method_label} round {round_id} task order differs from "
                    "the first validated round."
                )

        metrics_path = method_dir / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing metrics: {metrics_path}")
        metrics = _load_json(metrics_path)
        successful = sum(bool(item.get("success")) for item in results)
        if int(metrics.get("tasks", -1)) != len(results):
            raise ValueError(f"{method_label} metrics task count does not match JSONL.")
        if int(metrics.get("successful_tasks", -1)) != successful:
            raise ValueError(f"{method_label} metrics success count does not match JSONL.")
        if int(metrics.get("rounds", -1)) != expected_rounds:
            raise ValueError(f"{method_label} metrics round count is not {expected_rounds}.")

        results_by_method[method_key] = results
        metrics_by_method[method_key] = metrics

    if reference_task_names is None:
        raise ValueError("No task results were found.")
    return results_by_method, metrics_by_method, reference_task_names


def _round_summaries(
    *,
    records: list[dict[str, Any]],
    method_label: str,
) -> list[dict[str, Any]]:
    by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        by_round[int(item["round"])].append(item)

    summaries: list[dict[str, Any]] = []
    for round_id in sorted(by_round):
        round_items = by_round[round_id]
        tasks = len(round_items)
        successes = sum(bool(item.get("success")) for item in round_items)
        input_tokens = sum(int(item.get("input_tokens", 0)) for item in round_items)
        output_tokens = sum(int(item.get("output_tokens", 0)) for item in round_items)
        total_tokens = input_tokens + output_tokens
        total_steps = sum(int(item.get("steps", 0)) for item in round_items)
        summaries.append(
            {
                "method": method_label,
                "round": round_id,
                "tasks": tasks,
                "strict_successes": successes,
                "strict_success_rate": successes / tasks if tasks else 0.0,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "avg_tokens_per_task": total_tokens / tasks if tasks else 0.0,
                "total_steps": total_steps,
                "avg_steps_per_task": total_steps / tasks if tasks else 0.0,
                "runtime_errors": sum(_has_runtime_error(item) for item in round_items),
                "end_memory_size": int(round_items[-1].get("memory_size_after", 0)),
            }
        )
    return summaries


def _overall_summary(
    *,
    records: list[dict[str, Any]],
    method_label: str,
) -> dict[str, Any]:
    tasks = len(records)
    successes = sum(bool(item.get("success")) for item in records)
    total_tokens = sum(
        int(item.get("input_tokens", 0)) + int(item.get("output_tokens", 0))
        for item in records
    )
    total_steps = sum(int(item.get("steps", 0)) for item in records)
    return {
        "method": method_label,
        "tasks": tasks,
        "strict_successes": successes,
        "strict_success_rate": successes / tasks if tasks else 0.0,
        "total_tokens": total_tokens,
        "avg_tokens_per_task": total_tokens / tasks if tasks else 0.0,
        "total_steps": total_steps,
        "avg_steps_per_task": total_steps / tasks if tasks else 0.0,
        "runtime_errors": sum(_has_runtime_error(item) for item in records),
        "final_memory_size": int(records[-1].get("memory_size_after", 0))
        if records
        else 0,
    }


def _write_csv(
    *,
    path: Path,
    fieldnames: Iterable[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    names = list(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_method_lines(
    *,
    by_round: dict[str, list[dict[str, Any]]],
    rounds: list[int],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    percent: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10.67, 6.33))
    for method_key, label, color, marker in METHODS:
        values = [float(item[metric]) for item in by_round[method_key]]
        if percent:
            values = [value * 100.0 for value in values]
        ax.plot(
            rounds,
            values,
            marker=marker,
            markersize=8,
            linewidth=2.5,
            color=color,
            label=label,
        )
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Round", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xticks(rounds)
    if percent:
        ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_dms_memory(
    *,
    records: list[dict[str, Any]],
    tasks_per_round: int,
    output_path: Path,
) -> None:
    xs = list(range(1, len(records) + 1))
    ys = [int(item.get("memory_size_after", 0)) for item in records]
    color = "#54A24B"
    fig, ax = plt.subplots(figsize=(10.67, 5.6))
    ax.step(xs, ys, where="post", linewidth=2.5, color=color, label="DMS")
    ax.scatter(xs, ys, s=18, color=color, zorder=3)
    for boundary in range(tasks_per_round, len(records), tasks_per_round):
        ax.axvline(boundary, color="#999999", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_title("DMS Memory Size over Task Attempts", fontsize=20)
    ax.set_xlabel("Time (one completed task)", fontsize=13)
    ax.set_ylabel("Number of Memories", fontsize=13)
    ax.set_xlim(0, len(records))
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_task_success(
    *,
    task_name: str,
    rounds: list[int],
    results_by_method: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.67, 5.2))
    y_positions = {
        method_key: len(METHODS) - index - 1
        for index, (method_key, _, _, _) in enumerate(METHODS)
    }

    for method_key, _, color, _ in METHODS:
        y = y_positions[method_key]
        task_records = [
            item
            for item in results_by_method[method_key]
            if str(item["task_name"]) == task_name
        ]
        by_round = {int(item["round"]): bool(item.get("success")) for item in task_records}
        ax.plot(rounds, [y] * len(rounds), color=color, linewidth=2.0, alpha=0.65)
        for round_id in rounds:
            success = by_round[round_id]
            facecolor = "#2E7D32" if success else "#E6E6E6"
            text_color = "white" if success else "#333333"
            ax.scatter(
                [round_id],
                [y],
                marker="s",
                s=620,
                facecolor=facecolor,
                edgecolor=color,
                linewidth=2.8,
                zorder=3,
            )
            ax.text(
                round_id,
                y,
                "S" if success else "F",
                ha="center",
                va="center",
                color=text_color,
                fontsize=14,
                fontweight="bold",
                zorder=4,
            )

    ax.set_title(f"{task_name}: Success by Round", fontsize=20)
    ax.set_xlabel("Round", fontsize=13)
    ax.set_xticks(rounds)
    ax.set_yticks(
        [y_positions[method_key] for method_key, _, _, _ in METHODS],
        [label for _, label, _, _ in METHODS],
    )
    ax.set_ylim(-0.55, len(METHODS) - 0.45)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=(
            Patch(facecolor="#2E7D32", edgecolor="#2E7D32", label="Success (1)"),
            Patch(facecolor="#E6E6E6", edgecolor="#777777", label="Failure (0)"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=2,
        frameon=False,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_summary_markdown(
    *,
    path: Path,
    overall: dict[str, dict[str, Any]],
    by_round: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# Main Experiment Results",
        "",
        "Strict success requires AndroidWorld evaluator success within the official "
        "per-task `int(10 * complexity)` action budget.",
        "",
        "## Overall",
        "",
        "| Method | Tasks | Strict Successes | Success Rate | Avg Tokens/Task | Avg Steps/Task | Runtime Errors | Final Memory Size |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method_key, _, _, _ in METHODS:
        item = overall[method_key]
        lines.append(
            "| {method} | {tasks} | {strict_successes} | "
            "{strict_success_rate:.2%} | {avg_tokens_per_task:.1f} | "
            "{avg_steps_per_task:.2f} | {runtime_errors} | "
            "{final_memory_size} |".format(**item)
        )
    lines.extend(
        [
            "",
            "## By Round",
            "",
            "| Method | Round | Successes | Success Rate | Avg Tokens/Task | Avg Steps/Task | Runtime Errors | End Memory Size |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method_key, _, _, _ in METHODS:
        for item in by_round[method_key]:
            lines.append(
                "| {method} | {round} | {strict_successes}/{tasks} | "
                "{strict_success_rate:.2%} | {avg_tokens_per_task:.1f} | "
                "{avg_steps_per_task:.2f} | {runtime_errors} | "
                "{end_memory_size} |".format(**item)
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_run_root(
    *,
    run_root: Path,
    output_dir: Path | None = None,
    expected_rounds: int = 5,
    expected_tasks_per_round: int = 5,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    output_dir = (output_dir or run_root / "figs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_method, _, task_names = _validate_run_root(
        run_root=run_root,
        expected_rounds=expected_rounds,
        expected_tasks_per_round=expected_tasks_per_round,
    )
    rounds = list(range(1, expected_rounds + 1))

    overall: dict[str, dict[str, Any]] = {}
    by_round: dict[str, list[dict[str, Any]]] = {}
    individual_task_success: list[dict[str, Any]] = []
    task_csv_rows: list[dict[str, Any]] = []
    error_audit: list[dict[str, Any]] = []

    for method_key, method_label, _, _ in METHODS:
        records = results_by_method[method_key]
        overall[method_key] = _overall_summary(
            records=records,
            method_label=method_label,
        )
        by_round[method_key] = _round_summaries(
            records=records,
            method_label=method_label,
        )
        for item in records:
            individual_task_success.append(
                {
                    "method_key": method_key,
                    "method": method_label,
                    "task_name": str(item["task_name"]),
                    "round": int(item["round"]),
                    "success": int(bool(item.get("success"))),
                }
            )
            task_csv_rows.append(
                {
                    "method": method_label,
                    "task_id": str(item["task_id"]),
                    "task_name": str(item["task_name"]),
                    "round": int(item["round"]),
                    "success": bool(item.get("success")),
                    "failure_category": str(item.get("failure_category") or ""),
                    "steps": int(item.get("steps", 0)),
                    "control_turns": item.get("control_turns", ""),
                    "event_count": item.get("event_count", ""),
                    "input_tokens": int(item.get("input_tokens", 0)),
                    "output_tokens": int(item.get("output_tokens", 0)),
                    "memory_size_after": int(item.get("memory_size_after", 0)),
                    "error": _render_error(item.get("error")),
                }
            )
            if _has_runtime_error(item):
                error_audit.append(
                    {
                        "method_key": method_key,
                        "method": method_label,
                        "task_id": str(item["task_id"]),
                        "task_name": str(item["task_name"]),
                        "round": int(item["round"]),
                        "failure_category": str(item.get("failure_category") or ""),
                        "error": _render_error(item.get("error")),
                    }
                )

    summary = {
        "overall": overall,
        "by_round": by_round,
        "individual_task_success_by_round": individual_task_success,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_summary_markdown(
        path=output_dir / "summary.md",
        overall=overall,
        by_round=by_round,
    )
    _write_csv(
        path=output_dir / "round_metrics.csv",
        fieldnames=ROUND_FIELDS,
        rows=(
            item
            for method_key, _, _, _ in METHODS
            for item in by_round[method_key]
        ),
    )
    _write_csv(
        path=output_dir / "task_results.csv",
        fieldnames=TASK_FIELDS,
        rows=task_csv_rows,
    )

    dms_timeline = [
        {
            "time": index,
            "round": int(item["round"]),
            "task_id": str(item["task_id"]),
            "task_name": str(item["task_name"]),
            "memory_size": int(item.get("memory_size_after", 0)),
        }
        for index, item in enumerate(results_by_method["dms"], start=1)
    ]
    _write_csv(
        path=output_dir / "dms_memory_timeline.csv",
        fieldnames=("time", "round", "task_id", "task_name", "memory_size"),
        rows=dms_timeline,
    )
    (output_dir / "task_error_audit.json").write_text(
        json.dumps(error_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _plot_method_lines(
        by_round=by_round,
        rounds=rounds,
        metric="strict_success_rate",
        title="Success Rate by Round (Strict AndroidWorld Budget)",
        ylabel="Success Rate (%)",
        output_path=output_dir / "success_rate_by_round.png",
        percent=True,
    )
    _plot_method_lines(
        by_round=by_round,
        rounds=rounds,
        metric="avg_tokens_per_task",
        title="Average Token Usage per Task by Round",
        ylabel="Average Tokens per Task",
        output_path=output_dir / "avg_tokens_per_task_by_round.png",
    )
    _plot_method_lines(
        by_round=by_round,
        rounds=rounds,
        metric="avg_steps_per_task",
        title="Average Execution Steps per Task by Round",
        ylabel="Average Steps per Task",
        output_path=output_dir / "avg_steps_per_task_by_round.png",
    )
    _plot_dms_memory(
        records=results_by_method["dms"],
        tasks_per_round=expected_tasks_per_round,
        output_path=output_dir / "dms_memory_size_timeline.png",
    )

    task_output_dir = output_dir / "task_success_by_round"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    for task_name in task_names:
        safe_name = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in task_name
        )
        _plot_task_success(
            task_name=task_name,
            rounds=rounds,
            results_by_method=results_by_method,
            output_path=task_output_dir / f"{safe_name}_success_by_round.png",
        )

    return {
        "output_dir": str(output_dir),
        "overall": overall,
        "figure_count": 4 + len(task_names),
        "task_names": task_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze one selected-five, three-method experiment RunRoot."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-rounds", type=int, default=5)
    parser.add_argument("--expected-tasks-per-round", type=int, default=5)
    args = parser.parse_args()
    if args.expected_rounds < 1 or args.expected_tasks_per_round < 1:
        parser.error("expected rounds and tasks per round must be positive")

    result = analyze_run_root(
        run_root=Path(args.run_root),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        expected_rounds=args.expected_rounds,
        expected_tasks_per_round=args.expected_tasks_per_round,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
