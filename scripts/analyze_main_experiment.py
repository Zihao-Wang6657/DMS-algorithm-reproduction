#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import yaml


METHODS = (
    ("baseline_a", "Baseline A", "#4C78A8", "o"),
    ("baseline_b", "Baseline B", "#F58518", "s"),
    ("dms", "DMS", "#54A24B", "^"),
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _task_budgets(dataset_path: Path) -> dict[str, int]:
    from dms.android_tasks import TaskSpec, instantiate_task

    dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    budgets: dict[str, int] = {}
    for item in dataset["tasks"]:
        task = instantiate_task(TaskSpec.from_mapping(item))
        budgets[str(task.name)] = int(10 * task.complexity)
    return budgets


def _strict_success(item: dict[str, Any], budgets: dict[str, int]) -> bool:
    task_name = str(item["task_name"])
    if task_name not in budgets:
        raise KeyError(f"No AndroidWorld step budget for {task_name}")
    if item.get("protocol_id"):
        scoring_steps = int(item.get("scoring_attempt_steps", item.get("steps", 0)))
        return bool(item.get("success")) and scoring_steps < budgets[task_name]
    return bool(item.get("success")) and int(item.get("steps", 0)) <= budgets[task_name]


def _summarize_method(
    results: list[dict[str, Any]],
    *,
    label: str,
    budgets: dict[str, int],
    expected_rounds: int,
    expected_tasks_per_round: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_ids = [str(item["task_id"]) for item in results]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"{label} contains duplicate task IDs")

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        grouped[int(item["round"])].append(item)

    expected_round_ids = list(range(1, expected_rounds + 1))
    if sorted(grouped) != expected_round_ids:
        raise ValueError(
            f"{label} rounds are {sorted(grouped)}, expected {expected_round_ids}"
        )

    by_round: list[dict[str, Any]] = []
    for round_id in expected_round_ids:
        items = grouped[round_id]
        if len(items) != expected_tasks_per_round:
            raise ValueError(
                f"{label} round {round_id} has {len(items)} tasks, "
                f"expected {expected_tasks_per_round}"
            )
        strict_successes = sum(_strict_success(item, budgets) for item in items)
        input_tokens = sum(int(item.get("input_tokens", 0)) for item in items)
        output_tokens = sum(int(item.get("output_tokens", 0)) for item in items)
        total_tokens = input_tokens + output_tokens
        total_steps = sum(int(item.get("steps", 0)) for item in items)
        by_round.append(
            {
                "method": label,
                "round": round_id,
                "tasks": len(items),
                "strict_successes": strict_successes,
                "strict_success_rate": strict_successes / len(items),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "avg_tokens_per_task": total_tokens / len(items),
                "total_steps": total_steps,
                "avg_steps_per_task": total_steps / len(items),
                "runtime_errors": sum(bool(item.get("error")) for item in items),
                "infrastructure_retries": sum(
                    int(item.get("infrastructure_retry_count", 0)) for item in items
                ),
                "infrastructure_failure_after_retry": sum(
                    item.get("failure_category")
                    == "infrastructure_failure_after_retry"
                    for item in items
                ),
                "end_memory_size": int(items[-1].get("memory_size_after", 0)),
            }
        )

    total_tasks = len(results)
    strict_successes = sum(_strict_success(item, budgets) for item in results)
    total_tokens = sum(
        int(item.get("input_tokens", 0)) + int(item.get("output_tokens", 0))
        for item in results
    )
    total_steps = sum(int(item.get("steps", 0)) for item in results)
    overall = {
        "method": label,
        "tasks": total_tasks,
        "strict_successes": strict_successes,
        "strict_success_rate": strict_successes / total_tasks,
        "total_tokens": total_tokens,
        "avg_tokens_per_task": total_tokens / total_tasks,
        "total_steps": total_steps,
        "avg_steps_per_task": total_steps / total_tasks,
        "runtime_errors": sum(bool(item.get("error")) for item in results),
        "infrastructure_retries": sum(
            int(item.get("infrastructure_retry_count", 0)) for item in results
        ),
        "infrastructure_failure_after_retry": sum(
            item.get("failure_category") == "infrastructure_failure_after_retry"
            for item in results
        ),
        "final_memory_size": int(results[-1].get("memory_size_after", 0)),
    }
    return overall, by_round


def _plot_method_lines(
    *,
    summaries: dict[str, list[dict[str, Any]]],
    key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    percent: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for method_key, label, color, marker in METHODS:
        values = [float(item[key]) for item in summaries[method_key]]
        if percent:
            values = [value * 100 for value in values]
        rounds = [int(item["round"]) for item in summaries[method_key]]
        ax.plot(
            rounds,
            values,
            label=label,
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2.3,
        )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    all_rounds = sorted(
        {int(item["round"]) for items in summaries.values() for item in items}
    )
    ax.set_xticks(all_rounds)
    if percent:
        ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.32)
    ax.legend(frameon=False, ncol=3, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_dms_memory(
    results: list[dict[str, Any]], output_path: Path, tasks_per_round: int
) -> None:
    times = list(range(1, len(results) + 1))
    sizes = [int(item.get("memory_size_after", 0)) for item in results]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.step(times, sizes, where="post", color="#54A24B", linewidth=2.2, label="DMS")
    ax.scatter(times, sizes, color="#54A24B", s=13, alpha=0.75)
    for boundary in range(tasks_per_round, len(results), tasks_per_round):
        ax.axvline(boundary, color="#888888", linestyle=":", linewidth=1, alpha=0.65)
    ax.set_title("DMS Memory Size over Task Attempts", fontsize=14)
    ax.set_xlabel("Time (one completed task attempt)")
    ax.set_ylabel("Number of Memories")
    ax.set_xlim(1, len(results))
    ax.set_xticks(list(range(0, len(results) + 1, tasks_per_round)))
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.32)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _task_success_rows(
    results: dict[str, list[dict[str, Any]]],
    budgets: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_key, label, _, _ in METHODS:
        for item in results[method_key]:
            rows.append(
                {
                    "method_key": method_key,
                    "method": label,
                    "task_name": str(item["task_name"]),
                    "round": int(item["round"]),
                    "success": int(_strict_success(item, budgets)),
                }
            )
    return rows


def _plot_individual_tasks(
    *,
    rows: list[dict[str, Any]],
    task_order: list[str],
    rounds: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lookup = {
        (str(row["method_key"]), str(row["task_name"]), int(row["round"])): int(row["success"])
        for row in rows
    }
    for task_name in task_order:
        fig, ax = plt.subplots(figsize=(8.8, 3.6))
        xs = list(range(1, rounds + 1))
        y_positions = {
            method_key: 2 - index
            for index, (method_key, _, _, _) in enumerate(METHODS)
        }
        for method_key, _, color, _ in METHODS:
            y = y_positions[method_key]
            ax.plot(xs, [y] * len(xs), color=color, linewidth=1.7, alpha=0.7, zorder=1)
            for round_id in xs:
                success = lookup[(method_key, task_name, round_id)]
                ax.scatter(
                    round_id,
                    y,
                    s=520,
                    marker="s",
                    facecolor="#2E7D32" if success else "#E0E0E0",
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
                    color="white" if success else "#333333",
                    fontsize=11,
                    fontweight="bold",
                    zorder=3,
                )
        ax.set_title(f"{task_name}: Success by Round", fontsize=14)
        ax.set_xlabel("Round")
        ax.set_xticks(xs)
        ax.set_xlim(min(xs) - 0.45, max(xs) + 0.45)
        ax.set_yticks(
            [y_positions[method_key] for method_key, _, _, _ in METHODS],
            labels=[label for _, label, _, _ in METHODS],
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


def _write_summary(
    output_path: Path,
    overall: dict[str, dict[str, Any]],
    by_round: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# Main Experiment Results",
        "",
        "Strict success requires both AndroidWorld success and completion within "
        "the official per-task `int(10 * complexity)` action budget.",
        "",
        "## Overall",
        "",
        "| Method | Tasks | Strict Successes | Success Rate | Avg Tokens/Task | Avg Steps/Task | Infra Retries | Infra Failure After Retry | Final Memory Size |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method_key, _, _, _ in METHODS:
        item = overall[method_key]
        lines.append(
            f"| {item['method']} | {item['tasks']} | {item['strict_successes']} | "
            f"{item['strict_success_rate']:.2%} | {item['avg_tokens_per_task']:.1f} | "
            f"{item['avg_steps_per_task']:.2f} | {item['infrastructure_retries']} | "
            f"{item['infrastructure_failure_after_retry']} | "
            f"{item['final_memory_size']} |"
        )
    lines.extend(
        [
            "",
            "## By Round",
            "",
            "| Method | Round | Successes | Success Rate | Avg Tokens/Task | Avg Steps/Task | Infra Retries | Infra Failure After Retry | End Memory Size |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method_key, _, _, _ in METHODS:
        for item in by_round[method_key]:
            lines.append(
                f"| {item['method']} | {item['round']} | {item['strict_successes']}/{item['tasks']} | "
                f"{item['strict_success_rate']:.2%} | {item['avg_tokens_per_task']:.1f} | "
                f"{item['avg_steps_per_task']:.2f} | {item['infrastructure_retries']} | "
                f"{item['infrastructure_failure_after_retry']} | "
                f"{item['end_memory_size']} |"
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_readme(
    *,
    readme_path: Path,
    repo_root: Path,
    output_dir: Path,
    overall: dict[str, dict[str, Any]],
    by_round: dict[str, list[dict[str, Any]]],
    tasks_per_round: int,
    rounds: int,
    task_order: list[str],
) -> None:
    start_marker = "<!-- MAIN_EXPERIMENT_RESULTS_START -->"
    end_marker = "<!-- MAIN_EXPERIMENT_RESULTS_END -->"
    relative_output = output_dir.relative_to(repo_root).as_posix()
    total_infrastructure_retries = sum(
        item["infrastructure_retries"] for item in overall.values()
    )
    total_infrastructure_failures = sum(
        item["infrastructure_failure_after_retry"] for item in overall.values()
    )
    lines = [
        start_marker,
        f"## 平衡 Mini Benchmark 结果（{tasks_per_round} Tasks × {rounds} Rounds）",
        "",
        "本节数据来自本机 AndroidWorld 模拟器与 AutoDL Qwen2.5-VL-7B-Instruct",
        f"远程推理实验。每种方法运行 {tasks_per_round} 个固定任务，共 {rounds} 轮、{tasks_per_round * rounds} 次",
        f"任务尝试；三种方法合计 {tasks_per_round * rounds * 3} 次。成功率采用严格口径：AndroidWorld 判定成功且",
        "最终计分尝试的动作数严格小于该任务的官方 `int(10 × complexity)` 预算。",
        "",
        "| Method | Strict Successes | Success Rate | Avg Tokens/Task | Avg Steps/Task | Infra Retries | Infra Failure After Retry | Final Memory Size |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method_key, _, _, _ in METHODS:
        item = overall[method_key]
        lines.append(
            f"| {item['method']} | {item['strict_successes']}/{item['tasks']} | "
            f"{item['strict_success_rate']:.2%} | {item['avg_tokens_per_task']:.1f} | "
            f"{item['avg_steps_per_task']:.2f} | {item['infrastructure_retries']} | "
            f"{item['infrastructure_failure_after_retry']} | "
            f"{item['final_memory_size']} |"
        )
    baseline_a = overall["baseline_a"]
    baseline_b = overall["baseline_b"]
    dms = overall["dms"]
    token_reduction_a = 100 * (
        1 - dms["avg_tokens_per_task"] / baseline_a["avg_tokens_per_task"]
    )
    token_reduction_b = 100 * (
        1 - dms["avg_tokens_per_task"] / baseline_b["avg_tokens_per_task"]
    )
    step_reduction_a = 100 * (
        1 - dms["avg_steps_per_task"] / baseline_a["avg_steps_per_task"]
    )
    step_reduction_b = 100 * (
        1 - dms["avg_steps_per_task"] / baseline_b["avg_steps_per_task"]
    )
    lines.extend(
        [
            "",
            f"从严格成功率看，DMS 的 {dms['strict_success_rate']:.0%} 低于 Baseline A 的 "
            f"{baseline_a['strict_success_rate']:.0%} 和 Baseline B 的 "
            f"{baseline_b['strict_success_rate']:.0%}，因此本次 mini benchmark "
            "**没有验证 DMS 能提高任务成功率**。DMS 的优势体现在自调节效率：平均 "
            f"Token 用量比 Baseline A 低 {token_reduction_a:.1f}%、比 Baseline B 低 "
            f"{token_reduction_b:.1f}%，平均动作数分别低 {step_reduction_a:.1f}% 和 "
            f"{step_reduction_b:.1f}%；25 次任务后仅保留 "
            f"{dms['final_memory_size']} 条记忆，而静态记忆保留 "
            f"{baseline_b['final_memory_size']} 条。结果支持动态剪枝和上下文压缩机制有效，"
            "但不能把这种资源效率解释为整体任务能力提升。",
            "",
            f"本次共有 **{total_infrastructure_retries}** 次基础设施重试；其中第二次仍异常并按失败计分的任务数为 **{total_infrastructure_failures}**。",
            "Token 和 Step 均累计首次异常尝试与最终计分尝试的消耗。逐轮数值、原始 CSV 和",
            f"严格统计定义见 [`{relative_output}/summary.md`]({relative_output}/summary.md)。",
            "",
            "### 1. 三种算法的成功率随轮数变化",
            "",
            f"![Success rate by round]({relative_output}/success_rate_by_round.png)",
            "",
            "### 2. 三种算法的平均单任务 Token 用量随轮数变化",
            "",
            f"![Average tokens per task by round]({relative_output}/avg_tokens_per_task_by_round.png)",
            "",
            "### 3. 三种算法的平均单任务 Step 数随轮数变化",
            "",
            f"![Average steps per task by round]({relative_output}/avg_steps_per_task_by_round.png)",
            "",
            "### 4. DMS 记忆库大小随任务时间变化",
            "",
            f"横轴中的一个时间单位对应一个已经完成的任务尝试；每 {tasks_per_round} 个时间单位为一轮。",
            "",
            f"![DMS memory size timeline]({relative_output}/dms_memory_size_timeline.png)",
            "",
            "### 5. 五个单独任务的逐轮成功/失败（三种算法）",
            "",
            *[
                f"![{task_name} success by round]({relative_output}/task_success_by_round/{task_name}_success_by_round.png)"
                for task_name in task_order
            ],
            "",
            end_marker,
        ]
    )
    block = "\n".join(lines)
    readme = readme_path.read_text(encoding="utf-8")
    if start_marker in readme and end_marker in readme:
        prefix, rest = readme.split(start_marker, 1)
        _, suffix = rest.split(end_marker, 1)
        updated = prefix.rstrip() + "\n\n" + block + suffix
    else:
        anchor = "## 上游参考结果"
        if anchor not in readme:
            raise ValueError(f"README anchor is missing: {anchor}")
        updated = readme.replace(anchor, block + "\n\n" + anchor, 1)
    readme_path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--tasks-per-round", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    dataset_path = Path(args.dataset).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_root / "results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = _task_budgets(dataset_path)
    dataset_payload = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    task_order = [str(item["name"]) for item in dataset_payload["tasks"]]
    tasks_per_round = args.tasks_per_round or len(task_order)

    results: dict[str, list[dict[str, Any]]] = {}
    overall: dict[str, dict[str, Any]] = {}
    by_round: dict[str, list[dict[str, Any]]] = {}
    for method_key, label, _, _ in METHODS:
        method_results = _load_jsonl(run_root / method_key / "task_results.jsonl")
        results[method_key] = method_results
        overall[method_key], by_round[method_key] = _summarize_method(
            method_results,
            label=label,
            budgets=budgets,
            expected_rounds=args.rounds,
            expected_tasks_per_round=tasks_per_round,
        )

    task_success_rows = _task_success_rows(results, budgets)

    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "overall": overall,
                "by_round": by_round,
                "individual_task_success_by_round": task_success_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "round_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fieldnames = list(next(iter(by_round.values()))[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method_key, _, _, _ in METHODS:
            writer.writerows(by_round[method_key])

    task_result_fields = (
        "method", "task_id", "task_name", "round", "success", "failure_category",
        "steps", "control_turns", "event_count", "input_tokens", "output_tokens",
        "memory_size_after", "attempt_count", "infrastructure_retry_count",
    )
    with (output_dir / "task_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=task_result_fields)
        writer.writeheader()
        for method_key, label, _, _ in METHODS:
            for item in results[method_key]:
                writer.writerow({
                    field: (label if field == "method" else item.get(field))
                    for field in task_result_fields
                })

    retry_rows = []
    for method_key, label, _, _ in METHODS:
        for item in results[method_key]:
            if int(item.get("attempt_count", 1)) > 1 or item.get("error"):
                retry_rows.append({
                    "method": label,
                    "task_id": item.get("task_id"),
                    "task_name": item.get("task_name"),
                    "round": item.get("round"),
                    "attempt_count": item.get("attempt_count", 1),
                    "infrastructure_retry_count": item.get("infrastructure_retry_count", 0),
                    "failure_category": item.get("failure_category"),
                    "final_error": item.get("error"),
                })
    (output_dir / "retry_infrastructure_audit.json").write_text(
        json.dumps(retry_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (output_dir / "dms_memory_timeline.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("time", "round", "task_id", "task_name", "memory_size"),
        )
        writer.writeheader()
        for index, item in enumerate(results["dms"], start=1):
            writer.writerow(
                {
                    "time": index,
                    "round": item["round"],
                    "task_id": item["task_id"],
                    "task_name": item["task_name"],
                    "memory_size": item.get("memory_size_after", 0),
                }
            )

    _write_summary(output_dir / "summary.md", overall, by_round)
    _plot_method_lines(
        summaries=by_round,
        key="strict_success_rate",
        ylabel="Success Rate (%)",
        title="Success Rate by Round (Strict AndroidWorld Budget)",
        output_path=output_dir / "success_rate_by_round.png",
        percent=True,
    )
    _plot_method_lines(
        summaries=by_round,
        key="avg_tokens_per_task",
        ylabel="Average Tokens per Task",
        title="Average Token Usage per Task by Round",
        output_path=output_dir / "avg_tokens_per_task_by_round.png",
    )
    _plot_method_lines(
        summaries=by_round,
        key="avg_steps_per_task",
        ylabel="Average Steps per Task",
        title="Average Steps per Task by Round",
        output_path=output_dir / "avg_steps_per_task_by_round.png",
    )
    _plot_dms_memory(
        results["dms"],
        output_dir / "dms_memory_size_timeline.png",
        tasks_per_round,
    )
    _plot_individual_tasks(
        rows=task_success_rows,
        task_order=task_order,
        rounds=args.rounds,
        output_dir=output_dir / "task_success_by_round",
    )
    repo_root = dataset_path.parent.parent
    _update_readme(
        readme_path=repo_root / "README.md",
        repo_root=repo_root,
        output_dir=output_dir,
        overall=overall,
        by_round=by_round,
        tasks_per_round=tasks_per_round,
        rounds=args.rounds,
        task_order=task_order,
    )
    print(json.dumps({"output_dir": str(output_dir), "overall": overall}, indent=2))


if __name__ == "__main__":
    main()
