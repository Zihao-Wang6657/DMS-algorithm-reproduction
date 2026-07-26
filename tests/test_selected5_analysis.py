from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analysis.analyze_selected5_all_methods import analyze_run_root


TASK_NAMES = (
    "AudioRecorderRecordAudio",
    "RecipeAddSingleRecipe",
    "CameraTakePhoto",
    "BrowserDraw",
    "ClockStopWatchRunning",
)


def _write_method_results(
    *,
    run_root: Path,
    method_key: str,
    method_index: int,
) -> None:
    method_dir = run_root / method_key
    method_dir.mkdir(parents=True)
    records = []
    memory_size = 0
    for round_id in range(1, 6):
        for task_index, task_name in enumerate(TASK_NAMES):
            success = (round_id + task_index + method_index) % 3 == 0
            if method_key == "baseline_b":
                memory_size += 1
            elif method_key == "dms" and success:
                memory_size += 1
            record = {
                "task_id": f"r{round_id:02d}_{task_index:03d}_{task_name}",
                "task_name": task_name,
                "round": round_id,
                "success": success,
                "steps": 2 + task_index,
                "input_tokens": 1000 + round_id * 10 + task_index,
                "output_tokens": 50 + method_index,
                "memory_size_after": memory_size,
                "error": None,
            }
            records.append(record)

    (method_dir / "task_results.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    successful = sum(bool(record["success"]) for record in records)
    metrics = {
        "method": method_key,
        "rounds": 5,
        "tasks": len(records),
        "successful_tasks": successful,
        "success_rate": successful / len(records),
        "total_steps": sum(int(record["steps"]) for record in records),
        "input_tokens": sum(int(record["input_tokens"]) for record in records),
        "output_tokens": sum(int(record["output_tokens"]) for record in records),
        "memory_size": memory_size,
        "results": records,
    }
    (method_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


def _write_complete_run_root(run_root: Path) -> None:
    for method_index, method_key in enumerate(("baseline_a", "baseline_b", "dms")):
        _write_method_results(
            run_root=run_root,
            method_key=method_key,
            method_index=method_index,
        )


def test_selected5_analysis_generates_complete_artifact_set(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_complete_run_root(run_root)

    result = analyze_run_root(run_root=run_root)

    output_dir = run_root / "figs"
    expected_figures = {
        output_dir / "success_rate_by_round.png",
        output_dir / "avg_tokens_per_task_by_round.png",
        output_dir / "avg_steps_per_task_by_round.png",
        output_dir / "dms_memory_size_timeline.png",
        *{
            output_dir / "task_success_by_round" / f"{task_name}_success_by_round.png"
            for task_name in TASK_NAMES
        },
    }
    assert result["figure_count"] == 9
    assert result["task_names"] == list(TASK_NAMES)
    assert all(path.is_file() and path.stat().st_size > 1000 for path in expected_figures)

    for filename in (
        "summary.json",
        "summary.md",
        "round_metrics.csv",
        "task_results.csv",
        "dms_memory_timeline.csv",
        "task_error_audit.json",
    ):
        assert (output_dir / filename).is_file()

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["overall"]["baseline_a"]["tasks"] == 25
    assert len(summary["by_round"]["dms"]) == 5
    assert len(summary["individual_task_success_by_round"]) == 75
    assert (
        json.loads((output_dir / "task_error_audit.json").read_text(encoding="utf-8"))
        == []
    )
    assert len((output_dir / "round_metrics.csv").read_text().splitlines()) == 16
    assert len((output_dir / "task_results.csv").read_text().splitlines()) == 76


def test_selected5_analysis_rejects_incomplete_method(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_complete_run_root(run_root)
    results_path = run_root / "dms" / "task_results.jsonl"
    lines = results_path.read_text(encoding="utf-8").splitlines()
    results_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly 25 task results"):
        analyze_run_root(run_root=run_root)


def test_selected5_analysis_rejects_duplicate_task_name(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_complete_run_root(run_root)
    results_path = run_root / "baseline_a" / "task_results.jsonl"
    records = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    ]
    records[1]["task_name"] = records[0]["task_name"]
    results_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate task names"):
        analyze_run_root(run_root=run_root)
