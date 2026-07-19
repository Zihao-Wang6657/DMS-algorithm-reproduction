from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path


PROTOCOL_ID = "formal_recovery_bd_20apps_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--preflight-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    base_path = Path(args.base_manifest).resolve()
    run_root = Path(args.run_root).resolve()
    preflight = Path(args.preflight_summary).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = copy.deepcopy(json.loads(base_path.read_text(encoding="utf-8")))
    original_run_root = Path(manifest["formal_run_root"]).resolve()
    baseline_a_dir = original_run_root / "baseline_a"
    baseline_a_files = [
        baseline_a_dir / "task_results.jsonl",
        baseline_a_dir / "attempt_results.jsonl",
        baseline_a_dir / "metrics.json",
    ]
    for path in [preflight, *baseline_a_files]:
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest.update(
        {
            "protocol_id": PROTOCOL_ID,
            "frozen_at": datetime.now().astimezone().isoformat(),
            "formal_run_root": str(run_root),
            "rounds": 5,
            "tasks_per_round": 20,
            "method_order": [
                "baseline_b_static_memory",
                "dms_hierarchical_memory",
            ],
            "scored_record_count": 200,
            "cross_run_design": {
                "enabled": True,
                "baseline_a_reused": True,
                "baseline_a_run_root": str(original_run_root),
                "baseline_a_scored_records": 100,
                "baseline_b_and_dms_run_root": str(run_root),
                "disclosure": (
                    "Baseline A is reused from formal_main_20apps_v1. Baseline B "
                    "and DMS are rerun from empty memory in this recovery RunRoot. "
                    "Any three-method figure is therefore a disclosed cross-run comparison."
                ),
            },
            "excluded_prior_results": {
                "baseline_b": str(original_run_root / "baseline_b"),
                "reason": (
                    "AndroidWorld became persistently unavailable after r02_017; "
                    "the final 62 Baseline B records were consecutive infrastructure failures."
                ),
            },
            "infrastructure_round_isolation": {
                "emulator_cold_restart_before_each_round": True,
                "formal_runner_process_isolated_per_round": True,
                "round_limit_sequence": [1, 2, 3, 4, 5],
                "memory_persists_in_method_run_dir": True,
                "task_retry_rule_unchanged": True,
                "algorithm_prompt_model_budget_and_evaluator_unchanged": True,
                "purpose": (
                    "Prevent recurrence of the long-lived AndroidWorld reset failure "
                    "observed in the excluded Baseline B run."
                ),
            },
            "recovery_preflight_summary": relative(root, preflight),
            "preflight_summary": relative(root, preflight),
        }
    )

    paths = {
        (root / rel).resolve()
        for rel in manifest["sha256"]
        if (root / rel).is_file()
    }
    paths.update(
        {
            root / "scripts/freeze_recovery_bd_protocol.py",
            root / "scripts/run_formal_bd_recovery_windows.ps1",
            root / "scripts/start_local_androidworld.ps1",
            root / "src/dms/formal_runner.py",
            preflight,
            *baseline_a_files,
        }
    )
    manifest["sha256"] = {
        relative(root, path): sha256(path)
        for path in sorted(paths, key=lambda item: str(item).lower())
    }

    manifest_path = output_dir / "protocol_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "sha256sums.txt").write_text(
        "\n".join(
            f"{digest}  {path}" for path, digest in manifest["sha256"].items()
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "PROTOCOL.md").write_text(
        "\n".join(
            [
                "# Formal Recovery Experiment: Baseline B + DMS",
                "",
                f"- Protocol ID: `{PROTOCOL_ID}`",
                "- Scale: 20 tasks x 5 rounds x 2 methods = 200 scored records",
                "- Methods: Baseline B, then DMS; both memories start empty",
                "- Baseline A: reused from the v1 RunRoot and disclosed as cross-run",
                "- Frozen tasks, seeds, order, budgets, model, Prompt and evaluator: unchanged",
                "- Task-level infrastructure retry: unchanged, at most once",
                "- Infrastructure isolation: cold restart emulator before every round",
                "- Prior failed Baseline B data: excluded from recovered formal results",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), "protocol_id": PROTOCOL_ID}))


if __name__ == "__main__":
    main()
