from __future__ import annotations

import json
from pathlib import Path

from dms.formal_runner import (
    _aggregate_final_result,
    _recovery_action,
    _restore_memory,
    _snapshot_memory,
    verify_frozen_hashes,
)


def test_process_restart_never_retries_a_returned_normal_attempt() -> None:
    normal_failure = _attempt(
        number=1,
        success=False,
        steps=3,
        input_tokens=10,
        output_tokens=1,
    )
    assert _recovery_action(
        attempt_number=1,
        returned_record=normal_failure,
    ) == "finalize_returned"


def test_process_restart_retries_only_first_infrastructure_attempt() -> None:
    infrastructure_error = _attempt(
        number=1,
        success=False,
        steps=3,
        input_tokens=10,
        output_tokens=1,
        error="adb disconnected",
    )
    assert _recovery_action(
        attempt_number=1,
        returned_record=infrastructure_error,
    ) == "retry_after_restore"
    assert _recovery_action(
        attempt_number=1,
        returned_record=None,
    ) == "retry_after_restore"
    assert _recovery_action(
        attempt_number=2,
        returned_record=None,
    ) == "finalize_interrupted_after_restore"
    assert _recovery_action(
        attempt_number=2,
        returned_record=infrastructure_error,
    ) == "finalize_returned"


def _attempt(
    *,
    number: int,
    success: bool,
    steps: int,
    input_tokens: int,
    output_tokens: int,
    error: str | None = None,
) -> dict:
    return {
        "task_name": "ExampleTask",
        "goal": "Example goal",
        "attempt_number": number,
        "status": "normal_scored_attempt" if error is None else f"infrastructure_error_attempt_{number}",
        "success": success,
        "reward": 1.0 if success else 0.0,
        "steps": steps,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "memory_size_after": 3,
        "memory_stats": {},
        "trajectory": [],
        "error": error,
        "started_at": "start",
        "finished_at": "finish",
        "artifact_dir": "artifacts",
    }


def test_formal_retry_resources_are_cumulative_but_budget_uses_final_attempt() -> None:
    first = _attempt(
        number=1,
        success=False,
        steps=7,
        input_tokens=100,
        output_tokens=10,
        error="adb disconnected",
    )
    second = _attempt(
        number=2,
        success=True,
        steps=4,
        input_tokens=200,
        output_tokens=20,
    )
    result = _aggregate_final_result(
        canonical_task_id="r01_000_ExampleTask",
        final_attempt=second,
        attempts=[first, second],
        round_id=1,
        budget=10,
        protocol_id="formal_main_20apps_v1",
    )
    assert result["success"] is True
    assert result["scoring_attempt_steps"] == 4
    assert result["steps"] == 11
    assert result["input_tokens"] == 300
    assert result["output_tokens"] == 30
    assert result["control_turns"] == 0
    assert result["infrastructure_retry_overhead_steps"] == 7
    assert result["infrastructure_retry_overhead_tokens"] == 110


def test_formal_budget_is_strictly_less_than_official_budget() -> None:
    attempt = _attempt(
        number=1,
        success=True,
        steps=10,
        input_tokens=100,
        output_tokens=10,
    )
    result = _aggregate_final_result(
        canonical_task_id="r01_000_ExampleTask",
        final_attempt=attempt,
        attempts=[attempt],
        round_id=1,
        budget=10,
        protocol_id="formal_main_20apps_v1",
    )
    assert result["success"] is False
    assert result["failure_category"] == "model_failure"


def test_second_infrastructure_error_is_disclosed_and_not_retried() -> None:
    first = _attempt(
        number=1,
        success=False,
        steps=2,
        input_tokens=10,
        output_tokens=1,
        error="first infrastructure error",
    )
    second = _attempt(
        number=2,
        success=False,
        steps=3,
        input_tokens=20,
        output_tokens=2,
        error="second infrastructure error",
    )
    result = _aggregate_final_result(
        canonical_task_id="r01_000_ExampleTask",
        final_attempt=second,
        attempts=[first, second],
        round_id=1,
        budget=10,
        protocol_id="formal_main_20apps_v1",
    )
    assert result["success"] is False
    assert result["attempt_count"] == 2
    assert result["failure_category"] == "infrastructure_failure_after_retry"


def test_memory_checkpoint_restores_and_removes_attempt_changes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint = tmp_path / "checkpoint"
    run_dir.mkdir()
    (run_dir / "static_memory.jsonl").write_text("before\n", encoding="utf-8")
    _snapshot_memory(memory=None, run_dir=run_dir, checkpoint_dir=checkpoint)
    (run_dir / "static_memory.jsonl").write_text("after\n", encoding="utf-8")
    (run_dir / "dms_events.jsonl").write_text("attempt-only\n", encoding="utf-8")
    _restore_memory(run_dir=run_dir, checkpoint_dir=checkpoint)
    assert (run_dir / "static_memory.jsonl").read_text(encoding="utf-8") == "before\n"
    assert not (run_dir / "dms_events.jsonl").exists()


def test_frozen_hash_verification_detects_drift(tmp_path: Path) -> None:
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("frozen", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(tracked.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol_id": "formal_main_20apps_v1",
                "workspace_root": str(tmp_path),
                "sha256": {"tracked.txt": digest},
            }
        ),
        encoding="utf-8",
    )
    assert verify_frozen_hashes(manifest) == "formal_main_20apps_v1"
    tracked.write_text("drifted", encoding="utf-8")
    try:
        verify_frozen_hashes(manifest)
    except RuntimeError as exc:
        assert "hash verification failed" in str(exc)
    else:
        raise AssertionError("Expected protocol drift to be detected")
