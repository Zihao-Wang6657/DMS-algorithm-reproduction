#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK / "third_party" / "android_world"))
sys.path.insert(0, str(WORK / "src"))

import yaml

from dms.android_tasks import TaskSpec, generate_task_spec, task_summary, validate_specs
from dms.io_utils import now_iso, write_json
from dms.paths import workspace_path


BUG_TASKS = [
    ("OpenAppTaskEval", {"app_name": "settings"}),
    ("SystemWifiTurnOnVerify", {}),
    ("ClockStopWatchPausedVerify", {}),
]


MINI_BENCHMARK_TASKS = [
    ("AudioRecorderRecordAudio", "audio recorder"),
    ("RecipeAddSingleRecipe", "broccoli app"),
    ("CameraTakePhoto", "camera"),
    ("BrowserDraw", "chrome"),
    ("ClockStopWatchRunning", "clock"),
    ("ContactsNewContactDraft", "contacts"),
    ("FilesDeleteFile", "files"),
    ("NotesTodoItemCount", "joplin"),
    ("MarkorCreateFolder", "markor"),
    ("SportsTrackerActivityDuration", "open tracks sports tracker"),
    ("OsmAndFavorite", "osmand"),
    ("ExpenseAddSingle", "pro expense"),
    ("RetroCreatePlaylist", "retro music"),
    ("SimpleCalendarDeleteOneEvent", "simple calendar pro"),
    ("SimpleDrawProCreateDrawing", "simple draw pro"),
    ("SaveCopyOfReceiptTaskEval", "simple gallery pro"),
    ("SimpleSmsSend", "simple sms messenger"),
    ("SystemWifiTurnOn", "settings"),
    ("TasksDueOnDate", "tasks"),
    ("VlcCreatePlaylist", "vlc"),
]


def _with_fixed_params(name: str, seed: int, fixed_params: dict[str, object]) -> dict:
    return {"name": name, "seed": seed, "param_overrides": fixed_params}


def _generated(name: str, seed: int) -> dict:
    return {"name": name, "seed": seed}


def build_bug_suite(seed: int) -> dict:
    tasks = []
    for index, (name, fixed) in enumerate(BUG_TASKS):
        tasks.append(_with_fixed_params(name, seed + index, fixed))
    return {
        "name": "bug_suite",
        "created_at": now_iso(),
        "description": "Small simple task set for debugging AndroidWorld, Qwen, and action execution issues.",
        "source": "AndroidWorld task registry; selected for low complexity.",
        "tasks": tasks,
    }


def build_mini_benchmark(seed: int) -> dict:
    tasks = []
    app_coverage = OrderedDict()
    for index, (name, app) in enumerate(MINI_BENCHMARK_TASKS):
        task_seed = seed + index
        if name == "OpenAppTaskEval":
            item = _with_fixed_params(name, task_seed, {"app_name": app})
        else:
            item = _generated(name, task_seed)
        tasks.append(item)
        app_coverage.setdefault(app, []).append(name)
    return {
        "name": "mini_benchmark_20apps",
        "created_at": now_iso(),
        "description": "Mini-benchmark covering the real AndroidWorld app scenarios with one representative task per app scenario.",
        "source": "AndroidWorld task registry. Parameters are generated with fixed seeds.",
        "app_coverage": dict(app_coverage),
        "tasks": tasks,
    }


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=30)
    args = parser.parse_args()

    bug_suite = build_bug_suite(args.seed)
    mini_benchmark = build_mini_benchmark(args.seed + 1000)
    bug_path = workspace_path("datasets", "bug_suite.yaml")
    mini_path = workspace_path("datasets", "mini_benchmark_20apps.yaml")
    write_yaml(bug_path, bug_suite)
    write_yaml(mini_path, mini_benchmark)

    validation = {
        "timestamp": now_iso(),
        "bug_suite": {
            "path": str(bug_path.resolve()),
            "tasks": validate_specs([
                TaskSpec.from_mapping(item)
                for item in bug_suite["tasks"]
            ]),
        },
        "mini_benchmark": {
            "path": str(mini_path.resolve()),
            "declared_app_coverage": mini_benchmark["app_coverage"],
            "tasks": [task_summary(TaskSpec.from_mapping(item)) for item in mini_benchmark["tasks"]],
        },
    }
    write_json(workspace_path("logs", "dataset_build.json"), validation)
    print(f"Wrote {bug_path}")
    print(f"Wrote {mini_path}")


if __name__ == "__main__":
    main()
