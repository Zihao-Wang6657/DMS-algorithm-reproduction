#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / "third_party" / "android_world"))
sys.path.insert(0, str(WORK / "src"))

from dms.android_tasks import TaskSpec, validate_specs
from dms.config import load_yaml
from dms.io_utils import now_iso, write_json
from dms.paths import workspace_path


def validate_dataset(path: Path) -> dict:
    data = load_yaml(path)
    specs = [TaskSpec.from_mapping(item) for item in data["tasks"]]
    summaries = validate_specs(specs)
    apps = sorted({app for summary in summaries for app in summary["app_names"]})
    return {
        "path": str(path.resolve()),
        "name": data.get("name"),
        "task_count": len(summaries),
        "apps": apps,
        "app_count": len(apps),
        "tasks": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
    )
    args = parser.parse_args()
    dataset_paths = args.dataset or [
        str(workspace_path("datasets", "bug_suite.yaml")),
        str(workspace_path("datasets", "mini_benchmark_20apps.yaml")),
    ]
    result = {
        "timestamp": now_iso(),
        "datasets": [validate_dataset(Path(path)) for path in dataset_paths],
    }
    output = workspace_path("logs", "dataset_validation.json")
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
