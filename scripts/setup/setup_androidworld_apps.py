#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK / "third_party" / "android_world"))
sys.path.insert(0, str(WORK / "src"))

from android_world.env import adb_utils
from android_world.env import env_launcher

from dms.config import apply_runtime_environment, load_yaml
from dms.io_utils import now_iso, write_json
from dms.paths import workspace_path


def main() -> None:
    runtime_path = workspace_path("configs", "runtime.yaml")
    runtime = load_yaml(runtime_path)
    apply_runtime_environment(runtime)

    result: dict[str, object] = {
        "timestamp": now_iso(),
        "runtime_config": str(runtime_path.resolve()),
        "androidworld_setup_completed": False,
        "installed_package_count": None,
        "error": None,
    }
    env = None
    try:
        env = env_launcher.load_and_setup_env(
            console_port=int(runtime["android"]["console_port"]),
            emulator_setup=True,
            adb_path=runtime["android"]["adb_path"],
            grpc_port=int(runtime["android"]["grpc_port"]),
        )
        packages = adb_utils.get_all_package_names(env.controller.env)
        result["installed_package_count"] = len(packages)
        result["androidworld_setup_completed"] = True
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        if env is not None:
            env.close()

    output = workspace_path("logs", "androidworld_setup.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["androidworld_setup_completed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
