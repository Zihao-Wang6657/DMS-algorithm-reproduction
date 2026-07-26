#!/usr/bin/env python
from __future__ import annotations

import json
import subprocess
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK / "third_party" / "android_world"))
sys.path.insert(0, str(WORK / "src"))

from android_world.env import env_launcher

from dms.config import apply_runtime_environment, load_yaml
from dms.io_utils import now_iso, write_json
from dms.paths import workspace_path


def run_adb(adb: str, *args: str) -> str:
    return subprocess.run(
        [adb, *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def main() -> None:
    runtime_path = workspace_path("configs", "runtime.yaml")
    runtime = load_yaml(runtime_path)
    apply_runtime_environment(runtime)
    adb = runtime["android"]["adb_path"]
    env = None
    result = {
        "timestamp": now_iso(),
        "runtime_config": str(runtime_path.resolve()),
        "adb_devices": None,
        "boot_completed": None,
        "foreground_activity": None,
        "logical_screen_size": None,
        "ui_element_count": None,
        "androidworld_env_connected": False,
        "error": None,
    }
    try:
        result["adb_devices"] = run_adb(adb, "devices", "-l")
        result["boot_completed"] = run_adb(
            adb, "shell", "getprop", "sys.boot_completed"
        ).strip()
        env = env_launcher.load_and_setup_env(
            console_port=int(runtime["android"]["console_port"]),
            emulator_setup=False,
            adb_path=adb,
            grpc_port=int(runtime["android"]["grpc_port"]),
        )
        state = env.reset(go_home=True)
        result["foreground_activity"] = env.foreground_activity_name
        result["logical_screen_size"] = list(env.logical_screen_size)
        result["ui_element_count"] = len(state.ui_elements)
        result["androidworld_env_connected"] = True
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        if env is not None:
            env.close()
    output = workspace_path("logs", "androidworld_env_check.json")
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["androidworld_env_connected"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
