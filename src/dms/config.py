from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved_path = Path(path).resolve()
    data = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data.setdefault("__config_path__", str(resolved_path))
    return data


def resolve_path(
    value: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    if base_dir is not None:
        return (Path(base_dir) / candidate).resolve()
    return candidate.resolve()


def apply_runtime_environment(runtime_config: dict[str, Any]) -> None:
    config_path = runtime_config.get("__config_path__")
    config_dir = Path(str(config_path)).resolve().parent if config_path else None

    paths = dict(runtime_config["paths"])
    android = dict(runtime_config["android"])
    runtime = dict(runtime_config["runtime"])

    for key in (
        "workspace",
        "android_world_src",
        "android_sdk",
        "android_avd_home",
        "java_home",
        "conda_env",
    ):
        paths[key] = str(resolve_path(paths[key], base_dir=config_dir))

    for key in ("adb_path", "emulator_path", "accessibility_forwarder_apk"):
        android[key] = str(resolve_path(android[key], base_dir=config_dir))

    runtime["hf_home"] = str(resolve_path(runtime["hf_home"], base_dir=config_dir))

    runtime_config["paths"] = paths
    runtime_config["android"] = android
    runtime_config["runtime"] = runtime

    cuda_visible_devices = str(
        os.environ.get("GPU_ID")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
        or runtime["cuda_visible_devices"]
    )

    os.environ["DMS_ROOT"] = paths["workspace"]
    os.environ["JAVA_HOME"] = paths["java_home"]
    os.environ["ANDROID_HOME"] = paths["android_sdk"]
    os.environ["ANDROID_SDK_ROOT"] = paths["android_sdk"]
    os.environ["ANDROID_AVD_HOME"] = paths["android_avd_home"]
    os.environ["HF_HOME"] = runtime["hf_home"]
    os.environ["GPU_ID"] = cuda_visible_devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ.setdefault(
        "MODEL_REQUIRE_CUDA_VISIBLE_DEVICES",
        cuda_visible_devices,
    )
    if runtime.get("offline", False):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    path_entries = [
        Path(paths["java_home"]) / "bin",
        Path(paths["android_sdk"]) / "platform-tools",
        Path(paths["android_sdk"]) / "emulator",
        Path(paths["android_sdk"]) / "cmdline-tools" / "latest" / "bin",
        Path(paths["conda_env"]) / "bin",
    ]
    current = os.environ.get("PATH", "")
    os.environ["PATH"] = ":".join(str(item) for item in path_entries) + (
        f":{current}" if current else ""
    )
