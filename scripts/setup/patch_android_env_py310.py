#!/usr/bin/env python
from __future__ import annotations

import sys
import sysconfig
from pathlib import Path


OLD_IMPORT = "from typing import Self"
NEW_IMPORT = "from typing_extensions import Self"


def _candidate_package_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("purelib", "platlib"):
        value = sysconfig.get_paths().get(key)
        if value:
            roots.append(Path(value).resolve())
    for entry in sys.path:
        if "site-packages" in entry:
            roots.append(Path(entry).resolve())
    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        unique_roots.append(root)
    return unique_roots


def _find_android_env_package() -> Path:
    for root in _candidate_package_roots():
        candidate = root / "android_env"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("android_env package directory not found in site-packages")


def _patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if OLD_IMPORT not in text or NEW_IMPORT in text:
        return False
    path.write_text(text.replace(OLD_IMPORT, NEW_IMPORT), encoding="utf-8")
    return True


def main() -> None:
    package_root = _find_android_env_package()
    patched_files: list[Path] = []
    for py_file in sorted(package_root.rglob("*.py")):
        if _patch_file(py_file):
            patched_files.append(py_file)

    import android_env.components.app_screen_checker as app_screen_checker

    status = "patched" if patched_files else "already_ok"
    print(f"android_env_py310_compat={status}")
    print(f"android_env_import_target={app_screen_checker.__file__}")
    for patched_file in patched_files:
        print(f"patched_file={patched_file}")


if __name__ == "__main__":
    main()
