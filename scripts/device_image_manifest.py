#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(repo_root: Path, roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().lower()):
            if not path.is_file():
                continue
            rows.append(
                {
                    "path": path.resolve().relative_to(repo_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{row['sha256']}  {row['bytes']}  {row['path']}" for row in rows
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def freeze(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    roots = [Path(value).resolve() for value in args.asset_root]
    rows = inventory(repo_root, roots)
    payload = {
        "format": "dms-device-image-manifest-v1",
        "workspace_root": str(repo_root),
        "asset_roots": [root.relative_to(repo_root).as_posix() for root in roots],
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "tree_sha256": aggregate(rows),
        "files": rows,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("file_count", "total_bytes", "tree_sha256")}, indent=2))


def verify(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_root = Path(payload["workspace_root"]).resolve()
    errors: list[str] = []
    actual_rows: list[dict[str, Any]] = []
    for expected in payload["files"]:
        path = (repo_root / expected["path"]).resolve()
        actual = {
            "path": expected["path"],
            "bytes": path.stat().st_size if path.is_file() else -1,
            "sha256": sha256(path) if path.is_file() else "MISSING",
        }
        actual_rows.append(actual)
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            errors.append(expected["path"])
    actual_tree = aggregate(actual_rows)
    if actual_tree != payload["tree_sha256"]:
        errors.append("TREE_SHA256")
    if errors:
        raise RuntimeError("Device image integrity failure: " + ", ".join(errors[:20]))
    print(json.dumps({"verified": True, "tree_sha256": actual_tree, "file_count": len(actual_rows)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--repo-root", required=True)
    freeze_parser.add_argument("--asset-root", action="append", required=True)
    freeze_parser.add_argument("--output", required=True)
    freeze_parser.set_defaults(function=freeze)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", required=True)
    verify_parser.set_defaults(function=verify)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
