#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


WORK = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--target",
        default=str(
            WORK
            / "model_cache"
            / "modelscope"
            / "AI-ModelScope"
            / "all-MiniLM-L6-v2"
        ),
    )
    args = parser.parse_args()
    target = Path(args.target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    marker_files = ("config.json", "modules.json", "model.safetensors")
    if target.is_dir() and any((target / marker).is_file() for marker in marker_files):
        print(f"embedder_ready={target}")
        return

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(target),
    )
    print(f"embedder_ready={target}")


if __name__ == "__main__":
    main()
