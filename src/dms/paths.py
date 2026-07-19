from __future__ import annotations

from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE.parent


def workspace_path(*parts: str) -> Path:
    return WORKSPACE.joinpath(*parts)


def root_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)

