#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -x "$WORK/conda_envs/dms_py310/bin/python" ]]; then
  echo "python environment is missing: $WORK/conda_envs/dms_py310" >&2
  echo "run scripts/setup_python_env.sh first" >&2
  exit 2
fi

source "$WORK/scripts/activate_env.sh"

export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
export EMBED_MODEL_ID="${EMBED_MODEL_ID:-sentence-transformers/all-MiniLM-L6-v2}"
export EMBED_DIR="${EMBED_DIR:-$WORK/model_cache/modelscope/AI-ModelScope/all-MiniLM-L6-v2}"

python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

hf_home = Path(os.environ["HF_HOME"]).resolve()
qwen_model_id = os.environ["QWEN_MODEL_ID"]
embed_model_id = os.environ["EMBED_MODEL_ID"]
embed_dir = Path(os.environ["EMBED_DIR"]).resolve()

hf_home.mkdir(parents=True, exist_ok=True)
embed_dir.parent.mkdir(parents=True, exist_ok=True)

def ensure_cached_snapshot(repo_id: str, cache_dir: Path) -> str:
    try:
        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
        print(f"reusing_qwen_cache={path}")
        return path
    except Exception as exc:
        print(f"cached_qwen_lookup_miss={exc!r}")
    path = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
    )
    print(f"downloaded_qwen_cache={path}")
    return path


def ensure_embedder_dir(repo_id: str, local_dir: Path) -> str:
    marker_files = (
        "config.json",
        "modules.json",
        "tokenizer.json",
        "pytorch_model.bin",
        "model.safetensors",
    )
    if local_dir.is_dir() and any((local_dir / name).exists() for name in marker_files):
        print(f"reusing_embedder_dir={local_dir}")
        return str(local_dir)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            local_files_only=True,
        )
        print(f"restored_embedder_dir_from_cache={local_dir}")
        return str(local_dir)
    except Exception as exc:
        print(f"cached_embedder_lookup_miss={exc!r}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    print(f"downloaded_embedder_dir={local_dir}")
    return str(local_dir)


print(f"downloading_qwen={qwen_model_id}")
ensure_cached_snapshot(qwen_model_id, hf_home)

print(f"downloading_embedder={embed_model_id}")
ensure_embedder_dir(embed_model_id, embed_dir)

print(f"hf_home={hf_home}")
print(f"embed_dir={embed_dir}")
PY
