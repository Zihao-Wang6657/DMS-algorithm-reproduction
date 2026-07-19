from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class QwenVLClient:
    """Selects the local or remote transport without changing the agent API."""

    def __new__(cls, config_path: str | Path) -> Any:
        path = Path(config_path).resolve()
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        runtime = config.get("runtime", {}) if isinstance(config, dict) else {}
        backend = str(runtime.get("backend", "local_transformers")).strip().lower()

        if backend == "local_transformers":
            from .qwen_vl import QwenVLClient as LocalQwenVLClient

            return LocalQwenVLClient(path)
        if backend == "openai_compatible":
            from .remote_qwen_vl import RemoteQwenVLClient

            return RemoteQwenVLClient(path)
        raise ValueError(
            f"Unsupported model runtime backend {backend!r}; expected "
            "'local_transformers' or 'openai_compatible'"
        )
