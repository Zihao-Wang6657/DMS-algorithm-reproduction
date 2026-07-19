from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class GenerationResult:
    text: str
    parsed_json: dict[str, Any] | list[Any] | None
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    peak_gpu_memory_gib: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_json(text: str) -> dict[str, Any] | list[Any] | None:
    candidates = [text.strip()]
    tool_call = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL)
    if tool_call:
        candidates.insert(0, tool_call.group(1).strip())
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, (dict, list)):
                return value
        except json.JSONDecodeError:
            pass

        for index, char in enumerate(candidate):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, (dict, list)):
                return value
    return None
