#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / "third_party" / "android_world"))
sys.path.insert(0, str(WORK / "src"))

from dms.config import apply_runtime_environment, load_yaml
from model_client import QwenVLClient
from qwen_vl_smoke import make_test_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--idle-seconds", type=float, default=7.0)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runtime = load_yaml(args.runtime_config)
    apply_runtime_environment(runtime)
    image_path = WORK / "runs" / "smoke_assets" / "qwen_smoke.png"
    make_test_image(image_path)
    client = QwenVLClient(Path(args.model_config).resolve())

    records: list[dict[str, object]] = []
    for index in range(args.iterations):
        started = time.time()
        result = client.generate(
            image_path=image_path,
            prompt="Return exactly this JSON object: {\"transport_ok\": true}",
            system_prompt="Return strict JSON only, without Markdown fences.",
        )
        record = {
            "iteration": index + 1,
            "elapsed_seconds": round(time.time() - started, 3),
            "transport_ok": bool(
                isinstance(result.parsed_json, dict)
                and result.parsed_json.get("transport_ok") is True
            ),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        if not record["transport_ok"]:
            raise SystemExit(2)
        if index + 1 < args.iterations:
            time.sleep(args.idle_seconds)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"passed": True, "records": records}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
