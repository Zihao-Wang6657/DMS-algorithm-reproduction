#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / "third_party" / "android_world"))
sys.path.insert(0, str(WORK / "src"))

from PIL import Image, ImageDraw

from dms.config import apply_runtime_environment, load_yaml
from dms.io_utils import now_iso, write_json
from dms.paths import workspace_path
from model_client import QwenVLClient


def make_test_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (640, 480), color=(250, 250, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 80, 560, 400), outline=(40, 40, 40), width=4)
    draw.text((120, 180), "AndroidWorld DMS smoke test", fill=(10, 10, 10))
    draw.text((120, 230), "Return JSON: status/model_loaded", fill=(10, 10, 10))
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-config",
        default=str(workspace_path("configs", "runtime.yaml")),
    )
    parser.add_argument(
        "--model-config",
        default=str(workspace_path("configs", "model_qwen25vl_7b.yaml")),
    )
    args = parser.parse_args()

    runtime = load_yaml(args.runtime_config)
    apply_runtime_environment(runtime)
    model_config = Path(args.model_config).resolve()
    image_path = workspace_path("runs", "smoke_assets", "qwen_smoke.png")
    make_test_image(image_path)
    client = QwenVLClient(model_config)
    result = client.generate(
        image_path=image_path,
        prompt=(
            "Inspect the image and return exactly one JSON object with keys "
            "status, model_loaded, visible_text. model_loaded must be true."
        ),
        system_prompt="Return strict JSON only. Do not use Markdown fences.",
    )
    parsed = result.parsed_json
    ok = isinstance(parsed, dict) and parsed.get("model_loaded") is True
    record = {
        "timestamp": now_iso(),
        "model_config": str(model_config.resolve()),
        "image_path": str(image_path.resolve()),
        "structured_output_valid": ok,
        "generation": result.to_dict(),
    }
    output = workspace_path("logs", "qwen_vl_smoke.json")
    write_json(output, record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
