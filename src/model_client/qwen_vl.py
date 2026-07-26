from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


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


class QwenVLClient:
    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.model_config = config["model"]
        self.runtime_config = config["runtime"]
        self.generation_config = config.get("generation", {})
        self._validate_runtime()

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.model_config["dtype"]]
        model_name = self.model_config["name"]
        common_kwargs = {
            "cache_dir": os.environ.get("HF_HOME"),
            "local_files_only": self.runtime_config.get("local_files_only", False),
            "trust_remote_code": self.runtime_config.get("trust_remote_code", False),
        }
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=self.model_config["min_pixels"],
            max_pixels=self.model_config["max_pixels"],
            **common_kwargs,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation=self.model_config["attn_implementation"],
            device_map={"": 0},
            **common_kwargs,
        )
        if not self.model_config["do_sample"]:
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
        self.model.eval()

    def _validate_runtime(self) -> None:
        expected = os.environ.get("MODEL_REQUIRE_CUDA_VISIBLE_DEVICES")
        if expected is None and self.runtime_config.get("require_cuda_visible_devices") is not None:
            expected = str(self.runtime_config["require_cuda_visible_devices"])
        actual = os.environ.get("CUDA_VISIBLE_DEVICES")
        if expected is not None and actual != expected:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES must be {expected!r}, got {actual!r}"
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one CUDA device must be visible to the model client")

    @torch.inference_mode()
    def generate(
        self,
        image_path: str | Path,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> GenerationResult:
        image_path = Path(image_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        if tools:
            tool_protocol = (
                "\n\nAvailable tools are registered below. Call exactly one tool by "
                "returning `<tool_call>` followed by one JSON object and "
                "`</tool_call>`. The object must contain `name` and `arguments`.\n"
                f"<tools>\n{json.dumps(tools, ensure_ascii=False)}\n</tools>"
            )
            system_prompt = f"{system_prompt or ''}{tool_protocol}".strip()

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path.as_uri()},
                    {"type": "text", "text": prompt},
                ],
            }
        )
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started_at = time.perf_counter()
        generation_kwargs = {
            "max_new_tokens": self.model_config["max_new_tokens"],
            "do_sample": self.model_config["do_sample"],
        }
        if self.model_config["do_sample"]:
            generation_kwargs.update(
                temperature=self.generation_config.get("temperature"),
                top_p=self.generation_config.get("top_p"),
            )
        generated_ids = self.model.generate(**inputs, **generation_kwargs)
        torch.cuda.synchronize()
        latency = time.perf_counter() - started_at

        input_length = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_length:]
        text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return GenerationResult(
            text=text,
            parsed_json=extract_json(text),
            input_tokens=int(input_length),
            output_tokens=int(output_ids.shape[1]),
            latency_seconds=latency,
            peak_gpu_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
        )
