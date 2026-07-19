from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
import time
from typing import Any

import requests
import yaml

from .contracts import GenerationResult, extract_json


class RemoteModelError(RuntimeError):
    """Raised when the remote inference service cannot complete a request."""


class RemoteModelProtocolError(RemoteModelError):
    """Raised when a successful HTTP response lacks required model fields."""


class RemoteQwenVLClient:
    """Qwen client backed by an OpenAI-compatible multimodal endpoint."""

    _RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).resolve()
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.model_config = config["model"]
        self.runtime_config = config["runtime"]
        self.generation_config = config.get("generation", {})

        configured_base_url = str(
            self.runtime_config.get("base_url", "http://127.0.0.1:8000/v1")
        )
        self.base_url = os.environ.get(
            "DMS_MODEL_BASE_URL", configured_base_url
        ).rstrip("/")
        self.endpoint = f"{self.base_url}/chat/completions"
        self.timeout_seconds = float(
            self.runtime_config.get("request_timeout_seconds", 300)
        )
        self.max_retries = int(self.runtime_config.get("max_retries", 3))
        self.retry_backoff_seconds = float(
            self.runtime_config.get("retry_backoff_seconds", 2)
        )
        self.verify_tls = bool(self.runtime_config.get("verify_tls", True))
        self.reported_peak_gpu_memory_gib = float(
            self.runtime_config.get("reported_peak_gpu_memory_gib", 0.0)
        )

        api_key_env = str(self.runtime_config.get("api_key_env", "")).strip()
        api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.session = requests.Session()
        # The model is reached through a long-lived SSH local forward while
        # Uvicorn may close idle HTTP keep-alive sockets. Reusing one of those
        # stale sockets can lose an otherwise successful response and surface
        # as RemoteDisconnected. Close each transport connection explicitly;
        # task-level infrastructure retries remain the only scored retries.
        self.session.headers.update(
            {"Content-Type": "application/json", "Connection": "close"}
        )
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def healthcheck(self) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/models",
            timeout=min(self.timeout_seconds, 30),
            verify=self.verify_tls,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RemoteModelProtocolError("GET /models did not return a JSON object")
        return payload

    @staticmethod
    def _image_data_url(image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(image_path.name)
        mime_type = mime_type or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _response_text(message_content: Any) -> str:
        if isinstance(message_content, str):
            return message_content
        if isinstance(message_content, list):
            parts = []
            for item in message_content:
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                    parts.append(str(item.get("text", "")))
            return "".join(parts)
        raise RemoteModelProtocolError(
            "choices[0].message.content must be a string or a list of text parts"
        )

    def _post_with_retries(self, payload: dict[str, Any]) -> requests.Response:
        attempts = max(1, self.max_retries + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self.session.post(
                    self.endpoint,
                    json=payload,
                    timeout=self.timeout_seconds,
                    verify=self.verify_tls,
                )
                if response.status_code in self._RETRYABLE_STATUS_CODES:
                    detail = response.text[:500]
                    raise RemoteModelError(
                        f"remote model returned retryable HTTP {response.status_code}: "
                        f"{detail}"
                    )
                response.raise_for_status()
                return response
            except (requests.RequestException, RemoteModelError) as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                time.sleep(self.retry_backoff_seconds * (2**attempt))
        raise RemoteModelError(
            f"remote model request failed after {attempts} attempt(s): {last_error}"
        ) from last_error

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
                    {
                        "type": "image_url",
                        "image_url": {"url": self._image_data_url(image_path)},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        )

        payload: dict[str, Any] = {
            "model": self.model_config["name"],
            "messages": messages,
            "max_tokens": int(self.model_config["max_new_tokens"]),
        }
        if self.model_config["do_sample"]:
            payload["temperature"] = self.generation_config.get("temperature")
            payload["top_p"] = self.generation_config.get("top_p")
        else:
            payload["temperature"] = 0.0

        started_at = time.perf_counter()
        response = self._post_with_retries(payload)
        latency = time.perf_counter() - started_at
        try:
            data = response.json()
            choices = data["choices"]
            text = self._response_text(choices[0]["message"]["content"])
            usage = data["usage"]
            input_tokens = int(usage["prompt_tokens"])
            output_tokens = int(usage["completion_tokens"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RemoteModelProtocolError(
                "Remote response must contain choices[0].message.content and "
                "usage.prompt_tokens/completion_tokens so experiment metrics remain valid"
            ) from exc

        return GenerationResult(
            text=text,
            parsed_json=extract_json(text),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
            peak_gpu_memory_gib=self.reported_peak_gpu_memory_gib,
        )
