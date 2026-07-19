from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest
import requests
import yaml

from model_client import QwenVLClient
from model_client.remote_qwen_vl import (
    RemoteModelProtocolError,
    RemoteQwenVLClient,
)


class FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        status_code: int = 200,
        text: str = "",
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[Any]) -> None:
        self.headers: dict[str, str] = {}
        self.responses = list(responses)
        self.posts: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.posts.append({"url": url, **kwargs})
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse({"object": "list", "data": [{"id": "test-model"}]})


def write_config(path: Path, *, max_retries: int = 0) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "name": "test-model",
                    "min_pixels": 200704,
                    "max_pixels": 1003520,
                    "max_new_tokens": 192,
                    "do_sample": False,
                },
                "runtime": {
                    "backend": "openai_compatible",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "request_timeout_seconds": 30,
                    "max_retries": max_retries,
                    "retry_backoff_seconds": 0,
                },
                "generation": {"temperature": None, "top_p": None},
            }
        ),
        encoding="utf-8",
    )


def successful_response() -> FakeResponse:
    return FakeResponse(
        {
            "choices": [
                {"message": {"content": '<tool_call>{"name":"tap","arguments":{}}</tool_call>'}}
            ],
            "usage": {"prompt_tokens": 321, "completion_tokens": 17},
        }
    )


def test_remote_factory_preserves_generation_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "remote.yaml"
    image_path = tmp_path / "screen.png"
    write_config(config_path)
    image_bytes = b"\x89PNG\r\n\x1a\nfixture"
    image_path.write_bytes(image_bytes)
    fake_session = FakeSession([successful_response()])
    monkeypatch.setattr("model_client.remote_qwen_vl.requests.Session", lambda: fake_session)

    client = QwenVLClient(config_path)
    assert isinstance(client, RemoteQwenVLClient)
    result = client.generate(
        image_path=image_path,
        prompt="Choose one action.",
        system_prompt="Return one tool call.",
        tools=[{"name": "tap", "description": "Tap", "parameters": {}}],
    )

    assert result.input_tokens == 321
    assert result.output_tokens == 17
    assert result.parsed_json == {"name": "tap", "arguments": {}}
    assert result.peak_gpu_memory_gib == 0.0

    request = fake_session.posts[0]
    assert request["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    payload = request["json"]
    assert payload["model"] == "test-model"
    assert payload["max_tokens"] == 192
    assert payload["temperature"] == 0.0
    assert payload["messages"][0]["role"] == "system"
    assert "Available tools are registered below" in payload["messages"][0]["content"][0]["text"]
    user_content = payload["messages"][1]["content"]
    assert user_content[0]["type"] == "image_url"
    prefix, encoded = user_content[0]["image_url"]["url"].split(",", 1)
    assert prefix == "data:image/png;base64"
    assert base64.b64decode(encoded) == image_bytes
    assert user_content[1] == {"type": "text", "text": "Choose one action."}


def test_remote_client_retries_connection_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "remote.yaml"
    image_path = tmp_path / "screen.png"
    write_config(config_path, max_retries=1)
    image_path.write_bytes(b"image")
    fake_session = FakeSession(
        [requests.ConnectionError("temporary disconnect"), successful_response()]
    )
    monkeypatch.setattr("model_client.remote_qwen_vl.requests.Session", lambda: fake_session)
    monkeypatch.setattr("model_client.remote_qwen_vl.time.sleep", lambda _: None)

    result = QwenVLClient(config_path).generate(image_path, "prompt")

    assert result.output_tokens == 17
    assert len(fake_session.posts) == 2


def test_remote_client_rejects_missing_usage_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "remote.yaml"
    image_path = tmp_path / "screen.png"
    write_config(config_path)
    image_path.write_bytes(b"image")
    fake_session = FakeSession(
        [FakeResponse({"choices": [{"message": {"content": "{}"}}]})]
    )
    monkeypatch.setattr("model_client.remote_qwen_vl.requests.Session", lambda: fake_session)

    with pytest.raises(RemoteModelProtocolError, match="usage.prompt_tokens"):
        QwenVLClient(config_path).generate(image_path, "prompt")
