from __future__ import annotations

import json

import pytest

from connect.auth import BearerTokenAuth
from connect.client import AsyncLLMClient
from connect.registry import ModelRegistry, ProviderRegistry
from connect.types import GenerateRequest, ModelSpec, RequestOptions, UserMessage


class _FakeStreamResponse:
    request_id = "req_test"
    status = 200
    headers = {}
    url = "https://api.openai.com/v1/responses"

    def __init__(self) -> None:
        payload_lines: list[bytes] = []
        for event in (
            {"type": "response.created", "response": {"id": "resp_test"}},
            {"type": "response.content_part.added", "output_index": 0, "part": {"type": "output_text"}},
            {"type": "response.output_text.delta", "output_index": 0, "delta": "pong"},
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_test",
                    "status": "completed",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            },
        ):
            payload_lines.append(f"data: {json.dumps(event)}\n".encode())
            payload_lines.append(b"\n")
        payload_lines.append(b"data: [DONE]\n")
        payload_lines.append(b"\n")
        self.content = _FakeStreamContent(payload_lines)

    async def __aenter__(self) -> _FakeStreamResponse:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeStreamContent:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = iter(lines)

    async def readline(self) -> bytes:
        return next(self._lines, b"")


class _FakeClientSession:
    def __init__(self) -> None:
        self.closed = False
        self.requests: list[dict] = []

    async def request(self, method, url, **kwargs):
        self.requests.append({"method": method, "url": url, **kwargs})
        return _FakeStreamResponse()

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_async_client_generate_uses_default_provider_registry() -> None:
    session = _FakeClientSession()
    model = ModelSpec(provider="openai", model="gpt-4.1-mini", api_family="openai-responses")
    model_registry = ModelRegistry([model])
    provider_registry = ProviderRegistry()
    from connect.providers import OpenAIProvider

    provider_registry.register("openai", OpenAIProvider())

    async with AsyncLLMClient(
        http_client=session,
        model_registry=model_registry,
        provider_registry=provider_registry,
    ) as client:
        response = await client.generate(
            "openai/gpt-4.1-mini",
            GenerateRequest(messages=[UserMessage(content="ping")]),
            options=RequestOptions(auth=BearerTokenAuth("test-token")),
        )

    assert response.response_id == "resp_test"
    assert response.content[0].text == "pong"
    assert response.usage.total_tokens == 2
    assert session.requests[0]["headers"]["Authorization"] == "Bearer test-token"