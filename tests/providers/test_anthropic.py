from __future__ import annotations

import json

import pytest

from connect.providers import AnthropicProvider
from connect.types import (
    AssistantMessage,
    GenerateRequest,
    ImageBlock,
    ModelSpec,
    ReasoningBlock,
    ReasoningConfig,
    RequestOptions,
    SpecificToolChoice,
    TextBlock,
    ToolCallBlock,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)


class _FakeStreamResponse:
    def __init__(self, events: list[dict], request_id: str = "req_anthropic") -> None:
        self._events = events
        self.request_id = request_id

    async def __aenter__(self) -> _FakeStreamResponse:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def iter_lines(self):
        for event in self._events:
            yield f"data: {json.dumps(event)}"
            yield ""
        yield "data: [DONE]"
        yield ""


class _FakeHttpTransport:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def stream(self, method: str, url: str, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.response


def _anthropic_model(**updates) -> ModelSpec:
    data = {
        "provider": "anthropic",
        "model": "claude-3-7-sonnet-latest",
        "api_family": "anthropic-messages",
        "base_url": "https://api.anthropic.com/v1",
        "supports_reasoning": True,
        "supports_images": True,
        "max_output_tokens": 4096,
        "capabilities": {"tool_call_id_max_length": 64},
    }
    data.update(updates)
    return ModelSpec(**data)


def test_anthropic_build_payload_serializes_multimodal_history_tools_and_reasoning() -> None:
    provider = AnthropicProvider()
    model = _anthropic_model()
    request = GenerateRequest(
        messages=[
            UserMessage(
                content=[
                    TextBlock(text="What is in this image?"),
                    ImageBlock(data=b"image-bytes", mime_type="image/png"),
                ]
            ),
            AssistantMessage(
                content=[
                    ReasoningBlock(text="Need a tool", signature="sig_reasoning"),
                    ToolCallBlock(id="call 1", name="inspect_image", arguments={"detail": "high"}),
                ]
            ),
            ToolResultMessage(
                tool_call_id="call 1",
                tool_name="inspect_image",
                content=[
                    TextBlock(text="Found a red circle."),
                    ImageBlock(data=b"tool-image", mime_type="image/png"),
                ],
            ),
        ],
        system_prompt="Be concise.",
        tools=[
            ToolSpec(
                name="inspect_image",
                description="Inspect an image",
                input_schema={"type": "object", "properties": {"detail": {"type": "string"}}},
            )
        ],
        tool_choice=SpecificToolChoice(name="inspect_image"),
        reasoning=ReasoningConfig(effort="medium", max_tokens=2048),
        temperature=0.2,
        max_output_tokens=512,
    )

    options = RequestOptions(provider_options={"cache_retention": "short"})
    headers = provider.build_headers(model, request, options)
    payload = provider.build_payload(model, request, options)

    assert headers["anthropic-version"] == "2023-06-01"
    assert "fine-grained-tool-streaming-2025-05-14" in headers["anthropic-beta"]
    assert payload["model"] == "claude-3-7-sonnet-latest"
    assert payload["system"] == [
        {"type": "text", "text": "Be concise.", "cache_control": {"type": "ephemeral"}}
    ]
    assert payload["max_tokens"] == 512
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"][0] == {"type": "text", "text": "What is in this image?"}
    assert payload["messages"][0]["content"][1]["type"] == "image"
    assert payload["messages"][1]["role"] == "assistant"
    assert payload["messages"][1]["content"][0] == {
        "type": "thinking",
        "thinking": "Need a tool",
        "signature": "sig_reasoning",
    }
    assert payload["messages"][1]["content"][1] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "inspect_image",
        "input": {"detail": "high"},
    }
    assert payload["messages"][2]["role"] == "user"
    assert payload["messages"][2]["content"][0]["type"] == "tool_result"
    assert payload["messages"][2]["content"][0]["tool_use_id"] == "call_1"
    assert payload["messages"][2]["content"][0]["content"][1]["type"] == "image"
    assert payload["messages"][2]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert payload["tools"][0]["name"] == "inspect_image"
    assert payload["tool_choice"] == {"type": "tool", "name": "inspect_image"}
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "temperature" not in payload


def test_anthropic_build_payload_disables_reasoning_and_omits_tools_for_none_choice() -> None:
    provider = AnthropicProvider()
    model = _anthropic_model()
    request = GenerateRequest(
        messages=[UserMessage(content="hello")],
        tools=[
            ToolSpec(
                name="lookup",
                description="Lookup data",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ],
        tool_choice="none",
        reasoning=ReasoningConfig(enabled=False),
        temperature=0.4,
    )

    payload = provider.build_payload(model, request, RequestOptions())

    assert payload["thinking"] == {"type": "disabled"}
    assert payload["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}],
        }
    ]
    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert "temperature" not in payload


def test_anthropic_build_payload_uses_adaptive_thinking_and_long_cache_retention() -> None:
    provider = AnthropicProvider()
    model = _anthropic_model(model="claude-opus-4-6")
    request = GenerateRequest(
        messages=[UserMessage(content="hello")],
        system_prompt="System prompt",
        reasoning=ReasoningConfig(effort="xhigh"),
        metadata={"user_id": "user-123", "ignored": "value"},
    )
    options = RequestOptions(provider_options={"cache_retention": "long"})

    headers = provider.build_headers(model, request, options)
    payload = provider.build_payload(model, request, options)

    assert headers.get("anthropic-beta") is None
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"] == {"effort": "max"}
    assert payload["metadata"] == {"user_id": "user-123"}


def test_anthropic_build_payload_can_disable_beta_headers_explicitly() -> None:
    provider = AnthropicProvider()
    model = _anthropic_model()
    request = GenerateRequest(
        messages=[UserMessage(content="hello")],
        tools=[
            ToolSpec(
                name="lookup",
                description="Lookup data",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ],
        reasoning=ReasoningConfig(effort="medium"),
    )
    headers = provider.build_headers(
        model,
        request,
        RequestOptions(
            provider_options={
                "fine_grained_tool_streaming": False,
                "interleaved_thinking": False,
            }
        ),
    )

    assert "anthropic-beta" not in headers


def test_anthropic_build_payload_batches_consecutive_tool_results() -> None:
    provider = AnthropicProvider()
    model = _anthropic_model()
    request = GenerateRequest(
        messages=[
            AssistantMessage(
                content=[
                    ToolCallBlock(id="call_1", name="lookup", arguments={}),
                    ToolCallBlock(id="call_2", name="fetch", arguments={}),
                ]
            ),
            ToolResultMessage(tool_call_id="call_1", tool_name="lookup", content=[TextBlock(text="one")]),
            ToolResultMessage(tool_call_id="call_2", tool_name="fetch", content=[TextBlock(text="two")]),
        ]
    )

    payload = provider.build_payload(model, request, RequestOptions())

    assert payload["messages"][1] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "one", "is_error": False},
            {
                "type": "tool_result",
                "tool_use_id": "call_2",
                "content": "two",
                "is_error": False,
                "cache_control": {"type": "ephemeral"},
            },
        ],
    }


@pytest.mark.asyncio
async def test_anthropic_stream_response_normalizes_reasoning_tool_calls_and_usage() -> None:
    provider = AnthropicProvider()
    model = _anthropic_model()
    request = GenerateRequest(messages=[UserMessage(content="Hello")])
    response = _FakeStreamResponse(
        [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 1,
                        "cache_creation_input_tokens": 0,
                    },
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Need tool"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig_reasoning"}},
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"query":"weather"}'},
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "cache_read_input_tokens": 1,
                    "cache_creation_input_tokens": 0,
                },
            },
            {"type": "message_stop"},
        ]
    )

    events = [
        event
        async for event in provider.stream_response(
            model=model,
            request=request,
            options=RequestOptions(),
            http=_FakeHttpTransport(response),
        )
    ]

    assert [event.type for event in events] == [
        "response_start",
        "usage",
        "reasoning_start",
        "reasoning_delta",
        "reasoning_end",
        "tool_call_start",
        "tool_call_delta",
        "tool_call_end",
        "usage",
        "usage",
        "response_end",
    ]
    assert events[1].usage.completeness == "partial"
    assert events[-2].usage.completeness == "final"
    assert events[-1].response.response_id == "msg_123"
    assert events[-1].response.finish_reason == "tool_call"
    assert events[-1].response.content[0].text == "Need tool"
    assert events[-1].response.content[0].signature == "sig_reasoning"
    assert events[-1].response.content[0].protocol_meta["anthropic_provider"] == "anthropic"
    assert events[-1].response.content[0].protocol_meta["anthropic_model"] == "claude-3-7-sonnet-latest"
    assert events[-1].response.content[1].id == "call_1"
    assert events[-1].response.content[1].arguments == {"query": "weather"}
    assert events[-1].response.usage.input_tokens == 5
    assert events[-1].response.usage.output_tokens == 3
    assert events[-1].response.usage.cache_read_tokens == 1


@pytest.mark.asyncio
async def test_anthropic_stream_response_handles_redacted_thinking_and_stop_reason_mapping() -> None:
    provider = AnthropicProvider()
    model = _anthropic_model()
    request = GenerateRequest(messages=[UserMessage(content="Hello")])
    response = _FakeStreamResponse(
        [
            {"type": "message_start", "message": {"id": "msg_redacted"}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "redacted_thinking", "data": "sig_redacted"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": "Visible answer"},
            },
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "pause_turn"}},
            {"type": "message_stop"},
        ]
    )

    events = [
        event
        async for event in provider.stream_response(
            model=model,
            request=request,
            options=RequestOptions(),
            http=_FakeHttpTransport(response),
        )
    ]

    assert events[-1].response.finish_reason == "stop"
    assert events[-1].response.content[0].redacted is True
    assert events[-1].response.content[0].signature == "sig_redacted"
    assert events[-1].response.content[0].text == "[Reasoning redacted]"
    assert events[-1].response.content[1].text == "Visible answer"


@pytest.mark.asyncio
async def test_anthropic_stream_response_emits_error_for_malformed_tool_arguments() -> None:
    provider = AnthropicProvider()
    model = _anthropic_model()
    request = GenerateRequest(messages=[UserMessage(content="Hello")])
    response = _FakeStreamResponse(
        [
            {"type": "message_start", "message": {"id": "msg_bad"}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "call_bad", "name": "lookup", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"query":'},
            },
            {"type": "content_block_stop", "index": 0},
        ]
    )

    events = [
        event
        async for event in provider.stream_response(
            model=model,
            request=request,
            options=RequestOptions(),
            http=_FakeHttpTransport(response),
        )
    ]

    assert events[-1].type == "error"
    assert events[-1].error.code == "invalid_tool_arguments"