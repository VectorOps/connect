from __future__ import annotations

import json

import pytest

from connect.auth import ChatGPTAccessTokenAuth
from connect.providers import GeminiProvider, OpenAIProvider, OpenRouterProvider
from connect.registry import default_provider_registry
from connect.types import (
    AssistantMessage,
    GenerateRequest,
    ImageBlock,
    ModelSpec,
    ReasoningBlock,
    ReasoningConfig,
    ResponseFormat,
    RequestOptions,
    TextBlock,
    ToolCallBlock,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)


class _FakeStreamResponse:
    def __init__(self, events: list[dict], request_id: str = "req_123") -> None:
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


def _openai_model(**updates) -> ModelSpec:
    data = {
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "api_family": "openai-responses",
        "supports_reasoning": True,
        "supports_images": True,
        "supports_json_mode": True,
        "capabilities": {"supports_developer_role": True},
    }
    data.update(updates)
    return ModelSpec(**data)


def _chatgpt_token(account_id: str = "acct_test") -> str:
    def _encode(data: dict) -> str:
        raw = json.dumps(data, separators=(",", ":")).encode()
        return json.dumps("").encode() and __import__("base64").urlsafe_b64encode(raw).decode().rstrip("=")

    header = _encode({"alg": "none", "typ": "JWT"})
    payload = _encode({"https://api.openai.com/auth": {"chatgpt_account_id": account_id}})
    return f"{header}.{payload}.signature"


def test_default_provider_registry_exposes_implemented_providers() -> None:
    assert default_provider_registry.list() == ["chatgpt", "gemini", "openai", "openrouter"]


def test_gemini_provider_is_registered() -> None:
    assert isinstance(default_provider_registry.get("gemini"), GeminiProvider)


def test_chatgpt_build_headers_and_payload_use_account_and_instructions() -> None:
    from connect.providers import ChatGPTProvider

    provider = ChatGPTProvider()
    model = _openai_model(
        provider="chatgpt",
        model="gpt-5-codex",
        api_family="chatgpt-responses",
        base_url="https://chatgpt.com/backend-api",
        capabilities={"supports_developer_role": False},
    )
    request = GenerateRequest(
        messages=[UserMessage(content="ping")],
        system_prompt="Be concise.",
        tools=[
            ToolSpec(
                name="lookup",
                description="Lookup data",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ],
    )
    options = RequestOptions(
        auth=ChatGPTAccessTokenAuth(_chatgpt_token("acct_123")),
        provider_options={"session_id": "sess_123", "text_verbosity": "high"},
    )
    headers = options.auth and __import__("asyncio")

    payload = provider.build_payload(model, request, options)
    built_headers = provider.build_headers(model, request, options)

    assert payload["instructions"] == "Be concise."
    assert payload["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "ping"}]}]
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is True
    assert payload["prompt_cache_key"] == "sess_123"
    assert payload["store"] is False
    assert payload["text"]["verbosity"] == "high"
    assert built_headers["chatgpt-account-id"] == "acct_123"
    assert built_headers["session_id"] == "sess_123"
    assert built_headers["OpenAI-Beta"] == "responses=experimental"
    assert built_headers["originator"] == "connect"
    assert built_headers["User-Agent"].startswith("pi (")


def test_chatgpt_build_payload_includes_empty_instructions_when_missing() -> None:
    from connect.providers import ChatGPTProvider

    provider = ChatGPTProvider()
    model = _openai_model(
        provider="chatgpt",
        model="gpt-5.4-mini",
        api_family="chatgpt-responses",
        base_url="https://chatgpt.com/backend-api",
        capabilities={"supports_developer_role": False},
    )
    request = GenerateRequest(messages=[UserMessage(content="ping")])

    payload = provider.build_payload(model, request, RequestOptions())

    assert payload["instructions"] == ""
    assert payload["store"] is False


def test_openai_build_payload_serializes_multimodal_history_and_controls() -> None:
    provider = OpenAIProvider()
    model = _openai_model()
    request = GenerateRequest(
        messages=[
            UserMessage(
                content=[
                    TextBlock(text="What is in this image?"),
                    ImageBlock(data=b"png-bytes", mime_type="image/png"),
                ]
            ),
            AssistantMessage(
                content=[
                    ToolCallBlock(id="call_1", name="inspect_image", arguments={"detail": "high"})
                ]
            ),
            ToolResultMessage(
                tool_call_id="call_1",
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
        tool_choice="required",
        reasoning=ReasoningConfig(effort="medium", summary="detailed"),
        response_format=ResponseFormat(
            type="json_schema",
            name="inspection",
            json_schema={"type": "object", "properties": {"summary": {"type": "string"}}},
            strict=True,
        ),
        metadata={"request_kind": "vision"},
    )
    options = RequestOptions(provider_options={"store": False, "parallel_tool_calls": False})

    payload = provider.build_payload(model, request, options)

    assert payload["instructions"] == "Be concise."
    assert payload["stream"] is True
    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["strict"] is True
    assert payload["tool_choice"] == "required"
    assert payload["reasoning"] == {"effort": "medium", "summary": "detailed"}
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["metadata"] == {"request_kind": "vision"}
    assert payload["store"] is False
    assert payload["parallel_tool_calls"] is False
    assert "generate" not in payload
    assert payload["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")
    assert payload["input"][0]["content"][1]["detail"] == "auto"
    assert payload["input"][1]["type"] == "function_call"
    assert payload["input"][2]["type"] == "function_call_output"
    assert payload["input"][2]["output"][1]["image_url"].startswith("data:image/png;base64,")


def test_openai_build_payload_supports_empty_tool_result_output() -> None:
    provider = OpenAIProvider()
    model = _openai_model()
    request = GenerateRequest(
        messages=[
            AssistantMessage(content=[ToolCallBlock(id="call_1", name="lookup", arguments={"query": "ping"})]),
            ToolResultMessage(tool_call_id="call_1", tool_name="lookup", content=[]),
        ]
    )

    payload = provider.build_payload(model, request, RequestOptions())

    assert payload["input"][1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "",
    }


def test_openai_build_payload_preserves_reasoning_replay_metadata() -> None:
    provider = OpenAIProvider()
    model = _openai_model()
    request = GenerateRequest(
        messages=[
            AssistantMessage(
                content=[
                    TextBlock(
                        text="Prior assistant text",
                        protocol_meta={
                            "openai_message_id": "msg_123",
                            "openai_message_phase": "final_answer",
                        },
                    ),
                    ToolCallBlock(id="call_1", name="lookup", arguments={"q": "x"}),
                ]
            ),
            ToolResultMessage(tool_call_id="call_1", tool_name="lookup", content=[TextBlock(text="ok")]),
            AssistantMessage(
                content=[
                    TextBlock(text="Visible"),
                    ReasoningBlock(
                        text="Reasoning summary",
                        protocol_meta={
                            "openai_reasoning_id": "rs_123",
                            "openai_encrypted_content": "enc_abc",
                        },
                    ),
                ]
            ),
        ]
    )

    payload = provider.build_payload(model, request, RequestOptions())

    assert payload["input"][0]["type"] == "message"
    assert payload["input"][1]["type"] == "function_call"
    assert payload["input"][3]["type"] == "message"
    assert payload["input"][4]["type"] == "reasoning"
    assert payload["input"][4]["id"] == "rs_123"
    assert payload["input"][4]["encrypted_content"] == "enc_abc"


def test_openai_build_payload_requests_encrypted_reasoning_content() -> None:
    provider = OpenAIProvider()
    model = _openai_model()
    request = GenerateRequest(
        messages=[UserMessage(content="Hello")],
        reasoning=ReasoningConfig(effort="medium", summary="detailed"),
    )

    payload = provider.build_payload(
        model,
        request,
        RequestOptions(provider_options={"include": ["message.output_text.logprobs"]}),
    )

    assert payload["include"] == ["message.output_text.logprobs", "reasoning.encrypted_content"]


@pytest.mark.asyncio
async def test_openai_stream_response_maps_reasoning_summary_and_tool_use_finish_reason() -> None:
    provider = OpenAIProvider()
    model = _openai_model()
    request = GenerateRequest(messages=[UserMessage(content="Hello")])
    response = _FakeStreamResponse(
        [
            {"type": "response.created", "response": {"id": "resp_200", "status": "in_progress"}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"id": "rs_1", "type": "reasoning", "status": "in_progress"},
            },
            {
                "type": "response.reasoning_summary_part.added",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "summary_text", "text": ""},
            },
            {
                "type": "response.reasoning_summary_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "Need tool",
            },
            {
                "type": "response.reasoning_summary_part.done",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "summary_text", "text": "Need tool"},
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "rs_1",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Need tool"}],
                    "encrypted_content": "enc_reasoning",
                    "status": "completed",
                },
            },
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "lookup"},
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 1,
                "item_id": "fc_1",
                "delta": '{"query":"weather"}',
            },
            {
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": '{"query":"weather"}',
                    "status": "completed",
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_200",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 2,
                        "output_tokens": 3,
                        "total_tokens": 5,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 1},
                    },
                },
            },
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
        "reasoning_start",
        "reasoning_delta",
        "reasoning_delta",
        "reasoning_end",
        "tool_call_start",
        "tool_call_delta",
        "tool_call_end",
        "usage",
        "response_end",
    ]
    assert events[-1].response.finish_reason == "tool_call"
    assert events[-1].response.content[0].protocol_meta["openai_reasoning_id"] == "rs_1"
    assert events[-1].response.content[0].protocol_meta["openai_encrypted_content"] == "enc_reasoning"
    assert events[-1].response.content[1].protocol_meta["openai_item_id"] == "fc_1"


@pytest.mark.asyncio
async def test_openai_stream_response_preserves_message_metadata_and_cancelled_terminal() -> None:
    provider = OpenAIProvider()
    model = _openai_model()
    request = GenerateRequest(messages=[UserMessage(content="Hello")])
    response = _FakeStreamResponse(
        [
            {"type": "response.created", "response": {"id": "resp_300"}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"id": "msg_300", "type": "message", "phase": "final_answer"},
            },
            {
                "type": "response.content_part.added",
                "item_id": "msg_300",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text"},
            },
            {"type": "response.output_text.delta", "output_index": 0, "delta": "Hello"},
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "msg_300",
                    "type": "message",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "Hello"}],
                    "status": "completed",
                },
            },
            {
                "type": "response.cancelled",
                "response": {
                    "id": "resp_300",
                    "status": "cancelled",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            },
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

    assert events[-1].response.finish_reason == "cancelled"
    assert events[-1].response.content[0].protocol_meta["openai_message_id"] == "msg_300"
    assert events[-1].response.content[0].protocol_meta["openai_message_phase"] == "final_answer"


@pytest.mark.asyncio
async def test_openai_stream_response_emits_error_for_malformed_tool_arguments() -> None:
    provider = OpenAIProvider()
    model = _openai_model()
    request = GenerateRequest(messages=[UserMessage(content="Hello")])
    response = _FakeStreamResponse(
        [
            {"type": "response.created", "response": {"id": "resp_bad"}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"id": "fc_bad", "type": "function_call", "call_id": "call_bad", "name": "lookup"},
            },
            {
                "type": "response.function_call_arguments.done",
                "output_index": 0,
                "item_id": "fc_bad",
                "arguments": '{"query":',
            },
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


@pytest.mark.asyncio
async def test_openai_stream_response_normalizes_sse_events() -> None:
    provider = OpenAIProvider()
    model = _openai_model()
    request = GenerateRequest(messages=[UserMessage(content="Hello")])
    options = RequestOptions(headers={"authorization": "Bearer test-token"})
    response = _FakeStreamResponse(
        [
            {"type": "response.created", "response": {"id": "resp_123"}},
            {"type": "response.content_part.added", "output_index": 0, "part": {"type": "output_text"}},
            {"type": "response.output_text.delta", "output_index": 0, "delta": "Hello"},
            {"type": "response.output_text.delta", "output_index": 0, "delta": " world"},
            {"type": "response.output_text.done", "output_index": 0},
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "lookup"},
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 1,
                "item_id": "fc_1",
                "delta": '{"query":"weather"}',
            },
            {"type": "response.function_call_arguments.done", "output_index": 1, "item_id": "fc_1"},
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_123",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 4,
                        "total_tokens": 14,
                        "output_tokens_details": {"reasoning_tokens": 1},
                    },
                },
            },
        ]
    )
    http = _FakeHttpTransport(response)

    events = [event async for event in provider.stream_response(model=model, request=request, options=options, http=http)]

    assert [event.type for event in events] == [
        "response_start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "tool_call_start",
        "tool_call_delta",
        "tool_call_end",
        "usage",
        "response_end",
    ]
    assert events[-1].response.response_id == "resp_123"
    assert events[-1].response.content[0].text == "Hello world"
    assert events[-1].response.content[1].arguments == {"query": "weather"}
    assert events[-2].usage.completeness == "final"
    assert http.calls[0]["method"] == "POST"
    assert http.calls[0]["url"].endswith("/responses")


@pytest.mark.asyncio
async def test_openai_stream_response_handles_refusal_and_incomplete_events() -> None:
    provider = OpenAIProvider()
    model = _openai_model()
    request = GenerateRequest(messages=[UserMessage(content="Hello")])
    response = _FakeStreamResponse(
        [
            {"type": "response.created", "response": {"id": "resp_124"}},
            {
                "type": "response.content_part.added",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "refusal"},
            },
            {
                "type": "response.refusal.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "Sorry",
            },
            {
                "type": "response.refusal.done",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "refusal": "Sorry, I can't help with that.",
            },
            {
                "type": "response.incomplete",
                "response": {
                    "id": "resp_124",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "content_filter"},
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 6,
                        "total_tokens": 11,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            },
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
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "usage",
        "response_end",
    ]
    assert events[-1].response.finish_reason == "content_filter"
    assert events[-1].response.content[0].text == "Sorry, I can't help with that."


def test_openrouter_provider_applies_routing_headers_and_payload() -> None:
    provider = OpenRouterProvider()
    model = _openai_model(provider="openrouter")
    request = GenerateRequest(messages=[UserMessage(content="Hello")])
    options = RequestOptions(
        provider_options={
            "referer": "https://example.com",
            "title": "connect-test",
            "provider": {"order": ["openai"]},
        }
    )

    headers = provider.build_headers(model, request, options)
    payload = provider.build_payload(model, request, options)

    assert headers["http-referer"] == "https://example.com"
    assert headers["x-title"] == "connect-test"
    assert payload["provider"] == {"order": ["openai"]}