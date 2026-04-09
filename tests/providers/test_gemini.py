from __future__ import annotations

import json

import pytest

from connect.exceptions import AuthenticationError, exception_from_error_info
from connect.providers import GeminiProvider
from connect.transport.http import HttpResponse, HttpStatusError
from connect.types import (
    AssistantMessage,
    GenerateRequest,
    ImageBlock,
    ModelSpec,
    ReasoningBlock,
    ReasoningConfig,
    RequestOptions,
    ResponseFormat,
    SpecificToolChoice,
    TextBlock,
    ToolCallBlock,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)


class _FakeStreamResponse:
    def __init__(self, chunks: list[dict], request_id: str = "req_gemini") -> None:
        self._chunks = chunks
        self.request_id = request_id

    async def __aenter__(self) -> _FakeStreamResponse:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def iter_bytes(self):
        for chunk in self._chunks:
            yield json.dumps(chunk).encode("utf-8")


class _FakeArrayStreamResponse:
    def __init__(self, chunks: list[list[dict]], request_id: str = "req_gemini") -> None:
        self._chunks = chunks
        self.request_id = request_id

    async def __aenter__(self) -> _FakeArrayStreamResponse:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def iter_bytes(self):
        for chunk in self._chunks:
            yield json.dumps(chunk).encode("utf-8")


class _FakeHttpTransport:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def stream(self, method: str, url: str, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.response


class _ErrorHttpTransport:
    def __init__(self, response: HttpResponse) -> None:
        self.response = response

    async def stream(self, method: str, url: str, **kwargs):
        raise HttpStatusError(self.response)


def _gemini_model(**updates) -> ModelSpec:
    data = {
        "provider": "gemini",
        "model": "gemini-3-pro-preview",
        "api_family": "gemini-generate-content",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "supports_reasoning": True,
        "supports_images": True,
        "supports_json_mode": True,
        "capabilities": {"usage_final_only": False},
    }
    data.update(updates)
    return ModelSpec(**data)


def test_gemini_build_payload_serializes_multimodal_history_tools_and_reasoning() -> None:
    provider = GeminiProvider()
    model = _gemini_model()
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
                    TextBlock(
                        text="I should inspect the file.",
                        protocol_meta={
                            "gemini_thought_signature": "c2lnX3RleHQ=",
                            "gemini_provider": "gemini",
                            "gemini_model": "gemini-3-pro-preview",
                        },
                    ),
                    ReasoningBlock(
                        text="Need a tool",
                        signature="sig_reasoning",
                        protocol_meta={
                            "gemini_thought_signature": "c2lnX3JlYXNvbmluZw==",
                            "gemini_provider": "gemini",
                            "gemini_model": "gemini-3-pro-preview",
                        },
                    ),
                    ToolCallBlock(
                        id="call_1",
                        name="inspect_image",
                        arguments={"detail": "high"},
                        protocol_meta={
                            "gemini_provider": "gemini",
                            "gemini_model": "gemini-3-pro-preview",
                        },
                    ),
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
        tool_choice=SpecificToolChoice(name="inspect_image"),
        reasoning=ReasoningConfig(effort="medium", max_tokens=2048),
        response_format=ResponseFormat(
            type="json_schema",
            name="inspection",
            json_schema={"type": "object", "properties": {"summary": {"type": "string"}}},
        ),
        temperature=0.2,
        max_output_tokens=512,
    )

    payload = provider.build_payload(model, request, RequestOptions())

    assert payload["systemInstruction"]["parts"][0]["text"] == "Be concise."
    assert payload["contents"][0]["role"] == "user"
    assert payload["contents"][0]["parts"][0] == {"text": "What is in this image?"}
    assert payload["contents"][0]["parts"][1]["inlineData"]["mimeType"] == "image/png"
    assert payload["contents"][1]["role"] == "model"
    assert payload["contents"][1]["parts"][0]["thoughtSignature"] == "c2lnX3RleHQ="
    assert payload["contents"][1]["parts"][1]["thought"] is True
    assert payload["contents"][1]["parts"][1]["thoughtSignature"] == "c2lnX3JlYXNvbmluZw=="
    assert payload["contents"][1]["parts"][2]["functionCall"]["id"] == "call_1"
    assert payload["contents"][1]["parts"][2]["thoughtSignature"] == "skip_thought_signature_validator"
    assert payload["contents"][2]["parts"][0]["functionResponse"]["response"] == {"output": "Found a red circle."}
    assert payload["contents"][2]["parts"][0]["functionResponse"]["parts"][0]["inlineData"]["mimeType"] == "image/png"
    assert payload["tools"][0]["functionDeclarations"][0]["name"] == "inspect_image"
    assert payload["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"] == {
        "type": "object",
        "properties": {"detail": {"type": "string"}},
        "required": [],
        "additionalProperties": False,
    }
    assert payload["toolConfig"] == {
        "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["inspect_image"]}
    }
    assert payload["generationConfig"]["temperature"] == 0.2
    assert payload["generationConfig"]["maxOutputTokens"] == 512
    assert payload["generationConfig"]["responseMimeType"] == "application/json"
    assert payload["generationConfig"]["responseSchema"]["type"] == "object"
    assert payload["generationConfig"]["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingLevel": "HIGH",
    }


def test_gemini_build_payload_omits_foreign_reasoning_signature_and_downgrades_to_text() -> None:
    provider = GeminiProvider()
    model = _gemini_model()
    request = GenerateRequest(
        messages=[
            AssistantMessage(
                content=[
                    ReasoningBlock(
                        text="Foreign reasoning",
                        signature="Zm9yZWlnbg==",
                        protocol_meta={
                            "gemini_thought_signature": "Zm9yZWlnbg==",
                            "gemini_provider": "gemini",
                            "gemini_model": "gemini-2.5-flash",
                        },
                    )
                ]
            )
        ]
    )

    payload = provider.build_payload(model, request, RequestOptions())

    assert payload["contents"][0]["parts"][0] == {"text": "Foreign reasoning"}


def test_gemini_build_payload_uses_gemini2_budget_and_disable_rules() -> None:
    provider = GeminiProvider()
    model = _gemini_model(model="gemini-2.5-flash")
    enabled_request = GenerateRequest(
        messages=[UserMessage(content="hi")],
        reasoning=ReasoningConfig(effort="high"),
    )
    disabled_request = GenerateRequest(
        messages=[UserMessage(content="hi")],
        reasoning=ReasoningConfig(enabled=False),
    )

    enabled_payload = provider.build_payload(model, enabled_request, RequestOptions())
    disabled_payload = provider.build_payload(model, disabled_request, RequestOptions())

    assert enabled_payload["generationConfig"]["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingBudget": 24576,
    }
    assert disabled_payload["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


def test_gemini_build_payload_uses_gemini3_disable_rules() -> None:
    provider = GeminiProvider()
    pro_model = _gemini_model(model="gemini-3-pro-preview")
    flash_model = _gemini_model(model="gemini-3-flash-preview")
    request = GenerateRequest(
        messages=[UserMessage(content="hi")],
        reasoning=ReasoningConfig(enabled=False),
    )

    pro_payload = provider.build_payload(pro_model, request, RequestOptions())
    flash_payload = provider.build_payload(flash_model, request, RequestOptions())

    assert pro_payload["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "LOW"}
    assert flash_payload["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "MINIMAL"}


def test_gemini_build_payload_disables_default_thinking_when_not_requested() -> None:
    provider = GeminiProvider()
    model = _gemini_model(model="gemini-2.5-flash")
    request = GenerateRequest(messages=[UserMessage(content="hi")])

    payload = provider.build_payload(model, request, RequestOptions())

    assert payload["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


def test_gemini_build_payload_accepts_generated_assistant_message_verbatim() -> None:
    provider = GeminiProvider()
    model = _gemini_model()
    assistant = AssistantMessage(
        provider="gemini",
        model="gemini-3-pro-preview",
        api_family="gemini-generate-content",
        finish_reason="tool_call",
        response_id="resp_gemini",
        content=[
            ReasoningBlock(
                text="Need a tool",
                signature="sig_reasoning",
                protocol_meta={
                    "gemini_thought_signature": "c2lnX3JlYXNvbmluZw==",
                    "gemini_provider": "gemini",
                    "gemini_model": "gemini-3-pro-preview",
                },
            ),
            ToolCallBlock(
                id="call_1",
                name="lookup",
                arguments={"id": "alpha"},
                protocol_meta={
                    "gemini_provider": "gemini",
                    "gemini_model": "gemini-3-pro-preview",
                },
            ),
        ],
    )

    payload = provider.build_payload(
        model,
        GenerateRequest(
            messages=[
                UserMessage(content="first"),
                assistant,
                ToolResultMessage(tool_call_id="call_1", tool_name="lookup", content=[TextBlock(text="ok")]),
                UserMessage(content="next"),
            ]
        ),
        RequestOptions(),
    )

    assert payload["contents"][1]["role"] == "model"
    assert payload["contents"][1]["parts"][0]["thought"] is True
    assert payload["contents"][1]["parts"][1]["functionCall"]["id"] == "call_1"


@pytest.mark.asyncio
async def test_gemini_stream_response_normalizes_reasoning_tool_calls_and_usage() -> None:
    provider = GeminiProvider()
    model = _gemini_model()
    request = GenerateRequest(messages=[UserMessage(content="Hello")])
    response = _FakeStreamResponse(
        [
            {
                "responseId": "resp_gemini",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"thought": True, "text": "Thinking", "thoughtSignature": "sig_thought"}
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
            },
            {
                "responseId": "resp_gemini",
                "modelVersion": "gemini-3-pro-001",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"thought": True, "text": "Thinking more", "thoughtSignature": "sig_thought"},
                                {"text": "Answer ready"},
                                {
                                    "functionCall": {"name": "lookup", "args": {"query": "weather"}},
                                    "thoughtSignature": "sig_tool",
                                },
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 4,
                    "thoughtsTokenCount": 2,
                    "totalTokenCount": 9,
                    "cachedContentTokenCount": 1,
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
        "usage",
        "reasoning_delta",
        "reasoning_end",
        "text_start",
        "text_delta",
        "text_end",
        "tool_call_start",
        "tool_call_delta",
        "tool_call_end",
        "usage",
        "response_end",
    ]
    assert events[3].usage.completeness == "partial"
    assert events[-2].usage.input_tokens == 4
    assert events[-2].usage.output_tokens == 4
    assert events[-2].usage.reasoning_tokens == 2
    assert events[-2].usage.cache_read_tokens == 1
    assert events[-2].usage.total_tokens == 9
    assert events[-2].usage.completeness == "final"
    assert events[-1].response.response_id == "resp_gemini"
    assert events[-1].response.finish_reason == "tool_call"
    assert events[-1].response.provider_meta["gemini_model_version"] == "gemini-3-pro-001"
    assert events[-1].response.content[0].text == "Thinking more"
    assert events[-1].response.content[0].signature == "sig_thought"
    assert events[-1].response.content[0].protocol_meta["gemini_provider"] == "gemini"
    assert events[-1].response.content[0].protocol_meta["gemini_model"] == "gemini-3-pro-preview"
    assert events[-1].response.content[1].text == "Answer ready"
    assert events[-1].response.content[2].arguments == {"query": "weather"}
    assert events[-1].response.content[2].protocol_meta["gemini_thought_signature"] == "sig_tool"


@pytest.mark.asyncio
async def test_gemini_stream_response_captures_usage_without_candidates() -> None:
    provider = GeminiProvider()
    model = _gemini_model(model="gemini-2.5-flash")
    request = GenerateRequest(messages=[UserMessage(content="Hello")])
    response = _FakeStreamResponse(
        [
            {
                "responseId": "resp_usage_only",
                "promptFeedback": {"blockReason": "SAFETY"},
                "usageMetadata": {
                    "promptTokenCount": 6,
                    "candidatesTokenCount": 0,
                    "thoughtsTokenCount": 0,
                    "totalTokenCount": 6,
                    "cachedContentTokenCount": 2,
                },
            }
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

    assert [event.type for event in events] == ["response_start", "usage", "usage", "response_end"]
    assert events[1].usage.completeness == "partial"
    assert events[1].usage.input_tokens == 4
    assert events[1].usage.output_tokens == 0
    assert events[1].usage.cache_read_tokens == 2
    assert events[1].usage.total_tokens == 6
    assert events[2].usage.completeness == "final"
    assert events[2].usage.input_tokens == 4
    assert events[2].usage.output_tokens == 0
    assert events[2].usage.cache_read_tokens == 2
    assert events[2].usage.total_tokens == 6
    assert events[-1].response.response_id == "resp_usage_only"
    assert events[-1].response.finish_reason == "content_filter"
    assert events[-1].response.usage.input_tokens == 4
    assert events[-1].response.usage.cache_read_tokens == 2


@pytest.mark.asyncio
async def test_gemini_stream_response_defaults_missing_tool_args_to_empty_object() -> None:
    provider = GeminiProvider()
    model = _gemini_model(model="gemini-2.5-flash")
    request = GenerateRequest(messages=[UserMessage(content="status")])
    response = _FakeStreamResponse(
        [
            {
                "responseId": "resp_tool",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {"name": "get_status"},
                                }
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
            }
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

    assert events[-1].response.content[0].arguments == {}
    assert events[-1].response.finish_reason == "tool_call"


@pytest.mark.asyncio
async def test_gemini_stream_response_normalizes_duplicate_tool_call_ids() -> None:
    provider = GeminiProvider()
    model = _gemini_model(model="gemini-3-pro-preview")
    request = GenerateRequest(messages=[UserMessage(content="status")])
    response = _FakeStreamResponse(
        [
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"functionCall": {"id": "dup", "name": "one", "args": {}}},
                                {"functionCall": {"id": "dup", "name": "two", "args": {}}},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ]
            }
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

    tool_calls = events[-1].response.content
    assert tool_calls[0].id == "dup"
    assert tool_calls[1].id != "dup"


@pytest.mark.asyncio
async def test_gemini_stream_response_handles_reasoning_then_text_at_same_part_position() -> None:
    provider = GeminiProvider()
    model = _gemini_model(model="gemini-3-pro-preview")
    request = GenerateRequest(messages=[UserMessage(content="status")])
    response = _FakeStreamResponse(
        [
            {
                "responseId": "resp_transition",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "thought": True,
                                    "text": "Thinking",
                                    "thoughtSignature": "c2ln",
                                }
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
            },
            {
                "responseId": "resp_transition",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": "Answer",
                                }
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
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
        "reasoning_end",
        "text_start",
        "text_delta",
        "text_end",
        "response_end",
    ]
    assert events[-1].response.content[0].type == "reasoning"
    assert events[-1].response.content[0].text == "Thinking"
    assert events[-1].response.content[0].signature == "c2ln"
    assert events[-1].response.content[1].type == "text"
    assert events[-1].response.content[1].text == "Answer"


@pytest.mark.asyncio
async def test_gemini_stream_response_handles_text_then_tool_call_at_same_part_position() -> None:
    provider = GeminiProvider()
    model = _gemini_model(model="gemini-3-pro-preview")
    request = GenerateRequest(messages=[UserMessage(content="status")])
    response = _FakeStreamResponse(
        [
            {
                "responseId": "resp_transition_tool",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": "Let me check",
                                }
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
            },
            {
                "responseId": "resp_transition_tool",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {"id": "call_1", "name": "lookup", "args": {"id": "alpha"}},
                                }
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
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
        "text_end",
        "tool_call_start",
        "tool_call_delta",
        "tool_call_end",
        "response_end",
    ]
    assert events[-1].response.content[0].type == "text"
    assert events[-1].response.content[0].text == "Let me check"
    assert events[-1].response.content[1].type == "tool_call"
    assert events[-1].response.content[1].id == "call_1"
    assert events[-1].response.content[1].name == "lookup"
    assert events[-1].response.content[1].arguments == {"id": "alpha"}
    assert events[-1].response.finish_reason == "tool_call"


@pytest.mark.asyncio
async def test_gemini_stream_response_handles_reordered_text_and_reasoning_parts() -> None:
    provider = GeminiProvider()
    model = _gemini_model(model="gemini-3-pro-preview")
    request = GenerateRequest(messages=[UserMessage(content="status")])
    response = _FakeStreamResponse(
        [
            {
                "responseId": "resp_reordered",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"thought": True, "text": "Think", "thoughtSignature": "c2ln"},
                                {"text": "Ans"},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
            },
            {
                "responseId": "resp_reordered",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Answer"},
                                {"thought": True, "text": "Thinking", "thoughtSignature": "c2ln"},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
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
        "reasoning_end",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "reasoning_start",
        "reasoning_delta",
        "reasoning_end",
        "response_end",
    ]
    assert events[-1].response.content[0].type == "reasoning"
    assert events[-1].response.content[0].text == "Think"
    assert events[-1].response.content[1].type == "text"
    assert events[-1].response.content[1].text == "Answer"
    assert events[-1].response.content[2].type == "reasoning"
    assert events[-1].response.content[2].text == "Thinking"


@pytest.mark.asyncio
async def test_gemini_stream_response_accepts_array_payloads_from_live_api() -> None:
    provider = GeminiProvider()
    model = _gemini_model(model="gemini-2.5-flash")
    request = GenerateRequest(messages=[UserMessage(content="hello")])
    response = _FakeArrayStreamResponse(
        [
            [
                {
                    "responseId": "resp_array",
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "streamed"}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 2,
                        "candidatesTokenCount": 1,
                        "totalTokenCount": 3,
                    },
                }
            ]
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
        "text_start",
        "text_delta",
        "text_end",
        "usage",
        "response_end",
    ]
    assert events[-1].response.response_id == "resp_array"
    assert events[-1].response.content[0].text == "streamed"


@pytest.mark.asyncio
async def test_gemini_stream_response_decodes_http_error_body() -> None:
    provider = GeminiProvider()
    model = _gemini_model(model="gemini-2.5-flash")
    request = GenerateRequest(messages=[UserMessage(content="hello")])
    response = HttpResponse(
        status_code=400,
        headers={},
        content=(
            b'{"error":{"code":400,"message":"Detailed Gemini error","status":"INVALID_ARGUMENT"}}'
        ),
        url="https://example.test/gemini",
    )

    events = [
        event
        async for event in provider.stream_response(
            model=model,
            request=request,
            options=RequestOptions(),
            http=_ErrorHttpTransport(response),
        )
    ]

    assert [event.type for event in events] == ["error"]
    assert events[0].error.code == "invalid_argument"
    assert events[0].error.message == "Detailed Gemini error"
    assert events[0].error.status_code == 400


@pytest.mark.asyncio
async def test_gemini_stream_response_decodes_invalid_api_key_error_details() -> None:
    provider = GeminiProvider()
    model = _gemini_model(model="gemini-2.5-flash")
    request = GenerateRequest(messages=[UserMessage(content="hello")])
    response = HttpResponse(
        status_code=400,
        headers={},
        content=(
            b'[{"error":{"code":400,"status":"INVALID_ARGUMENT","details":['
            b'{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"API_KEY_INVALID","domain":"googleapis.com"},'
            b'{"@type":"type.googleapis.com/google.rpc.LocalizedMessage","locale":"en-US","message":"API key expired. Please renew the API key."}'
            b']}}]'
        ),
        url="https://example.test/gemini",
    )

    events = [
        event
        async for event in provider.stream_response(
            model=model,
            request=request,
            options=RequestOptions(),
            http=_ErrorHttpTransport(response),
        )
    ]

    assert [event.type for event in events] == ["error"]
    assert events[0].error.code == "authentication_error"
    assert events[0].error.message == "API key expired. Please renew the API key."
    assert isinstance(exception_from_error_info(events[0].error), AuthenticationError)