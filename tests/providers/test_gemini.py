from __future__ import annotations

import json

import pytest

from connect.providers import GeminiProvider
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


class _FakeHttpTransport:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def stream(self, method: str, url: str, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.response


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
        "reasoning_delta",
        "reasoning_end",
        "text_start",
        "text_delta",
        "text_end",
        "tool_call_start",
        "tool_call_delta",
        "usage",
        "tool_call_end",
        "usage",
        "response_end",
    ]
    assert events[-4].usage.completeness == "partial"
    assert events[-2].usage.input_tokens == 4
    assert events[-2].usage.output_tokens == 4
    assert events[-2].usage.reasoning_tokens == 2
    assert events[-2].usage.cache_read_tokens == 1
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