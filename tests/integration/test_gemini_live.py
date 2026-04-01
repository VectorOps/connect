from __future__ import annotations

import os

import pytest

from connect import (
    AssistantMessage,
    AsyncLLMClient,
    GenerateRequest,
    ImageBlock,
    RequestOptions,
    SpecificToolChoice,
    TextBlock,
    ToolCallBlock,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)
from connect.auth_env import resolve_transport_auth_from_env


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("INTEGRATION_TEST"),
        reason="Set INTEGRATION_TEST to run live integration tests",
    ),
]


GEMINI_TEXT_MODEL = os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", GEMINI_TEXT_MODEL)
RED_PIXEL_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAGUlEQVR4nGP4z8DwnxLMMGrAqAGjBgwXAwAwxP4QHCfkAAAAAABJRU5ErkJggg=="


def _require_gemini_key() -> None:
    assert os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"), (
        "GEMINI_API_KEY or GOOGLE_API_KEY must be set when INTEGRATION_TEST is enabled"
    )


def _gemini_auth():
    auth = resolve_transport_auth_from_env("gemini", env=os.environ)
    assert auth is not None, "Gemini auth could not be resolved from environment"
    return auth


def _text_from_response(response) -> str:
    return "\n".join(block.text for block in response.content if block.type == "text")


def _tool_calls_from_response(response) -> list:
    return [block for block in response.content if block.type == "tool_call"]


def _contains_any_color(text: str) -> bool:
    return any(
        color in text.lower()
        for color in (
            "red",
            "green",
            "blue",
            "yellow",
            "orange",
            "purple",
            "square",
            "pixel",
            "image",
        )
    )


@pytest.mark.asyncio
async def test_gemini_generate_live() -> None:
    _require_gemini_key()

    async with AsyncLLMClient() as client:
        response = await client.generate(
            GEMINI_TEXT_MODEL,
            GenerateRequest(
                messages=[UserMessage(content="Reply with exactly the word: pong")],
                max_output_tokens=16,
            ),
            options=RequestOptions(auth=_gemini_auth()),
        )

    text = _text_from_response(response).lower()
    assert response.provider == "gemini"
    assert response.api_family == "gemini-generate-content"
    assert response.usage.completeness in {"final", "partial", "none"}
    assert "pong" in text


@pytest.mark.asyncio
async def test_gemini_stream_live_emits_text_events_and_final_response() -> None:
    _require_gemini_key()

    async with AsyncLLMClient() as client:
        stream = client.stream(
            GEMINI_TEXT_MODEL,
            GenerateRequest(
                messages=[UserMessage(content="Reply with exactly the word: streamed")],
                max_output_tokens=16,
            ),
            options=RequestOptions(auth=_gemini_auth()),
        )

        event_types: list[str] = []
        async for event in stream:
            event_types.append(event.type)

        response = await stream.final_response()

    assert event_types[0] == "response_start"
    assert "text_delta" in event_types or "text_end" in event_types
    assert event_types[-1] == "response_end"
    assert "streamed" in _text_from_response(response).lower()


@pytest.mark.asyncio
async def test_gemini_tool_call_and_tool_result_round_trip_live() -> None:
    _require_gemini_key()

    tool_prompt = "Use the lookup_status tool for id 'alpha'. Do not answer directly."
    tool_spec = ToolSpec(
        name="lookup_status",
        description="Lookup a status string for a known identifier.",
        input_schema={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        },
    )

    async with AsyncLLMClient() as client:
        tool_response = await client.generate(
            GEMINI_TEXT_MODEL,
            GenerateRequest(
                messages=[UserMessage(content=tool_prompt)],
                tools=[tool_spec],
                tool_choice=SpecificToolChoice(name="lookup_status"),
                max_output_tokens=128,
            ),
            options=RequestOptions(auth=_gemini_auth()),
        )

        tool_calls = _tool_calls_from_response(tool_response)
        assert tool_calls, f"expected a tool call, got content: {tool_response.content!r}"
        tool_call = tool_calls[0]
        assert tool_call.name == "lookup_status"
        assert tool_call.arguments.get("id") == "alpha"

        final_response = await client.generate(
            GEMINI_TEXT_MODEL,
            GenerateRequest(
                messages=[
                    UserMessage(content=tool_prompt),
                    AssistantMessage(content=tool_response.content),
                    ToolResultMessage(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        content=[TextBlock(text="status=green")],
                    ),
                ],
                system_prompt="Answer in one short sentence and include the exact status value.",
                max_output_tokens=64,
            ),
            options=RequestOptions(auth=_gemini_auth()),
        )

    final_text = _text_from_response(final_response).lower()
    assert "green" in final_text


@pytest.mark.asyncio
async def test_gemini_multimodal_user_image_and_tool_result_image_live() -> None:
    _require_gemini_key()

    async with AsyncLLMClient() as client:
        image_response = await client.generate(
            GEMINI_VISION_MODEL,
            GenerateRequest(
                messages=[
                    UserMessage(
                        content=[
                            TextBlock(text="Answer with a single lowercase color word for this solid-color image."),
                            ImageBlock(data=RED_PIXEL_PNG_BASE64, mime_type="image/png"),
                        ]
                    )
                ],
                max_output_tokens=16,
            ),
            options=RequestOptions(auth=_gemini_auth()),
        )

        image_text = _text_from_response(image_response).lower()
        assert image_text
        assert _contains_any_color(image_text)

        tool_image_response = await client.generate(
            GEMINI_VISION_MODEL,
            GenerateRequest(
                messages=[
                    UserMessage(content="Use the provided tool result to answer what color the sample image is."),
                    AssistantMessage(
                        content=[
                            ToolCallBlock(
                                id="call_visual_1",
                                name="inspect_image",
                                arguments={"target": "sample"},
                            )
                        ]
                    ),
                    ToolResultMessage(
                        tool_call_id="call_visual_1",
                        tool_name="inspect_image",
                        content=[
                            TextBlock(text="The tool attached a solid red square image."),
                            ImageBlock(data=RED_PIXEL_PNG_BASE64, mime_type="image/png"),
                        ],
                    ),
                ],
                max_output_tokens=32,
            ),
            options=RequestOptions(auth=_gemini_auth()),
        )

    tool_image_text = _text_from_response(tool_image_response).lower()
    assert tool_image_text
    assert "red" in tool_image_text or "square" in tool_image_text