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
from connect.auth_env import resolve_env_auth
from tests.integration._live_test_utils import (
    build_lookup_status_tool_result,
    build_lookup_status_tool_spec,
)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("INTEGRATION_TEST"),
        reason="Set INTEGRATION_TEST to run live integration tests",
    ),
]


ANTHROPIC_TEXT_MODEL = os.getenv("ANTHROPIC_MODEL", "anthropic/claude-sonnet-4-6")
ANTHROPIC_VISION_MODEL = os.getenv("ANTHROPIC_VISION_MODEL", ANTHROPIC_TEXT_MODEL)
RED_PIXEL_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAGUlEQVR4nGP4z8DwnxLMMGrAqAGjBgwXAwAwxP4QHCfkAAAAAABJRU5ErkJggg=="


def _require_anthropic_key() -> None:
    assert os.getenv("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY must be set when INTEGRATION_TEST is enabled"


def _anthropic_auth():
    auth = resolve_env_auth("anthropic", env=os.environ)
    assert auth is not None, "ANTHROPIC_API_KEY must resolve to transport auth"
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


def _print_usage(label: str, response) -> None:
    print(f"{label} usage: {response.usage.model_dump()}")


@pytest.mark.asyncio
async def test_anthropic_generate_live() -> None:
    _require_anthropic_key()

    async with AsyncLLMClient() as client:
        response = await client.generate(
            ANTHROPIC_TEXT_MODEL,
            GenerateRequest(
                messages=[UserMessage(content="Reply with exactly the word: pong")],
                max_output_tokens=16,
            ),
            options=RequestOptions(auth=_anthropic_auth()),
        )

    text = _text_from_response(response).lower()
    _print_usage("anthropic generate", response)
    assert response.provider == "anthropic"
    assert response.api_family == "anthropic-messages"
    assert response.usage.completeness in {"final", "partial", "none"}
    assert "pong" in text


@pytest.mark.asyncio
async def test_anthropic_stream_live_emits_text_events_and_final_response() -> None:
    _require_anthropic_key()

    async with AsyncLLMClient() as client:
        stream = client.stream(
            ANTHROPIC_TEXT_MODEL,
            GenerateRequest(
                messages=[UserMessage(content="Reply with exactly the word: streamed")],
                max_output_tokens=16,
            ),
            options=RequestOptions(auth=_anthropic_auth()),
        )

        event_types: list[str] = []
        async for event in stream:
            event_types.append(event.type)

        response = await stream.final_response()

    _print_usage("anthropic stream", response)
    assert event_types[0] == "response_start"
    assert "text_delta" in event_types or "text_end" in event_types
    assert event_types[-1] == "response_end"
    assert "streamed" in _text_from_response(response).lower()


@pytest.mark.asyncio
async def test_anthropic_tool_call_and_tool_result_round_trip_live() -> None:
    _require_anthropic_key()

    first_prompt = (
        "I am checking a rollout. Call lookup_status exactly once for id 'alpha' "
        "and wait for the tool result before answering."
    )
    second_prompt = (
        "Thanks. Now check id 'beta' with the same tool and then compare beta with alpha "
        "in one short sentence."
    )
    tool_spec = build_lookup_status_tool_spec()

    async with AsyncLLMClient() as client:
        first_tool_response = await client.generate(
            ANTHROPIC_TEXT_MODEL,
            GenerateRequest(
                messages=[UserMessage(content=first_prompt)],
                tools=[tool_spec],
                tool_choice=SpecificToolChoice(name="lookup_status"),
                max_output_tokens=128,
            ),
            options=RequestOptions(auth=_anthropic_auth()),
        )

        first_tool_calls = _tool_calls_from_response(first_tool_response)
        assert first_tool_calls, f"expected a tool call, got content: {first_tool_response.content!r}"
        first_tool_call = first_tool_calls[0]
        assert first_tool_call.name == "lookup_status"
        assert first_tool_call.arguments.get("id") == "alpha"

        first_history = [
            UserMessage(content=first_prompt),
            first_tool_response,
            build_lookup_status_tool_result(first_tool_call, status="green"),
        ]
        first_answer_response = await client.generate(
            ANTHROPIC_TEXT_MODEL,
            GenerateRequest(
                messages=first_history,
                system_prompt="Answer in one short sentence and include the exact status value.",
                max_output_tokens=64,
            ),
            options=RequestOptions(auth=_anthropic_auth()),
        )

        second_history = [
            *first_history,
            first_answer_response,
            UserMessage(content=second_prompt),
        ]
        second_tool_response = await client.generate(
            ANTHROPIC_TEXT_MODEL,
            GenerateRequest(
                messages=second_history,
                tools=[tool_spec],
                tool_choice=SpecificToolChoice(name="lookup_status"),
                max_output_tokens=128,
            ),
            options=RequestOptions(auth=_anthropic_auth()),
        )

        second_tool_calls = _tool_calls_from_response(second_tool_response)
        assert second_tool_calls, f"expected a tool call, got content: {second_tool_response.content!r}"
        second_tool_call = second_tool_calls[0]
        assert second_tool_call.name == "lookup_status"
        assert second_tool_call.arguments.get("id") == "beta"

        final_response = await client.generate(
            ANTHROPIC_TEXT_MODEL,
            GenerateRequest(
                messages=[
                    *second_history,
                    second_tool_response,
                    build_lookup_status_tool_result(second_tool_call, status="yellow"),
                ],
                system_prompt=(
                    "Answer naturally in one short sentence and mention alpha and beta with their exact "
                    "status values."
                ),
                max_output_tokens=96,
            ),
            options=RequestOptions(auth=_anthropic_auth()),
        )

    _print_usage("anthropic tool call alpha", first_tool_response)
    _print_usage("anthropic tool result alpha", first_answer_response)
    _print_usage("anthropic tool call beta", second_tool_response)
    _print_usage("anthropic tool result", final_response)
    final_text = _text_from_response(final_response).lower()
    assert "alpha" in final_text
    assert "beta" in final_text
    assert "green" in final_text
    assert "yellow" in final_text


@pytest.mark.asyncio
async def test_anthropic_multimodal_user_image_and_tool_result_image_live() -> None:
    _require_anthropic_key()

    async with AsyncLLMClient() as client:
        image_response = await client.generate(
            ANTHROPIC_VISION_MODEL,
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
            options=RequestOptions(auth=_anthropic_auth()),
        )

        image_text = _text_from_response(image_response).lower()
        assert image_text
        assert _contains_any_color(image_text)

        tool_image_response = await client.generate(
            ANTHROPIC_VISION_MODEL,
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
            options=RequestOptions(auth=_anthropic_auth()),
        )

    _print_usage("anthropic image", image_response)
    _print_usage("anthropic tool image", tool_image_response)
    tool_image_text = _text_from_response(tool_image_response).lower()
    assert tool_image_text
    assert "red" in tool_image_text or "square" in tool_image_text