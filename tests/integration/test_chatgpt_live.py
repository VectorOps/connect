from __future__ import annotations

import os
import pathlib
import tempfile

import pytest

from connect import AsyncLLMClient, GenerateRequest, RequestOptions, UserMessage


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("INTEGRATION_TEST"),
        reason="Set INTEGRATION_TEST to run live integration tests",
    ),
]


CHATGPT_TEXT_MODEL = os.getenv("CHATGPT_MODEL", "chatgpt/gpt-5.4-mini")


def _credentials_file_path() -> pathlib.Path:
    return pathlib.Path(
        os.getenv(
            "CHATGPT_CREDENTIALS_FILE",
            str(pathlib.Path(tempfile.gettempdir()) / "connect-chatgpt-oauth-test.json"),
        )
    )


def _text_from_response(response) -> str:
    return "\n".join(block.text for block in response.content if block.type == "text")


@pytest.mark.asyncio
async def test_chatgpt_generate_live_uses_oauth_access_token() -> None:
    os.environ.setdefault("CHATGPT_CREDENTIALS_FILE", str(_credentials_file_path()))

    async with AsyncLLMClient() as client:
        response = await client.generate(
            CHATGPT_TEXT_MODEL,
            GenerateRequest(
                messages=[UserMessage(content="Reply with exactly the word: pong")],
            ),
            options=RequestOptions(provider_options={"session_id": "connect-integration-chatgpt-generate"}),
        )

    text = _text_from_response(response).lower()
    assert response.provider == "chatgpt"
    assert response.api_family == "chatgpt-responses"
    assert response.response_id is not None
    assert response.usage.completeness in {"final", "partial", "none"}
    assert "pong" in text


@pytest.mark.asyncio
async def test_chatgpt_stream_live_emits_text_events_and_final_response() -> None:
    os.environ.setdefault("CHATGPT_CREDENTIALS_FILE", str(_credentials_file_path()))

    async with AsyncLLMClient() as client:
        stream = client.stream(
            CHATGPT_TEXT_MODEL,
            GenerateRequest(
                messages=[UserMessage(content="Reply with exactly the word: streamed")],
            ),
            options=RequestOptions(provider_options={"session_id": "connect-integration-chatgpt-stream"}),
        )

        event_types: list[str] = []
        async for event in stream:
            event_types.append(event.type)

        response = await stream.final_response()

    assert event_types[0] == "response_start"
    assert "text_delta" in event_types or "text_end" in event_types
    assert event_types[-1] == "response_end"
    assert "streamed" in _text_from_response(response).lower()