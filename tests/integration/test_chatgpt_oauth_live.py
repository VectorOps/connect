from __future__ import annotations

import os
import pathlib
import tempfile

import pytest

from connect import AsyncLLMClient, GenerateRequest, RequestOptions, UserMessage
from connect.auth import ChatGPTAccessTokenAuth
from connect.credentials import (
    DEFAULT_MANUAL_REDIRECT_URL,
    CredentialManager,
    OAuthAuthInfo,
    OAuthLoginCallbacks,
    OAuthPrompt,
)


CHATGPT_TEXT_MODEL = os.getenv("CHATGPT_MODEL", "chatgpt/codex-mini-latest")


def _text_from_response(response) -> str:
    return "\n".join(block.text for block in response.content if block.type == "text")


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("INTEGRATION_TEST"),
        reason="Set INTEGRATION_TEST to run interactive ChatGPT OAuth tests",
    ),
]


@pytest.mark.asyncio
async def test_chatgpt_oauth_login_live() -> None:
    manager = CredentialManager()
    seen: list[OAuthAuthInfo] = []
    redirect_url = os.getenv("CHATGPT_OAUTH_REDIRECT_URL", DEFAULT_MANUAL_REDIRECT_URL)
    persist_path = pathlib.Path(tempfile.gettempdir()) / "connect-chatgpt-oauth-test.json"

    async def _prompt(prompt: OAuthPrompt) -> str:
        return redirect_url

    def _on_auth(info: OAuthAuthInfo) -> None:
        seen.append(info)
        print(f"Open this URL in your browser to authenticate with ChatGPT: {info.url}")

    credentials = await manager.login(
        "chatgpt",
        OAuthLoginCallbacks(
            on_auth=_on_auth,
            on_prompt=_prompt,
        ),
        persist_path=persist_path,
    )

    assert seen
    assert credentials.provider == "chatgpt"
    assert credentials.access_token
    assert credentials.refresh_token
    assert credentials.account_id

    print(f"Stored ChatGPT OAuth credentials at: {persist_path}")

    async with AsyncLLMClient() as client:
        response = await client.generate(
            CHATGPT_TEXT_MODEL,
            GenerateRequest(
                messages=[UserMessage(content="Reply with exactly the word: pong")],
                max_output_tokens=16,
            ),
            options=RequestOptions(
                auth=ChatGPTAccessTokenAuth(credentials.access_token, credentials.account_id),
                provider_options={"session_id": "connect-integration-chatgpt-oauth-e2e"},
            ),
        )

    assert response.provider == "chatgpt"
    assert response.api_family == "chatgpt-responses"
    assert response.response_id is not None
    assert "pong" in _text_from_response(response).lower()