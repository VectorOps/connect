from __future__ import annotations

import json

import pytest

from connect.auth import AuthContext, ResolvedAuth
from connect.auth import BearerTokenAuth
from connect.auth_env import resolve_env_auth
from connect.auth_router import DynamicAuthRouter, EnvironmentCredentialManager
from connect.client import AsyncLLMClient, StreamHandle
from connect.credentials import ChatGPTCredentials, CredentialManager, OAuthLoginCallbacks, OAuthPrompt
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


class _ContextAwareAuth:
    def __init__(self) -> None:
        self.contexts: list[AuthContext | None] = []

    async def resolve(self, context: AuthContext | None = None) -> ResolvedAuth:
        self.contexts.append(context)
        return ResolvedAuth(headers={"Authorization": "Bearer test-token"})

    async def refresh(self, context: AuthContext | None = None) -> bool:
        self.contexts.append(context)
        return False


class _MemoryCredentialManager:
    def __init__(self) -> None:
        self.tokens: dict[str, str] = {}
        self.oauth: dict[str, ChatGPTCredentials] = {}
        self.login_requests: list[str] = []

    async def get_token(self, name: str, *, context: AuthContext | None = None) -> str | None:
        return self.tokens.get(name)

    async def set_token(self, name: str, value: str | None, *, context: AuthContext | None = None) -> None:
        if value is None:
            self.tokens.pop(name, None)
            return
        self.tokens[name] = value

    async def get_oauth2_credentials(self, provider: str, *, context: AuthContext | None = None):
        return self.oauth.get(provider)

    async def set_oauth2_credentials(self, provider: str, credentials, *, context: AuthContext | None = None) -> None:
        if credentials is None:
            self.oauth.pop(provider, None)
            return
        self.oauth[provider] = credentials

    async def get_oauth_login_callbacks(
        self,
        provider: str,
        *,
        context: AuthContext | None = None,
    ) -> OAuthLoginCallbacks | None:
        self.login_requests.append(provider)

        async def _prompt(prompt: OAuthPrompt) -> str:
            return ""

        return OAuthLoginCallbacks(on_auth=lambda info: None, on_prompt=_prompt)


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


def test_credential_manager_registers_chatgpt_provider_by_default() -> None:
    manager = CredentialManager()

    assert "chatgpt" in manager.registry.list()


def test_resolve_env_auth_returns_openai_auth() -> None:
    auth = resolve_env_auth("openai", env={"OPENAI_API_KEY": "sk-test"})

    assert isinstance(auth, BearerTokenAuth)


@pytest.mark.asyncio
async def test_async_client_passes_model_context_to_auth() -> None:
    session = _FakeClientSession()
    model = ModelSpec(provider="openai", model="gpt-4.1-mini", api_family="openai-responses")
    model_registry = ModelRegistry([model])
    provider_registry = ProviderRegistry()
    from connect.providers import OpenAIProvider

    provider_registry.register("openai", OpenAIProvider())
    auth = _ContextAwareAuth()

    async with AsyncLLMClient(
        http_client=session,
        model_registry=model_registry,
        provider_registry=provider_registry,
    ) as client:
        await client.generate(
            "openai/gpt-4.1-mini",
            GenerateRequest(messages=[UserMessage(content="ping")]),
            options=RequestOptions(auth=auth),
        )

    assert auth.contexts
    assert any(context and context.provider == "openai" for context in auth.contexts)
    assert any(context and context.model == "gpt-4.1-mini" for context in auth.contexts)


@pytest.mark.asyncio
async def test_dynamic_auth_router_uses_credential_manager_token() -> None:
    manager = _MemoryCredentialManager()
    await manager.set_token("OPENAI_API_KEY", "sk-router")
    router = DynamicAuthRouter(credential_manager=manager)

    resolved = await router.resolve(AuthContext(provider="openai", model="gpt-4.1-mini"))

    assert resolved.headers["Authorization"] == "Bearer sk-router"


@pytest.mark.asyncio
async def test_async_client_uses_router_when_explicit_auth_missing() -> None:
    session = _FakeClientSession()
    model = ModelSpec(provider="openai", model="gpt-4.1-mini", api_family="openai-responses")
    model_registry = ModelRegistry([model])
    provider_registry = ProviderRegistry()
    from connect.providers import OpenAIProvider

    provider_registry.register("openai", OpenAIProvider())
    manager = _MemoryCredentialManager()
    await manager.set_token("OPENAI_API_KEY", "sk-router")

    async with AsyncLLMClient(
        http_client=session,
        credential_manager=manager,
        model_registry=model_registry,
        provider_registry=provider_registry,
    ) as client:
        await client.generate(
            "openai/gpt-4.1-mini",
            GenerateRequest(messages=[UserMessage(content="ping")]),
        )

    assert session.requests[0]["headers"]["Authorization"] == "Bearer sk-router"


def test_environment_credential_manager_reads_and_writes_tokens() -> None:
    env: dict[str, str] = {}
    manager = EnvironmentCredentialManager(env)

    import asyncio

    asyncio.run(manager.set_token("OPENAI_API_KEY", "sk-env"))

    assert env["OPENAI_API_KEY"] == "sk-env"


@pytest.mark.asyncio
async def test_async_client_emits_observability_events() -> None:
    session = _FakeClientSession()
    model = ModelSpec(provider="openai", model="gpt-4.1-mini", api_family="openai-responses")
    model_registry = ModelRegistry([model])
    provider_registry = ProviderRegistry()
    from connect.providers import OpenAIProvider

    provider_registry.register("openai", OpenAIProvider())
    observed: list[dict] = []

    async with AsyncLLMClient(
        http_client=session,
        model_registry=model_registry,
        provider_registry=provider_registry,
        event_hook=observed.append,
    ) as client:
        response = await client.generate(
            "openai/gpt-4.1-mini",
            GenerateRequest(messages=[UserMessage(content="ping")]),
            options=RequestOptions(auth=BearerTokenAuth("test-token")),
        )

    assert response.content[0].text == "pong"
    assert [event["type"] for event in observed] == [
        "request_start",
        "response_headers",
        "first_token",
        "request_end",
    ]
    assert observed[0]["provider"] == "openai"
    assert observed[-1]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_stream_handle_invokes_close_callback_on_terminal_event() -> None:
    closed: list[str] = []

    async def _iterator():
        from connect.types import AssistantResponse, ResponseEndEvent, Usage

        yield ResponseEndEvent(
            response=AssistantResponse(
                provider="openai",
                model="gpt-4.1-mini",
                api_family="openai-responses",
                content=[],
                finish_reason="stop",
                usage=Usage(),
                response_id="resp_test",
            )
        )

    async def _close() -> None:
        closed.append("done")

    stream = StreamHandle(_iterator(), close_callback=_close)
    response = await stream.final_response()

    assert response.response_id == "resp_test"
    assert closed == ["done"]

