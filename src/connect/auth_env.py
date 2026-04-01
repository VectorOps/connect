from __future__ import annotations

import os
import typing

from .auth import BearerTokenAuth, ChatGPTAccessTokenAuth, HeaderAPIKeyAuth, TransportAuth
from .auth_router import DynamicAuthRouter, EnvironmentCredentialManager
from .credentials.base import FileCredentialManager, build_console_login_callbacks


def openai_api_key_from_env(env: typing.Mapping[str, str] | None = None) -> BearerTokenAuth | None:
    source = env or os.environ
    token = source.get("OPENAI_API_KEY")
    if not token:
        return None
    return BearerTokenAuth(token)


def anthropic_api_key_from_env(env: typing.Mapping[str, str] | None = None) -> HeaderAPIKeyAuth | None:
    source = env or os.environ
    token = source.get("ANTHROPIC_API_KEY")
    if not token:
        return None
    return HeaderAPIKeyAuth(api_key=token, header_name="x-api-key", prefix="")


def gemini_api_key_from_env(env: typing.Mapping[str, str] | None = None) -> HeaderAPIKeyAuth | None:
    source = env or os.environ
    token = source.get("GEMINI_API_KEY") or source.get("GOOGLE_API_KEY")
    if not token:
        return None
    return HeaderAPIKeyAuth(api_key=token, header_name="x-goog-api-key", prefix="")


def openrouter_api_key_from_env(env: typing.Mapping[str, str] | None = None) -> BearerTokenAuth | None:
    source = env or os.environ
    token = source.get("OPENROUTER_API_KEY")
    if not token:
        return None
    return BearerTokenAuth(token)


def chatgpt_access_token_from_env(env: typing.Mapping[str, str] | None = None) -> ChatGPTAccessTokenAuth | None:
    source = env or os.environ
    token = source.get("CHATGPT_ACCESS_TOKEN")
    if not token:
        return None
    return ChatGPTAccessTokenAuth(token, account_id=source.get("CHATGPT_ACCOUNT_ID"))


def chatgpt_credentials_file_auth(env: typing.Mapping[str, str] | None = None) -> TransportAuth | None:
    source = env or os.environ
    path = source.get("CHATGPT_CREDENTIALS_FILE")
    if not path:
        return None

    manager = FileCredentialManager(path)

    class _ManagerBackedAuth:
        async def resolve(self, context=None):
            credentials = await manager.get("chatgpt")
            if credentials is None:
                await manager.login(
                    "chatgpt",
                    build_console_login_callbacks(
                        provider="chatgpt",
                        env=source,
                        manual_input_env_vars=("CHATGPT_OAUTH_REDIRECT_URL",),
                    ),
                )
            return await manager.resolve("chatgpt")

        async def refresh(self, context=None) -> bool:
            auth = await manager.auth("chatgpt")
            return await auth.refresh(context)

    return _ManagerBackedAuth()


def resolve_env_auth(
    provider: str | None = None,
    *,
    model: str | None = None,
    api_family: str | None = None,
    env: typing.Mapping[str, str] | None = None,
) -> TransportAuth | None:
    source = env or os.environ
    if provider is None and model is not None:
        return DynamicAuthRouter(credential_manager=EnvironmentCredentialManager(dict(source)))
    provider_name = provider or _provider_from_model_reference(model)
    factories: dict[str, typing.Callable[[], TransportAuth | None]] = {
        "chatgpt": lambda: chatgpt_credentials_file_auth(source) or chatgpt_access_token_from_env(source),
        "openai": lambda: openai_api_key_from_env(source),
        "anthropic": lambda: anthropic_api_key_from_env(source),
        "gemini": lambda: gemini_api_key_from_env(source),
        "openrouter": lambda: openrouter_api_key_from_env(source),
    }
    factory = factories.get(provider_name)
    if factory is None:
        return None
    return factory()


def _provider_from_model_reference(model: str | None) -> str | None:
    if not model or "/" not in model:
        return None
    provider, _, _ = model.partition("/")
    return provider or None