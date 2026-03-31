from __future__ import annotations

import base64
import json
import os
import typing

from .types import AuthStrategy


class AccessToken:
    def __init__(self, token: str, token_type: str = "Bearer") -> None:
        self.token = token
        self.token_type = token_type


class NoAuth:
    async def apply(self, request) -> None:
        return None


class HeaderAPIKeyAuth:
    def __init__(
        self,
        api_key: str,
        header_name: str = "Authorization",
        prefix: str = "Bearer ",
    ) -> None:
        self.api_key = api_key
        self.header_name = header_name
        self.prefix = prefix

    async def apply(self, request) -> None:
        request.headers[self.header_name] = f"{self.prefix}{self.api_key}"


class BearerTokenAuth(HeaderAPIKeyAuth):
    def __init__(self, token: str) -> None:
        self.api_key = token
        self.header_name = "Authorization"
        self.prefix = "Bearer "

    @property
    def token(self) -> str:
        return self.api_key


def _decode_jwt_payload(token: str) -> dict[str, typing.Any] | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None

    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(f"{payload}{padding}")
        data = json.loads(decoded.decode("utf-8"))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def extract_chatgpt_account_id(token: str) -> str | None:
    payload = _decode_jwt_payload(token)
    if payload is None:
        return None

    auth_payload = payload.get("https://api.openai.com/auth")
    if not isinstance(auth_payload, dict):
        return None

    account_id = auth_payload.get("chatgpt_account_id")
    if not isinstance(account_id, str) or not account_id:
        return None
    return account_id


class ChatGPTAccessTokenAuth(BearerTokenAuth):
    def __init__(self, token: str, account_id: str | None = None) -> None:
        super().__init__(token)
        self.account_id = account_id or extract_chatgpt_account_id(token)

    async def apply(self, request) -> None:
        await super().apply(request)
        if self.account_id:
            request.headers["chatgpt-account-id"] = self.account_id


class QueryAPIKeyAuth:
    def __init__(self, api_key: str, param_name: str = "key") -> None:
        self.api_key = api_key
        self.param_name = param_name

    async def apply(self, request) -> None:
        params = dict(getattr(request, "params", {}) or {})
        params[self.param_name] = self.api_key
        request.params = params


class CallableTokenAuth:
    def __init__(self, get_token: typing.Callable[[], typing.Awaitable[str] | str]) -> None:
        self.get_token = get_token

    async def apply(self, request) -> None:
        token = self.get_token()
        if typing.isawaitable(token):
            token = await token
        request.headers["Authorization"] = f"Bearer {token}"


class RefreshingOAuthAuth:
    def __init__(
        self,
        get_access_token: typing.Callable[[], typing.Awaitable[AccessToken] | AccessToken],
    ) -> None:
        self.get_access_token = get_access_token

    async def apply(self, request) -> None:
        token = self.get_access_token()
        if typing.isawaitable(token):
            token = await token
        request.headers["Authorization"] = f"{token.token_type} {token.token}"


class CompositeAuth:
    def __init__(self, strategies: list[AuthStrategy]) -> None:
        self.strategies = strategies

    async def apply(self, request) -> None:
        for strategy in self.strategies:
            await strategy.apply(request)


class AuthRegistry:
    def __init__(self) -> None:
        self._auth_by_provider: dict[str, AuthStrategy] = {}

    def register(self, provider: str, auth: AuthStrategy) -> None:
        self._auth_by_provider[provider] = auth

    def get(self, provider: str) -> AuthStrategy | None:
        return self._auth_by_provider.get(provider)


class _MutableRequest:
    def __init__(self, headers: dict[str, str], params: dict[str, str]) -> None:
        self.headers = headers
        self.params = params


async def resolve_request_auth(
    auth: AuthStrategy | None,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    resolved_headers = dict(headers or {})
    resolved_params = dict(params or {})
    if auth is None:
        return resolved_headers, resolved_params

    request = _MutableRequest(headers=resolved_headers, params=resolved_params)
    await auth.apply(request)
    return request.headers, request.params


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


def chatgpt_credentials_file_auth(env: typing.Mapping[str, str] | None = None) -> AuthStrategy | None:
    source = env or os.environ
    path = source.get("CHATGPT_CREDENTIALS_FILE")
    if not path:
        return None

    from .credentials.base import CredentialManager, build_console_login_callbacks

    manager = CredentialManager()
    return manager.auth_from_file_or_login(
        "chatgpt",
        path,
        login_callbacks_factory=lambda provider: build_console_login_callbacks(
            provider=provider,
            env=source,
            manual_input_env_vars=("CHATGPT_OAUTH_REDIRECT_URL",),
        ),
    )


def default_env_auth(provider: str) -> AuthStrategy | None:
    factories: dict[str, typing.Callable[[], AuthStrategy | None]] = {
        "chatgpt": lambda: chatgpt_credentials_file_auth() or chatgpt_access_token_from_env(),
        "openai": openai_api_key_from_env,
        "anthropic": anthropic_api_key_from_env,
        "gemini": gemini_api_key_from_env,
        "openrouter": openrouter_api_key_from_env,
    }
    factory = factories.get(provider)
    if factory is None:
        return None
    return factory()