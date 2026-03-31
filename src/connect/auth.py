from __future__ import annotations

import dataclasses
import os
import typing

from .types import AuthStrategy


@dataclasses.dataclass(slots=True)
class AccessToken:
    token: str
    token_type: str = "Bearer"


class NoAuth:
    async def apply(self, request) -> None:
        return None


@dataclasses.dataclass(slots=True)
class HeaderAPIKeyAuth:
    api_key: str
    header_name: str = "Authorization"
    prefix: str = "Bearer "

    async def apply(self, request) -> None:
        request.headers[self.header_name] = f"{self.prefix}{self.api_key}"


@dataclasses.dataclass(slots=True)
class BearerTokenAuth(HeaderAPIKeyAuth):
    def __init__(self, token: str) -> None:
        self.api_key = token
        self.header_name = "Authorization"
        self.prefix = "Bearer "

    @property
    def token(self) -> str:
        return self.api_key


@dataclasses.dataclass(slots=True)
class QueryAPIKeyAuth:
    api_key: str
    param_name: str = "key"

    async def apply(self, request) -> None:
        params = dict(getattr(request, "params", {}) or {})
        params[self.param_name] = self.api_key
        request.params = params


@dataclasses.dataclass(slots=True)
class CallableTokenAuth:
    get_token: typing.Callable[[], typing.Awaitable[str] | str]

    async def apply(self, request) -> None:
        token = self.get_token()
        if typing.isawaitable(token):
            token = await token
        request.headers["Authorization"] = f"Bearer {token}"


@dataclasses.dataclass(slots=True)
class RefreshingOAuthAuth:
    get_access_token: typing.Callable[[], typing.Awaitable[AccessToken] | AccessToken]

    async def apply(self, request) -> None:
        token = self.get_access_token()
        if typing.isawaitable(token):
            token = await token
        request.headers["Authorization"] = f"{token.token_type} {token.token}"


@dataclasses.dataclass(slots=True)
class CompositeAuth:
    strategies: list[AuthStrategy]

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


@dataclasses.dataclass(slots=True)
class _MutableRequest:
    headers: dict[str, str]
    params: dict[str, str]


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


def default_env_auth(provider: str) -> AuthStrategy | None:
    factories: dict[str, typing.Callable[[], AuthStrategy | None]] = {
        "openai": openai_api_key_from_env,
        "anthropic": anthropic_api_key_from_env,
        "gemini": gemini_api_key_from_env,
        "openrouter": openrouter_api_key_from_env,
    }
    factory = factories.get(provider)
    if factory is None:
        return None
    return factory()