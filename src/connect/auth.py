from __future__ import annotations

import base64
import inspect
import json
import typing

import pydantic


class ResolvedAuth(pydantic.BaseModel):
    headers: dict[str, str] = pydantic.Field(default_factory=dict)
    params: dict[str, str] = pydantic.Field(default_factory=dict)


class AuthContext(pydantic.BaseModel):
    provider: str | None = None
    model: str | None = None
    api_family: str | None = None
    method: str | None = None
    url: str | None = None


@typing.runtime_checkable
class TransportAuth(typing.Protocol):
    async def resolve(self, context: AuthContext | None = None) -> ResolvedAuth:
        ...

    async def refresh(self, context: AuthContext | None = None) -> bool:
        ...


class AccessToken:
    def __init__(self, token: str, token_type: str = "Bearer") -> None:
        self.token = token
        self.token_type = token_type


class NoAuth:
    async def resolve(self, context: AuthContext | None = None) -> ResolvedAuth:
        return ResolvedAuth()

    async def refresh(self, context: AuthContext | None = None) -> bool:
        return False


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

    async def resolve(self, context: AuthContext | None = None) -> ResolvedAuth:
        return ResolvedAuth(headers={self.header_name: f"{self.prefix}{self.api_key}"})

    async def refresh(self, context: AuthContext | None = None) -> bool:
        return False


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

    async def resolve(self, context: AuthContext | None = None) -> ResolvedAuth:
        resolved = await super().resolve(context)
        if self.account_id:
            resolved.headers["chatgpt-account-id"] = self.account_id
        return resolved


class QueryAPIKeyAuth:
    def __init__(self, api_key: str, param_name: str = "key") -> None:
        self.api_key = api_key
        self.param_name = param_name

    async def resolve(self, context: AuthContext | None = None) -> ResolvedAuth:
        return ResolvedAuth(params={self.param_name: self.api_key})

    async def refresh(self, context: AuthContext | None = None) -> bool:
        return False


class CallableTokenAuth:
    def __init__(self, get_token: typing.Callable[[], typing.Awaitable[str] | str]) -> None:
        self.get_token = get_token

    async def resolve(self, context: AuthContext | None = None) -> ResolvedAuth:
        token = self.get_token()
        if inspect.isawaitable(token):
            token = await token
        return ResolvedAuth(headers={"Authorization": f"Bearer {token}"})

    async def refresh(self, context: AuthContext | None = None) -> bool:
        return False


class RefreshingOAuthAuth:
    def __init__(
        self,
        get_access_token: typing.Callable[[], typing.Awaitable[AccessToken] | AccessToken],
    ) -> None:
        self.get_access_token = get_access_token

    async def resolve(self, context: AuthContext | None = None) -> ResolvedAuth:
        token = self.get_access_token()
        if inspect.isawaitable(token):
            token = await token
        return ResolvedAuth(headers={"Authorization": f"{token.token_type} {token.token}"})

    async def refresh(self, context: AuthContext | None = None) -> bool:
        return False


class CompositeAuth:
    def __init__(self, strategies: list[TransportAuth]) -> None:
        self.strategies = strategies

    async def resolve(self, context: AuthContext | None = None) -> ResolvedAuth:
        resolved = ResolvedAuth()
        for strategy in self.strategies:
            current = await strategy.resolve(context)
            resolved.headers.update(current.headers)
            resolved.params.update(current.params)
        return resolved

    async def refresh(self, context: AuthContext | None = None) -> bool:
        refreshed = False
        for strategy in self.strategies:
            refreshed = await strategy.refresh(context) or refreshed
        return refreshed


async def resolve_transport_auth(
    auth: TransportAuth | None,
    *,
    context: AuthContext | None = None,
) -> ResolvedAuth:
    if auth is None:
        return ResolvedAuth()
    return await auth.resolve(context)