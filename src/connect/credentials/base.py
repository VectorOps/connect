from __future__ import annotations

import asyncio
import inspect
import json
import pathlib
import time
import typing

import pydantic

from ..auth import AuthContext, ResolvedAuth, TransportAuth


class OAuthAuthInfo(pydantic.BaseModel):
    url: str
    instructions: str | None = None


class OAuthPrompt(pydantic.BaseModel):
    message: str
    placeholder: str | None = None
    allow_empty: bool = False


class OAuthLoginCallbacks(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    on_auth: typing.Callable[[OAuthAuthInfo], None]
    on_prompt: typing.Callable[[OAuthPrompt], typing.Awaitable[str]]
    on_progress: typing.Callable[[str], None] | None = None
    on_manual_code_input: typing.Callable[[], typing.Awaitable[str]] | None = None


class OAuth2Credentials(pydantic.BaseModel):
    provider: str
    access_token: str
    refresh_token: str | None = None
    expires_at: float | None = None
    token_type: str = "Bearer"
    metadata: dict[str, typing.Any] = pydantic.Field(default_factory=dict)

    def is_expired(self, *, skew_seconds: float = 30.0) -> bool:
        if self.expires_at is None:
            return False
        return time.time() >= (self.expires_at - skew_seconds)


OAuthCredentials = OAuth2Credentials


class StoredCredentialsDocument(pydantic.BaseModel):
    version: int = 1
    credentials: dict[str, dict[str, typing.Any]] = pydantic.Field(default_factory=dict)


class CredentialProvider(typing.Protocol):
    provider_name: str
    credentials_type: type[OAuth2Credentials]

    async def login(self, callbacks: OAuthLoginCallbacks) -> OAuth2Credentials:
        ...

    async def refresh(self, credentials: OAuth2Credentials) -> OAuth2Credentials:
        ...

    def build_resolved_auth(self, credentials: OAuth2Credentials) -> ResolvedAuth:
        ...


LoginCallbacksFactory = typing.Callable[[str], OAuthLoginCallbacks]


class CredentialRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, CredentialProvider] = {}

    def register(self, provider: CredentialProvider) -> None:
        self._providers[provider.provider_name] = provider

    def get(self, provider: str) -> CredentialProvider:
        try:
            return self._providers[provider]
        except KeyError as exc:
            raise ValueError(f"Credential provider '{provider}' is not registered") from exc

    def list(self) -> list[str]:
        return sorted(self._providers)


class CredentialStore:
    def load_document(self, path: str | pathlib.Path) -> StoredCredentialsDocument:
        file_path = pathlib.Path(path)
        if not file_path.exists():
            return StoredCredentialsDocument()
        return StoredCredentialsDocument.model_validate_json(file_path.read_text(encoding="utf-8"))

    def save_document(self, path: str | pathlib.Path, document: StoredCredentialsDocument) -> None:
        file_path = pathlib.Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")

    def save(self, path: str | pathlib.Path, credentials: OAuthCredentials) -> None:
        document = self.load_document(path)
        document.credentials[credentials.provider] = credentials.model_dump(mode="json")
        self.save_document(path, document)

    def load(
        self,
        path: str | pathlib.Path,
        *,
        provider: str,
        registry: CredentialRegistry,
    ) -> OAuthCredentials:
        document = self.load_document(path)
        payload = document.credentials.get(provider)
        if payload is None:
            raise ValueError(f"No stored credentials for provider '{provider}'")
        provider_adapter = registry.get(provider)
        return provider_adapter.credentials_type.model_validate(payload)

    def delete(self, path: str | pathlib.Path, *, provider: str) -> None:
        document = self.load_document(path)
        if provider not in document.credentials:
            return
        del document.credentials[provider]
        self.save_document(path, document)


PersistCallback = typing.Callable[[OAuth2Credentials], typing.Awaitable[None] | None]


class OAuth2RefreshableAuth:
    def __init__(
        self,
        *,
        provider: CredentialProvider,
        credentials: OAuth2Credentials,
        persist_callback: PersistCallback | None = None,
    ) -> None:
        self._provider = provider
        self._credentials = credentials
        self._persist_callback = persist_callback

    async def resolve(self, context: AuthContext | None = None) -> ResolvedAuth:
        if self._credentials.is_expired() and self._credentials.refresh_token:
            await self._refresh_credentials()
        return self._provider.build_resolved_auth(self._credentials)

    async def refresh(self, context: AuthContext | None = None) -> bool:
        if not self._credentials.refresh_token:
            return False
        await self._refresh_credentials()
        return True

    async def _refresh_credentials(self) -> None:
        self._credentials = await self._provider.refresh(self._credentials)
        if self._persist_callback is None:
            return
        result = self._persist_callback(self._credentials)
        if inspect.isawaitable(result):
            await result


class PersistedCredentialAuth(TransportAuth):
    def __init__(
        self,
        *,
        provider: CredentialProvider,
        store: CredentialStore,
        path: str | pathlib.Path,
        login_callbacks_factory: LoginCallbacksFactory | None = None,
        auto_login_if_missing: bool = False,
    ) -> None:
        self._provider = provider
        self._store = store
        self._path = pathlib.Path(path)
        self._login_callbacks_factory = login_callbacks_factory
        self._auto_login_if_missing = auto_login_if_missing
        self._auth: OAuth2RefreshableAuth | None = None

    async def resolve(self, context: AuthContext | None = None) -> ResolvedAuth:
        auth = await self._ensure_auth()
        return await auth.resolve(context)

    async def refresh(self, context: AuthContext | None = None) -> bool:
        auth = await self._ensure_auth()
        return await auth.refresh(context)

    async def _ensure_auth(self) -> OAuth2RefreshableAuth:
        if self._auth is not None:
            return self._auth

        registry = CredentialRegistry()
        registry.register(self._provider)
        try:
            credentials = self._store.load(self._path, provider=self._provider.provider_name, registry=registry)
        except ValueError:
            if not self._auto_login_if_missing or self._login_callbacks_factory is None:
                raise
            credentials = await self._provider.login(self._login_callbacks_factory(self._provider.provider_name))
            self._store.save(self._path, credentials)

        self._auth = OAuth2RefreshableAuth(
            provider=self._provider,
            credentials=credentials,
            persist_callback=lambda updated: self._store.save(self._path, updated),
        )
        return self._auth


class CredentialManager:
    def __init__(
        self,
        *,
        registry: CredentialRegistry | None = None,
        store: CredentialStore | None = None,
    ) -> None:
        self.registry = registry or build_default_credential_registry()
        self.store = store or CredentialStore()
        self._credentials: dict[str, OAuth2Credentials] = {}

    async def get(self, provider: str) -> OAuth2Credentials | None:
        return self._credentials.get(provider)

    async def set(self, provider: str, credentials: OAuth2Credentials | None) -> None:
        if credentials is None:
            self._credentials.pop(provider, None)
            return
        self._credentials[provider] = credentials

    async def login(
        self,
        provider: str,
        callbacks: OAuthLoginCallbacks,
    ) -> OAuth2Credentials:
        adapter = self.registry.get(provider)
        credentials = await adapter.login(callbacks)
        await self.set(provider, credentials)
        return credentials

    async def refresh(
        self,
        provider: str,
        credentials: OAuthCredentials,
    ) -> OAuth2Credentials:
        adapter = self.registry.get(provider)
        refreshed = await adapter.refresh(credentials)
        await self.set(provider, refreshed)
        return refreshed

    async def resolve(self, provider: str) -> ResolvedAuth:
        auth = await self.auth(provider)
        return await auth.resolve()

    async def auth(self, provider: str) -> OAuth2RefreshableAuth:
        adapter = self.registry.get(provider)
        credentials = await self.get(provider)
        if credentials is None:
            raise ValueError(f"No stored credentials for provider '{provider}'")
        return OAuth2RefreshableAuth(
            provider=adapter,
            credentials=credentials,
            persist_callback=lambda updated: self.set(provider, updated),
        )

    def save(self, path: str | pathlib.Path, credentials: OAuthCredentials) -> None:
        self.store.save(path, credentials)

    def load(self, provider: str, path: str | pathlib.Path) -> OAuthCredentials:
        return self.store.load(path, provider=provider, registry=self.registry)


class FileCredentialManager(CredentialManager):
    def __init__(
        self,
        path: str | pathlib.Path,
        *,
        registry: CredentialRegistry | None = None,
        store: CredentialStore | None = None,
    ) -> None:
        super().__init__(registry=registry, store=store)
        self.path = pathlib.Path(path)

    async def get(self, provider: str) -> OAuth2Credentials | None:
        try:
            return self.load(provider, self.path)
        except ValueError:
            return None

    async def set(self, provider: str, credentials: OAuth2Credentials | None) -> None:
        if credentials is None:
            self.store.delete(self.path, provider=provider)
            return
        self.store.save(self.path, credentials)

    async def auth(self, provider: str) -> OAuth2RefreshableAuth:
        adapter = self.registry.get(provider)
        credentials = await self.get(provider)
        if credentials is None:
            raise ValueError(f"No stored credentials for provider '{provider}'")
        return OAuth2RefreshableAuth(
            provider=adapter,
            credentials=credentials,
            persist_callback=lambda updated: self.set(provider, updated),
        )


def build_console_login_callbacks(
    *,
    provider: str,
    env: typing.Mapping[str, str] | None = None,
    manual_input_env_vars: tuple[str, ...] = (),
) -> OAuthLoginCallbacks:
    source = env or {}

    def _on_auth(info: OAuthAuthInfo) -> None:
        print(f"Open this URL in your browser to authenticate with {provider}: {info.url}")
        if info.instructions:
            print(info.instructions)

    async def _on_prompt(prompt: OAuthPrompt) -> str:
        for env_var in manual_input_env_vars:
            value = source.get(env_var)
            if value:
                return value
        return await asyncio.to_thread(input, f"{prompt.message} ")

    def _on_progress(message: str) -> None:
        print(message)

    return OAuthLoginCallbacks(
        on_auth=_on_auth,
        on_prompt=_on_prompt,
        on_progress=_on_progress,
    )


def build_default_credential_registry() -> CredentialRegistry:
    from .chatgpt import ChatGPTCredentialProvider

    registry = CredentialRegistry()
    registry.register(ChatGPTCredentialProvider())
    return registry


default_credential_registry = build_default_credential_registry()