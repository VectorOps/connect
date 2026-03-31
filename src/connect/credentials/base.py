from __future__ import annotations

import json
import pathlib
import time
import typing

import pydantic

from ..types import AuthStrategy


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


class OAuthCredentials(pydantic.BaseModel):
    provider: str
    access_token: str
    refresh_token: str
    expires_at: float
    token_type: str = "Bearer"
    account_id: str | None = None
    metadata: dict[str, typing.Any] = pydantic.Field(default_factory=dict)

    def is_expired(self, *, skew_seconds: float = 30.0) -> bool:
        return time.time() >= (self.expires_at - skew_seconds)


class StoredCredentialsDocument(pydantic.BaseModel):
    version: int = 1
    credentials: dict[str, dict[str, typing.Any]] = pydantic.Field(default_factory=dict)


class CredentialProvider(typing.Protocol):
    provider_name: str
    credentials_type: type[OAuthCredentials]

    async def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
        ...

    async def refresh(self, credentials: OAuthCredentials) -> OAuthCredentials:
        ...

    def to_auth(self, credentials: OAuthCredentials) -> AuthStrategy:
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


class PersistedCredentialAuth:
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

    async def apply(self, request) -> None:
        registry = CredentialRegistry()
        registry.register(self._provider)
        try:
            credentials = self._store.load(self._path, provider=self._provider.provider_name, registry=registry)
        except ValueError:
            if not self._auto_login_if_missing or self._login_callbacks_factory is None:
                raise
            credentials = await self._provider.login(self._login_callbacks_factory(self._provider.provider_name))
            self._store.save(self._path, credentials)

        if credentials.is_expired():
            credentials = await self._provider.refresh(credentials)
            self._store.save(self._path, credentials)

        auth = self._provider.to_auth(credentials)
        await auth.apply(request)


class CredentialManager:
    def __init__(
        self,
        *,
        registry: CredentialRegistry | None = None,
        store: CredentialStore | None = None,
    ) -> None:
        self.registry = registry or build_default_credential_registry()
        self.store = store or CredentialStore()

    async def login(
        self,
        provider: str,
        callbacks: OAuthLoginCallbacks,
        *,
        persist_path: str | pathlib.Path | None = None,
    ) -> OAuthCredentials:
        adapter = self.registry.get(provider)
        credentials = await adapter.login(callbacks)
        if persist_path is not None:
            self.store.save(persist_path, credentials)
        return credentials

    async def refresh(
        self,
        provider: str,
        credentials: OAuthCredentials,
        *,
        persist_path: str | pathlib.Path | None = None,
    ) -> OAuthCredentials:
        adapter = self.registry.get(provider)
        refreshed = await adapter.refresh(credentials)
        if persist_path is not None:
            self.store.save(persist_path, refreshed)
        return refreshed

    def save(self, path: str | pathlib.Path, credentials: OAuthCredentials) -> None:
        self.store.save(path, credentials)

    def load(self, provider: str, path: str | pathlib.Path) -> OAuthCredentials:
        return self.store.load(path, provider=provider, registry=self.registry)

    def auth_from_file(self, provider: str, path: str | pathlib.Path) -> PersistedCredentialAuth:
        return PersistedCredentialAuth(provider=self.registry.get(provider), store=self.store, path=path)

    def auth_from_file_or_login(
        self,
        provider: str,
        path: str | pathlib.Path,
        *,
        login_callbacks_factory: LoginCallbacksFactory,
    ) -> PersistedCredentialAuth:
        return PersistedCredentialAuth(
            provider=self.registry.get(provider),
            store=self.store,
            path=path,
            login_callbacks_factory=login_callbacks_factory,
            auto_login_if_missing=True,
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