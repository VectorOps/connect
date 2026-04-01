from __future__ import annotations

import os
import typing

from .auth import AuthContext, BearerTokenAuth, ChatGPTAccessTokenAuth, HeaderAPIKeyAuth, ResolvedAuth, TransportAuth
from .credentials.base import (
    CredentialRegistry,
    CredentialStore,
    OAuth2Credentials,
    OAuth2RefreshableAuth,
    OAuthLoginCallbacks,
    build_console_login_callbacks,
    build_default_credential_registry,
)


class AuthCredentialManager(typing.Protocol):
    async def get_token(self, name: str, *, context: AuthContext | None = None) -> str | None:
        ...

    async def set_token(self, name: str, value: str | None, *, context: AuthContext | None = None) -> None:
        ...

    async def get_oauth2_credentials(
        self,
        provider: str,
        *,
        context: AuthContext | None = None,
    ) -> OAuth2Credentials | None:
        ...

    async def set_oauth2_credentials(
        self,
        provider: str,
        credentials: OAuth2Credentials | None,
        *,
        context: AuthContext | None = None,
    ) -> None:
        ...

    async def get_oauth_login_callbacks(
        self,
        provider: str,
        *,
        context: AuthContext | None = None,
    ) -> OAuthLoginCallbacks | None:
        ...


class EnvironmentCredentialManager:
    def __init__(
        self,
        env: typing.MutableMapping[str, str] | None = None,
        *,
        credential_registry: CredentialRegistry | None = None,
        credential_store: CredentialStore | None = None,
    ) -> None:
        self._env = env if env is not None else os.environ
        self._credential_registry = credential_registry or build_default_credential_registry()
        self._credential_store = credential_store or CredentialStore()

    async def get_token(self, name: str, *, context: AuthContext | None = None) -> str | None:
        value = self._env.get(name)
        if not value:
            return None
        return value

    async def set_token(self, name: str, value: str | None, *, context: AuthContext | None = None) -> None:
        if value is None:
            self._env.pop(name, None)
            return
        self._env[name] = value

    async def get_oauth2_credentials(
        self,
        provider: str,
        *,
        context: AuthContext | None = None,
    ) -> OAuth2Credentials | None:
        path = self._oauth_credentials_path(provider)
        if path is None:
            return None
        try:
            return self._credential_store.load(path, provider=provider, registry=self._credential_registry)
        except ValueError:
            return None

    async def set_oauth2_credentials(
        self,
        provider: str,
        credentials: OAuth2Credentials | None,
        *,
        context: AuthContext | None = None,
    ) -> None:
        path = self._oauth_credentials_path(provider)
        if path is None:
            return
        if credentials is None:
            self._credential_store.delete(path, provider=provider)
            return
        self._credential_store.save(path, credentials)

    async def get_oauth_login_callbacks(
        self,
        provider: str,
        *,
        context: AuthContext | None = None,
    ) -> OAuthLoginCallbacks | None:
        manual_input_env_vars: dict[str, tuple[str, ...]] = {
            "chatgpt": ("CHATGPT_OAUTH_REDIRECT_URL",),
        }
        return build_console_login_callbacks(
            provider=provider,
            env=self._env,
            manual_input_env_vars=manual_input_env_vars.get(provider, ()),
        )

    def _oauth_credentials_path(self, provider: str) -> str | None:
        env_var_by_provider = {
            "chatgpt": "CHATGPT_CREDENTIALS_FILE",
        }
        env_var = env_var_by_provider.get(provider)
        if env_var is None:
            return None
        value = self._env.get(env_var)
        if not value:
            return None
        return value


class DynamicAuthRouter(TransportAuth):
    def __init__(
        self,
        credential_manager: AuthCredentialManager | None = None,
        *,
        credential_registry: CredentialRegistry | None = None,
    ) -> None:
        self.credential_manager = credential_manager or EnvironmentCredentialManager()
        self.credential_registry = credential_registry or build_default_credential_registry()
        self._auth_by_provider: dict[str, TransportAuth] = {}

    async def resolve(self, context: AuthContext | None = None) -> ResolvedAuth:
        auth = await self._get_auth(context)
        if auth is None:
            return ResolvedAuth()
        return await auth.resolve(context)

    async def refresh(self, context: AuthContext | None = None) -> bool:
        auth = await self._get_auth(context)
        if auth is None:
            return False
        return await auth.refresh(context)

    async def _get_auth(self, context: AuthContext | None) -> TransportAuth | None:
        provider = self._resolve_provider(context)
        if provider is None:
            return None
        cached = self._auth_by_provider.get(provider)
        if cached is not None:
            return cached

        auth = await self._build_auth(provider, context)
        if auth is not None:
            self._auth_by_provider[provider] = auth
        return auth

    async def _build_auth(self, provider: str, context: AuthContext | None) -> TransportAuth | None:
        oauth_auth = await self._build_oauth_auth(provider, context)
        if oauth_auth is not None:
            return oauth_auth

        token_auth = await self._build_token_auth(provider, context)
        if token_auth is not None:
            return token_auth

        login_callbacks = await self.credential_manager.get_oauth_login_callbacks(provider, context=context)
        if login_callbacks is None:
            return None
        try:
            provider_adapter = self.credential_registry.get(provider)
        except ValueError:
            return None

        credentials = await provider_adapter.login(login_callbacks)
        await self.credential_manager.set_oauth2_credentials(provider, credentials, context=context)
        return OAuth2RefreshableAuth(
            provider=provider_adapter,
            credentials=credentials,
            persist_callback=lambda updated: self.credential_manager.set_oauth2_credentials(
                provider,
                updated,
                context=context,
            ),
        )

    async def _build_oauth_auth(self, provider: str, context: AuthContext | None) -> TransportAuth | None:
        try:
            provider_adapter = self.credential_registry.get(provider)
        except ValueError:
            return None

        credentials = await self.credential_manager.get_oauth2_credentials(provider, context=context)
        if credentials is None:
            return None
        return OAuth2RefreshableAuth(
            provider=provider_adapter,
            credentials=credentials,
            persist_callback=lambda updated: self.credential_manager.set_oauth2_credentials(
                provider,
                updated,
                context=context,
            ),
        )

    async def _build_token_auth(self, provider: str, context: AuthContext | None) -> TransportAuth | None:
        if provider == "openai":
            token = await self.credential_manager.get_token("OPENAI_API_KEY", context=context)
            return BearerTokenAuth(token) if token else None
        if provider == "anthropic":
            token = await self.credential_manager.get_token("ANTHROPIC_API_KEY", context=context)
            return HeaderAPIKeyAuth(api_key=token, header_name="x-api-key", prefix="") if token else None
        if provider == "gemini":
            token = await self.credential_manager.get_token("GEMINI_API_KEY", context=context)
            if not token:
                token = await self.credential_manager.get_token("GOOGLE_API_KEY", context=context)
            return HeaderAPIKeyAuth(api_key=token, header_name="x-goog-api-key", prefix="") if token else None
        if provider == "openrouter":
            token = await self.credential_manager.get_token("OPENROUTER_API_KEY", context=context)
            return BearerTokenAuth(token) if token else None
        if provider == "chatgpt":
            token = await self.credential_manager.get_token("CHATGPT_ACCESS_TOKEN", context=context)
            if not token:
                return None
            account_id = await self.credential_manager.get_token("CHATGPT_ACCOUNT_ID", context=context)
            return ChatGPTAccessTokenAuth(token, account_id=account_id)
        return None

    def _resolve_provider(self, context: AuthContext | None) -> str | None:
        if context is None:
            return None
        if context.provider:
            return context.provider
        if context.model and "/" in context.model:
            provider, _, _ = context.model.partition("/")
            return provider or None
        return None