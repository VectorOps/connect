from __future__ import annotations

import asyncio
import base64
import json
import pathlib
import time
import urllib.parse

import pytest

from connect.auth import ResolvedAuth
from connect.auth_router import EnvironmentCredentialManager
from connect.auth_env import chatgpt_access_token_from_env
from connect.credentials import (
    ChatGPTCredentialProvider,
    ChatGPTCredentials,
    ChatGPTOAuthSettings,
    CredentialManager,
    CredentialStore,
    FileCredentialManager,
    OAuth2RefreshableAuth,
    OAuthCredentials,
    OAuthAuthInfo,
    OAuthLoginCallbacks,
    OAuthPrompt,
    build_console_login_callbacks,
    build_chatgpt_authorization_url,
    generate_pkce_pair,
    parse_authorization_input,
)


def _jwt(account_id: str = "acct_123") -> str:
    def _encode(payload: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")

    return (
        f"{_encode({'alg': 'none', 'typ': 'JWT'})}."
        f"{_encode({'https://api.openai.com/auth': {'chatgpt_account_id': account_id}})}."
        "signature"
    )


def test_generate_pkce_pair_returns_verifier_and_challenge() -> None:
    verifier, challenge = generate_pkce_pair()

    assert verifier
    assert challenge
    assert verifier != challenge
    assert "+" not in verifier
    assert "/" not in verifier


def test_parse_authorization_input_supports_callback_url() -> None:
    parsed = parse_authorization_input("http://localhost:1455/auth/callback?code=abc&state=xyz")

    assert parsed == {"code": "abc", "state": "xyz"}


def test_build_chatgpt_authorization_url_contains_expected_query_params() -> None:
    url = build_chatgpt_authorization_url(
        code_challenge="challenge_123",
        state="state_123",
        settings=ChatGPTOAuthSettings(originator="connect-test"),
    )
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert parsed.netloc == "auth.openai.com"
    assert params["client_id"] == ["app_EMoamEEZ73f0CkXaXp7hrann"]
    assert params["code_challenge"] == ["challenge_123"]
    assert params["state"] == ["state_123"]
    assert params["originator"] == ["connect-test"]


def test_credential_store_save_and_load_round_trip(tmp_path: pathlib.Path) -> None:
    manager = FileCredentialManager(tmp_path / "credentials.json")
    path = tmp_path / "credentials.json"
    credentials = ChatGPTCredentials(
        access_token=_jwt(),
        refresh_token="refresh_123",
        expires_at=time.time() + 300,
        account_id="acct_123",
    )

    asyncio.run(manager.set("chatgpt", credentials))
    loaded = asyncio.run(manager.get("chatgpt"))

    assert isinstance(loaded, ChatGPTCredentials)
    assert loaded.access_token == credentials.access_token
    assert loaded.account_id == "acct_123"


@pytest.mark.asyncio
async def test_credential_manager_auth_from_file_refreshes_expired_credentials(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "credentials.json"
    store = CredentialStore()

    class _Provider(ChatGPTCredentialProvider):
        async def refresh(self, credentials):
            return ChatGPTCredentials(
                access_token=_jwt("acct_new"),
                refresh_token="refresh_new",
                expires_at=time.time() + 600,
                account_id="acct_new",
            )

    from connect.credentials.base import CredentialRegistry

    registry = CredentialRegistry()
    registry.register(_Provider())
    manager = FileCredentialManager(path, registry=registry, store=store)
    store.save(
        path,
        ChatGPTCredentials(
            access_token=_jwt("acct_old"),
            refresh_token="refresh_old",
            expires_at=time.time() - 10,
            account_id="acct_old",
        ),
    )

    auth = await manager.auth("chatgpt")
    resolved = await auth.resolve()

    assert resolved.headers["Authorization"].startswith("Bearer ")
    assert resolved.headers["chatgpt-account-id"] == "acct_new"
    loaded = await manager.get("chatgpt")
    assert loaded.account_id == "acct_new"


@pytest.mark.asyncio
async def test_chatgpt_credential_provider_to_auth_returns_chatgpt_auth() -> None:
    provider = ChatGPTCredentialProvider()
    auth = provider.build_resolved_auth(
        ChatGPTCredentials(
            access_token=_jwt("acct_555"),
            refresh_token="refresh_555",
            expires_at=time.time() + 60,
            account_id="acct_555",
        )
    )

    assert isinstance(auth, ResolvedAuth)
    assert auth.headers["chatgpt-account-id"] == "acct_555"


@pytest.mark.asyncio
async def test_file_credential_manager_login_creates_credentials_when_missing(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "credentials.json"

    class _Provider(ChatGPTCredentialProvider):
        async def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
            callbacks.on_auth(OAuthAuthInfo(url="https://example.test/auth"))
            return ChatGPTCredentials(
                access_token=_jwt("acct_login"),
                refresh_token="refresh_login",
                expires_at=time.time() + 600,
                account_id="acct_login",
            )

    from connect.credentials.base import CredentialRegistry

    registry = CredentialRegistry()
    registry.register(_Provider())
    manager = FileCredentialManager(path, registry=registry)
    credentials = await manager.login(
        "chatgpt",
        build_console_login_callbacks(provider="chatgpt", env={}),
    )
    resolved = await manager.resolve("chatgpt")

    assert path.exists()
    assert credentials.account_id == "acct_login"
    assert resolved.headers["chatgpt-account-id"] == "acct_login"


@pytest.mark.asyncio
async def test_in_memory_credential_manager_login_and_resolve() -> None:
    class _Provider(ChatGPTCredentialProvider):
        async def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
            return ChatGPTCredentials(
                access_token=_jwt("acct_memory"),
                refresh_token="refresh_memory",
                expires_at=time.time() + 600,
                account_id="acct_memory",
            )

    from connect.credentials.base import CredentialRegistry

    registry = CredentialRegistry()
    registry.register(_Provider())
    manager = CredentialManager(registry=registry)

    credentials = await manager.login(
        "chatgpt",
        build_console_login_callbacks(provider="chatgpt", env={}),
    )
    resolved = await manager.resolve("chatgpt")

    assert credentials.account_id == "acct_memory"
    assert resolved.headers["chatgpt-account-id"] == "acct_memory"


@pytest.mark.asyncio
async def test_oauth2_refreshable_auth_refreshes_expired_credentials() -> None:
    saved: list[ChatGPTCredentials] = []

    class _Provider(ChatGPTCredentialProvider):
        async def refresh(self, credentials):
            return ChatGPTCredentials(
                access_token=_jwt("acct_refreshed"),
                refresh_token="refresh_refreshed",
                expires_at=time.time() + 600,
                account_id="acct_refreshed",
            )

    auth = OAuth2RefreshableAuth(
        provider=_Provider(),
        credentials=ChatGPTCredentials(
            access_token=_jwt("acct_expired"),
            refresh_token="refresh_expired",
            expires_at=time.time() - 60,
            account_id="acct_expired",
        ),
        persist_callback=saved.append,
    )

    resolved = await auth.resolve()

    assert resolved.headers["chatgpt-account-id"] == "acct_refreshed"
    assert saved[-1].account_id == "acct_refreshed"


def test_chatgpt_access_token_from_env_returns_auth() -> None:
    auth = chatgpt_access_token_from_env(
        {"CHATGPT_ACCESS_TOKEN": _jwt("acct_env"), "CHATGPT_ACCOUNT_ID": "acct_env"}
    )

    assert auth is not None
    assert auth.account_id == "acct_env"


@pytest.mark.asyncio
async def test_chatgpt_login_uses_manual_prompt_when_callback_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
    from connect.credentials import chatgpt as chatgpt_credentials

    async def _fake_exchange(*, code: str, verifier: str, redirect_uri: str, session=None):
        assert code == "manual_code"
        assert verifier
        assert redirect_uri == "http://localhost:1455/auth/callback"
        return ChatGPTCredentials(
            access_token=_jwt("acct_manual"),
            refresh_token="refresh_manual",
            expires_at=time.time() + 600,
            account_id="acct_manual",
        )

    monkeypatch.setattr(chatgpt_credentials, "exchange_chatgpt_authorization_code", _fake_exchange)
    monkeypatch.setattr(chatgpt_credentials.LocalOAuthCallbackServer, "start", classmethod(lambda cls, **kwargs: (_ for _ in ()).throw(OSError("bind failed"))))

    seen: list[OAuthAuthInfo] = []

    async def _prompt(prompt: OAuthPrompt) -> str:
        assert "authorization code" in prompt.message.lower()
        return "manual_code"

    credentials = await chatgpt_credentials.login_chatgpt(
        OAuthLoginCallbacks(
            on_auth=seen.append,
            on_prompt=_prompt,
        )
    )

    assert seen
    assert seen[0].url.startswith("https://auth.openai.com/oauth/authorize?")
    assert credentials.account_id == "acct_manual"


@pytest.mark.asyncio
async def test_environment_credential_manager_loads_and_saves_oauth_credentials(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "credentials.json"
    manager = EnvironmentCredentialManager({"CHATGPT_CREDENTIALS_FILE": str(path)})
    credentials = ChatGPTCredentials(
        access_token=_jwt("acct_store"),
        refresh_token="refresh_store",
        expires_at=time.time() + 600,
        account_id="acct_store",
    )

    await manager.set_oauth2_credentials("chatgpt", credentials)
    loaded = await manager.get_oauth2_credentials("chatgpt")

    assert loaded is not None
    assert loaded.provider == "chatgpt"
    assert loaded.access_token == credentials.access_token