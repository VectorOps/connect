from __future__ import annotations

import asyncio
import time
import typing
import urllib.parse

import aiohttp
import pydantic

from ..auth import ChatGPTAccessTokenAuth, extract_chatgpt_account_id
from ..types import AuthStrategy
from .base import CredentialProvider, OAuthAuthInfo, OAuthCredentials, OAuthLoginCallbacks, OAuthPrompt
from .helpers import LocalOAuthCallbackServer, create_oauth_state, generate_pkce_pair, parse_authorization_input


CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
DEFAULT_REDIRECT_HOST = "localhost"
DEFAULT_REDIRECT_PORT = 1455
DEFAULT_CALLBACK_PATH = "/auth/callback"
DEFAULT_SCOPE = "openid profile email offline_access"
DEFAULT_MANUAL_REDIRECT_URL = (
    f"http://{DEFAULT_REDIRECT_HOST}:{DEFAULT_REDIRECT_PORT}{DEFAULT_CALLBACK_PATH}"
    "?code=replace-me&state=replace-me"
)


class ChatGPTCredentials(OAuthCredentials):
    provider: typing.Literal["chatgpt"] = "chatgpt"
    account_id: str


class ChatGPTOAuthSettings(pydantic.BaseModel):
    redirect_host: str = DEFAULT_REDIRECT_HOST
    redirect_port: int = DEFAULT_REDIRECT_PORT
    callback_path: str = DEFAULT_CALLBACK_PATH
    originator: str = "connect"

    @property
    def redirect_uri(self) -> str:
        return f"http://{self.redirect_host}:{self.redirect_port}{self.callback_path}"


def build_chatgpt_authorization_url(
    *,
    code_challenge: str,
    state: str,
    settings: ChatGPTOAuthSettings | None = None,
) -> str:
    oauth_settings = settings or ChatGPTOAuthSettings()
    url = urllib.parse.urlparse(AUTHORIZE_URL)
    query = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": oauth_settings.redirect_uri,
            "scope": DEFAULT_SCOPE,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": oauth_settings.originator,
        }
    )
    return urllib.parse.urlunparse(url._replace(query=query))


async def exchange_chatgpt_authorization_code(
    *,
    code: str,
    verifier: str,
    redirect_uri: str,
    session: aiohttp.ClientSession | None = None,
) -> ChatGPTCredentials:
    return await _token_request(
        {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
        },
        session=session,
    )


async def refresh_chatgpt_access_token(
    refresh_token: str,
    *,
    session: aiohttp.ClientSession | None = None,
) -> ChatGPTCredentials:
    return await _token_request(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLIENT_ID,
        },
        session=session,
    )


async def login_chatgpt(
    callbacks: OAuthLoginCallbacks,
    *,
    settings: ChatGPTOAuthSettings | None = None,
) -> ChatGPTCredentials:
    oauth_settings = settings or ChatGPTOAuthSettings()
    verifier, challenge = generate_pkce_pair()
    state = create_oauth_state()
    authorization_url = build_chatgpt_authorization_url(
        code_challenge=challenge,
        state=state,
        settings=oauth_settings,
    )

    server: LocalOAuthCallbackServer | None = None
    try:
        try:
            server = await LocalOAuthCallbackServer.start(
                host=oauth_settings.redirect_host,
                port=oauth_settings.redirect_port,
                callback_path=oauth_settings.callback_path,
                state=state,
            )
            redirect_uri = server.callback_url
        except OSError:
            redirect_uri = oauth_settings.redirect_uri
            if callbacks.on_progress is not None:
                callbacks.on_progress("Failed to bind OAuth callback server; falling back to manual code paste.")

        if redirect_uri != oauth_settings.redirect_uri:
            authorization_url = build_chatgpt_authorization_url(
                code_challenge=challenge,
                state=state,
                settings=oauth_settings.model_copy(
                    update={
                        "redirect_host": oauth_settings.redirect_host,
                        "redirect_port": int(urllib.parse.urlparse(redirect_uri).port or oauth_settings.redirect_port),
                    }
                ),
            )

        callbacks.on_auth(
            OAuthAuthInfo(
                url=authorization_url,
                instructions="Open the URL in a browser and complete login. The local callback server will capture the code if available.",
            )
        )

        code = await _obtain_authorization_code(callbacks, server=server, state=state)
        return await exchange_chatgpt_authorization_code(code=code, verifier=verifier, redirect_uri=redirect_uri)
    finally:
        if server is not None:
            await server.close()


async def _obtain_authorization_code(
    callbacks: OAuthLoginCallbacks,
    *,
    server: LocalOAuthCallbackServer | None,
    state: str,
) -> str:
    manual_error: BaseException | None = None
    manual_input: str | None = None

    async def wait_for_manual_input() -> None:
        nonlocal manual_input, manual_error
        if callbacks.on_manual_code_input is None:
            return
        try:
            manual_input = await callbacks.on_manual_code_input()
        except BaseException as exc:  # pragma: no cover - defensive propagation path
            manual_error = exc
        finally:
            if server is not None:
                server.cancel_wait()

    manual_task = None
    if callbacks.on_manual_code_input is not None:
        manual_task = asyncio.create_task(wait_for_manual_input())

    try:
        callback_result = await server.wait_for_callback() if server is not None else None
        if manual_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await manual_task
    finally:
        if manual_task is not None and not manual_task.done():
            manual_task.cancel()

    import contextlib

    if manual_error is not None:
        raise manual_error

    if callback_result and callback_result.get("error"):
        raise RuntimeError(callback_result.get("error_description") or callback_result["error"])

    if callback_result and callback_result.get("code"):
        return callback_result["code"]

    if manual_input:
        parsed = parse_authorization_input(manual_input)
        if parsed.get("state") and parsed["state"] != state:
            raise RuntimeError("State mismatch")
        if parsed.get("error"):
            raise RuntimeError(parsed.get("error_description") or parsed["error"])
        if parsed.get("code"):
            return parsed["code"]

    prompted = await callbacks.on_prompt(
        OAuthPrompt(
            message="Paste the authorization code or full redirect URL:",
            placeholder="http://localhost:1455/auth/callback?code=...&state=...",
        )
    )
    parsed = parse_authorization_input(prompted)
    if parsed.get("state") and parsed["state"] != state:
        raise RuntimeError("State mismatch")
    if parsed.get("error"):
        raise RuntimeError(parsed.get("error_description") or parsed["error"])
    if not parsed.get("code"):
        raise RuntimeError("Missing authorization code")
    return parsed["code"]


async def _token_request(
    form_data: dict[str, str],
    *,
    session: aiohttp.ClientSession | None,
) -> ChatGPTCredentials:
    owns_session = session is None
    http = session or aiohttp.ClientSession()
    try:
        async with http.post(
            TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=form_data,
        ) as response:
            payload = await response.json(content_type=None)
            if response.status >= 400:
                message = "Token request failed"
                if isinstance(payload, dict):
                    message = str(payload.get("error_description") or payload.get("error") or message)
                raise RuntimeError(message)
    finally:
        if owns_session:
            await http.close()

    if not isinstance(payload, dict):
        raise RuntimeError("Token response was not a JSON object")

    access_token = payload.get("access_token")
    refresh_token = payload.get("refresh_token")
    expires_in = payload.get("expires_in")
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("Token response missing access_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise RuntimeError("Token response missing refresh_token")
    if not isinstance(expires_in, int | float):
        raise RuntimeError("Token response missing expires_in")

    account_id = extract_chatgpt_account_id(access_token)
    if not account_id:
        raise RuntimeError("Failed to extract chatgpt_account_id from access token")

    return ChatGPTCredentials(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=time.time() + float(expires_in) - 300,
        account_id=account_id,
    )


class ChatGPTCredentialProvider:
    provider_name = "chatgpt"
    credentials_type = ChatGPTCredentials

    async def login(self, callbacks: OAuthLoginCallbacks) -> ChatGPTCredentials:
        return await login_chatgpt(callbacks)

    async def refresh(self, credentials: OAuthCredentials) -> ChatGPTCredentials:
        if not isinstance(credentials, ChatGPTCredentials):
            credentials = ChatGPTCredentials.model_validate(credentials.model_dump(mode="json"))
        return await refresh_chatgpt_access_token(credentials.refresh_token)

    def to_auth(self, credentials: OAuthCredentials) -> AuthStrategy:
        if not isinstance(credentials, ChatGPTCredentials):
            credentials = ChatGPTCredentials.model_validate(credentials.model_dump(mode="json"))
        return ChatGPTAccessTokenAuth(credentials.access_token, credentials.account_id)