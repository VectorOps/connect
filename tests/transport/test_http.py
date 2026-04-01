from __future__ import annotations

import pytest

from connect.auth import AuthContext, ResolvedAuth
from connect.transport.http import HttpTransport


class _RefreshableAuth:
    def __init__(self) -> None:
        self.token = "expired"
        self.refresh_calls = 0
        self.contexts: list[AuthContext | None] = []

    async def resolve(self, context: AuthContext | None = None) -> ResolvedAuth:
        self.contexts.append(context)
        return ResolvedAuth(headers={"Authorization": f"Bearer {self.token}"})

    async def refresh(self, context: AuthContext | None = None) -> bool:
        self.refresh_calls += 1
        self.contexts.append(context)
        self.token = "fresh"
        return True


class _FakeResponse:
    def __init__(self, status: int, headers: dict[str, str] | None = None, body: bytes = b"{}") -> None:
        self.status = status
        self.headers = headers or {}
        self._body = body
        self.url = "https://example.test"
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def read(self) -> bytes:
        return self._body

    async def json(self, content_type=None):
        return {}

    async def text(self) -> str:
        return self._body.decode()

    def close(self) -> None:
        self.closed = True

    def release(self) -> None:
        self.closed = True


class _FakeSession:
    def __init__(self) -> None:
        self.closed = False
        self.calls: list[dict] = []

    async def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        auth_header = kwargs["headers"].get("Authorization")
        if auth_header == "Bearer expired":
            return _FakeResponse(401)
        return _FakeResponse(200, body=b'{"ok": true}')

    async def close(self) -> None:
        self.closed = True


def test_extract_error_message_prefers_nested_error_message() -> None:
    transport = HttpTransport.__new__(HttpTransport)

    message = transport._extract_error_message(
        {
            "error": {
                "error": {
                    "message": "vendor nested message",
                }
            }
        }
    )

    assert message == "vendor nested message"


def test_extract_error_message_uses_detail_when_message_missing() -> None:
    transport = HttpTransport.__new__(HttpTransport)

    message = transport._extract_error_message({"detail": "provider detail text"})

    assert message == "provider detail text"


@pytest.mark.asyncio
async def test_http_transport_refreshes_auth_and_retries_once() -> None:
    session = _FakeSession()
    auth = _RefreshableAuth()
    transport = HttpTransport(session=session, auth=auth)

    response = await transport.request(
        "GET",
        "https://example.test",
        provider="openai",
        model="gpt-4.1-mini",
        api_family="openai-responses",
    )

    assert response.status_code == 200
    assert auth.refresh_calls == 1
    assert len(session.calls) == 2
    assert session.calls[0]["headers"]["Authorization"] == "Bearer expired"
    assert session.calls[1]["headers"]["Authorization"] == "Bearer fresh"
    assert auth.contexts[0] is not None
    assert auth.contexts[0].provider == "openai"
    assert auth.contexts[0].model == "gpt-4.1-mini"