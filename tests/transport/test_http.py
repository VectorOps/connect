from __future__ import annotations

import aiohttp
import pytest

from connect.auth import AuthContext, ResolvedAuth
from connect.exceptions import TransientProviderError
from connect.transport.http import HttpStatusError, HttpTransport


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
        import json

        return json.loads(self._body.decode())

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


class _BrokenContent:
    async def readany(self) -> bytes:
        raise aiohttp.ClientPayloadError(
            "Response payload is not completed: <TransferEncodingError: 400, message='Not enough data to satisfy transfer length header.'>"
        )

    async def readline(self) -> bytes:
        raise aiohttp.ClientPayloadError(
            "Response payload is not completed: <TransferEncodingError: 400, message='Not enough data to satisfy transfer length header.'>"
        )


class _BrokenStreamResponse:
    def __init__(self) -> None:
        self.status = 200
        self.headers = {}
        self.url = "https://example.test"
        self.content = _BrokenContent()

    def close(self) -> None:
        return None

    async def read(self) -> bytes:
        raise aiohttp.ClientPayloadError(
            "Response payload is not completed: <TransferEncodingError: 400, message='Not enough data to satisfy transfer length header.'>"
        )

    async def text(self) -> str:
        raise aiohttp.ClientPayloadError(
            "Response payload is not completed: <TransferEncodingError: 400, message='Not enough data to satisfy transfer length header.'>"
        )

    async def json(self, content_type=None):
        raise aiohttp.ClientPayloadError(
            "Response payload is not completed: <TransferEncodingError: 400, message='Not enough data to satisfy transfer length header.'>"
        )


@pytest.mark.asyncio
async def test_http_transport_raises_raw_status_error_with_response_body() -> None:
    class _ErrorSession:
        closed = False

        async def request(self, method, url, **kwargs):
            return _FakeResponse(400, body=b'{"error":{"message":"bad request"}}')

        async def close(self) -> None:
            self.closed = True

    transport = HttpTransport(session=_ErrorSession())

    with pytest.raises(HttpStatusError) as exc_info:
        await transport.request("GET", "https://example.test")

    assert exc_info.value.response.status_code == 400
    assert exc_info.value.response.json() == {"error": {"message": "bad request"}}


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


@pytest.mark.asyncio
async def test_http_transport_maps_incomplete_transfer_payload_as_retryable_connection_error() -> None:
    class _BrokenSession:
        closed = False

        async def request(self, method, url, **kwargs):
            raise aiohttp.ClientPayloadError(
                "Response payload is not completed: <TransferEncodingError: 400, message='Not enough data to satisfy transfer length header.'>"
            )

        async def close(self) -> None:
            self.closed = True

    transport = HttpTransport(session=_BrokenSession())

    with pytest.raises(TransientProviderError) as exc_info:
        await transport.request("GET", "https://example.test", provider="openai", api_family="openai-responses")

    assert exc_info.value.error.code == "connection_error"
    assert exc_info.value.error.retryable is True
    assert "Response payload is not completed" in exc_info.value.error.message


@pytest.mark.asyncio
async def test_http_stream_response_maps_midstream_payload_failure_as_retryable_connection_error() -> None:
    class _BrokenStreamSession:
        closed = False

        async def request(self, method, url, **kwargs):
            return _BrokenStreamResponse()

        async def close(self) -> None:
            self.closed = True

    transport = HttpTransport(session=_BrokenStreamSession())
    response = await transport.stream("GET", "https://example.test", provider="openai", api_family="openai-responses")

    with pytest.raises(TransientProviderError) as exc_info:
        async for _ in response.iter_bytes():
            pass

    assert exc_info.value.error.code == "connection_error"
    assert exc_info.value.error.retryable is True
    assert "Response payload is not completed" in exc_info.value.error.message