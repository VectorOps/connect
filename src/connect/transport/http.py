from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlsplit

import aiohttp

from ..auth import AuthContext, TransportAuth, resolve_transport_auth
from ..exceptions import exception_from_error_info, make_error_info


@dataclass(slots=True)
class HttpResponse:
    status_code: int
    headers: Mapping[str, str]
    content: bytes
    url: str

    @property
    def request_id(self) -> str | None:
        for header_name in ("x-request-id", "request-id"):
            if header_name in self.headers:
                return self.headers[header_name]
        return None

    def text(self, encoding: str = "utf-8") -> str:
        return self.content.decode(encoding, errors="replace")

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8"))


class HttpStatusError(Exception):
    def __init__(self, response: HttpResponse):
        self.response = response
        super().__init__(f"HTTP {response.status_code}")


class HttpStreamResponse:
    def __init__(self, response: aiohttp.ClientResponse):
        self._response = response

    @property
    def status_code(self) -> int:
        return self._response.status

    @property
    def headers(self) -> Mapping[str, str]:
        return self._response.headers

    @property
    def url(self) -> str:
        return str(self._response.url)

    @property
    def request_id(self) -> str | None:
        for header_name in ("x-request-id", "request-id"):
            if header_name in self._response.headers:
                return self._response.headers[header_name]
        return None

    async def read(self) -> bytes:
        return await self._response.read()

    async def text(self) -> str:
        return await self._response.text()

    async def json(self) -> Any:
        return await self._response.json()

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._response.content.readany()
            if chunk == b"":
                break
            if chunk:
                yield chunk

    async def iter_lines(self) -> AsyncIterator[str]:
        while True:
            line = await self._response.content.readline()
            if line == b"":
                break
            yield line.decode("utf-8", errors="replace").rstrip("\r\n")

    async def close(self) -> None:
        self._response.close()

    async def __aenter__(self) -> HttpStreamResponse:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


class HttpTransport:
    def __init__(
        self,
        *,
        session: aiohttp.ClientSession | None = None,
        base_url: str | None = None,
        headers: Mapping[str, str] | None = None,
        auth: TransportAuth | None = None,
    ) -> None:
        self._session = session or aiohttp.ClientSession()
        self._owns_session = session is None
        self._base_url = base_url
        self._headers = dict(headers or {})
        self._auth = auth

    async def close(self) -> None:
        if self._owns_session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> HttpTransport:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def request(
        self,
        method: str,
        url: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_family: str | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        auth: TransportAuth | None = None,
        json_body: Any = None,
        data: Any = None,
        timeout: float | aiohttp.ClientTimeout | None = None,
        expected_status: int | set[int] | None = None,
    ) -> HttpResponse:
        try:
            response = await self._request_with_optional_refresh(
                method,
                url,
                provider=provider,
                model=model,
                api_family=api_family,
                params=params,
                headers=headers,
                auth=auth,
                json_body=json_body,
                data=data,
                timeout=timeout,
                expected_status=expected_status,
            )
            async with response:
                await self._raise_for_status(
                    response,
                    expected_status=expected_status,
                )
                content = await response.read()
                return HttpResponse(
                    status_code=response.status,
                    headers=response.headers,
                    content=content,
                    url=str(response.url),
                )
        except Exception as exc:
            raise self._map_transport_exception(exc, provider=provider, api_family=api_family) from exc

    async def stream(
        self,
        method: str,
        url: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_family: str | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        auth: TransportAuth | None = None,
        json_body: Any = None,
        data: Any = None,
        timeout: float | aiohttp.ClientTimeout | None = None,
        expected_status: int | set[int] | None = None,
    ) -> HttpStreamResponse:
        try:
            response = await self._request_with_optional_refresh(
                method,
                url,
                provider=provider,
                model=model,
                api_family=api_family,
                params=params,
                headers=headers,
                auth=auth,
                json_body=json_body,
                data=data,
                timeout=timeout,
                expected_status=expected_status,
            )
            try:
                await self._raise_for_status(
                    response,
                    expected_status=expected_status,
                )
            except Exception:
                response.close()
                raise
            return HttpStreamResponse(response)
        except Exception as exc:
            raise self._map_transport_exception(exc, provider=provider, api_family=api_family) from exc

    async def _request_with_optional_refresh(
        self,
        method: str,
        url: str,
        *,
        provider: str | None,
        model: str | None,
        api_family: str | None,
        params: Mapping[str, str] | None,
        headers: Mapping[str, str] | None,
        auth: TransportAuth | None,
        json_body: Any,
        data: Any,
        timeout: float | aiohttp.ClientTimeout | None,
        expected_status: int | set[int] | None,
    ) -> aiohttp.ClientResponse:
        resolved_auth = auth or self._auth
        response = await self._perform_request(
            method,
            url,
            context=AuthContext(provider=provider, model=model, api_family=api_family, method=method, url=url),
            params=params,
            headers=headers,
            auth=resolved_auth,
            json_body=json_body,
            data=data,
            timeout=timeout,
        )
        if response.status not in {401, 403} or resolved_auth is None:
            return response

        refreshed = await resolved_auth.refresh(
            AuthContext(provider=provider, model=model, api_family=api_family, method=method, url=url)
        )
        if not refreshed:
            return response

        response.release()
        return await self._perform_request(
            method,
            url,
            context=AuthContext(provider=provider, model=model, api_family=api_family, method=method, url=url),
            params=params,
            headers=headers,
            auth=resolved_auth,
            json_body=json_body,
            data=data,
            timeout=timeout,
        )

    async def _perform_request(
        self,
        method: str,
        url: str,
        *,
        context: AuthContext,
        params: Mapping[str, str] | None,
        headers: Mapping[str, str] | None,
        auth: TransportAuth | None,
        json_body: Any,
        data: Any,
        timeout: float | aiohttp.ClientTimeout | None,
    ) -> aiohttp.ClientResponse:
        resolved = await resolve_transport_auth(auth, context=context)
        merged_params = dict(params or {})
        merged_params.update({str(key): str(value) for key, value in resolved.params.items()})
        merged_headers = self._merge_headers(headers)
        merged_headers.update(resolved.headers)
        return await self._session.request(
            method,
            self._resolve_url(url),
            params=merged_params or None,
            headers=merged_headers,
            json=json_body,
            data=data,
            timeout=self._normalize_timeout(timeout),
        )

    async def _raise_for_status(
        self,
        response: aiohttp.ClientResponse,
        *,
        expected_status: int | set[int] | None,
    ) -> None:
        if expected_status is None:
            allowed_statuses = {200}
        elif isinstance(expected_status, int):
            allowed_statuses = {expected_status}
        else:
            allowed_statuses = expected_status

        if response.status in allowed_statuses:
            return

        content = await response.read()
        raise HttpStatusError(
            HttpResponse(
                status_code=response.status,
                headers=response.headers,
                content=content,
                url=str(response.url),
            )
        )

    def _resolve_url(self, url: str) -> str:
        if self._base_url is None or urlsplit(url).scheme:
            return url
        return urljoin(f"{self._base_url.rstrip('/')}/", url.lstrip("/"))

    def _merge_headers(self, headers: Mapping[str, str] | None) -> dict[str, str]:
        merged = dict(self._headers)
        if headers:
            merged.update(headers)
        return merged

    def _normalize_timeout(
        self,
        timeout: float | aiohttp.ClientTimeout | None,
    ) -> aiohttp.ClientTimeout | None:
        if timeout is None or isinstance(timeout, aiohttp.ClientTimeout):
            return timeout
        return aiohttp.ClientTimeout(total=timeout)

    def _map_transport_exception(
        self,
        exc: Exception,
        *,
        provider: str | None,
        api_family: str | None,
    ):
        if isinstance(exc, HttpStatusError):
            return exc

        if hasattr(exc, "error"):
            return exc

        if isinstance(exc, (asyncio.TimeoutError, aiohttp.ServerTimeoutError)):
            return exception_from_error_info(
                make_error_info(
                    code="timeout",
                    message="The provider request timed out",
                    provider=provider,
                    api_family=api_family,
                    retryable=True,
                )
            )

        if isinstance(
            exc,
            (
                aiohttp.ClientConnectionError,
                aiohttp.ClientOSError,
                aiohttp.ClientPayloadError,
                aiohttp.ClientConnectorError,
            ),
        ):
            return exception_from_error_info(
                make_error_info(
                    code="connection_error",
                    message=str(exc) or "The provider connection failed",
                    provider=provider,
                    api_family=api_family,
                    retryable=True,
                )
            )

        if isinstance(exc, aiohttp.InvalidURL):
            return exception_from_error_info(
                make_error_info(
                    code="invalid_url",
                    message=str(exc),
                    provider=provider,
                    api_family=api_family,
                )
            )

        return exception_from_error_info(
            make_error_info(
                code="transport_error",
                message=str(exc) or "Unexpected transport error",
                provider=provider,
                api_family=api_family,
                retryable=False,
            )
        )
