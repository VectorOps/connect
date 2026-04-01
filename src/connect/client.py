from __future__ import annotations

import typing

import aiohttp

from .auth import AuthContext, resolve_transport_auth
from .auth_router import AuthCredentialManager, DynamicAuthRouter
from .exceptions import ConnectError, exception_from_error_info
from .registry import ModelRegistry, ProviderRegistry, default_model_registry, default_provider_registry
from .transport import HttpTransport
from .types import AssistantResponse, GenerateRequest, ModelSpec, RequestOptions, StreamEvent, validate_request_for_model


class StreamHandle:
    def __init__(self, iterator: typing.AsyncIterator[StreamEvent]) -> None:
        self._iterator = iterator
        self._done = False
        self._final_response: AssistantResponse | None = None
        self._error: ConnectError | None = None

    def __aiter__(self) -> StreamHandle:
        return self

    async def __anext__(self) -> StreamEvent:
        if self._done:
            raise StopAsyncIteration

        event = await anext(self._iterator)
        if event.type == "response_end":
            self._final_response = event.response
            self._done = True
        elif event.type == "error":
            self._error = exception_from_error_info(event.error)
            self._done = True
        return event

    async def final_response(self) -> AssistantResponse:
        if self._final_response is not None:
            return self._final_response
        if self._error is not None:
            raise self._error

        async for _ in self:
            pass

        if self._final_response is not None:
            return self._final_response
        if self._error is not None:
            raise self._error
        raise RuntimeError("Stream ended without a terminal response")


class AsyncLLMClient:
    def __init__(
        self,
        *,
        http_client: aiohttp.ClientSession | None = None,
        auth_router: DynamicAuthRouter | None = None,
        credential_manager: AuthCredentialManager | None = None,
        model_registry: ModelRegistry | None = None,
        provider_registry: ProviderRegistry | None = None,
    ) -> None:
        self.auth_router = auth_router or DynamicAuthRouter(credential_manager=credential_manager)
        self.model_registry = model_registry or default_model_registry
        self.provider_registry = provider_registry or default_provider_registry
        self.http = HttpTransport(session=http_client)

    async def close(self) -> None:
        await self.http.close()

    async def __aenter__(self) -> AsyncLLMClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def generate(
        self,
        model: str | ModelSpec,
        request: GenerateRequest,
        *,
        provider: str | None = None,
        options: RequestOptions | None = None,
    ) -> AssistantResponse:
        stream = self.stream(model, request, provider=provider, options=options)
        return await stream.final_response()

    def stream(
        self,
        model: str | ModelSpec,
        request: GenerateRequest,
        *,
        provider: str | None = None,
        options: RequestOptions | None = None,
    ) -> StreamHandle:
        resolved_options = options or RequestOptions()
        return StreamHandle(self._stream(model, request, provider=provider, options=resolved_options))

    async def _stream(
        self,
        model: str | ModelSpec,
        request: GenerateRequest,
        *,
        provider: str | None,
        options: RequestOptions,
    ) -> typing.AsyncIterator[StreamEvent]:
        resolved_model = self._resolve_model(model, provider=provider)
        validate_request_for_model(resolved_model, request)

        provider_adapter = self.provider_registry.get(resolved_model.provider)
        auth = options.auth or self.auth_router
        auth_context = AuthContext(
            provider=resolved_model.provider,
            model=resolved_model.model,
            api_family=resolved_model.api_family,
        )
        resolved_auth = await resolve_transport_auth(auth, context=auth_context)
        headers = {**resolved_auth.headers, **options.headers}
        params = resolved_auth.params
        effective_options = options.model_copy(
            update={
                "auth": auth,
                "headers": headers,
                "transport_options": {**options.transport_options, "query_params": params},
            }
        )

        async for event in provider_adapter.stream_response(
            model=resolved_model,
            request=request,
            options=effective_options,
            http=self.http,
        ):
            yield event

    def _resolve_model(self, model: str | ModelSpec, *, provider: str | None) -> ModelSpec:
        if isinstance(model, ModelSpec):
            return model
        return self.model_registry.resolve(model, provider=provider)

async def generate(
    model: str | ModelSpec,
    request: GenerateRequest,
    *,
    provider: str | None = None,
    options: RequestOptions | None = None,
) -> AssistantResponse:
    async with AsyncLLMClient() as client:
        return await client.generate(model, request, provider=provider, options=options)


def stream(
    model: str | ModelSpec,
    request: GenerateRequest,
    *,
    provider: str | None = None,
    options: RequestOptions | None = None,
) -> StreamHandle:
    client = AsyncLLMClient()
    return client.stream(model, request, provider=provider, options=options)