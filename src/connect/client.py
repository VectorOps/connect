from __future__ import annotations

import inspect
import time
import typing

import aiohttp

from .auth import AuthContext, resolve_transport_auth
from .auth_router import AuthCredentialManager, DynamicAuthRouter
from .exceptions import ConnectError, exception_from_error_info
from .registry import ModelRegistry, ProviderRegistry, default_model_registry, default_provider_registry
from .transport import HttpTransport
from .types import AssistantResponse, GenerateRequest, ModelSpec, RequestOptions, StreamEvent, validate_request_for_model


class StreamHandle:
    def __init__(
        self,
        iterator: typing.AsyncIterator[StreamEvent],
        *,
        close_callback: typing.Callable[[], typing.Awaitable[None] | None] | None = None,
    ) -> None:
        self._iterator = iterator
        self._close_callback = close_callback
        self._done = False
        self._final_response: AssistantResponse | None = None
        self._error: ConnectError | None = None
        self._closed = False

    def __aiter__(self) -> StreamHandle:
        return self

    async def __anext__(self) -> StreamEvent:
        if self._done:
            raise StopAsyncIteration

        event = await anext(self._iterator)
        if event.type == "response_end":
            self._final_response = event.response
            self._done = True
            await self._maybe_close()
        elif event.type == "error":
            self._error = exception_from_error_info(event.error)
            self._done = True
            await self._maybe_close()
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

    async def _maybe_close(self) -> None:
        if self._closed or self._close_callback is None:
            return
        self._closed = True
        result = self._close_callback()
        if inspect.isawaitable(result):
            await result


class AsyncLLMClient:
    def __init__(
        self,
        *,
        http_client: aiohttp.ClientSession | None = None,
        auth_router: DynamicAuthRouter | None = None,
        credential_manager: AuthCredentialManager | None = None,
        model_registry: ModelRegistry | None = None,
        provider_registry: ProviderRegistry | None = None,
        event_hook: typing.Callable[[dict[str, typing.Any]], typing.Awaitable[None] | None] | None = None,
    ) -> None:
        self.auth_router = auth_router or DynamicAuthRouter(credential_manager=credential_manager)
        self.model_registry = model_registry or default_model_registry
        self.provider_registry = provider_registry or default_provider_registry
        self.http = HttpTransport(session=http_client)
        self.event_hook = event_hook

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
        request_started_at = time.perf_counter()
        first_token_emitted = False

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

        await self._emit_event(
            {
                "type": "request_start",
                "provider": resolved_model.provider,
                "model": resolved_model.model,
                "api_family": resolved_model.api_family,
            }
        )

        try:
            async for event in provider_adapter.stream_response(
                model=resolved_model,
                request=request,
                options=effective_options,
                http=self.http,
            ):
                if event.type == "response_start":
                    await self._emit_event(
                        {
                            "type": "response_headers",
                            "provider": resolved_model.provider,
                            "model": resolved_model.model,
                            "api_family": resolved_model.api_family,
                            "request_id": None,
                            "response_id": event.response_id,
                        }
                    )

                if not first_token_emitted and event.type in {"text_delta", "reasoning_delta", "tool_call_delta"}:
                    first_token_emitted = True
                    await self._emit_event(
                        {
                            "type": "first_token",
                            "provider": resolved_model.provider,
                            "model": resolved_model.model,
                            "api_family": resolved_model.api_family,
                            "latency_s": time.perf_counter() - request_started_at,
                        }
                    )

                if event.type == "response_end":
                    await self._emit_event(
                        {
                            "type": "request_end",
                            "provider": resolved_model.provider,
                            "model": resolved_model.model,
                            "api_family": resolved_model.api_family,
                            "latency_s": time.perf_counter() - request_started_at,
                            "finish_reason": event.response.finish_reason,
                            "request_id": event.response.request_id,
                            "response_id": event.response.response_id,
                            "usage": event.response.usage.model_dump(),
                        }
                    )
                elif event.type == "error":
                    await self._emit_event(
                        {
                            "type": "error",
                            "provider": resolved_model.provider,
                            "model": resolved_model.model,
                            "api_family": resolved_model.api_family,
                            "latency_s": time.perf_counter() - request_started_at,
                            "error": event.error.model_dump(),
                        }
                    )

                yield event
        except Exception as exc:
            await self._emit_event(
                {
                    "type": "error",
                    "provider": resolved_model.provider,
                    "model": resolved_model.model,
                    "api_family": resolved_model.api_family,
                    "latency_s": time.perf_counter() - request_started_at,
                    "error": {"message": str(exc)},
                }
            )
            raise

    def _resolve_model(self, model: str | ModelSpec, *, provider: str | None) -> ModelSpec:
        if isinstance(model, ModelSpec):
            return model
        return self.model_registry.resolve(model, provider=provider)

    async def _emit_event(self, event: dict[str, typing.Any]) -> None:
        if self.event_hook is None:
            return
        result = self.event_hook(event)
        if inspect.isawaitable(result):
            await result

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
    resolved_options = options or RequestOptions()
    return StreamHandle(
        client._stream(model, request, provider=provider, options=resolved_options),
        close_callback=client.close,
    )