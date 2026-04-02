from __future__ import annotations

import typing

from ..exceptions import make_error_info
from ..transport.http import HttpResponse
from ..types import ErrorInfo, GenerateRequest, ModelSpec, RequestOptions, SpecificToolChoice, Usage


class ProviderAdapter(typing.Protocol):
    provider_name: str
    api_family: str

    def build_headers(
        self,
        model: ModelSpec,
        request: GenerateRequest,
        options: RequestOptions,
    ) -> dict[str, str]:
        ...

    def build_payload(
        self,
        model: ModelSpec,
        request: GenerateRequest,
        options: RequestOptions,
    ) -> dict[str, typing.Any]:
        ...

    async def stream_response(
        self,
        *,
        model: ModelSpec,
        request: GenerateRequest,
        options: RequestOptions,
        http,
    ) -> typing.AsyncIterator:
        ...


class BaseProviderAdapter:
    provider_name = ""
    api_family = ""
    default_base_url: str | None = None
    stream_path = "/"

    def resolve_base_url(self, model: ModelSpec) -> str:
        return model.base_url or self.default_base_url or ""

    def request_url(self, model: ModelSpec) -> str:
        base_url = self.resolve_base_url(model).rstrip("/")
        if not base_url:
            return self.stream_path
        return f"{base_url}{self.stream_path}"

    def normalize_finish_reason(self, value: str | None) -> str:
        if value in {None, "completed", "stop", "end_turn"}:
            return "stop"
        if value in {"max_output_tokens", "max_tokens", "length"}:
            return "length"
        if value in {"tool_use", "tool_call", "function_call"}:
            return "tool_call"
        if value in {"content_filter", "content_filtered"}:
            return "content_filter"
        if value in {"cancelled", "canceled"}:
            return "cancelled"
        if value == "error":
            return "error"
        return "unknown"

    def build_usage(
        self,
        payload: typing.Any,
        *,
        completeness: typing.Literal["final", "partial", "none"],
    ) -> Usage | None:
        if not isinstance(payload, dict):
            return None

        input_details = payload.get("input_tokens_details") or {}
        output_details = payload.get("output_tokens_details") or {}

        input_tokens = int(payload.get("input_tokens") or 0)
        output_tokens = int(payload.get("output_tokens") or 0)
        reasoning_tokens = int(output_details.get("reasoning_tokens") or payload.get("reasoning_tokens") or 0)
        cache_read_tokens = int(input_details.get("cached_tokens") or payload.get("cache_read_tokens") or 0)
        cache_write_tokens = int(payload.get("cache_write_tokens") or 0)
        total_tokens = int(payload.get("total_tokens") or (input_tokens + output_tokens))

        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            total_tokens=total_tokens,
            completeness=completeness,
        )

    def build_error(
        self,
        payload: typing.Any,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> ErrorInfo:
        raw = payload if isinstance(payload, dict) else None
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            payload = payload["error"]

        code = "provider_error"
        message = "Provider request failed"
        if isinstance(payload, dict):
            code = str(payload.get("code") or code)
            message = str(payload.get("message") or message)
            status_code = status_code or payload.get("status_code")
            retryable = bool(payload.get("retryable", retryable))

        return make_error_info(
            code=code,
            message=message,
            provider=self.provider_name,
            api_family=self.api_family,
            status_code=status_code,
            retryable=retryable,
            raw=raw,
        )

    def build_http_error(self, response: HttpResponse) -> ErrorInfo:
        retryable = response.status_code in {408, 429} or response.status_code >= 500

        try:
            payload = response.json()
        except Exception:
            payload = None

        if payload is not None:
            normalized_payload = self._normalize_http_error_payload(payload)
            error = self.build_error(
                normalized_payload,
                status_code=response.status_code,
                retryable=retryable,
            )
            if error.raw is not None:
                return error.model_copy(update={"raw": payload if isinstance(payload, dict) else {"body": payload}})

            raw = payload if isinstance(payload, dict) else {"body": payload}
            return error.model_copy(update={"raw": raw})

        body_text = response.text().strip()
        return make_error_info(
            code=self._status_code_to_code(response.status_code),
            message=body_text or f"HTTP {response.status_code}",
            provider=self.provider_name,
            api_family=self.api_family,
            status_code=response.status_code,
            retryable=retryable,
            raw={"body": body_text} if body_text else None,
        )

    def _status_code_to_code(self, status_code: int) -> str:
        if status_code in {401, 403}:
            return "authentication_error"
        if status_code in {408, 429}:
            return "rate_limit"
        if status_code >= 500:
            return "server_error"
        return f"http_{status_code}"

    def _normalize_http_error_payload(self, payload: typing.Any) -> typing.Any:
        if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
            candidate = payload[0]
            if any(key in candidate for key in ("error", "message", "detail")):
                return candidate
        return payload

    def map_tool_choice(self, tool_choice: str | SpecificToolChoice | None) -> typing.Any:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, SpecificToolChoice):
            return {"type": "function", "name": tool_choice.name}
        return tool_choice