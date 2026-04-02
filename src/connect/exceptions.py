from __future__ import annotations

from .types import ErrorInfo


class ConnectError(Exception):
    def __init__(self, error: ErrorInfo):
        self.error = error
        super().__init__(error.message)


class AuthenticationError(ConnectError):
    pass


class RateLimitError(ConnectError):
    pass


class ContextLengthError(ConnectError):
    pass


class ProviderProtocolError(ConnectError):
    pass


class TransientProviderError(ConnectError):
    pass


class PermanentProviderError(ConnectError):
    pass


def exception_from_error_info(error: ErrorInfo) -> ConnectError:
    code = error.code.lower()
    message = error.message.lower()

    if error.status_code in {401, 403}:
        return AuthenticationError(error)

    if code in {"authentication_error", "unauthenticated", "api_key_invalid", "invalid_api_key"}:
        return AuthenticationError(error)

    if error.status_code in {408, 429}:
        return RateLimitError(error)

    if code in {"rate_limit", "resource_exhausted"}:
        return RateLimitError(error)

    if "context" in message and ("limit" in message or "length" in message):
        return ContextLengthError(error)
    if "context" in code and ("limit" in code or "length" in code):
        return ContextLengthError(error)

    if error.status_code is not None and error.status_code >= 500:
        return TransientProviderError(error)

    if error.retryable:
        return TransientProviderError(error)

    if error.status_code is not None and error.status_code >= 400:
        return PermanentProviderError(error)

    return ProviderProtocolError(error)


def make_error_info(
    *,
    code: str,
    message: str,
    provider: str | None = None,
    api_family: str | None = None,
    status_code: int | None = None,
    retryable: bool = False,
    raw: dict | None = None,
) -> ErrorInfo:
    return ErrorInfo(
        code=code,
        message=message,
        provider=provider,
        api_family=api_family,
        status_code=status_code,
        retryable=retryable,
        raw=raw,
    )