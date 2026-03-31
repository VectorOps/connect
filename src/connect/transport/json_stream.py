from __future__ import annotations

import json
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

from ..exceptions import ProviderProtocolError, make_error_info


class JSONStreamDecoder:
    def __init__(self) -> None:
        self._buffer = ""
        self._decoder = json.JSONDecoder()

    def feed(self, chunk: bytes | str) -> list[Any]:
        if isinstance(chunk, bytes):
            text = chunk.decode("utf-8", errors="replace")
        else:
            text = chunk

        self._buffer += text
        values: list[Any] = []

        while True:
            stripped = self._buffer.lstrip()
            if not stripped:
                self._buffer = ""
                break

            try:
                value, index = self._decoder.raw_decode(stripped)
            except json.JSONDecodeError:
                break

            values.append(value)
            self._buffer = stripped[index:]

        return values

    def finalize(
        self,
        *,
        provider: str | None = None,
        api_family: str | None = None,
    ) -> None:
        if self._buffer.strip():
            raise ProviderProtocolError(
                make_error_info(
                    code="invalid_json_stream",
                    message="The JSON stream ended with an incomplete payload",
                    provider=provider,
                    api_family=api_family,
                    raw={"buffer": self._buffer},
                )
            )


async def iter_json_values(
    chunks: AsyncIterable[bytes | str],
    *,
    provider: str | None = None,
    api_family: str | None = None,
) -> AsyncIterator[Any]:
    decoder = JSONStreamDecoder()
    async for chunk in chunks:
        for value in decoder.feed(chunk):
            yield value
    decoder.finalize(provider=provider, api_family=api_family)