from __future__ import annotations

import pytest

from connect.exceptions import ProviderProtocolError
from connect.transport.json_stream import JSONStreamDecoder, iter_json_values


@pytest.mark.asyncio
async def test_iter_json_values_decodes_multiple_streamed_objects() -> None:
    async def chunks():
        yield b'{"a": 1}'
        yield b'{"b":'
        yield b' 2}'

    values = [value async for value in iter_json_values(chunks())]

    assert values == [{"a": 1}, {"b": 2}]


def test_json_stream_decoder_buffers_partial_json_until_complete() -> None:
    decoder = JSONStreamDecoder()

    assert decoder.feed('{"a":') == []
    assert decoder.feed(' 1}') == [{"a": 1}]


def test_json_stream_decoder_finalize_rejects_incomplete_payload() -> None:
    decoder = JSONStreamDecoder()
    decoder.feed('{"a":')

    with pytest.raises(ProviderProtocolError):
        decoder.finalize(provider="gemini", api_family="gemini-generate-content")