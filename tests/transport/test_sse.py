from __future__ import annotations

import pytest

from connect.transport.sse import iter_sse_frames, iter_sse_response


async def _collect(lines: list[str]):
    async def generator():
        for line in lines:
            yield line

    return [frame async for frame in iter_sse_frames(generator())]


@pytest.mark.asyncio
async def test_iter_sse_frames_parses_multiline_frames_and_done_marker() -> None:
    frames = await _collect(
        [
            ": keepalive",
            "event: response.output_text.delta",
            "id: evt_1",
            "retry: 1000",
            "data: hello",
            "data: world",
            "",
            "data: [DONE]",
            "",
        ]
    )

    assert len(frames) == 2
    assert frames[0].event == "response.output_text.delta"
    assert frames[0].id == "evt_1"
    assert frames[0].retry == 1000
    assert frames[0].data == "hello\nworld"
    assert frames[0].is_done is False
    assert frames[1].data == "[DONE]"
    assert frames[1].is_done is True


@pytest.mark.asyncio
async def test_iter_sse_frames_flushes_trailing_frame_without_blank_line() -> None:
    frames = await _collect(["data: tail"])

    assert len(frames) == 1
    assert frames[0].data == "tail"


@pytest.mark.asyncio
async def test_iter_sse_frames_ignores_comments_and_empty_events() -> None:
    frames = await _collect([": comment", "", ": second comment"])

    assert frames == []


class _ChunkedResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_bytes(self):
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_iter_sse_response_parses_large_chunked_event_without_readline_limits() -> None:
    large_data = "x" * 200_000
    encoded = f"data: {large_data}\n\ndata: [DONE]\n\n".encode("utf-8")
    response = _ChunkedResponse(
        [encoded[:65536], encoded[65536:131072], encoded[131072:]]
    )

    frames = [frame async for frame in iter_sse_response(response)]

    assert len(frames) == 2
    assert frames[0].data == large_data
    assert frames[0].is_done is False
    assert frames[1].data == "[DONE]"
    assert frames[1].is_done is True