from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass


@dataclass(slots=True)
class SSEFrame:
    data: str
    event: str | None = None
    id: str | None = None
    retry: int | None = None

    @property
    def is_done(self) -> bool:
        return self.data == "[DONE]"


async def iter_sse_frames(lines: AsyncIterable[str]) -> AsyncIterator[SSEFrame]:
    data_lines: list[str] = []
    event: str | None = None
    event_id: str | None = None
    retry: int | None = None
    saw_fields = False

    async for line in lines:
        if line == "":
            if saw_fields:
                yield SSEFrame(
                    data="\n".join(data_lines),
                    event=event,
                    id=event_id,
                    retry=retry,
                )
            data_lines = []
            event = None
            event_id = None
            retry = None
            saw_fields = False
            continue

        if line.startswith(":"):
            continue

        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]

        saw_fields = True
        if field == "data":
            data_lines.append(value)
        elif field == "event":
            event = value
        elif field == "id":
            event_id = value
        elif field == "retry":
            try:
                retry = int(value)
            except ValueError:
                pass

    if saw_fields:
        yield SSEFrame(
            data="\n".join(data_lines),
            event=event,
            id=event_id,
            retry=retry,
        )


async def iter_sse_response(response) -> AsyncIterator[SSEFrame]:
    async for frame in iter_sse_frames(response.iter_lines()):
        yield frame