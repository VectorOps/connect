from __future__ import annotations

from .types import AssistantMessage


def assistant_message_text(
    message: AssistantMessage,
    *,
    include_reasoning: bool = False,
    separator: str = "",
) -> str:
    parts: list[str] = []
    for block in message.content:
        if block.type == "text":
            parts.append(block.text)
        elif include_reasoning and block.type == "reasoning":
            parts.append(block.text)
    return separator.join(parts).strip()