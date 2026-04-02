from __future__ import annotations

from connect import AssistantMessage, TextBlock, ToolCallBlock, ToolResultMessage, ToolSpec


def build_lookup_status_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="lookup_status",
        description="Lookup a status string for a known identifier.",
        input_schema={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        },
    )


def build_history_assistant_message(response) -> AssistantMessage:
    content: list[TextBlock | ToolCallBlock] = []
    for block in response.content:
        if block.type == "text":
            content.append(TextBlock(text=block.text))
        elif block.type == "tool_call":
            content.append(
                ToolCallBlock(
                    id=block.id,
                    name=block.name,
                    arguments=dict(block.arguments),
                )
            )

    if not content:
        raise AssertionError(f"expected replayable assistant content, got: {response.content!r}")

    return AssistantMessage(content=content)


def build_lookup_status_tool_result(tool_call: ToolCallBlock, *, status: str) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=[TextBlock(text=f"status={status}")],
    )