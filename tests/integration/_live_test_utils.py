from __future__ import annotations

from connect import TextBlock, ToolCallBlock, ToolResultMessage, ToolSpec


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
def build_lookup_status_tool_result(tool_call: ToolCallBlock, *, status: str) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=[TextBlock(text=f"status={status}")],
    )