from __future__ import annotations

import base64
import binascii
import decimal
import re
import typing

import aiohttp
import pydantic

from .auth import TransportAuth
from .tool_schema import ToolSchemaError, normalize_canonical_tool_schema


TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")


class MetadataModel(pydantic.BaseModel):
    provider_meta: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    protocol_meta: dict[str, typing.Any] = pydantic.Field(default_factory=dict)


class AnnotatedMetadataModel(MetadataModel):
    annotations: list[dict[str, typing.Any]] | dict[str, typing.Any] | None = None


class TextBlock(AnnotatedMetadataModel):
    type: typing.Literal["text"] = "text"
    text: str


class ImageBlock(AnnotatedMetadataModel):
    type: typing.Literal["image"] = "image"
    mime_type: str
    data: str
    detail: typing.Literal["high", "low", "auto", "original"] = "auto"

    @pydantic.field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, value: str) -> str:
        mime_type = value.strip().lower()
        if not mime_type.startswith("image/"):
            raise ValueError("ImageBlock.mime_type must be an image media type")
        return mime_type

    @pydantic.field_validator("data", mode="before")
    @classmethod
    def normalize_data(cls, value: str | bytes) -> str:
        if isinstance(value, bytes):
            if not value:
                raise ValueError("ImageBlock.data must not be empty")
            return base64.b64encode(value).decode("ascii")

        normalized = "".join(str(value).split())
        if not normalized:
            raise ValueError("ImageBlock.data must not be empty")
        if normalized.startswith("data:"):
            raise ValueError("ImageBlock.data must not include a data URL prefix")

        try:
            base64.b64decode(normalized, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("ImageBlock.data must be valid base64") from exc

        return normalized


class ReasoningBlock(AnnotatedMetadataModel):
    type: typing.Literal["reasoning"] = "reasoning"
    text: str
    signature: str | None = None
    redacted: bool = False


class ToolCallBlock(AnnotatedMetadataModel):
    type: typing.Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, typing.Any]

    @pydantic.field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        tool_call_id = value.strip()
        if not tool_call_id:
            raise ValueError("Tool call id must not be empty")
        return tool_call_id

    @pydantic.field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        tool_name = value.strip()
        if not TOOL_NAME_PATTERN.match(tool_name):
            raise ValueError("Tool names must match ^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
        return tool_name


UserContentBlock = typing.Annotated[
    TextBlock | ImageBlock,
    pydantic.Field(discriminator="type"),
]
AssistantContentBlock = typing.Annotated[
    TextBlock | ReasoningBlock | ToolCallBlock,
    pydantic.Field(discriminator="type"),
]
ToolResultContentBlock = typing.Annotated[
    TextBlock | ImageBlock,
    pydantic.Field(discriminator="type"),
]


class UserMessage(MetadataModel):
    role: typing.Literal["user"] = "user"
    content: str | list[UserContentBlock]

    @pydantic.field_validator("content")
    @classmethod
    def validate_content(cls, value: str | list[UserContentBlock]) -> str | list[UserContentBlock]:
        if isinstance(value, list) and not value:
            raise ValueError("UserMessage.content must not be an empty list")
        return value


class AssistantMessage(MetadataModel):
    role: typing.Literal["assistant"] = "assistant"
    content: list[AssistantContentBlock]

    @pydantic.field_validator("content")
    @classmethod
    def validate_content(cls, value: list[AssistantContentBlock]) -> list[AssistantContentBlock]:
        if not value:
            raise ValueError("AssistantMessage.content must not be empty")
        return value


class ToolResultMessage(MetadataModel):
    role: typing.Literal["tool"] = "tool"
    tool_call_id: str
    tool_name: str
    content: list[ToolResultContentBlock]
    is_error: bool = False

    @pydantic.field_validator("tool_call_id")
    @classmethod
    def validate_tool_call_id(cls, value: str) -> str:
        tool_call_id = value.strip()
        if not tool_call_id:
            raise ValueError("Tool result messages must reference a tool_call_id")
        return tool_call_id

    @pydantic.field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        tool_name = value.strip()
        if not TOOL_NAME_PATTERN.match(tool_name):
            raise ValueError("Tool names must match ^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
        return tool_name

    @pydantic.field_validator("content")
    @classmethod
    def validate_content(cls, value: list[ToolResultContentBlock]) -> list[ToolResultContentBlock]:
        return value


Message = typing.Annotated[
    UserMessage | AssistantMessage | ToolResultMessage,
    pydantic.Field(discriminator="role"),
]


class ReasoningConfig(pydantic.BaseModel):
    effort: typing.Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    summary: typing.Literal["auto", "concise", "detailed"] | None = None
    max_tokens: int | None = None
    enabled: bool | None = None

    @pydantic.field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("ReasoningConfig.max_tokens must be greater than zero")
        return value


class ToolSpec(pydantic.BaseModel):
    name: str
    description: str
    input_schema: dict[str, typing.Any]
    strict: bool | None = True

    @pydantic.field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        tool_name = value.strip()
        if not TOOL_NAME_PATTERN.match(tool_name):
            raise ValueError("Tool names must match ^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
        return tool_name

    @pydantic.field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        description = value.strip()
        if not description:
            raise ValueError("Tool descriptions must not be empty")
        return description

    @pydantic.field_validator("input_schema")
    @classmethod
    def validate_input_schema(cls, value: dict[str, typing.Any]) -> dict[str, typing.Any]:
        if not isinstance(value, dict) or not value:
            raise ValueError("Tool specs must include a non-empty input_schema")
        try:
            return normalize_canonical_tool_schema(value)
        except ToolSchemaError as exc:
            raise ValueError(str(exc)) from exc


class SpecificToolChoice(pydantic.BaseModel):
    type: typing.Literal["tool"] = "tool"
    name: str

    @pydantic.field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        tool_name = value.strip()
        if not TOOL_NAME_PATTERN.match(tool_name):
            raise ValueError("Tool names must match ^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
        return tool_name


ToolChoice = typing.Literal["auto", "none", "required"] | SpecificToolChoice


class SessionHints(pydantic.BaseModel):
    session_id: str | None = None
    continue_from: str | None = None
    metadata: dict[str, typing.Any] = pydantic.Field(default_factory=dict)


class ResponseFormat(pydantic.BaseModel):
    type: typing.Literal["text", "json_object", "json_schema"] = "text"
    name: str | None = None
    json_schema: dict[str, typing.Any] | None = None
    strict: bool | None = None

    @pydantic.model_validator(mode="after")
    def validate_shape(self) -> ResponseFormat:
        if self.type == "json_schema" and not self.json_schema:
            raise ValueError("ResponseFormat.json_schema is required when type='json_schema'")
        return self


class GenerateRequest(pydantic.BaseModel):
    messages: list[Message]
    system_prompt: str | None = None
    tools: list[ToolSpec] = pydantic.Field(default_factory=list)
    tool_choice: ToolChoice | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning: ReasoningConfig | None = None
    response_format: ResponseFormat | None = None
    metadata: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    session: SessionHints | None = None
    protocol_hints: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    extension_data: dict[str, typing.Any] = pydantic.Field(default_factory=dict)

    @pydantic.field_validator("messages")
    @classmethod
    def validate_messages(cls, value: list[Message]) -> list[Message]:
        if not value:
            raise ValueError("GenerateRequest.messages must not be empty")
        return value

    @pydantic.field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float | None) -> float | None:
        if value is not None and not 0 <= value <= 2:
            raise ValueError("GenerateRequest.temperature must be between 0 and 2")
        return value

    @pydantic.field_validator("max_output_tokens")
    @classmethod
    def validate_max_output_tokens(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("GenerateRequest.max_output_tokens must be greater than zero")
        return value

    @pydantic.model_validator(mode="after")
    def validate_tool_history(self) -> GenerateRequest:
        pending_tool_calls: dict[str, str] = {}

        for message in self.messages:
            if message.role == "assistant":
                if pending_tool_calls:
                    unresolved = ", ".join(sorted(pending_tool_calls))
                    raise ValueError(
                        f"Assistant messages cannot appear before tool results resolve prior tool calls: {unresolved}"
                    )

                for block in message.content:
                    if block.type == "tool_call":
                        if block.id in pending_tool_calls:
                            raise ValueError(f"Duplicate tool call id in transcript: {block.id}")
                        pending_tool_calls[block.id] = block.name

            elif message.role == "tool":
                if message.tool_call_id not in pending_tool_calls:
                    raise ValueError(
                        f"Tool result references unknown tool call id: {message.tool_call_id}"
                    )

                expected_tool_name = pending_tool_calls.pop(message.tool_call_id)
                if message.tool_name != expected_tool_name:
                    raise ValueError(
                        f"Tool result name '{message.tool_name}' does not match prior tool call '{expected_tool_name}'"
                    )

            else:
                if pending_tool_calls:
                    unresolved = ", ".join(sorted(pending_tool_calls))
                    raise ValueError(
                        f"User messages cannot appear before tool results resolve prior tool calls: {unresolved}"
                    )

        if pending_tool_calls:
            unresolved = ", ".join(sorted(pending_tool_calls))
            raise ValueError(f"Transcript ends with unresolved tool calls: {unresolved}")

        tool_names = [tool.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("GenerateRequest.tools must not contain duplicate tool names")

        if isinstance(self.tool_choice, SpecificToolChoice) and self.tool_choice.name not in set(tool_names):
            raise ValueError("Specific tool choices must reference a declared tool")

        return self


class RequestOptions(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    timeout: float | aiohttp.ClientTimeout | None = 60.0
    headers: dict[str, str] = pydantic.Field(default_factory=dict)
    auth: TransportAuth | None = None
    idempotency_key: str | None = None
    provider_options: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    transport_options: dict[str, typing.Any] = pydantic.Field(default_factory=dict)


class Usage(pydantic.BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    completeness: typing.Literal["final", "partial", "none"] = "none"


class CostBreakdown(pydantic.BaseModel):
    input_cost: decimal.Decimal = decimal.Decimal("0")
    output_cost: decimal.Decimal = decimal.Decimal("0")
    cache_read_cost: decimal.Decimal = decimal.Decimal("0")
    cache_write_cost: decimal.Decimal = decimal.Decimal("0")
    total_cost: decimal.Decimal = decimal.Decimal("0")


class ErrorInfo(pydantic.BaseModel):
    code: str
    message: str
    provider: str | None = None
    api_family: str | None = None
    status_code: int | None = None
    retryable: bool = False
    raw: dict[str, typing.Any] | None = None


FinishReason = typing.Literal[
    "stop",
    "length",
    "tool_use",
    "tool_call",
    "content_filter",
    "cancelled",
    "error",
    "unknown",
]


class AssistantResponse(pydantic.BaseModel):
    provider: str
    model: str
    api_family: str
    content: list[AssistantContentBlock]
    finish_reason: FinishReason
    usage: Usage = pydantic.Field(default_factory=Usage)
    response_id: str | None = None
    request_id: str | None = None
    protocol_state: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    provider_meta: dict[str, typing.Any] = pydantic.Field(default_factory=dict)


class ResponseStartEvent(pydantic.BaseModel):
    type: typing.Literal["response_start"] = "response_start"
    provider: str
    model: str
    response_id: str | None = None


class TextStartEvent(pydantic.BaseModel):
    type: typing.Literal["text_start"] = "text_start"
    index: int


class TextDeltaEvent(pydantic.BaseModel):
    type: typing.Literal["text_delta"] = "text_delta"
    index: int
    delta: str


class TextEndEvent(pydantic.BaseModel):
    type: typing.Literal["text_end"] = "text_end"
    index: int
    text: str


class ReasoningStartEvent(pydantic.BaseModel):
    type: typing.Literal["reasoning_start"] = "reasoning_start"
    index: int


class ReasoningDeltaEvent(pydantic.BaseModel):
    type: typing.Literal["reasoning_delta"] = "reasoning_delta"
    index: int
    delta: str


class ReasoningEndEvent(pydantic.BaseModel):
    type: typing.Literal["reasoning_end"] = "reasoning_end"
    index: int
    text: str
    signature: str | None = None
    redacted: bool = False


class ToolCallStartEvent(pydantic.BaseModel):
    type: typing.Literal["tool_call_start"] = "tool_call_start"
    index: int
    id: str
    name: str


class ToolCallDeltaEvent(pydantic.BaseModel):
    type: typing.Literal["tool_call_delta"] = "tool_call_delta"
    index: int
    delta: str


class ToolCallEndEvent(pydantic.BaseModel):
    type: typing.Literal["tool_call_end"] = "tool_call_end"
    index: int
    tool_call: ToolCallBlock


class UsageEvent(pydantic.BaseModel):
    type: typing.Literal["usage"] = "usage"
    usage: Usage


class ResponseEndEvent(pydantic.BaseModel):
    type: typing.Literal["response_end"] = "response_end"
    response: AssistantResponse


class ErrorEvent(pydantic.BaseModel):
    type: typing.Literal["error"] = "error"
    error: ErrorInfo
    partial_response: AssistantResponse | None = None


StreamEvent = typing.Annotated[
    ResponseStartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ReasoningStartEvent
    | ReasoningDeltaEvent
    | ReasoningEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | UsageEvent
    | ResponseEndEvent
    | ErrorEvent,
    pydantic.Field(discriminator="type"),
]


class ModelPricing(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    input_per_million: decimal.Decimal | None = None
    output_per_million: decimal.Decimal | None = None
    cache_read_per_million: decimal.Decimal | None = None
    cache_write_per_million: decimal.Decimal | None = None


class ModelSpec(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    provider: str
    model: str
    api_family: str
    base_url: str | None = None
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_reasoning: bool = False
    supports_images: bool = False
    supports_image_outputs: bool = False
    supports_json_mode: bool = False
    supports_prompt_caching: bool = False
    context_window: int | None = None
    max_output_tokens: int | None = None
    pricing: ModelPricing | None = None
    capabilities: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    protocol_defaults: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    extra: dict[str, typing.Any] = pydantic.Field(default_factory=dict)


def request_uses_images(request: GenerateRequest) -> bool:
    for message in request.messages:
        if message.role == "user":
            if isinstance(message.content, list) and any(block.type == "image" for block in message.content):
                return True
            continue

        if message.role == "tool" and any(block.type == "image" for block in message.content):
            return True

    return False


def validate_request_for_model(model: ModelSpec, request: GenerateRequest) -> None:
    if request.tools and not model.supports_tools:
        raise ValueError(f"Model '{model.provider}:{model.model}' does not support tool use")

    if request.reasoning and not model.supports_reasoning:
        raise ValueError(f"Model '{model.provider}:{model.model}' does not support reasoning controls")

    if request_uses_images(request) and not model.supports_images:
        raise ValueError(
            f"Model '{model.provider}:{model.model}' does not support image inputs or image-bearing tool results"
        )

    if request.response_format and request.response_format.type != "text" and not model.supports_json_mode:
        raise ValueError(
            f"Model '{model.provider}:{model.model}' does not support structured response formats"
        )