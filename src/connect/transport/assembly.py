from __future__ import annotations

import json

from ..exceptions import ProviderProtocolError, make_error_info
from ..types import (
    AssistantResponse,
    ErrorEvent,
    ErrorInfo,
    ReasoningBlock,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    ResponseEndEvent,
    ResponseStartEvent,
    TextBlock,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCallBlock,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
    UsageEvent,
)


class ResponseAssembler:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_family: str,
        response_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_family = api_family
        self.response_id = response_id
        self.request_id = request_id
        self.usage = Usage()
        self.protocol_state: dict = {}
        self.provider_meta: dict = {}
        self._content: list[TextBlock | ReasoningBlock | ToolCallBlock | None] = []
        self._tool_call_buffers: dict[int, str] = {}

    def response_start(self) -> ResponseStartEvent:
        return ResponseStartEvent(
            provider=self.provider,
            model=self.model,
            response_id=self.response_id,
        )

    def set_usage(self, usage: Usage) -> UsageEvent:
        self.usage = usage
        return UsageEvent(usage=usage)

    def update_usage(self, **values: int | str) -> UsageEvent:
        self.usage = self.usage.model_copy(update=values)
        return UsageEvent(usage=self.usage)

    def update_protocol_state(self, **values: object) -> None:
        self.protocol_state.update(values)

    def update_provider_meta(self, **values: object) -> None:
        self.provider_meta.update(values)

    def text_start(self, index: int) -> TextStartEvent:
        self._set_block(index, TextBlock(text=""), expected_type="text")
        return TextStartEvent(index=index)

    def text_delta(self, index: int, delta: str) -> TextDeltaEvent:
        block = self._require_block(index, TextBlock, "text")
        block.text += delta
        return TextDeltaEvent(index=index, delta=delta)

    def text_end(self, index: int) -> TextEndEvent:
        block = self._require_block(index, TextBlock, "text")
        return TextEndEvent(index=index, text=block.text)

    def reasoning_start(self, index: int) -> ReasoningStartEvent:
        self._set_block(index, ReasoningBlock(text=""), expected_type="reasoning")
        return ReasoningStartEvent(index=index)

    def reasoning_delta(self, index: int, delta: str) -> ReasoningDeltaEvent:
        block = self._require_block(index, ReasoningBlock, "reasoning")
        block.text += delta
        return ReasoningDeltaEvent(index=index, delta=delta)

    def reasoning_end(
        self,
        index: int,
        *,
        signature: str | None = None,
        redacted: bool | None = None,
    ) -> ReasoningEndEvent:
        block = self._require_block(index, ReasoningBlock, "reasoning")
        if signature is not None:
            block.signature = signature
        if redacted is not None:
            block.redacted = redacted
        return ReasoningEndEvent(
            index=index,
            text=block.text,
            signature=block.signature,
            redacted=block.redacted,
        )

    def tool_call_start(self, index: int, *, tool_call_id: str, name: str) -> ToolCallStartEvent:
        self._set_block(
            index,
            ToolCallBlock(id=tool_call_id, name=name, arguments={}),
            expected_type="tool_call",
        )
        self._tool_call_buffers[index] = ""
        return ToolCallStartEvent(index=index, id=tool_call_id, name=name)

    def tool_call_delta(self, index: int, delta: str) -> ToolCallDeltaEvent:
        self._require_block(index, ToolCallBlock, "tool_call")
        self._tool_call_buffers[index] = self._tool_call_buffers.get(index, "") + delta
        return ToolCallDeltaEvent(index=index, delta=delta)

    def tool_call_end(
        self,
        index: int,
        *,
        arguments: dict | None = None,
    ) -> ToolCallEndEvent:
        block = self._require_block(index, ToolCallBlock, "tool_call")
        if arguments is None:
            raw_arguments = self._tool_call_buffers.pop(index, "").strip()
            if not raw_arguments:
                block.arguments = {}
            else:
                try:
                    block.arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    raise ProviderProtocolError(
                        make_error_info(
                            code="invalid_tool_arguments",
                            message=f"Invalid streamed tool-call arguments at index {index}",
                            provider=self.provider,
                            api_family=self.api_family,
                            retryable=False,
                            raw={"arguments": raw_arguments},
                        )
                    ) from exc
        else:
            self._tool_call_buffers.pop(index, None)
            block.arguments = arguments

        return ToolCallEndEvent(index=index, tool_call=block)

    def build_response(self, *, finish_reason: str) -> AssistantResponse:
        return AssistantResponse(
            provider=self.provider,
            model=self.model,
            api_family=self.api_family,
            content=self._materialize_content(),
            finish_reason=finish_reason,
            usage=self.usage,
            response_id=self.response_id,
            request_id=self.request_id,
            protocol_state=self.protocol_state,
            provider_meta=self.provider_meta,
        )

    def response_end(self, *, finish_reason: str) -> ResponseEndEvent:
        return ResponseEndEvent(response=self.build_response(finish_reason=finish_reason))

    def error(self, error: ErrorInfo) -> ErrorEvent:
        partial_response = AssistantResponse(
            provider=self.provider,
            model=self.model,
            api_family=self.api_family,
            content=self._materialize_content(allow_incomplete=True),
            finish_reason="error",
            usage=self.usage,
            response_id=self.response_id,
            request_id=self.request_id,
            protocol_state=self.protocol_state,
            provider_meta=self.provider_meta,
        )
        return ErrorEvent(error=error, partial_response=partial_response)

    def _set_block(
        self,
        index: int,
        block: TextBlock | ReasoningBlock | ToolCallBlock,
        *,
        expected_type: str,
    ) -> None:
        self._ensure_index(index)
        existing = self._content[index]
        if existing is not None and existing.type != expected_type:
            raise ProviderProtocolError(
                make_error_info(
                    code="invalid_content_index",
                    message=f"Content index {index} already contains a different block type",
                    provider=self.provider,
                    api_family=self.api_family,
                )
            )
        self._content[index] = block

    def _require_block(self, index: int, expected_class: type, expected_type: str):
        self._ensure_index(index)
        block = self._content[index]
        if not isinstance(block, expected_class):
            raise ProviderProtocolError(
                make_error_info(
                    code="missing_content_block",
                    message=f"Expected {expected_type} block at content index {index}",
                    provider=self.provider,
                    api_family=self.api_family,
                )
            )
        return block

    def _ensure_index(self, index: int) -> None:
        while len(self._content) <= index:
            self._content.append(None)

    def _materialize_content(self, *, allow_incomplete: bool = False) -> list[TextBlock | ReasoningBlock | ToolCallBlock]:
        if allow_incomplete:
            return [block for block in self._content if block is not None]

        missing_indices = [index for index, block in enumerate(self._content) if block is None]
        if missing_indices:
            raise ProviderProtocolError(
                make_error_info(
                    code="incomplete_response",
                    message=f"Missing content blocks at indices: {', '.join(str(index) for index in missing_indices)}",
                    provider=self.provider,
                    api_family=self.api_family,
                )
            )
        return [block for block in self._content if block is not None]