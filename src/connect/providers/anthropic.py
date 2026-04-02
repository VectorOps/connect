from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ..exceptions import ConnectError, ProviderProtocolError, make_error_info
from ..transport.http import HttpStatusError
from ..transport.assembly import ResponseAssembler
from ..transport.sse import iter_sse_response
from ..types import (
    AssistantMessage,
    GenerateRequest,
    ImageBlock,
    Message,
    ModelSpec,
    RequestOptions,
    ResponseEndEvent,
    SpecificToolChoice,
    StreamEvent,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from .base import BaseProviderAdapter


class AnthropicProvider(BaseProviderAdapter):
    provider_name = "anthropic"
    api_family = "anthropic-messages"
    default_base_url = "https://api.anthropic.com/v1"
    stream_path = "/messages"

    def build_headers(
        self,
        model: ModelSpec,
        request: GenerateRequest,
        options: RequestOptions,
    ) -> dict[str, str]:
        headers = {
            "accept": "text/event-stream",
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        beta_features: list[str] = []
        if self._should_enable_fine_grained_tool_streaming(request, options):
            beta_features.append("fine-grained-tool-streaming-2025-05-14")
        if self._should_enable_interleaved_thinking(model, request, options):
            beta_features.append("interleaved-thinking-2025-05-14")
        if beta_features:
            headers["anthropic-beta"] = ",".join(beta_features)

        default_headers = model.protocol_defaults.get("headers")
        if isinstance(default_headers, dict):
            headers.update({str(key): str(value) for key, value in default_headers.items()})
        return headers

    def build_usage(
        self,
        payload: Any,
        *,
        completeness: str,
    ) -> Usage | None:
        if not isinstance(payload, dict):
            return None

        input_tokens = int(payload.get("input_tokens") or 0)
        output_tokens = int(payload.get("output_tokens") or 0)
        cache_read_tokens = int(payload.get("cache_read_input_tokens") or 0)
        cache_write_tokens = int(payload.get("cache_creation_input_tokens") or 0)
        total_tokens = input_tokens + output_tokens + cache_read_tokens + cache_write_tokens

        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=0,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            total_tokens=total_tokens,
            completeness=completeness,
        )

    def build_error(
        self,
        payload: Any,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> Any:
        raw = payload if isinstance(payload, dict) else None
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            payload = payload["error"]

        code = "provider_error"
        message = "Provider request failed"
        if isinstance(payload, dict):
            code = str(payload.get("type") or payload.get("code") or code)
            message = str(payload.get("message") or message)

        return make_error_info(
            code=code,
            message=message,
            provider=self.provider_name,
            api_family=self.api_family,
            status_code=status_code,
            retryable=retryable,
            raw=raw,
        )

    def build_payload(
        self,
        model: ModelSpec,
        request: GenerateRequest,
        options: RequestOptions,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model.model,
            "stream": True,
            "messages": self._build_messages(model, request, options),
            "max_tokens": request.max_output_tokens or model.max_output_tokens or 1024,
        }

        system_blocks = self._build_system_blocks(model, request, options)
        if system_blocks is not None:
            payload["system"] = system_blocks

        thinking_payload, output_config = self._build_thinking_payload(model, request)
        if thinking_payload is not None:
            payload["thinking"] = thinking_payload
        if output_config is not None:
            payload["output_config"] = output_config

        if request.temperature is not None and thinking_payload is None:
            payload["temperature"] = request.temperature

        metadata = self._build_metadata(request, options)
        if metadata is not None:
            payload["metadata"] = metadata

        if request.tools and request.tool_choice != "none":
            payload["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": self._convert_tool_schema(tool.input_schema),
                }
                for tool in request.tools
            ]
            tool_choice = self._build_tool_choice(request.tool_choice)
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        for option_name in ("service_tier", "stop_sequences"):
            if option_name in options.provider_options and option_name not in payload:
                payload[option_name] = options.provider_options[option_name]

        protocol_defaults = model.protocol_defaults
        for option_name, option_value in protocol_defaults.items():
            if option_name not in {"headers"} and option_name not in payload:
                payload[option_name] = option_value

        return payload

    async def stream_response(
        self,
        *,
        model: ModelSpec,
        request: GenerateRequest,
        options: RequestOptions,
        http,
    ) -> AsyncIterator[StreamEvent]:
        payload = self.build_payload(model, request, options)
        headers = self.build_headers(model, request, options)
        assembler = ResponseAssembler(
            provider=self.provider_name,
            model=model.model,
            api_family=model.api_family,
        )
        block_types: dict[int, str] = {}
        reasoning_signatures: dict[int, str] = {}
        reasoning_redacted: set[int] = set()
        tool_call_initial_arguments: dict[int, dict[str, Any]] = {}
        tool_call_streamed_arguments: set[int] = set()
        usage_payload: dict[str, Any] = {}
        finish_reason = "stop"

        try:
            stream_response = await http.stream(
                "POST",
                self.request_url(model),
                provider=self.provider_name,
                model=model.model,
                api_family=model.api_family,
                auth=options.auth,
                headers={**headers, **options.headers},
                json_body=payload,
                timeout=options.timeout,
                expected_status=200,
            )
        except HttpStatusError as exc:
            yield assembler.error(self.build_http_error(exc.response))
            return
        except ConnectError as exc:
            yield assembler.error(exc.error)
            return

        async with stream_response as response:
            assembler.request_id = response.request_id
            yield assembler.response_start()

            async for frame in iter_sse_response(response):
                if frame.is_done:
                    break
                if not frame.data:
                    continue

                try:
                    event = json.loads(frame.data)
                except json.JSONDecodeError:
                    yield assembler.error(
                        make_error_info(
                            code="invalid_sse_json",
                            message="Anthropic stream emitted invalid JSON",
                            provider=self.provider_name,
                            api_family=model.api_family,
                            raw={"data": frame.data},
                        )
                    )
                    return

                event_type = event.get("type")
                if event_type == "ping":
                    continue

                if event_type == "error":
                    yield assembler.error(self.build_error(event))
                    return

                if event_type == "message_start":
                    message = event.get("message") or {}
                    response_id = message.get("id")
                    if isinstance(response_id, str) and response_id:
                        assembler.response_id = response_id
                        assembler.update_protocol_state(anthropic_response_id=response_id)
                    message_usage = message.get("usage")
                    if isinstance(message_usage, dict):
                        usage_payload.update(message_usage)
                        usage = self.build_usage(usage_payload, completeness="partial")
                        if usage is not None:
                            yield assembler.set_usage(usage)
                    stop_reason = message.get("stop_reason")
                    normalized = self.normalize_finish_reason(stop_reason)
                    if normalized != "unknown":
                        finish_reason = normalized
                    continue

                if event_type == "content_block_start":
                    index = self._event_index(event)
                    block = event.get("content_block") or {}
                    block_type = str(block.get("type") or "")
                    block_types[index] = block_type

                    if block_type == "text":
                        yield assembler.text_start(index)
                        text = str(block.get("text") or "")
                        if text:
                            yield assembler.text_delta(index, text)
                        continue

                    if block_type == "thinking":
                        yield assembler.reasoning_start(index)
                        text = str(block.get("thinking") or "")
                        if text:
                            yield assembler.reasoning_delta(index, text)
                        signature = block.get("signature")
                        if isinstance(signature, str) and signature:
                            reasoning_signatures[index] = signature
                        continue

                    if block_type == "redacted_thinking":
                        yield assembler.reasoning_start(index)
                        yield assembler.reasoning_delta(index, "[Reasoning redacted]")
                        signature = block.get("data")
                        if isinstance(signature, str) and signature:
                            reasoning_signatures[index] = signature
                        reasoning_redacted.add(index)
                        continue

                    if block_type == "tool_use":
                        tool_call_id = self._normalize_tool_call_id(model, block.get("id"))
                        tool_name = str(block.get("name") or "tool")
                        yield assembler.tool_call_start(index, tool_call_id=tool_call_id, name=tool_name)
                        input_payload = block.get("input")
                        if isinstance(input_payload, dict):
                            tool_call_initial_arguments[index] = input_payload
                        continue

                if event_type == "content_block_delta":
                    index = self._event_index(event)
                    delta = event.get("delta") or {}
                    delta_type = delta.get("type")

                    if delta_type == "text_delta":
                        yield assembler.text_delta(index, str(delta.get("text") or ""))
                        continue

                    if delta_type == "thinking_delta":
                        yield assembler.reasoning_delta(index, str(delta.get("thinking") or ""))
                        continue

                    if delta_type == "signature_delta":
                        signature = str(delta.get("signature") or "")
                        if signature:
                            reasoning_signatures[index] = f"{reasoning_signatures.get(index, '')}{signature}"
                        continue

                    if delta_type == "input_json_delta":
                        tool_call_streamed_arguments.add(index)
                        yield assembler.tool_call_delta(index, str(delta.get("partial_json") or ""))
                        continue

                if event_type == "content_block_stop":
                    index = self._event_index(event)
                    block_type = block_types.get(index)
                    if block_type == "text":
                        yield assembler.text_end(index)
                        continue

                    if block_type in {"thinking", "redacted_thinking"}:
                        signature = reasoning_signatures.get(index)
                        if signature:
                            assembler.update_block_metadata(
                                index,
                                protocol_meta={
                                    "anthropic_signature": signature,
                                    "anthropic_provider": self.provider_name,
                                    "anthropic_model": model.model,
                                },
                            )
                        yield assembler.reasoning_end(
                            index,
                            signature=signature,
                            redacted=index in reasoning_redacted,
                        )
                        continue

                    if block_type == "tool_use":
                        try:
                            if index in tool_call_streamed_arguments:
                                yield assembler.tool_call_end(
                                    index,
                                    arguments=self.parse_tool_call_arguments(
                                        assembler.take_tool_call_buffer(index),
                                        index=index,
                                    ),
                                )
                            else:
                                yield assembler.tool_call_end(
                                    index,
                                    arguments=tool_call_initial_arguments.get(index, {}),
                                )
                        except ProviderProtocolError as exc:
                            yield assembler.error(exc.error)
                            return
                        continue

                if event_type == "message_delta":
                    delta = event.get("delta") or {}
                    next_finish_reason = self.normalize_finish_reason(delta.get("stop_reason"))
                    if next_finish_reason != "unknown":
                        finish_reason = next_finish_reason
                    message_usage = event.get("usage")
                    if isinstance(message_usage, dict):
                        usage_payload.update(message_usage)
                        usage = self.build_usage(usage_payload, completeness="partial")
                        if usage is not None:
                            yield assembler.set_usage(usage)
                    continue

                if event_type == "message_stop":
                    final_usage = self.build_usage(usage_payload, completeness="final")
                    if final_usage is not None:
                        yield assembler.set_usage(final_usage)
                    yield assembler.response_end(finish_reason=finish_reason)
                    return

        final_usage = self.build_usage(usage_payload, completeness="partial")
        if final_usage is not None:
            yield assembler.set_usage(final_usage)
        yield assembler.response_end(finish_reason=finish_reason)

    def normalize_finish_reason(self, value: str | None) -> str:
        normalized = (value or "").lower()
        if normalized in {"", "end_turn", "stop_sequence", "pause_turn"}:
            return "stop"
        if normalized == "max_tokens":
            return "length"
        if normalized == "tool_use":
            return "tool_call"
        if normalized in {"refusal", "sensitive"}:
            return "content_filter"
        return super().normalize_finish_reason(normalized or value)

    def _build_messages(self, model: ModelSpec, request: GenerateRequest, options: RequestOptions) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        index = 0
        while index < len(request.messages):
            message = request.messages[index]

            if isinstance(message, ToolResultMessage):
                tool_results: list[dict[str, Any]] = []
                while index < len(request.messages) and isinstance(request.messages[index], ToolResultMessage):
                    current = request.messages[index]
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": self._normalize_tool_call_id(model, current.tool_call_id),
                            "content": self._serialize_tool_result_content(current.content),
                            "is_error": current.is_error,
                        }
                    )
                    index += 1
                messages.append({"role": "user", "content": tool_results})
                continue

            serialized = self._serialize_message(model, message)
            if serialized is not None:
                messages.append(serialized)
            index += 1

        self._apply_prompt_cache_control(messages=messages, options=options)

        return messages

    def _serialize_message(self, model: ModelSpec, message: Message) -> dict[str, Any] | None:
        if isinstance(message, UserMessage):
            content = message.content
            if isinstance(content, str):
                value = content.strip()
                if not value:
                    return None
                return {"role": "user", "content": value}
            blocks = [self._serialize_user_block(block) for block in content if not self._is_empty_text_block(block)]
            if not blocks:
                return None
            return {"role": "user", "content": blocks}

        if isinstance(message, AssistantMessage):
            blocks: list[dict[str, Any]] = []
            for block in message.content:
                if block.type == "text":
                    if not block.text:
                        continue
                    blocks.append({"type": "text", "text": block.text})
                    continue

                if block.type == "reasoning":
                    replay_signature = self._resolve_replay_signature(model, block)
                    if block.redacted and isinstance(block.signature, str) and block.signature:
                        blocks.append({"type": "redacted_thinking", "data": block.signature})
                        continue
                    if replay_signature is not None:
                        blocks.append(
                            {
                                "type": "thinking",
                                "thinking": block.text,
                                "signature": replay_signature,
                            }
                        )
                        continue
                    if block.text:
                        blocks.append({"type": "text", "text": block.text})
                    continue

                if block.type == "tool_call":
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": self._normalize_tool_call_id(model, block.id),
                            "name": block.name,
                            "input": block.arguments or {},
                        }
                    )

            if not blocks:
                return None
            return {"role": "assistant", "content": blocks}

        return None

    def _serialize_user_block(self, block) -> dict[str, Any]:
        if block.type == "text":
            return {"type": "text", "text": block.text}
        if isinstance(block, ImageBlock):
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": block.mime_type,
                    "data": block.data,
                },
            }
        raise TypeError(f"Unsupported Anthropic block type: {block.type!r}")

    def _serialize_tool_result_content(self, blocks: list[Any]) -> str | list[dict[str, Any]]:
        text_blocks = [block for block in blocks if block.type == "text" and block.text]
        image_blocks = [block for block in blocks if isinstance(block, ImageBlock)]

        if not image_blocks:
            return "\n".join(block.text for block in text_blocks)

        content: list[dict[str, Any]] = [{"type": "text", "text": block.text} for block in text_blocks]
        if not content:
            content.append({"type": "text", "text": "(see attached image)"})
        content.extend(self._serialize_user_block(block) for block in image_blocks)
        return content

    def _build_tool_choice(self, tool_choice: str | SpecificToolChoice | None) -> dict[str, Any] | None:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, SpecificToolChoice):
            return {"type": "tool", "name": tool_choice.name}
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "required":
            return {"type": "any"}
        return None

    def _convert_tool_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(schema))

    def _build_thinking_payload(
        self,
        model: ModelSpec,
        request: GenerateRequest,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        reasoning = request.reasoning
        if reasoning is None:
            return None, None

        if reasoning.enabled is False or reasoning.effort == "none":
            return {"type": "disabled"}, None

        if self._supports_adaptive_thinking(model):
            output_config = None
            effort = self._adaptive_effort(model, reasoning.effort)
            if effort is not None:
                output_config = {"effort": effort}
            return {"type": "adaptive"}, output_config

        budget_tokens = reasoning.max_tokens or self._thinking_budget_for_effort(reasoning.effort or "medium")
        return {"type": "enabled", "budget_tokens": budget_tokens}, None

    def _supports_adaptive_thinking(self, model: ModelSpec) -> bool:
        model_name = model.model.lower()
        return any(token in model_name for token in ("opus-4-6", "opus-4.6", "sonnet-4-6", "sonnet-4.6"))

    def _adaptive_effort(self, model: ModelSpec, effort: str | None) -> str | None:
        if effort in {None, "minimal"}:
            return "low" if effort == "minimal" else None
        mapping = {
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "max" if "opus-4-6" in model.model.lower() or "opus-4.6" in model.model.lower() else "high",
        }
        return mapping.get(effort)

    def _thinking_budget_for_effort(self, effort: str) -> int:
        budgets = {
            "minimal": 1024,
            "low": 2048,
            "medium": 4096,
            "high": 8192,
            "xhigh": 16384,
        }
        return budgets.get(effort, 4096)

    def _normalize_tool_call_id(self, model: ModelSpec, value: Any) -> str:
        normalized = str(value or "call").strip() or "call"
        normalized = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in normalized)
        max_length = model.capabilities.get("tool_call_id_max_length")
        if not isinstance(max_length, int) or max_length <= 0:
            max_length = 64
        normalized = normalized[:max_length]
        return normalized or "call"

    def _event_index(self, event: dict[str, Any]) -> int:
        value = event.get("index")
        return value if isinstance(value, int) and value >= 0 else 0

    def _is_empty_text_block(self, block: Any) -> bool:
        return block.type == "text" and not block.text

    def _resolve_cache_retention(self, options: RequestOptions) -> str:
        value = options.provider_options.get("cache_retention", "short")
        return value if value in {"none", "short", "long"} else "short"

    def _cache_control(self, model: ModelSpec, options: RequestOptions) -> dict[str, Any] | None:
        retention = self._resolve_cache_retention(options)
        if retention == "none":
            return None
        if retention == "long" and self.resolve_base_url(model).startswith("https://api.anthropic.com"):
            return {"type": "ephemeral", "ttl": "1h"}
        return {"type": "ephemeral"}

    def _build_system_blocks(
        self,
        model: ModelSpec,
        request: GenerateRequest,
        options: RequestOptions,
    ) -> list[dict[str, Any]] | None:
        if request.system_prompt is None:
            return None
        blocks: list[dict[str, Any]] = [{"type": "text", "text": request.system_prompt}]
        cache_control = self._cache_control(model, options)
        if cache_control is not None:
            blocks[-1]["cache_control"] = cache_control
        return blocks

    def _apply_prompt_cache_control(self, *, messages: list[dict[str, Any]], options: RequestOptions) -> None:
        if self._resolve_cache_retention(options) == "none" or not messages:
            return
        cache_control = self._cache_control(ModelSpec(provider="anthropic", model="x", api_family="anthropic-messages"), options)
        if cache_control is None:
            return
        last_message = messages[-1]
        content = last_message.get("content")
        if isinstance(content, str):
            last_message["content"] = [{"type": "text", "text": content, "cache_control": cache_control}]
            return
        if isinstance(content, list) and content:
            last_block = content[-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = cache_control

    def _build_metadata(self, request: GenerateRequest, options: RequestOptions) -> dict[str, Any] | None:
        metadata = options.provider_options.get("anthropic_metadata")
        if isinstance(metadata, dict):
            result = {str(key): value for key, value in metadata.items() if key == "user_id" and isinstance(value, str)}
            return result or None
        user_id = request.metadata.get("user_id")
        if isinstance(user_id, str):
            return {"user_id": user_id}
        return None

    def _resolve_replay_signature(self, model: ModelSpec, block: Any) -> str | None:
        protocol_meta = getattr(block, "protocol_meta", {}) or {}
        provider_name = protocol_meta.get("anthropic_provider")
        model_name = protocol_meta.get("anthropic_model")
        signature = protocol_meta.get("anthropic_signature") or block.signature
        if block.redacted:
            return block.signature if isinstance(block.signature, str) and block.signature else None
        if provider_name is None and model_name is None:
            return signature if isinstance(signature, str) and signature else None
        if provider_name == self.provider_name and model_name == model.model and isinstance(signature, str) and signature:
            return signature
        return None

    def _should_enable_interleaved_thinking(
        self,
        model: ModelSpec,
        request: GenerateRequest,
        options: RequestOptions,
    ) -> bool:
        explicit = options.provider_options.get("interleaved_thinking")
        if isinstance(explicit, bool):
            return explicit and request.reasoning is not None and not self._supports_adaptive_thinking(model)
        return request.reasoning is not None and not self._supports_adaptive_thinking(model)

    def _should_enable_fine_grained_tool_streaming(
        self,
        request: GenerateRequest,
        options: RequestOptions,
    ) -> bool:
        explicit = options.provider_options.get("fine_grained_tool_streaming")
        if isinstance(explicit, bool):
            return explicit and bool(request.tools) and request.tool_choice != "none"
        return bool(request.tools) and request.tool_choice != "none"