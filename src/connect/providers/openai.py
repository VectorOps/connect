from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ..exceptions import ConnectError, PermanentProviderError, ProviderProtocolError, make_error_info
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
    StreamEvent,
    ToolResultMessage,
    UsageEvent,
    UserMessage,
)
from .base import BaseProviderAdapter


class OpenAIProvider(BaseProviderAdapter):
    provider_name = "openai"
    api_family = "openai-responses"
    default_base_url = "https://api.openai.com/v1"
    stream_path = "/responses"

    def build_headers(
        self,
        model: ModelSpec,
        request: GenerateRequest,
        options: RequestOptions,
    ) -> dict[str, str]:
        headers = {
            "accept": "text/event-stream",
            "content-type": "application/json",
        }
        if options.idempotency_key:
            headers["idempotency-key"] = options.idempotency_key

        default_headers = model.protocol_defaults.get("headers")
        if isinstance(default_headers, dict):
            headers.update({str(key): str(value) for key, value in default_headers.items()})
        return headers

    def build_payload(
        self,
        model: ModelSpec,
        request: GenerateRequest,
        options: RequestOptions,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model.model,
            "stream": True,
            "input": self._build_input(model, request),
        }

        if request.system_prompt and model.capabilities.get("supports_developer_role", True):
            payload["instructions"] = request.system_prompt

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload["max_output_tokens"] = request.max_output_tokens
        if request.metadata:
            payload["metadata"] = request.metadata
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                    "strict": True if tool.strict is None else tool.strict,
                }
                for tool in request.tools
            ]

        tool_choice = self.map_tool_choice(request.tool_choice)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        reasoning = self._build_reasoning_payload(model, request)
        if reasoning is not None:
            payload["reasoning"] = reasoning

        text_config = self._build_text_config(request)
        if text_config is not None:
            payload["text"] = text_config

        include = self._build_include(request, options)
        if include is not None:
            payload["include"] = include

        for option_name in (
            "background",
            "context_management",
            "conversation",
            "max_tool_calls",
            "parallel_tool_calls",
            "previous_response_id",
            "prompt_cache_key",
            "prompt_cache_retention",
            "safety_identifier",
            "service_tier",
            "store",
            "stream_options",
            "top_p",
            "truncation",
            "user",
        ):
            if option_name in options.provider_options:
                payload[option_name] = options.provider_options[option_name]

        if request.session and request.session.continue_from and "previous_response_id" not in payload:
            payload["previous_response_id"] = request.session.continue_from

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
        transport = options.provider_options.get("transport", "sse")
        if transport == "websocket":
            raise PermanentProviderError(
                make_error_info(
                    code="unsupported_transport",
                    message="OpenAI WebSocket transport is not implemented yet",
                    provider=self.provider_name,
                    api_family=self.api_family,
                )
            )

        payload = self.build_payload(model, request, options)
        headers = self.build_headers(model, request, options)
        assembler = ResponseAssembler(
            provider=self.provider_name,
            model=model.model,
            api_family=model.api_family,
        )
        started_text: set[int] = set()
        ended_text: set[int] = set()
        started_reasoning: set[int] = set()
        ended_reasoning: set[int] = set()
        started_tool_calls: set[int] = set()
        tool_call_metadata: dict[str, dict[str, str]] = {}

        try:
            stream_response = await http.stream(
                "POST",
                self.request_url(model),
                provider=self.provider_name,
                api_family=model.api_family,
                headers={**headers, **options.headers},
                json_body=payload,
                timeout=options.timeout,
                expected_status=200,
            )
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
                except json.JSONDecodeError as exc:
                    yield assembler.error(
                        make_error_info(
                            code="invalid_sse_json",
                            message="OpenAI stream emitted invalid JSON",
                            provider=self.provider_name,
                            api_family=model.api_family,
                            raw={"data": frame.data},
                        )
                    )
                    return

                try:
                    emitted_events, terminal = self._map_stream_event(
                        event,
                        assembler=assembler,
                        started_text=started_text,
                        ended_text=ended_text,
                        started_reasoning=started_reasoning,
                        ended_reasoning=ended_reasoning,
                        started_tool_calls=started_tool_calls,
                        tool_call_metadata=tool_call_metadata,
                    )
                except ProviderProtocolError as exc:
                    yield assembler.error(exc.error)
                    return

                for emitted_event in emitted_events:
                    yield emitted_event
                    if isinstance(emitted_event, ResponseEndEvent) or emitted_event.type == "error":
                        return

                if terminal:
                    return

        yield assembler.response_end(finish_reason="unknown")

    def _build_input(self, model: ModelSpec, request: GenerateRequest) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if request.system_prompt and not model.capabilities.get("supports_developer_role", True):
            items.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": request.system_prompt}],
                }
            )

        for message in request.messages:
            serialized = self._serialize_message(message)
            if isinstance(serialized, list):
                items.extend(serialized)
            else:
                items.append(serialized)
        return items

    def _serialize_message(self, message: Message) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(message, UserMessage):
            content = message.content
            if isinstance(content, str):
                content_items = [{"type": "input_text", "text": content}]
            else:
                content_items = [self._serialize_input_block(block) for block in content]
            return {"role": "user", "content": content_items}

        if isinstance(message, AssistantMessage):
            output_items: list[dict[str, Any]] = []
            text_blocks = [block for block in message.content if block.type == "text"]
            if text_blocks:
                output_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [self._serialize_assistant_block(block) for block in text_blocks],
                    }
                )

            for block_index, block in enumerate(message.content):
                if block.type == "reasoning":
                    reasoning_item = self._serialize_assistant_block(block, block_index=block_index)
                    output_items.append(reasoning_item)
                elif block.type == "tool_call":
                    output_items.append(self._serialize_assistant_block(block, block_index=block_index))

            return output_items

        if isinstance(message, ToolResultMessage):
            output_item = {
                "type": "function_call_output",
                "call_id": message.tool_call_id,
                "output": [self._serialize_input_block(block) for block in message.content],
            }
            if isinstance(output_item["output"], list) and not output_item["output"]:
                output_item["output"] = ""
            return output_item

        raise TypeError(f"Unsupported message type: {type(message)!r}")

    def _serialize_input_block(self, block) -> dict[str, Any]:
        if block.type == "text":
            return {"type": "input_text", "text": block.text}
        if isinstance(block, ImageBlock):
            return {
                "type": "input_image",
                "image_url": f"data:{block.mime_type};base64,{block.data}",
                "detail": block.detail,
            }
        raise TypeError(f"Unsupported input block type: {block.type!r}")

    def _serialize_assistant_block(self, block, *, block_index: int | None = None) -> dict[str, Any]:
        if block.type == "text":
            return {"type": "output_text", "text": block.text}
        if block.type == "reasoning":
            payload = {
                "type": "reasoning",
                "id": self._reasoning_item_id(block, block_index=block_index),
                "summary": [{"type": "summary_text", "text": block.text}],
            }
            if block.text:
                payload["content"] = [{"type": "reasoning_text", "text": block.text}]
            encrypted_content = block.protocol_meta.get("openai_encrypted_content")
            if not isinstance(encrypted_content, str) or not encrypted_content:
                if block.redacted and isinstance(block.signature, str) and block.signature:
                    encrypted_content = block.signature
            if isinstance(encrypted_content, str) and encrypted_content:
                payload["encrypted_content"] = encrypted_content
            return payload
        if block.type == "tool_call":
            return {
                "type": "function_call",
                "call_id": block.id,
                "name": block.name,
                "arguments": json.dumps(block.arguments),
            }
        raise TypeError(f"Unsupported assistant block type: {block.type!r}")

    def _build_reasoning_payload(self, model: ModelSpec, request: GenerateRequest) -> dict[str, Any] | None:
        reasoning = request.reasoning
        if reasoning is None:
            return None

        payload: dict[str, Any] = {}
        if reasoning.effort is not None:
            payload["effort"] = reasoning.effort
        if reasoning.summary is not None:
            payload["summary"] = reasoning.summary
        return payload or None

    def _reasoning_item_id(self, block, *, block_index: int | None) -> str:
        protocol_reasoning_id = block.protocol_meta.get("openai_reasoning_id")
        if isinstance(protocol_reasoning_id, str) and protocol_reasoning_id:
            return protocol_reasoning_id
        if block.signature:
            return f"reasoning_{block.signature}"
        return f"reasoning_{block_index or 0}"

    def _build_text_config(self, request: GenerateRequest) -> dict[str, Any] | None:
        if request.response_format is None or request.response_format.type == "text":
            return None

        if request.response_format.type == "json_object":
            return {"format": {"type": "json_object"}}

        return {
            "format": {
                "type": "json_schema",
                "name": request.response_format.name or "response",
                "schema": request.response_format.json_schema,
                "strict": bool(request.response_format.strict),
            }
        }

    def _build_include(self, request: GenerateRequest, options: RequestOptions) -> list[str] | None:
        include = options.provider_options.get("include")
        values: list[str] = []

        if isinstance(include, str) and include:
            values.append(include)
        elif isinstance(include, (list, tuple, set)):
            values.extend(str(value) for value in include if str(value))

        if request.reasoning is not None and "reasoning.encrypted_content" not in values:
            values.append("reasoning.encrypted_content")

        return values or None

    def _map_stream_event(
        self,
        event: dict[str, Any],
        *,
        assembler: ResponseAssembler,
        started_text: set[int],
        ended_text: set[int],
        started_reasoning: set[int],
        ended_reasoning: set[int],
        started_tool_calls: set[int],
        tool_call_metadata: dict[str, dict[str, str]],
    ) -> tuple[list[StreamEvent], bool]:
        emitted: list[StreamEvent] = []
        event_type = event.get("type")

        if event_type == "error":
            return [assembler.error(self.build_error(event))], True

        if event_type == "response.failed":
            error_payload = event.get("response", {}).get("error") or event.get("error") or event
            return [assembler.error(self.build_error(error_payload))], True

        if event_type in {"response.created", "response.in_progress"}:
            response = event.get("response") or {}
            response_id = response.get("id")
            if isinstance(response_id, str) and response_id:
                assembler.response_id = response_id
                assembler.update_protocol_state(previous_response_id=response_id)
            response_status = response.get("status")
            if isinstance(response_status, str) and response_status:
                assembler.update_protocol_state(openai_response_status=response_status)
            usage_event = self._maybe_usage_event(response.get("usage"), assembler=assembler, completeness="partial")
            if usage_event is not None:
                emitted.append(usage_event)
            return emitted, False

        if event_type == "response.output_item.added":
            item = event.get("item") or {}
            index = self._event_index(event)
            item_id = item.get("id")

            if item.get("type") == "reasoning" and index not in started_reasoning:
                started_reasoning.add(index)
                emitted.append(assembler.reasoning_start(index))
                protocol_meta = {}
                if isinstance(item_id, str) and item_id:
                    protocol_meta["openai_reasoning_id"] = item_id
                if protocol_meta:
                    assembler.update_block_metadata(index, protocol_meta=protocol_meta)
                return emitted, False

            if item.get("type") == "message" and index not in started_text:
                started_text.add(index)
                emitted.append(assembler.text_start(index))
                protocol_meta = {}
                if isinstance(item_id, str) and item_id:
                    protocol_meta["openai_message_id"] = item_id
                phase = item.get("phase")
                if isinstance(phase, str) and phase:
                    protocol_meta["openai_message_phase"] = phase
                if protocol_meta:
                    assembler.update_block_metadata(index, protocol_meta=protocol_meta)
                return emitted, False

            if item.get("type") == "function_call" and index not in started_tool_calls:
                if isinstance(item_id, str) and item_id:
                    tool_call_metadata[item_id] = {
                        "id": str(item.get("call_id") or item_id),
                        "name": str(item.get("name") or "tool"),
                    }
                started_tool_calls.add(index)
                emitted.append(
                    assembler.tool_call_start(
                        index,
                        tool_call_id=str(item.get("call_id") or item.get("id") or f"call_{index}"),
                        name=str(item.get("name") or "tool"),
                    )
                )
                protocol_meta = {}
                if isinstance(item_id, str) and item_id:
                    protocol_meta["openai_item_id"] = item_id
                call_status = item.get("status")
                if isinstance(call_status, str) and call_status:
                    protocol_meta["openai_status"] = call_status
                if protocol_meta:
                    assembler.update_block_metadata(index, protocol_meta=protocol_meta)
            return emitted, False

        if event_type == "response.reasoning_summary_part.added":
            index = self._event_index(event)
            if index not in started_reasoning:
                started_reasoning.add(index)
                emitted.append(assembler.reasoning_start(index))
            return emitted, False

        if event_type == "response.content_part.added":
            index = self._event_index(event)
            part = event.get("part") or {}
            part_type = part.get("type")
            if part_type in {"output_text", "text", "refusal"} and index not in started_text:
                started_text.add(index)
                emitted.append(assembler.text_start(index))
            elif part_type in {"reasoning", "reasoning_text", "reasoning_summary_text", "summary_text"} and index not in started_reasoning:
                started_reasoning.add(index)
                emitted.append(assembler.reasoning_start(index))
            protocol_meta = {}
            item_id = event.get("item_id")
            if isinstance(item_id, str) and item_id:
                protocol_meta["openai_item_id"] = item_id
            if protocol_meta and (index in started_text or index in started_reasoning):
                assembler.update_block_metadata(index, protocol_meta=protocol_meta)
            return emitted, False

        if event_type == "response.content_part.done":
            index = self._event_index(event)
            part = event.get("part") or {}
            part_type = part.get("type")
            if part_type in {"output_text", "text", "refusal"} and index not in ended_text:
                final_text = str(part.get("text") or part.get("refusal") or "")
                emitted.extend(self._finalize_text_event(assembler, index=index, started_text=started_text, ended_text=ended_text, final_text=final_text))
            elif part_type in {"reasoning", "reasoning_text", "reasoning_summary_text", "summary_text"} and index not in ended_reasoning:
                final_text = str(part.get("text") or "")
                emitted.extend(
                    self._finalize_reasoning_event(
                        assembler,
                        index=index,
                        started_reasoning=started_reasoning,
                        ended_reasoning=ended_reasoning,
                        final_text=final_text,
                        signature=part.get("signature"),
                        redacted=part.get("redacted"),
                    )
                )
            return emitted, False

        if event_type == "response.output_text.delta":
            index = self._event_index(event)
            if index not in started_text:
                started_text.add(index)
                emitted.append(assembler.text_start(index))
            emitted.append(assembler.text_delta(index, str(event.get("delta") or "")))
            return emitted, False

        if event_type == "response.output_text.done":
            index = self._event_index(event)
            text = event.get("text")
            emitted.extend(
                self._finalize_text_event(
                    assembler,
                    index=index,
                    started_text=started_text,
                    ended_text=ended_text,
                    final_text=str(text or ""),
                )
            )
            return emitted, False

        if event_type == "response.refusal.delta":
            index = self._event_index(event)
            if index not in started_text:
                started_text.add(index)
                emitted.append(assembler.text_start(index))
            emitted.append(assembler.text_delta(index, str(event.get("delta") or "")))
            return emitted, False

        if event_type == "response.refusal.done":
            index = self._event_index(event)
            emitted.extend(
                self._finalize_text_event(
                    assembler,
                    index=index,
                    started_text=started_text,
                    ended_text=ended_text,
                    final_text=str(event.get("refusal") or ""),
                )
            )
            return emitted, False

        if event_type in {"response.reasoning_text.delta", "response.reasoning_summary_text.delta"}:
            index = self._event_index(event)
            if index not in started_reasoning:
                started_reasoning.add(index)
                emitted.append(assembler.reasoning_start(index))
            emitted.append(assembler.reasoning_delta(index, str(event.get("delta") or "")))
            return emitted, False

        if event_type == "response.reasoning_summary_part.done":
            index = self._event_index(event)
            if index not in started_reasoning:
                started_reasoning.add(index)
                emitted.append(assembler.reasoning_start(index))

            current_text = assembler.current_reasoning(index) or ""
            if current_text:
                emitted.append(assembler.reasoning_delta(index, "\n\n"))
            return emitted, False

        if event_type in {"response.reasoning_text.done", "response.reasoning_summary_text.done"}:
            index = self._event_index(event)
            emitted.extend(
                self._finalize_reasoning_event(
                    assembler,
                    index=index,
                    started_reasoning=started_reasoning,
                    ended_reasoning=ended_reasoning,
                    final_text=str(event.get("text") or ""),
                    signature=event.get("signature"),
                    redacted=event.get("redacted"),
                )
            )
            return emitted, False

        if event_type == "response.function_call_arguments.delta":
            index = self._event_index(event)
            if index not in started_tool_calls:
                item_metadata = tool_call_metadata.get(str(event.get("item_id") or ""), {})
                started_tool_calls.add(index)
                emitted.append(
                    assembler.tool_call_start(
                        index,
                        tool_call_id=str(item_metadata.get("id") or event.get("call_id") or event.get("item_id") or f"call_{index}"),
                        name=str(item_metadata.get("name") or event.get("name") or "tool"),
                    )
                )
            emitted.append(assembler.tool_call_delta(index, str(event.get("delta") or "")))
            return emitted, False

        if event_type in {"response.function_call_arguments.done", "response.output_item.done"}:
            item = event.get("item") or {}
            item_type = item.get("type")

            if event_type == "response.output_item.done" and item_type == "reasoning":
                index = self._event_index(event)
                summary = item.get("summary") or []
                summary_text = "\n\n".join(
                    str(part.get("text") or "") for part in summary if isinstance(part, dict)
                )
                content = item.get("content") or []
                content_text = "".join(
                    str(part.get("text") or "") for part in content if isinstance(part, dict)
                )
                final_text = content_text or summary_text
                emitted.extend(
                    self._finalize_reasoning_event(
                        assembler,
                        index=index,
                        started_reasoning=started_reasoning,
                        ended_reasoning=ended_reasoning,
                        final_text=final_text,
                        signature=item.get("encrypted_content") or item.get("signature"),
                        redacted=item.get("encrypted_content") is not None,
                    )
                )
                protocol_meta = {}
                item_id = item.get("id")
                if isinstance(item_id, str) and item_id:
                    protocol_meta["openai_reasoning_id"] = item_id
                encrypted_content = item.get("encrypted_content")
                if isinstance(encrypted_content, str) and encrypted_content:
                    protocol_meta["openai_encrypted_content"] = encrypted_content
                if protocol_meta:
                    assembler.update_block_metadata(index, protocol_meta=protocol_meta)
                return emitted, False

            if event_type == "response.output_item.done" and item_type == "message":
                index = self._event_index(event)
                content = item.get("content") or []
                final_text = "".join(
                    str(part.get("text") or part.get("refusal") or "")
                    for part in content
                    if isinstance(part, dict)
                )
                emitted.extend(
                    self._finalize_text_event(
                        assembler,
                        index=index,
                        started_text=started_text,
                        ended_text=ended_text,
                        final_text=final_text,
                    )
                )
                protocol_meta = {}
                item_id = item.get("id")
                if isinstance(item_id, str) and item_id:
                    protocol_meta["openai_message_id"] = item_id
                phase = item.get("phase")
                if isinstance(phase, str) and phase:
                    protocol_meta["openai_message_phase"] = phase
                if protocol_meta:
                    assembler.update_block_metadata(index, protocol_meta=protocol_meta)
                return emitted, False

            if event_type == "response.output_item.done" and item_type not in {None, "function_call"}:
                return emitted, False

            index = self._event_index(event)
            item_id = str(event.get("item_id") or item.get("id") or "")
            if item_id:
                tool_call_metadata[item_id] = {
                    "id": str(event.get("call_id") or item.get("call_id") or item.get("id") or f"call_{index}"),
                    "name": str(event.get("name") or item.get("name") or "tool"),
                }
            item_metadata = tool_call_metadata.get(item_id, {})
            if index not in started_tool_calls:
                started_tool_calls.add(index)
                emitted.append(
                    assembler.tool_call_start(
                        index,
                        tool_call_id=str(item_metadata.get("id") or event.get("call_id") or item.get("call_id") or item.get("id") or f"call_{index}"),
                        name=str(item_metadata.get("name") or event.get("name") or item.get("name") or "tool"),
                    )
                )
                protocol_meta = {}
                if item_id:
                    protocol_meta["openai_item_id"] = item_id
                if protocol_meta:
                    assembler.update_block_metadata(index, protocol_meta=protocol_meta)

            arguments = event.get("arguments") or item.get("arguments")
            if isinstance(arguments, str) and arguments:
                try:
                    parsed_arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    emitted.append(assembler.tool_call_delta(index, arguments))
                    parsed_arguments = None
                else:
                    emitted.append(assembler.tool_call_end(index, arguments=parsed_arguments))
                    protocol_meta = {}
                    if item_id:
                        protocol_meta["openai_item_id"] = item_id
                    call_status = item.get("status")
                    if isinstance(call_status, str) and call_status:
                        protocol_meta["openai_status"] = call_status
                    if protocol_meta:
                        assembler.update_block_metadata(index, protocol_meta=protocol_meta)
                    return emitted, False
            emitted.append(assembler.tool_call_end(index, arguments=arguments if isinstance(arguments, dict) else None))
            protocol_meta = {}
            if item_id:
                protocol_meta["openai_item_id"] = item_id
            call_status = item.get("status")
            if isinstance(call_status, str) and call_status:
                protocol_meta["openai_status"] = call_status
            if protocol_meta:
                assembler.update_block_metadata(index, protocol_meta=protocol_meta)
            return emitted, False

        if event_type == "response.incomplete":
            response = event.get("response") or {}
            response_id = response.get("id")
            if isinstance(response_id, str) and response_id:
                assembler.response_id = response_id
                assembler.update_protocol_state(previous_response_id=response_id)

            usage_event = self._maybe_usage_event(response.get("usage"), assembler=assembler, completeness="final")
            if usage_event is not None:
                emitted.append(usage_event)

            incomplete_reason = (response.get("incomplete_details") or {}).get("reason")
            finish_reason = self.normalize_finish_reason(incomplete_reason or response.get("status") or event.get("reason"))
            if assembler.has_tool_calls() and finish_reason == "stop":
                finish_reason = "tool_call"
            emitted.append(assembler.response_end(finish_reason=finish_reason))
            return emitted, True

        if event_type == "response.cancelled":
            response = event.get("response") or {}
            response_id = response.get("id")
            if isinstance(response_id, str) and response_id:
                assembler.response_id = response_id
                assembler.update_protocol_state(previous_response_id=response_id)

            usage_event = self._maybe_usage_event(response.get("usage"), assembler=assembler, completeness="final")
            if usage_event is not None:
                emitted.append(usage_event)

            emitted.append(assembler.response_end(finish_reason="cancelled"))
            return emitted, True

        if event_type == "response.completed":
            response = event.get("response") or {}
            response_id = response.get("id")
            if isinstance(response_id, str) and response_id:
                assembler.response_id = response_id
                assembler.update_protocol_state(previous_response_id=response_id)

            usage_event = self._maybe_usage_event(response.get("usage"), assembler=assembler, completeness="final")
            if usage_event is not None:
                emitted.append(usage_event)

            finish_reason = self.normalize_finish_reason(
                response.get("finish_reason") or response.get("status") or event.get("reason")
            )
            if assembler.has_tool_calls() and finish_reason == "stop":
                finish_reason = "tool_call"
            emitted.append(assembler.response_end(finish_reason=finish_reason))
            return emitted, True

        return emitted, False

    def _finalize_text_event(
        self,
        assembler: ResponseAssembler,
        *,
        index: int,
        started_text: set[int],
        ended_text: set[int],
        final_text: str,
    ) -> list[StreamEvent]:
        if index in ended_text:
            return []

        emitted: list[StreamEvent] = []
        if index not in started_text:
            started_text.add(index)
            emitted.append(assembler.text_start(index))

        current_text = assembler.current_text(index) or ""
        if final_text and final_text.startswith(current_text):
            suffix = final_text[len(current_text) :]
            if suffix:
                emitted.append(assembler.text_delta(index, suffix))
        elif final_text and not current_text:
            emitted.append(assembler.text_delta(index, final_text))

        ended_text.add(index)
        emitted.append(assembler.text_end(index))
        return emitted

    def _finalize_reasoning_event(
        self,
        assembler: ResponseAssembler,
        *,
        index: int,
        started_reasoning: set[int],
        ended_reasoning: set[int],
        final_text: str,
        signature: Any,
        redacted: Any,
    ) -> list[StreamEvent]:
        if index in ended_reasoning:
            return []

        emitted: list[StreamEvent] = []
        if index not in started_reasoning:
            started_reasoning.add(index)
            emitted.append(assembler.reasoning_start(index))

        current_text = assembler.current_reasoning(index) or ""
        if final_text and final_text.startswith(current_text):
            suffix = final_text[len(current_text) :]
            if suffix:
                emitted.append(assembler.reasoning_delta(index, suffix))
        elif final_text and not current_text:
            emitted.append(assembler.reasoning_delta(index, final_text))

        ended_reasoning.add(index)
        emitted.append(
            assembler.reasoning_end(
                index,
                signature=signature if isinstance(signature, str) else None,
                redacted=bool(redacted) if redacted is not None else None,
            )
        )
        return emitted

    def _maybe_usage_event(
        self,
        payload: Any,
        *,
        assembler: ResponseAssembler,
        completeness: str,
    ) -> UsageEvent | None:
        usage = self.build_usage(payload, completeness=completeness)
        if usage is None:
            return None
        return assembler.set_usage(usage)

    def _event_index(self, event: dict[str, Any]) -> int:
        for field_name in ("output_index", "content_index", "index"):
            value = event.get(field_name)
            if isinstance(value, int):
                return value
        return 0