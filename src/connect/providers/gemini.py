from __future__ import annotations

import copy
import base64
import binascii
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

from ..exceptions import ConnectError, ProviderProtocolError, make_error_info
from ..transport.http import HttpStatusError
from ..transport.assembly import ResponseAssembler
from ..transport.json_stream import iter_json_values
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


class GeminiProvider(BaseProviderAdapter):
    provider_name = "gemini"
    api_family = "gemini-generate-content"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"
    skip_thought_signature = "skip_thought_signature_validator"

    def request_url(self, model: ModelSpec) -> str:
        base_url = self.resolve_base_url(model).rstrip("/")
        return f"{base_url}/models/{model.model}:streamGenerateContent"

    def normalize_finish_reason(self, value: str | None) -> str:
        normalized = (value or "").upper()
        if normalized in {"", "STOP"}:
            return "stop"
        if normalized == "MAX_TOKENS":
            return "length"
        if normalized in {
            "SAFETY",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "RECITATION",
            "SPII",
            "IMAGE_SAFETY",
            "IMAGE_PROHIBITED_CONTENT",
            "IMAGE_RECITATION",
            "NO_IMAGE",
            "LANGUAGE",
        }:
            return "content_filter"
        if normalized in {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"}:
            return "error"
        if normalized in {"CANCELLED", "CANCELED"}:
            return "cancelled"
        return super().normalize_finish_reason(value.lower() if value else value)

    def build_usage(
        self,
        payload: Any,
        *,
        completeness: str,
    ) -> Usage | None:
        if not isinstance(payload, dict):
            return None

        prompt_tokens = int(payload.get("promptTokenCount") or 0)
        cache_read_tokens = int(payload.get("cachedContentTokenCount") or 0)
        output_tokens = int(payload.get("candidatesTokenCount") or 0)
        reasoning_tokens = int(payload.get("thoughtsTokenCount") or 0)
        total_tokens = int(payload.get("totalTokenCount") or (prompt_tokens + output_tokens))

        return Usage(
            input_tokens=max(prompt_tokens - cache_read_tokens, 0),
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=0,
            total_tokens=total_tokens,
            completeness=completeness,
        )

    def build_headers(
        self,
        model: ModelSpec,
        request: GenerateRequest,
        options: RequestOptions,
    ) -> dict[str, str]:
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }
        default_headers = model.protocol_defaults.get("headers")
        if isinstance(default_headers, dict):
            headers.update({str(key): str(value) for key, value in default_headers.items()})
        return headers

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

        details: list[dict[str, Any]] = []
        if isinstance(payload, dict) and isinstance(payload.get("details"), list):
            details = [item for item in payload["details"] if isinstance(item, dict)]

        code = "provider_error"
        message = "Provider request failed"
        if isinstance(payload, dict):
            google_status = payload.get("status")
            google_reason = None
            for detail in details:
                detail_type = str(detail.get("@type") or "")
                if detail_type.endswith("google.rpc.ErrorInfo") and isinstance(detail.get("reason"), str):
                    google_reason = detail["reason"]
                    break

            status_value = str(google_status).lower() if isinstance(google_status, str) and google_status else None
            reason_value = str(google_reason).lower() if isinstance(google_reason, str) and google_reason else None

            if reason_value in {"api_key_invalid", "invalid_api_key"} or status_value in {
                "unauthenticated",
                "permission_denied",
            }:
                code = "authentication_error"
            elif status_value == "resource_exhausted":
                code = "rate_limit"
                retryable = True
            elif status_value in {"unavailable", "deadline_exceeded"}:
                code = status_value
                retryable = True
            else:
                code = status_value or reason_value or str(payload.get("code") or code).lower()

            if isinstance(payload.get("message"), str) and payload.get("message"):
                message = str(payload["message"])
            else:
                for detail in details:
                    detail_message = detail.get("message")
                    if isinstance(detail_message, str) and detail_message:
                        message = detail_message
                        break

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
        generation_config: dict[str, Any] = {}
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_output_tokens

        response_format = request.response_format
        if response_format is not None and response_format.type != "text":
            generation_config["responseMimeType"] = "application/json"
            if response_format.type == "json_schema":
                generation_config["responseSchema"] = response_format.json_schema

        thinking_config = self._build_thinking_config(model, request)
        if thinking_config is not None:
            generation_config["thinkingConfig"] = thinking_config

        payload: dict[str, Any] = {
            "contents": self._build_contents(model, request),
        }
        if generation_config:
            payload["generationConfig"] = generation_config
        if request.system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": request.system_prompt}],
            }
        if request.tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parametersJsonSchema": self._convert_tool_schema(tool.input_schema),
                        }
                        for tool in request.tools
                    ]
                }
            ]
            tool_config = self._build_tool_config(request.tool_choice)
            if tool_config is not None:
                payload["toolConfig"] = tool_config

        protocol_defaults = model.protocol_defaults
        for option_name, option_value in protocol_defaults.items():
            if option_name not in {"headers"} and option_name not in payload:
                payload[option_name] = option_value

        for option_name in ("cachedContent", "labels", "safetySettings"):
            if option_name in options.provider_options:
                payload[option_name] = options.provider_options[option_name]

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
        part_states: list[dict[str, Any] | None] = []
        finish_reason = "unknown"
        last_usage_payload: Mapping[str, Any] | None = None

        params = options.transport_options.get("query_params")
        if not isinstance(params, Mapping):
            params = None

        try:
            stream_response = await http.stream(
                "POST",
                self.request_url(model),
                provider=self.provider_name,
                model=model.model,
                api_family=model.api_family,
                auth=options.auth,
                params=params,
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

            try:
                async for payload_chunk in iter_json_values(
                    response.iter_bytes(),
                    provider=self.provider_name,
                    api_family=model.api_family,
                ):
                    for chunk in self._iter_stream_chunks(payload_chunk, model=model):

                        error_payload = chunk.get("error")
                        if isinstance(error_payload, dict):
                            yield assembler.error(self.build_error(error_payload))
                            return

                        response_id = chunk.get("responseId")
                        if isinstance(response_id, str) and response_id:
                            assembler.response_id = response_id
                            assembler.update_protocol_state(gemini_response_id=response_id)

                        model_version = chunk.get("modelVersion")
                        if isinstance(model_version, str) and model_version:
                            assembler.update_provider_meta(gemini_model_version=model_version)

                        usage_payload = chunk.get("usageMetadata")
                        partial_usage: Usage | None = None
                        if isinstance(usage_payload, dict):
                            last_usage_payload = usage_payload
                            partial_usage = self.build_usage(usage_payload, completeness="partial")

                        prompt_feedback = chunk.get("promptFeedback")
                        if isinstance(prompt_feedback, dict) and prompt_feedback.get("blockReason"):
                            finish_reason = "content_filter"

                        if partial_usage is not None:
                            yield assembler.set_usage(partial_usage)

                        candidates = chunk.get("candidates")
                        candidate = candidates[0] if isinstance(candidates, list) and candidates else None
                        if not isinstance(candidate, dict):
                            continue

                        candidate_finish_reason = self.normalize_finish_reason(candidate.get("finishReason"))
                        if candidate_finish_reason != "unknown":
                            finish_reason = candidate_finish_reason

                        content = candidate.get("content")
                        parts = content.get("parts") if isinstance(content, dict) else None
                        if not isinstance(parts, list):
                            continue

                        for emitted_event in self._consume_parts(
                            parts,
                            model=model,
                            assembler=assembler,
                            part_states=part_states,
                        ):
                            yield emitted_event
                            if isinstance(emitted_event, ResponseEndEvent) or emitted_event.type == "error":
                                return
            except ProviderProtocolError as exc:
                yield assembler.error(exc.error)
                return

        for emitted_event in self._finalize_open_parts(assembler=assembler, part_states=part_states):
            yield emitted_event

        final_usage = self.build_usage(last_usage_payload, completeness="final")
        if final_usage is not None:
            yield assembler.set_usage(final_usage)

        if assembler.has_tool_calls() and finish_reason == "stop":
            finish_reason = "tool_call"
        yield assembler.response_end(finish_reason=finish_reason)

    def _build_contents(self, model: ModelSpec, request: GenerateRequest) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for message in request.messages:
            serialized = self._serialize_message(model, message)
            if isinstance(serialized, list):
                contents.extend(serialized)
            else:
                contents.append(serialized)
        return contents

    def _serialize_message(self, model: ModelSpec, message: Message) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(message, UserMessage):
            content = message.content
            if isinstance(content, str):
                parts = [{"text": content}]
            else:
                parts = [self._serialize_user_or_tool_block(block) for block in content]
            return {"role": "user", "parts": parts}

        if isinstance(message, AssistantMessage):
            parts: list[dict[str, Any]] = []
            for block in message.content:
                if block.type == "text":
                    part = {"text": block.text}
                    signature = self._resolve_replay_signature(model, block.protocol_meta)
                    if isinstance(signature, str) and signature:
                        part["thoughtSignature"] = signature
                    parts.append(part)
                elif block.type == "reasoning":
                    signature = self._resolve_replay_signature(model, block.protocol_meta, fallback=block.signature)
                    part = {"text": block.text}
                    if signature is not None:
                        part["thought"] = True
                        part["thoughtSignature"] = signature
                    parts.append(part)
                elif block.type == "tool_call":
                    signature = self._resolve_replay_signature(model, block.protocol_meta)
                    effective_signature = signature or (
                        self.skip_thought_signature if self._is_gemini3_model(model) else None
                    )
                    part = {
                        "functionCall": {
                            "name": block.name,
                            "args": block.arguments or {},
                            "id": self._normalize_tool_call_id(model, block.id),
                        }
                    }
                    if isinstance(effective_signature, str) and effective_signature:
                        part["thoughtSignature"] = effective_signature
                    parts.append(part)
            return {"role": "model", "parts": parts}

        if isinstance(message, ToolResultMessage):
            text_blocks = [block for block in message.content if block.type == "text"]
            image_blocks = [block for block in message.content if isinstance(block, ImageBlock)]
            output_text = "\n".join(block.text for block in text_blocks)
            function_response: dict[str, Any] = {
                "name": message.tool_name,
                "response": {"error" if message.is_error else "output": output_text or ""},
                "id": self._normalize_tool_call_id(model, message.tool_call_id),
            }
            if image_blocks and self._supports_multimodal_function_response(model):
                function_response["parts"] = [self._serialize_user_or_tool_block(block) for block in image_blocks]

            content_items: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "parts": [{"functionResponse": function_response}],
                }
            ]
            if image_blocks and not self._supports_multimodal_function_response(model):
                content_items.append(
                    {
                        "role": "user",
                        "parts": [{"text": "Tool result image:"}] + [
                            self._serialize_user_or_tool_block(block) for block in image_blocks
                        ],
                    }
                )
            return content_items

        raise TypeError(f"Unsupported message type: {type(message)!r}")

    def _serialize_user_or_tool_block(self, block) -> dict[str, Any]:
        if block.type == "text":
            return {"text": block.text}
        if isinstance(block, ImageBlock):
            return {
                "inlineData": {
                    "mimeType": block.mime_type,
                    "data": block.data,
                }
            }
        raise TypeError(f"Unsupported Gemini block type: {block.type!r}")

    def _build_tool_config(self, tool_choice: str | SpecificToolChoice | None) -> dict[str, Any] | None:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, SpecificToolChoice):
            return {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": [tool_choice.name],
                }
            }
        mapping = {
            "auto": "AUTO",
            "none": "NONE",
            "required": "ANY",
        }
        return {"functionCallingConfig": {"mode": mapping.get(tool_choice, "AUTO")}}

    def _convert_tool_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(schema)

    def _build_thinking_config(self, model: ModelSpec, request: GenerateRequest) -> dict[str, Any] | None:
        reasoning = request.reasoning
        if reasoning is None:
            return self._disabled_thinking_config(model)
        if reasoning.enabled is False or reasoning.effort == "none":
            return self._disabled_thinking_config(model)

        effort = reasoning.effort or "medium"
        if self._is_gemini3_model(model):
            return {
                "includeThoughts": True,
                "thinkingLevel": self._thinking_level_for_effort(model, effort),
            }
        return {
            "includeThoughts": True,
            "thinkingBudget": reasoning.max_tokens or self._thinking_budget_for_effort(model, effort),
        }

    def _supports_multimodal_function_response(self, model: ModelSpec) -> bool:
        major_version = self._gemini_major_version(model)
        if major_version is None:
            return True
        return major_version >= 3

    def _consume_parts(
        self,
        parts: list[Any],
        *,
        model: ModelSpec,
        assembler: ResponseAssembler,
        part_states: list[dict[str, Any] | None],
    ) -> list[StreamEvent]:
        emitted: list[StreamEvent] = []
        for part_position, part in enumerate(parts):
            if not isinstance(part, dict):
                continue
            kind = self._part_kind(part)
            if kind is None:
                continue

            while len(part_states) <= part_position:
                part_states.append(None)
            state = part_states[part_position]

            if state is not None and state["kind"] != kind:
                emitted.extend(self._finalize_part_state(state, assembler=assembler))
                state = self._new_part_state(model=model, kind=kind, part=part, part_states=part_states)
                part_states[part_position] = state
                if kind == "text":
                    emitted.append(assembler.text_start(state["content_index"]))
                elif kind == "reasoning":
                    emitted.append(assembler.reasoning_start(state["content_index"]))
                else:
                    emitted.append(
                        assembler.tool_call_start(
                            state["content_index"],
                            tool_call_id=state["tool_call_id"],
                            name=state["tool_name"],
                        )
                    )
            elif state is None:
                emitted.extend(self._finalize_parts_before(part_position, assembler=assembler, part_states=part_states))
                state = self._new_part_state(model=model, kind=kind, part=part, part_states=part_states)
                part_states[part_position] = state
                if kind == "text":
                    emitted.append(assembler.text_start(state["content_index"]))
                elif kind == "reasoning":
                    emitted.append(assembler.reasoning_start(state["content_index"]))
                else:
                    emitted.append(
                        assembler.tool_call_start(
                            state["content_index"],
                            tool_call_id=state["tool_call_id"],
                            name=state["tool_name"],
                        )
                    )

            signature = part.get("thoughtSignature")
            if isinstance(signature, str) and signature:
                assembler.update_block_metadata(
                    state["content_index"],
                    protocol_meta={
                        "gemini_thought_signature": signature,
                        "gemini_provider": self.provider_name,
                        "gemini_model": model.model,
                    },
                )
                state["signature"] = signature

            if kind == "tool_call":
                function_call = part.get("functionCall") or {}
                arguments = function_call.get("args")
                if not isinstance(arguments, dict):
                    arguments = {}
                arguments_json = self._encode_tool_arguments(arguments)
                previous = state["buffer"]
                delta = arguments_json[len(previous) :] if arguments_json.startswith(previous) else arguments_json
                if delta:
                    emitted.append(assembler.tool_call_delta(state["content_index"], delta))
                state["buffer"] = arguments_json
                state["arguments"] = arguments
                continue

            text = str(part.get("text") or "")
            previous_text = state["buffer"]
            delta = text[len(previous_text) :] if text.startswith(previous_text) else text
            if delta:
                if kind == "text":
                    emitted.append(assembler.text_delta(state["content_index"], delta))
                else:
                    emitted.append(assembler.reasoning_delta(state["content_index"], delta))
            state["buffer"] = text

        return emitted

    def _finalize_parts_before(
        self,
        part_position: int,
        *,
        assembler: ResponseAssembler,
        part_states: list[dict[str, Any] | None],
    ) -> list[StreamEvent]:
        emitted: list[StreamEvent] = []
        for index in range(part_position):
            state = part_states[index]
            if state is None or state["finalized"]:
                continue
            emitted.extend(self._finalize_part_state(state, assembler=assembler))
        return emitted

    def _finalize_open_parts(
        self,
        *,
        assembler: ResponseAssembler,
        part_states: list[dict[str, Any] | None],
    ) -> list[StreamEvent]:
        emitted: list[StreamEvent] = []
        for state in part_states:
            if state is None or state["finalized"]:
                continue
            emitted.extend(self._finalize_part_state(state, assembler=assembler))
        return emitted

    def _finalize_part_state(self, state: dict[str, Any], *, assembler: ResponseAssembler) -> list[StreamEvent]:
        state["finalized"] = True
        content_index = state["content_index"]
        if state["kind"] == "text":
            return [assembler.text_end(content_index)]
        if state["kind"] == "reasoning":
            return [assembler.reasoning_end(content_index, signature=state.get("signature"))]
        return [assembler.tool_call_end(content_index, arguments=state.get("arguments") or {})]

    def _new_part_state(
        self,
        *,
        model: ModelSpec,
        kind: str,
        part: dict[str, Any],
        part_states: list[dict[str, Any] | None],
    ) -> dict[str, Any]:
        content_index = max(
            (state["content_index"] for state in part_states if state is not None),
            default=-1,
        ) + 1
        state: dict[str, Any] = {
            "kind": kind,
            "content_index": content_index,
            "buffer": "",
            "signature": None,
            "finalized": False,
        }
        if kind == "tool_call":
            function_call = part.get("functionCall") or {}
            tool_name = str(function_call.get("name") or "tool")
            tool_call_id = function_call.get("id")
            tool_call_id = tool_call_id if isinstance(tool_call_id, str) else None
            tool_call_id = self._normalize_tool_call_id(model, tool_call_id)
            existing_ids = {
                state["tool_call_id"]
                for state in part_states
                if state is not None and state.get("kind") == "tool_call"
            }
            if tool_call_id in existing_ids:
                tool_call_id = f"call_{content_index}"
            state.update(
                {
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "arguments": {},
                }
            )
        return state

    def _part_kind(self, part: dict[str, Any]) -> str | None:
        if isinstance(part.get("functionCall"), dict):
            return "tool_call"
        if "text" not in part:
            return None
        return "reasoning" if part.get("thought") is True else "text"

    def _encode_tool_arguments(self, arguments: dict[str, Any]) -> str:
        return json.dumps(arguments or {}, separators=(",", ":"), sort_keys=True)

    def _iter_stream_chunks(self, payload: Any, *, model: ModelSpec) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            return [payload]
        if isinstance(payload, list):
            chunks: list[dict[str, Any]] = []
            for item in payload:
                if not isinstance(item, dict):
                    raise ProviderProtocolError(
                        make_error_info(
                            code="invalid_gemini_chunk",
                            message="Gemini stream emitted a non-object JSON value",
                            provider=self.provider_name,
                            api_family=model.api_family,
                            raw={"chunk": item},
                        )
                    )
                chunks.append(item)
            return chunks
        raise ProviderProtocolError(
            make_error_info(
                code="invalid_gemini_chunk",
                message="Gemini stream emitted a non-object JSON value",
                provider=self.provider_name,
                api_family=model.api_family,
                raw={"chunk": payload},
            )
        )

    def _gemini_major_version(self, model: ModelSpec) -> int | None:
        model_name = model.model.lower()
        if not model_name.startswith("gemini-"):
            return None
        prefix = model_name.split("-", 2)
        if len(prefix) < 2:
            return None
        try:
            return int(prefix[1].split(".", 1)[0])
        except ValueError:
            return None

    def _is_gemini3_model(self, model: ModelSpec) -> bool:
        return self._gemini_major_version(model) == 3

    def _is_gemini3_pro_model(self, model: ModelSpec) -> bool:
        model_name = model.model.lower()
        return self._is_gemini3_model(model) and "-pro" in model_name

    def _is_gemini3_flash_model(self, model: ModelSpec) -> bool:
        model_name = model.model.lower()
        return self._is_gemini3_model(model) and "-flash" in model_name

    def _disabled_thinking_config(self, model: ModelSpec) -> dict[str, Any]:
        if self._is_gemini3_pro_model(model):
            return {"thinkingLevel": "LOW"}
        if self._is_gemini3_flash_model(model):
            return {"thinkingLevel": "MINIMAL"}
        return {"thinkingBudget": 0}

    def _thinking_level_for_effort(self, model: ModelSpec, effort: str) -> str:
        if self._is_gemini3_pro_model(model):
            if effort in {"minimal", "low"}:
                return "LOW"
            return "HIGH"
        mapping = {
            "minimal": "MINIMAL",
            "low": "LOW",
            "medium": "MEDIUM",
            "high": "HIGH",
            "xhigh": "HIGH",
        }
        return mapping.get(effort, "MEDIUM")

    def _thinking_budget_for_effort(self, model: ModelSpec, effort: str) -> int:
        model_name = model.model.lower()
        if "2.5-pro" in model_name:
            budgets = {
                "minimal": 128,
                "low": 2048,
                "medium": 8192,
                "high": 32768,
                "xhigh": 32768,
            }
            return budgets.get(effort, 8192)
        if "2.5-flash" in model_name:
            budgets = {
                "minimal": 128,
                "low": 2048,
                "medium": 8192,
                "high": 24576,
                "xhigh": 24576,
            }
            return budgets.get(effort, 8192)
        return -1

    def _normalize_tool_call_id(self, model: ModelSpec, tool_call_id: str | None) -> str:
        normalized = (tool_call_id or "").strip() or "call"
        charset = model.capabilities.get("tool_call_id_charset")
        if charset == "alnum_-":
            normalized = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in normalized)
        max_length = model.capabilities.get("tool_call_id_max_length")
        if isinstance(max_length, int) and max_length > 0:
            normalized = normalized[:max_length]
        return normalized or "call"

    def _resolve_replay_signature(
        self,
        model: ModelSpec,
        protocol_meta: Mapping[str, Any],
        *,
        fallback: str | None = None,
    ) -> str | None:
        provider_name = protocol_meta.get("gemini_provider")
        model_name = protocol_meta.get("gemini_model")
        if provider_name != self.provider_name or model_name != model.model:
            return None
        signature = protocol_meta.get("gemini_thought_signature") or fallback
        if not isinstance(signature, str) or not signature:
            return None
        return signature if self._is_valid_thought_signature(signature) else None

    def _is_valid_thought_signature(self, signature: str) -> bool:
        if len(signature) % 4 != 0:
            return False
        try:
            base64.b64decode(signature, validate=True)
        except (binascii.Error, ValueError):
            return False
        return True