from __future__ import annotations

import platform
from typing import Any

from ..auth import ChatGPTAccessTokenAuth, extract_chatgpt_account_id
from ..types import GenerateRequest, ModelSpec, RequestOptions
from .openai import OpenAIProvider


class ChatGPTProvider(OpenAIProvider):
    provider_name = "chatgpt"
    api_family = "chatgpt-responses"
    default_base_url = "https://chatgpt.com/backend-api"
    stream_path = "/codex/responses"
    excluded_payload_parameters = {
        "temperature",
        "store",
    }

    def _default_user_agent(self) -> str:
        return f"pi ({platform.system().lower()} {platform.release()}; {platform.machine().lower()})"

    def build_headers(
        self,
        model: ModelSpec,
        request: GenerateRequest,
        options: RequestOptions,
    ) -> dict[str, str]:
        headers = super().build_headers(model, request, options)
        headers["OpenAI-Beta"] = "responses=experimental"
        headers.setdefault(
            "originator", str(options.provider_options.get("originator", "connect"))
        )
        headers.setdefault("User-Agent", self._default_user_agent())

        session_id = self._session_id(request, options)
        if session_id is not None:
            headers["session_id"] = session_id

        account_id = self._resolve_account_id(options)
        if account_id is not None:
            headers["chatgpt-account-id"] = account_id

        return headers

    def build_payload(
        self,
        model: ModelSpec,
        request: GenerateRequest,
        options: RequestOptions,
    ) -> dict[str, Any]:
        payload_model = model.model_copy(
            update={
                "capabilities": {**model.capabilities, "supports_developer_role": True},
            }
        )
        payload = super().build_payload(payload_model, request, options)

        for parameter_name in self.excluded_payload_parameters:
            payload.pop(parameter_name, None)

        if request.system_prompt:
            payload["instructions"] = request.system_prompt
        else:
            payload.setdefault("instructions", "")

        payload.setdefault("store", False)

        session_id = self._session_id(request, options)
        if session_id is not None and "prompt_cache_key" not in payload:
            payload["prompt_cache_key"] = session_id

        if request.tools:
            payload.setdefault("tool_choice", "auto")
            payload.setdefault("parallel_tool_calls", True)

        text_verbosity = options.provider_options.get("text_verbosity")
        if isinstance(text_verbosity, str) and text_verbosity:
            text_payload = payload.get("text")
            if not isinstance(text_payload, dict):
                text_payload = {}
                payload["text"] = text_payload
            text_payload["verbosity"] = text_verbosity

        reasoning_summary = options.provider_options.get("reasoning_summary")
        if isinstance(reasoning_summary, str) and reasoning_summary:
            reasoning_payload = payload.get("reasoning")
            if not isinstance(reasoning_payload, dict):
                reasoning_payload = {}
                payload["reasoning"] = reasoning_payload
            reasoning_payload.setdefault("summary", reasoning_summary)

        return payload

    def _session_id(
        self, request: GenerateRequest, options: RequestOptions
    ) -> str | None:
        session_id = options.provider_options.get("session_id")
        if isinstance(session_id, str) and session_id:
            return session_id
        if (
            request.session
            and isinstance(request.session.session_id, str)
            and request.session.session_id
        ):
            return request.session.session_id
        return None

    def _resolve_account_id(self, options: RequestOptions) -> str | None:
        existing = options.headers.get("chatgpt-account-id") or options.headers.get(
            "ChatGPT-Account-Id"
        )
        if isinstance(existing, str) and existing:
            return existing

        auth = options.auth
        if isinstance(auth, ChatGPTAccessTokenAuth) and auth.account_id:
            return auth.account_id

        authorization = options.headers.get("Authorization") or options.headers.get(
            "authorization"
        )
        if not isinstance(authorization, str):
            return None
        prefix = "Bearer "
        if not authorization.startswith(prefix):
            return None
        return extract_chatgpt_account_id(authorization[len(prefix) :])
