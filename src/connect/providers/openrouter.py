from __future__ import annotations

from typing import Any

from ..types import GenerateRequest, ModelSpec, RequestOptions
from .openai import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    provider_name = "openrouter"
    default_base_url = "https://openrouter.ai/api/v1"

    def build_headers(
        self,
        model: ModelSpec,
        request: GenerateRequest,
        options: RequestOptions,
    ) -> dict[str, str]:
        headers = super().build_headers(model, request, options)
        if "referer" in options.provider_options:
            headers["http-referer"] = str(options.provider_options["referer"])
        if "title" in options.provider_options:
            headers["x-title"] = str(options.provider_options["title"])
        return headers

    def build_payload(
        self,
        model: ModelSpec,
        request: GenerateRequest,
        options: RequestOptions,
    ) -> dict[str, Any]:
        payload = super().build_payload(model, request, options)
        if "provider" in options.provider_options:
            payload["provider"] = options.provider_options["provider"]
        return payload