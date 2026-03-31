from __future__ import annotations

import decimal
import json
from importlib import resources
from typing import Any

import pydantic

from .types import ModelPricing, ModelSpec, request_uses_images, validate_request_for_model


class GeneratedModelRecord(pydantic.BaseModel):
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
    pricing: dict[str, str | None] | None = None
    capabilities: dict[str, Any] = pydantic.Field(default_factory=dict)
    protocol_defaults: dict[str, Any] = pydantic.Field(default_factory=dict)
    extra: dict[str, Any] = pydantic.Field(default_factory=dict)


class GeneratedModelsDocument(pydantic.BaseModel):
    generated_by: str
    version: int
    source: str | None = None
    models: list[GeneratedModelRecord]


def _load_models_document() -> GeneratedModelsDocument:
    with resources.files("connect").joinpath("data/models.json").open("r", encoding="utf-8") as handle:
        return GeneratedModelsDocument.model_validate(json.load(handle))


def _coerce_pricing(pricing: dict | None) -> ModelPricing | None:
    if not pricing:
        return None

    values = {
        key: decimal.Decimal(value)
        for key, value in pricing.items()
        if value is not None
    }
    return ModelPricing(**values)


def load_builtin_models() -> tuple[ModelSpec, ...]:
    document = _load_models_document()
    models: list[ModelSpec] = []

    for item in document.models:
        payload = item.model_dump()
        payload["pricing"] = _coerce_pricing(payload.get("pricing"))
        models.append(ModelSpec(**payload))

    return tuple(models)


BUILTIN_MODELS = load_builtin_models()