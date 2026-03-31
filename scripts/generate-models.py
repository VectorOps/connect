from __future__ import annotations

import json
from pathlib import Path
import urllib.request


SOURCE_URL = "https://models.dev/api.json"

TARGET_PROVIDERS = (
    {
        "provider": "openai",
        "source_provider": "openai",
        "api_family": "openai-responses",
        "base_url": "https://api.openai.com/v1",
    },
    {
        "provider": "chatgpt",
        "source_provider": "openai",
        "api_family": "chatgpt-responses",
        "base_url": "https://chatgpt.com/backend-api",
        "model_filter": lambda model_id, model: "codex" in model_id,
    },
    {
        "provider": "anthropic",
        "source_provider": "anthropic",
        "api_family": "anthropic-messages",
        "base_url": "https://api.anthropic.com/v1",
    },
    {
        "provider": "gemini",
        "source_provider": "google",
        "api_family": "gemini-generate-content",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model_filter": lambda model_id, model: model_id.startswith(("gemini-", "gemma-")),
    },
    {
        "provider": "openrouter",
        "source_provider": "openrouter",
        "api_family": "openai-responses",
        "base_url": "https://openrouter.ai/api/v1",
        "model_filter": lambda model_id, model: True,
    },
)


def _fetch_source_document() -> dict:
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "connect-model-generator/0.1"},
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def _get_source_model(document: dict, source_provider: str, model_id: str) -> dict:
    return document[source_provider]["models"][model_id]


def _iter_source_models(document: dict, source_provider: str):
    return document[source_provider]["models"].items()


def _normalize_pricing(cost: dict | None) -> dict | None:
    if not cost:
        return None

    return {
        "input_per_million": _decimal_or_none(cost.get("input")),
        "output_per_million": _decimal_or_none(cost.get("output")),
        "cache_read_per_million": _decimal_or_none(cost.get("cache_read")),
        "cache_write_per_million": _decimal_or_none(cost.get("cache_write")),
    }


def _decimal_or_none(value: int | float | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _build_record(document: dict, target_provider: str, source_provider: str, model_id: str, api_family: str, base_url: str) -> dict:
    source_model = _get_source_model(document, source_provider, model_id)
    input_modalities = source_model.get("modalities", {}).get("input", [])
    output_modalities = source_model.get("modalities", {}).get("output", [])
    limit = source_model.get("limit", {})

    return {
        "provider": target_provider,
        "model": model_id,
        "api_family": api_family,
        "base_url": base_url,
        "supports_streaming": True,
        "supports_tools": bool(source_model.get("tool_call", False)),
        "supports_reasoning": bool(source_model.get("reasoning", False)),
        "supports_images": "image" in input_modalities,
        "supports_image_outputs": "image" in output_modalities,
        "supports_json_mode": bool(source_model.get("structured_output", False)),
        "supports_prompt_caching": source_model.get("cost", {}).get("cache_read") is not None,
        "context_window": limit.get("context"),
        "max_output_tokens": limit.get("output"),
        "pricing": _normalize_pricing(source_model.get("cost")),
        "capabilities": {
            "supports_developer_role": target_provider in {"openai", "openrouter"},
            "requires_explicit_reasoning_disable": target_provider == "anthropic",
            "usage_final_only": target_provider in {"chatgpt", "anthropic"},
            "supports_parallel_tool_calls": bool(source_model.get("tool_call", False)),
        },
        "protocol_defaults": {},
        "extra": {
            "family": source_model.get("family"),
            "source_provider": source_provider,
            "source_model_id": source_model.get("id", model_id),
            "source_name": source_model.get("name"),
        },
    }


def _supports_text_generation(model: dict) -> bool:
    modalities = model.get("modalities", {})
    output_modalities = modalities.get("output", [])
    if "text" not in output_modalities:
        return False

    family = str(model.get("family") or "")
    if "embedding" in family:
        return False

    return True


def _select_models(document: dict, provider_config: dict) -> list[dict]:
    source_provider = provider_config["source_provider"]
    model_filter = provider_config.get("model_filter")
    selected: list[dict] = []

    for model_id, model in _iter_source_models(document, source_provider):
        if not _supports_text_generation(model):
            continue
        if model_filter is not None and not model_filter(model_id, model):
            continue
        selected.append(
            _build_record(
                document,
                provider_config["provider"],
                source_provider,
                model_id,
                provider_config["api_family"],
                provider_config["base_url"],
            )
        )

    selected.sort(key=lambda item: item["model"])
    return selected


def build_models_document() -> dict:
    source_document = _fetch_source_document()
    models: list[dict] = []
    for provider_config in TARGET_PROVIDERS:
        models.extend(_select_models(source_document, provider_config))
    return {
        "generated_by": "scripts/generate-models.py",
        "version": 1,
        "source": SOURCE_URL,
        "models": models,
    }


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "src" / "connect" / "data" / "models.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_models_document(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()