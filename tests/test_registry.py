from __future__ import annotations

import decimal

import pytest

from connect.models import BUILTIN_MODELS, GeneratedModelsDocument, ModelPricing, calculate_usage_cost
from connect.registry import AmbiguousModelError, ModelNotFoundError, ModelRegistry, get_model, list_models
from connect.types import ModelSpec, Usage


def test_builtin_registry_exposes_required_providers() -> None:
    providers = {model.provider for model in list_models()}

    assert {"openai", "chatgpt", "anthropic", "gemini", "openrouter"}.issubset(providers)


def test_builtin_registry_loads_multiple_models_per_vendor() -> None:
    all_models = list_models()

    assert len([model for model in all_models if model.provider == "openai"]) > 1
    assert len([model for model in all_models if model.provider == "anthropic"]) > 1
    assert len([model for model in all_models if model.provider == "gemini"]) > 1
    assert len([model for model in all_models if model.provider == "openrouter"]) > 1


def test_chatgpt_registry_only_contains_codex_family_models() -> None:
    chatgpt_models = list_models("chatgpt")

    assert chatgpt_models
    assert all("codex" in model.model for model in chatgpt_models)


def test_get_model_returns_registered_builtin_model() -> None:
    model = get_model("openai", "gpt-4.1-mini")

    assert model.provider == "openai"
    assert model.api_family == "openai-responses"


def test_model_registry_resolves_unambiguous_bare_model() -> None:
    registry = ModelRegistry(BUILTIN_MODELS)

    model = registry.resolve("claude-3-7-sonnet-latest")

    assert model.provider == "anthropic"


def test_model_registry_resolves_provider_model_string_with_slash_separator() -> None:
    registry = ModelRegistry(BUILTIN_MODELS)

    model = registry.resolve("openai/gpt-4.1-mini")

    assert model.provider == "openai"
    assert model.model == "gpt-4.1-mini"


def test_model_registry_rejects_ambiguous_bare_model() -> None:
    model = ModelSpec(provider="test-provider", model="shared-model", api_family="test")
    other_model = ModelSpec(provider="other-provider", model="shared-model", api_family="test")
    registry = ModelRegistry([model, other_model])

    with pytest.raises(AmbiguousModelError):
        registry.resolve("shared-model")


def test_model_registry_reports_available_models_for_unknown_provider_model() -> None:
    registry = ModelRegistry(BUILTIN_MODELS)

    with pytest.raises(ModelNotFoundError) as exc:
        registry.get("openai", "missing-model")

    assert "gpt-4.1-mini" in str(exc.value)


def test_model_registry_supports_runtime_overrides() -> None:
    registry = ModelRegistry(BUILTIN_MODELS)
    overridden = registry.with_overrides(
        pricing={
            ("openai", "gpt-4.1-mini"): ModelPricing(input_per_million=decimal.Decimal("9.99"))
        },
        base_urls={("openai", "gpt-4.1-mini"): "https://example.invalid/v1"},
        capabilities={("openai", "gpt-4.1-mini"): {"supports_developer_role": False}},
    )

    model = overridden.get("openai", "gpt-4.1-mini")

    assert model.pricing is not None
    assert model.pricing.input_per_million == decimal.Decimal("9.99")
    assert model.base_url == "https://example.invalid/v1"
    assert model.capabilities["supports_developer_role"] is False


def test_generated_models_document_schema_matches_checked_in_data() -> None:
    document = GeneratedModelsDocument.model_validate(
        {
            "generated_by": "scripts/generate-models.py",
            "version": 1,
            "source": "https://models.dev/api.json",
            "models": [
                {
                    "provider": "openai",
                    "model": "gpt-4.1-mini",
                    "api_family": "openai-responses",
                    "pricing": {
                        "input_per_million": "0.4",
                        "output_per_million": "1.6",
                        "cache_read_per_million": "0.1",
                        "cache_write_per_million": None,
                    },
                }
            ],
        }
    )

    assert document.models[0].provider == "openai"


def test_calculate_usage_cost_returns_generic_cost_breakdown() -> None:
    model = ModelSpec(
        provider="openai",
        model="gpt-4.1-mini",
        api_family="openai-responses",
        pricing=ModelPricing(
            input_per_million=decimal.Decimal("0.40"),
            output_per_million=decimal.Decimal("1.60"),
            cache_read_per_million=decimal.Decimal("0.10"),
            cache_write_per_million=decimal.Decimal("0.80"),
        ),
    )
    usage = Usage(
        input_tokens=2_000,
        output_tokens=500,
        cache_read_tokens=1_000,
        cache_write_tokens=250,
        total_tokens=3_750,
        completeness="final",
    )

    cost = calculate_usage_cost(model, usage)

    assert cost == {
        "input_cost": decimal.Decimal("0.0008"),
        "output_cost": decimal.Decimal("0.0008"),
        "cache_read_cost": decimal.Decimal("0.0001"),
        "cache_write_cost": decimal.Decimal("0.0002"),
        "total_cost": decimal.Decimal("0.0019"),
    }


def test_calculate_usage_cost_returns_none_without_model_pricing() -> None:
    model = ModelSpec(provider="openai", model="gpt-4.1-mini", api_family="openai-responses")

    assert calculate_usage_cost(model, Usage()) is None