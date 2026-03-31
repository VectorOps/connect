from __future__ import annotations

from collections import defaultdict

from .models import BUILTIN_MODELS
from .types import ModelSpec


class RegistryError(ValueError):
    pass


class ModelNotFoundError(RegistryError):
    pass


class AmbiguousModelError(RegistryError):
    pass


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, object] = {}

    def register(self, name: str, provider: object) -> None:
        self._providers[name] = provider

    def get(self, name: str) -> object:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise ModelNotFoundError(f"Provider '{name}' is not registered") from exc

    def list(self) -> list[str]:
        return sorted(self._providers)


class ModelRegistry:
    def __init__(self, models: list[ModelSpec] | tuple[ModelSpec, ...] | None = None) -> None:
        self._models_by_provider: dict[str, dict[str, ModelSpec]] = defaultdict(dict)
        if models:
            self.register_many(models)

    def register(self, model: ModelSpec) -> None:
        self._models_by_provider[model.provider][model.model] = model

    def register_many(self, models: list[ModelSpec] | tuple[ModelSpec, ...]) -> None:
        for model in models:
            self.register(model)

    def get(self, provider: str, model: str) -> ModelSpec:
        try:
            return self._models_by_provider[provider][model]
        except KeyError as exc:
            available = sorted(self._models_by_provider.get(provider, {}))
            if available:
                raise ModelNotFoundError(
                    f"Model '{provider}/{model}' is not registered. Available models: {', '.join(available)}"
                ) from exc
            raise ModelNotFoundError(f"Provider '{provider}' has no registered models") from exc

    def resolve(self, model: str, provider: str | None = None) -> ModelSpec:
        if provider is not None:
            return self.get(provider, model)

        if "/" in model:
            explicit_provider, explicit_model = model.split("/", 1)
            return self.get(explicit_provider, explicit_model)

        matches = [
            registered
            for provider_models in self._models_by_provider.values()
            for registered_model, registered in provider_models.items()
            if registered_model == model
        ]

        if not matches:
            raise ModelNotFoundError(f"Model '{model}' is not registered")

        if len(matches) > 1:
            candidates = ", ".join(sorted(f"{candidate.provider}/{candidate.model}" for candidate in matches))
            raise AmbiguousModelError(f"Model '{model}' is ambiguous. Candidates: {candidates}")

        return matches[0]

    def list_models(self, provider: str | None = None) -> list[ModelSpec]:
        if provider is not None:
            return sorted(self._models_by_provider.get(provider, {}).values(), key=lambda item: item.model)

        return sorted(
            (model for provider_models in self._models_by_provider.values() for model in provider_models.values()),
            key=lambda item: (item.provider, item.model),
        )

    def providers(self) -> list[str]:
        return sorted(self._models_by_provider)

    def with_overrides(
        self,
        *,
        pricing: dict[tuple[str, str], object] | None = None,
        base_urls: dict[tuple[str, str], str] | None = None,
        capabilities: dict[tuple[str, str], dict] | None = None,
    ) -> ModelRegistry:
        overridden = ModelRegistry()
        for model in self.list_models():
            key = (model.provider, model.model)
            update: dict = {}
            if pricing and key in pricing:
                update["pricing"] = pricing[key]
            if base_urls and key in base_urls:
                update["base_url"] = base_urls[key]
            if capabilities and key in capabilities:
                update["capabilities"] = {**model.capabilities, **capabilities[key]}
            overridden.register(model.model_copy(update=update) if update else model)
        return overridden


default_model_registry = ModelRegistry(BUILTIN_MODELS)


def get_model(provider: str, model: str) -> ModelSpec:
    return default_model_registry.get(provider, model)


def list_models(provider: str | None = None) -> list[ModelSpec]:
    return default_model_registry.list_models(provider)


def build_default_provider_registry() -> ProviderRegistry:
    from .providers import ChatGPTProvider, OpenAIProvider, OpenRouterProvider

    registry = ProviderRegistry()
    registry.register("chatgpt", ChatGPTProvider())
    registry.register("openai", OpenAIProvider())
    registry.register("openrouter", OpenRouterProvider())
    return registry


default_provider_registry = build_default_provider_registry()
