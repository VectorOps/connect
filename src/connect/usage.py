from __future__ import annotations

from collections.abc import Mapping

from .models import calculate_usage_cost
from .types import CostBreakdown, ModelSpec, Usage


def accumulate_usage(existing: Usage | None, new: Usage | None) -> Usage:
    if existing is None and new is None:
        return Usage()
    if existing is None:
        return new.model_copy() if new is not None else Usage()
    if new is None:
        return existing.model_copy()

    return Usage(
        input_tokens=existing.input_tokens + new.input_tokens,
        output_tokens=existing.output_tokens + new.output_tokens,
        reasoning_tokens=existing.reasoning_tokens + new.reasoning_tokens,
        cache_read_tokens=existing.cache_read_tokens + new.cache_read_tokens,
        cache_write_tokens=existing.cache_write_tokens + new.cache_write_tokens,
        total_tokens=existing.total_tokens + new.total_tokens,
        completeness=_merge_completeness(existing.completeness, new.completeness),
    )


def estimate_cost(model: ModelSpec, usage: Usage) -> CostBreakdown | None:
    values = calculate_usage_cost(model, usage)
    if values is None:
        return None
    return CostBreakdown(**values)


class ConversationUsageTracker:
    def __init__(self) -> None:
        self._usage = Usage()

    @property
    def usage(self) -> Usage:
        return self._usage.model_copy()

    def add(self, usage: Usage | None) -> Usage:
        self._usage = accumulate_usage(self._usage, usage)
        return self.usage

    def add_response(self, response) -> Usage:
        return self.add(getattr(response, "usage", None))

    def reset(self) -> Usage:
        self._usage = Usage()
        return self.usage

    def estimate_cost(self, model: ModelSpec) -> CostBreakdown | None:
        return estimate_cost(model, self._usage)


def _merge_completeness(left: str, right: str) -> str:
    values = {left, right}
    if "partial" in values:
        return "partial"
    if values == {"final"}:
        return "final"
    if "final" in values and "none" in values:
        return "partial"
    return "none"