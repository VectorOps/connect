from __future__ import annotations

import decimal

from connect.types import ModelPricing, ModelSpec, Usage
from connect.usage import ConversationUsageTracker, accumulate_usage, estimate_cost


def test_accumulate_usage_combines_token_fields() -> None:
    left = Usage(
        input_tokens=1,
        output_tokens=2,
        reasoning_tokens=3,
        cache_read_tokens=4,
        cache_write_tokens=5,
        total_tokens=6,
        completeness="final",
    )
    right = Usage(
        input_tokens=10,
        output_tokens=20,
        reasoning_tokens=30,
        cache_read_tokens=40,
        cache_write_tokens=50,
        total_tokens=60,
        completeness="final",
    )

    combined = accumulate_usage(left, right)

    assert combined == Usage(
        input_tokens=11,
        output_tokens=22,
        reasoning_tokens=33,
        cache_read_tokens=44,
        cache_write_tokens=55,
        total_tokens=66,
        completeness="final",
    )


def test_accumulate_usage_marks_partial_when_completeness_is_mixed() -> None:
    combined = accumulate_usage(
        Usage(total_tokens=1, completeness="final"),
        Usage(total_tokens=2, completeness="none"),
    )

    assert combined.total_tokens == 3
    assert combined.completeness == "partial"


def test_estimate_cost_uses_centralized_cost_breakdown() -> None:
    model = ModelSpec(
        provider="openai",
        model="gpt-test",
        api_family="openai-responses",
        pricing=ModelPricing(
            input_per_million=decimal.Decimal("1.0"),
            output_per_million=decimal.Decimal("2.0"),
            cache_read_per_million=decimal.Decimal("0.5"),
            cache_write_per_million=decimal.Decimal("4.0"),
        ),
    )
    usage = Usage(
        input_tokens=1_000,
        output_tokens=500,
        cache_read_tokens=200,
        cache_write_tokens=100,
        total_tokens=1_800,
        completeness="partial",
    )

    cost = estimate_cost(model, usage)

    assert cost is not None
    assert cost.input_cost == decimal.Decimal("0.001")
    assert cost.output_cost == decimal.Decimal("0.001")
    assert cost.cache_read_cost == decimal.Decimal("0.0001")
    assert cost.cache_write_cost == decimal.Decimal("0.0004")
    assert cost.total_cost == decimal.Decimal("0.0025")


def test_conversation_usage_tracker_accumulates_and_resets() -> None:
    tracker = ConversationUsageTracker()

    tracker.add(Usage(input_tokens=2, total_tokens=2, completeness="partial"))
    tracker.add(Usage(output_tokens=3, total_tokens=3, completeness="final"))

    assert tracker.usage == Usage(input_tokens=2, output_tokens=3, total_tokens=5, completeness="partial")
    assert tracker.reset() == Usage()