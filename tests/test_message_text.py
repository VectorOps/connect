from __future__ import annotations

from connect import AssistantMessage, ReasoningBlock, TextBlock, ToolCallBlock, assistant_message_text


def test_assistant_message_text_concatenates_fragmented_text_blocks() -> None:
    message = AssistantMessage(
        content=[
            TextBlock(text="Qu"),
            TextBlock(text="icksort is a highly efficient, \"divide and conquer\" sorting algorithm. Developed"),
            TextBlock(text=" by Tony Ho"),
        ]
    )

    assert assistant_message_text(message) == (
        'Quicksort is a highly efficient, "divide and conquer" sorting algorithm. Developed by Tony Ho'
    )


def test_assistant_message_text_ignores_non_text_blocks_by_default() -> None:
    message = AssistantMessage(
        content=[
            ReasoningBlock(text="Need a tool"),
            TextBlock(text="Final answer."),
            ToolCallBlock(id="call_1", name="lookup", arguments={"id": "alpha"}),
        ]
    )

    assert assistant_message_text(message) == "Final answer."


def test_assistant_message_text_can_include_reasoning_with_separator() -> None:
    message = AssistantMessage(
        content=[
            ReasoningBlock(text="Need a tool"),
            TextBlock(text="Final answer."),
        ]
    )

    assert assistant_message_text(message, include_reasoning=True, separator="\n\n") == (
        "Need a tool\n\nFinal answer."
    )