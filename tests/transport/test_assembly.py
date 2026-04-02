from __future__ import annotations

import json

import pytest

from connect.transport.assembly import ResponseAssembler
from connect.types import ErrorInfo, Usage


def test_response_assembler_builds_complete_response() -> None:
    assembler = ResponseAssembler(
        provider="openai",
        model="gpt-test",
        api_family="openai-responses",
        response_id="resp_1",
        request_id="req_1",
    )

    start = assembler.response_start()
    assert start.response_id == "resp_1"

    assembler.text_start(0)
    assembler.text_delta(0, "hello")
    assembler.text_delta(0, " world")
    assert assembler.text_end(0).text == "hello world"

    assembler.reasoning_start(1)
    assembler.reasoning_delta(1, "thinking")
    reasoning_end = assembler.reasoning_end(1, signature="sig_1", redacted=True)
    assert reasoning_end.signature == "sig_1"
    assert reasoning_end.redacted is True

    assembler.tool_call_start(2, tool_call_id="call_1", name="search")
    assembler.tool_call_delta(2, '{"query":"docs"}')
    tool_end = assembler.tool_call_end(2, arguments=json.loads(assembler.take_tool_call_buffer(2)))
    assert tool_end.tool_call.arguments == {"query": "docs"}

    usage_event = assembler.set_usage(Usage(input_tokens=1, output_tokens=2, total_tokens=3, completeness="final"))
    assert usage_event.usage.total_tokens == 3

    assembler.update_protocol_state(previous_response_id="resp_0")
    assembler.update_provider_meta(account="acct_1")
    response = assembler.response_end(finish_reason="stop").response

    assert response.finish_reason == "stop"
    assert [block.type for block in response.content] == ["text", "reasoning", "tool_call"]
    assert response.protocol_state == {"previous_response_id": "resp_0"}
    assert response.provider_meta == {"account": "acct_1"}


def test_response_assembler_exposes_raw_tool_argument_buffer() -> None:
    assembler = ResponseAssembler(
        provider="openai",
        model="gpt-test",
        api_family="openai-responses",
    )

    assembler.tool_call_start(0, tool_call_id="call_1", name="search")
    assembler.tool_call_delta(0, '{"query":')

    assert assembler.take_tool_call_buffer(0) == '{"query":'
    assert assembler.tool_call_end(0).tool_call.arguments == {}


def test_response_assembler_returns_partial_response_on_error() -> None:
    assembler = ResponseAssembler(
        provider="anthropic",
        model="claude-test",
        api_family="anthropic-messages",
    )
    assembler.text_start(0)
    assembler.text_delta(0, "partial")

    error_event = assembler.error(
        ErrorInfo(
            code="stream_error",
            message="stream failed",
            provider="anthropic",
            api_family="anthropic-messages",
        )
    )

    assert error_event.partial_response is not None
    assert error_event.partial_response.content[0].text == "partial"