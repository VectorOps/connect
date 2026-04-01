VectorOps Connect
=================

VectorOps Connect is a pure-Python, async-first client for working with multiple LLM providers through one consistent API. It lets applications use OpenAI, ChatGPT, Anthropic, Gemini, and OpenRouter without depending on vendor SDKs, while preserving important provider-specific behavior such as streaming, tool calling, multimodal inputs, replay metadata, and authentication differences.

The project provides:

- typed public request and response models
- shared HTTP and streaming transports built on `aiohttp`
- one provider adapter per backend
- pluggable authentication
- registry-driven model metadata
- normalized streaming events and usage tracking

Supported providers
-------------------

- OpenAI
- ChatGPT
- Anthropic
- Gemini
- OpenRouter

Installation
------------

This repository uses `uv` for development and execution.

```bash
uv sync
```

For local development and tests:

```bash
uv run pytest
```

Quick start
-----------

The main entry point is `AsyncLLMClient`. You can call `generate()` for a complete response or `stream()` for incremental events.

If you do not pass `options.auth`, the client automatically falls back to environment-based credential resolution through its default auth router. That means provider credentials from environment variables or configured credential files can be picked up without passing an explicit auth object on every call.

Simplified example
------------------

```python
import asyncio

from connect import AsyncLLMClient, GenerateRequest, RequestOptions, UserMessage
from connect.auth_env import resolve_env_auth


async def main() -> None:
    auth = resolve_env_auth("openai")
    if auth is None:
        raise RuntimeError("OPENAI_API_KEY is not set")

    async with AsyncLLMClient() as client:
        response = await client.generate(
            "openai/gpt-4.1-mini",
            GenerateRequest(
                messages=[UserMessage(content="Explain what VectorOps Connect does in one sentence.")],
                max_output_tokens=64,
            ),
            options=RequestOptions(auth=auth),
        )

    for block in response.content:
        if block.type == "text":
            print(block.text)


asyncio.run(main())
```

Streaming example
-----------------

```python
import asyncio

from connect import AsyncLLMClient, GenerateRequest, RequestOptions, UserMessage
from connect.auth_env import resolve_env_auth


async def main() -> None:
    auth = resolve_env_auth("anthropic")
    if auth is None:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    async with AsyncLLMClient() as client:
        stream = client.stream(
            "anthropic/claude-3-7-sonnet-latest",
            GenerateRequest(messages=[UserMessage(content="Reply with exactly the word: streamed")]),
            options=RequestOptions(auth=auth),
        )

        async for event in stream:
            if event.type == "text_delta":
                print(event.delta, end="", flush=True)

        response = await stream.final_response()
        print("\nfinish_reason:", response.finish_reason)


asyncio.run(main())
```

Core concepts
-------------

Unified request model
~~~~~~~~~~~~~~~~~~~~~

Requests are provider-agnostic and use typed message/content models. A request can include:

- plain text chat turns
- reasoning configuration
- tool definitions and tool results
- image inputs in user messages and tool results
- structured response format hints

Unified response model
~~~~~~~~~~~~~~~~~~~~~~

Responses normalize provider output into:

- assistant content blocks
- finish reason
- usage information
- response and request identifiers when available
- provider-specific replay metadata for same-provider continuation

Streaming-first design
~~~~~~~~~~~~~~~~~~~~~~

Streaming is the canonical execution path. `generate()` internally consumes `stream()` and returns the final assembled `AssistantResponse`.

Authentication management
-------------------------

Authentication is explicit, request-scoped, and independent from provider adapters.

Token-based auth
~~~~~~~~~~~~~~~~

The simplest pattern is to resolve credentials from environment variables:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `OPENROUTER_API_KEY`
- `CHATGPT_ACCESS_TOKEN`

Example:

```python
from connect import AsyncLLMClient, GenerateRequest, RequestOptions, UserMessage
from connect.auth_env import resolve_env_auth


auth = resolve_env_auth("gemini")

async with AsyncLLMClient() as client:
    response = await client.generate(
        "gemini/gemini-2.5-flash",
        GenerateRequest(messages=[UserMessage(content="Say hello")]),
        options=RequestOptions(auth=auth),
    )
```

If you omit `options.auth`, the client will try default environment-based auth resolution automatically. For example, this works when `OPENAI_API_KEY` is set:

```python
from connect import AsyncLLMClient, GenerateRequest, UserMessage


async with AsyncLLMClient() as client:
    response = await client.generate(
        "openai/gpt-4.1-mini",
        GenerateRequest(messages=[UserMessage(content="Say hello")]),
    )
```

You can also supply auth objects directly when you do not want environment-based resolution:

```python
from connect import AsyncLLMClient, BearerTokenAuth, GenerateRequest, RequestOptions, UserMessage


auth = BearerTokenAuth("your-openai-token")

async with AsyncLLMClient() as client:
    response = await client.generate(
        "openai/gpt-4.1-mini",
        GenerateRequest(messages=[UserMessage(content="Say hello")]),
        options=RequestOptions(auth=auth),
    )
```

Interactive and persisted OAuth flows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The repository includes credential-management helpers for OAuth-backed providers. Today, the interactive login path is implemented for ChatGPT credential management.

Typical flow:

1. create a `CredentialManager`
2. run provider login callbacks
3. persist credentials to disk
4. create transport auth from the saved credentials

Short example:

```python
import asyncio
from pathlib import Path

from connect.credentials import FileCredentialManager, build_console_login_callbacks


async def main() -> None:
    manager = FileCredentialManager(Path(".secrets/chatgpt-credentials.json"))

    credentials = await manager.login(
        "chatgpt",
        build_console_login_callbacks(provider="chatgpt"),
    )

    print("stored account id:", credentials.account_id)

    resolved = await manager.resolve("chatgpt")
    print("auth headers:", sorted(resolved.headers))


asyncio.run(main())
```

If you already have a saved ChatGPT credentials file, the environment helper can use it:

```bash
export CHATGPT_CREDENTIALS_FILE=.secrets/chatgpt-credentials.json
```

```python
from connect.auth_env import resolve_env_auth


auth = resolve_env_auth("chatgpt")
```

Note:

- ChatGPT interactive OAuth support is available through the credential helpers in this repository.
- Anthropic interactive OAuth credential management is planned for later.

Tools and multimodal inputs
---------------------------

VectorOps Connect supports tool calling and multimodal turns through the shared request model.

Example tool definition:

```python
from connect import GenerateRequest, SpecificToolChoice, ToolSpec, UserMessage


request = GenerateRequest(
    messages=[UserMessage(content="Use the lookup_status tool for id 'alpha'.")],
    tools=[
        ToolSpec(
            name="lookup_status",
            description="Return a status for a known identifier.",
            input_schema={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
                "additionalProperties": False,
            },
        )
    ],
    tool_choice=SpecificToolChoice(name="lookup_status"),
)
```

Image input example:

```python
from connect import GenerateRequest, ImageBlock, TextBlock, UserMessage


request = GenerateRequest(
    messages=[
        UserMessage(
            content=[
                TextBlock(text="What is in this image?"),
                ImageBlock(data=base64_png, mime_type="image/png"),
            ]
        )
    ]
)
```

Observability
-------------

The client supports a lightweight event hook for request lifecycle events.

Emitted events include:

- `request_start`
- `response_headers`
- `first_token`
- `request_end`
- `error`

Example:

```python
async def on_event(event: dict) -> None:
    print(event["type"], event)


client = AsyncLLMClient(event_hook=on_event)
```

Usage helpers
-------------

Centralized usage helpers are available in `connect.usage`.

```python
from connect import ConversationUsageTracker, estimate_cost


tracker = ConversationUsageTracker()
tracker.add_response(response)

cost = estimate_cost(model_spec, tracker.usage)
```

Development
-----------

Useful commands:

```bash
uv sync
uv run pytest
```

Live integration tests are gated behind `INTEGRATION_TEST=1` and the relevant provider credentials.
