# Unified Async LLM Client Design

## Purpose

This document specifies a Python library that provides a single async interface over major LLM vendors while remaining vendor-SDK-free. The initial implementation target is:

- OpenAI API
- OpenAI Codex via ChatGPT subscription
- Anthropic API
- Anthropic subscription access
- Gemini API
- OpenRouter

The library must be:

- pure Python
- async-first
- implemented without vendor SDKs
- based on standard HTTP tooling with streaming support
- extensible to additional providers and auth mechanisms

The document is intended to be an implementation specification, not a tutorial.

## Goals

- Provide one developer-facing API for text generation, reasoning, tool calling, multimodal input, and streaming.
- Hide provider-specific request and stream formats behind a stable internal model.
- Support multiple authentication strategies without coupling auth logic to any single provider.
- Expose normalized usage and cost accounting.
- Preserve enough provider-specific metadata to support replay and multi-turn continuity within the same provider session.
- Allow provider-specific escape hatches without contaminating the common API.

## Non-goals

- Do not implement prompt orchestration, agent loops, memory stores, retries across providers, or tool execution frameworks.
- Do not depend on vendor SDKs.
- Do not make synchronous APIs the primary interface.
- Do not normalize away all provider differences. Some differences must remain explicit through capability flags and provider options.

## Primary design principles

1. Common path first, provider escape hatch second.
2. Streaming is the core API; non-streaming is derived from it.
3. The transport layer is separate from provider mapping logic.
4. Authentication is pluggable and request-scoped.
5. Message and event models are provider-agnostic, but retain opaque provider metadata needed for same-provider replay.
6. A chat session is provider-bound. Switching providers starts a new session.
7. All code changes should add or update automated tests where there is a reasonable place to cover the behavior.
8. Development commands should be run through `uv`.

## Implementation stack

Use only small general-purpose Python libraries.

- `aiohttp` for async HTTP, connection pooling, timeouts, proxies, and streaming
- `websockets` only when a provider transport requires persistent WebSocket connections
- `pydantic` for public data models, validation, and serialization
- Python standard library for JSON, typing, base64, time, hmac/hashlib, and URL handling

Do not use provider SDKs.

The baseline runtime dependency set is `aiohttp` and `pydantic`. If OpenAI WebSocket mode is implemented, `websockets` must be an optional extra rather than a mandatory dependency for all users. SSE parsing must be implemented internally over streamed response line iteration. Do not add a dedicated SSE dependency.

## High-level package structure

Required package layout:

```text
src/connect/
    __init__.py
    client.py
    auth.py
    types.py
    models.py
    usage.py
    exceptions.py
    transport/
        __init__.py
        http.py
        sse.py
        json_stream.py
    providers/
        __init__.py
        base.py
        openai.py
        chatgpt.py
        anthropic.py
        gemini.py
        openrouter.py
    registry.py
```

Responsibilities:

- `client.py`: public async client and convenience entry points
- `auth.py`: auth abstractions and built-in auth strategies
- `types.py`: unified request, response, message, content, and stream event types
- `models.py`: model metadata and cost tables
- `usage.py`: usage aggregation and cost computation
- `transport/*`: shared streaming and HTTP helpers
- `providers/*`: vendor-specific request/response mappers
- `registry.py`: provider registration and capability lookup

The first implementation must include concrete provider modules for:

- `openai.py`
- `chatgpt.py`
- `anthropic.py`
- `gemini.py`
- `openrouter.py`

## Core concepts

### 1. Provider

A provider converts unified inputs into vendor-specific HTTP requests and converts streamed vendor responses back into normalized events.

Provider interface:

```python
class Provider(Protocol):
    name: str

    async def stream(
        self,
        *,
        model: ModelSpec,
        request: GenerateRequest,
        options: RequestOptions,
        http: HttpTransport,
    ) -> AsyncIterator[StreamEvent]:
        ...
```

Each provider owns:

- endpoint selection
- request serialization
- header construction
- auth application rules
- stream decoding
- usage extraction
- stop reason mapping
- provider-specific metadata retention

Provider contract requirements:

- provider code must accept only already-validated public models plus request options
- provider code must not mutate caller-owned request objects in place
- provider code must emit a terminal `response_end` or `error` event exactly once
- provider code must return enough provider metadata in the final response to support same-provider continuation
- provider code may retain internal assembly state, but must strip internal-only fields before returning public models

### Provider confidence requirements

Every provider adapter must satisfy the same minimum confidence bar before it is considered complete enough for general use.

Required standards:

- request mapping is deterministic and driven by model capabilities, protocol defaults, and explicit provider options
- unsupported common features fail early with clear validation errors rather than being silently dropped
- provider-specific optional fields are either explicitly supported, explicitly ignored by design, or rejected with a structured error
- stream handling covers all response item types and terminal conditions used by the target provider contract
- malformed provider payloads produce `ProviderProtocolError` or a normalized `error` event instead of being repaired silently unless the design explicitly permits normalization
- same-provider replay metadata is preserved intentionally and documented, including which IDs, signatures, or encrypted artifacts are required for continuation
- usage normalization documents whether counts are final-only, partial, cache-adjusted, or best-effort
- tests cover request serialization, normal streaming, terminal failure cases, incomplete responses, tool-call round trips, and multimodal edge cases where the provider supports them

Each provider should have a short implementation note in code or docs that answers the following:

- what is the canonical upstream API surface for this adapter
- which reasoning controls are supported and how they are mapped
- which provider metadata must be preserved for replay
- which stream events are expected and which are intentionally unsupported
- which provider options are accepted and what defaults are applied

Required provider module shape:

```python
class ProviderAdapter(Protocol):
    provider_name: str
    api_family: str

    def build_headers(self, model: ModelSpec, request: GenerateRequest, options: RequestOptions) -> dict[str, str]:
        ...

    def build_payload(self, model: ModelSpec, request: GenerateRequest, options: RequestOptions) -> dict[str, Any]:
        ...

    async def stream_response(
        self,
        *,
        model: ModelSpec,
        request: GenerateRequest,
        options: RequestOptions,
        http: HttpTransport,
    ) -> AsyncIterator[StreamEvent]:
        ...
```

### 2. ModelSpec

`ModelSpec` describes model identity and capabilities.

Required fields:

```python
from pydantic import BaseModel, ConfigDict, Field


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

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
    pricing: ModelPricing | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    protocol_defaults: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)
```

Notes:

- `provider` identifies vendor namespace such as `openai`, `anthropic`, `gemini`, `vertex-gemini`.
- `api_family` identifies wire protocol family such as `openai-responses`, `anthropic-messages`, `gemini-generate-content`.
- `base_url` is overridable for proxies and compatible endpoints.
- `capabilities` stores protocol and model feature flags that do not justify dedicated top-level fields.
- `protocol_defaults` stores provider-specific default knobs that may need to be merged into requests.
- `extra` stores non-portable metadata without polluting the common shape.

Modality semantics:

- `supports_images` means the model can accept image inputs in normalized message content
- `supports_image_outputs` means the model can return normalized assistant image output blocks
- phase 1 requires image input support and image-bearing tool results
- normalized assistant image output is not part of the phase 1 required surface and must be gated behind `supports_image_outputs`

Required capability flags include, at minimum where applicable:

- `supports_developer_role`
- `requires_explicit_reasoning_disable`
- `usage_final_only`
- `tool_call_id_max_length`
- `tool_call_id_charset`
- `supports_parallel_tool_calls`

Generic client logic must prefer capability-driven behavior over hardcoded branching on provider names whenever a behavior difference is representable as data.

### 3. GenerateRequest

Unified request shape:

```python
class GenerateRequest(BaseModel):
    messages: list[Message]
    system_prompt: str | None = None
    tools: list[ToolSpec] = Field(default_factory=list)
    tool_choice: ToolChoice | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning: ReasoningConfig | None = None
    response_format: ResponseFormat | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    session: SessionHints | None = None
    protocol_hints: dict[str, Any] = Field(default_factory=dict)
    extension_data: dict[str, Any] = Field(default_factory=dict)
```

This shape is provider-agnostic. Provider-specific request tuning belongs in `RequestOptions.provider_options`.

Notes:

- `protocol_hints` is for soft, forward-compatible hints that may influence provider behavior without becoming hard API guarantees.
- `extension_data` is reserved for protocol oddities that need request-scoped structured data without immediately expanding the stable top-level API.

### 4. RequestOptions

Per-call transport and execution controls:

```python
class RequestOptions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    timeout: float | aiohttp.ClientTimeout | None = 60.0
    headers: dict[str, str] = Field(default_factory=dict)
    auth: AuthStrategy | None = None
    idempotency_key: str | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)
    transport_options: dict[str, Any] = Field(default_factory=dict)
```

Do not overload `GenerateRequest` with transport concerns.

`transport_options` exists for future wire-level quirks that do not belong in the stable request model, for example WebSocket tuning, SSE parsing hints, or provider-specific connection parameters.

For providers that support multiple transports, the preferred transport must be expressed in `provider_options`, for example:

- OpenAI: `{"transport": "sse" | "websocket" | "auto"}`
- ChatGPT: `{"transport": "sse" | "websocket" | "auto"}`

`transport_options` remains reserved for lower-level mechanics such as connection reuse, heartbeat settings, or reconnect tuning.

### 5. Message model

Use a content-block message model from the start. A plain string-only interface will become a limitation immediately once tools, images, or reasoning are supported.

Use Pydantic discriminated unions for message and content block types so validation and serialization are explicit and stable.

Required public model structure:

```python
class TextBlock(BaseModel):
    type: Literal["text"]
    text: str
    provider_meta: dict[str, Any] = Field(default_factory=dict)
    protocol_meta: dict[str, Any] = Field(default_factory=dict)
    annotations: list[dict[str, Any]] | dict[str, Any] | None = None


class ImageBlock(BaseModel):
    type: Literal["image"]
    mime_type: str
    data: str
    provider_meta: dict[str, Any] = Field(default_factory=dict)
    protocol_meta: dict[str, Any] = Field(default_factory=dict)
    annotations: list[dict[str, Any]] | dict[str, Any] | None = None


class ReasoningBlock(BaseModel):
    type: Literal["reasoning"]
    text: str
    signature: str | None = None
    redacted: bool = False
    provider_meta: dict[str, Any] = Field(default_factory=dict)
    protocol_meta: dict[str, Any] = Field(default_factory=dict)
    annotations: list[dict[str, Any]] | dict[str, Any] | None = None


class ToolCallBlock(BaseModel):
    type: Literal["tool_call"]
    id: str
    name: str
    arguments: dict[str, Any]
    provider_meta: dict[str, Any] = Field(default_factory=dict)
    protocol_meta: dict[str, Any] = Field(default_factory=dict)
    annotations: list[dict[str, Any]] | dict[str, Any] | None = None
```

Required message structure:

```python
class UserMessage(BaseModel):
    role: Literal["user"]
    content: str | list[TextBlock | ImageBlock]
    provider_meta: dict[str, Any] = Field(default_factory=dict)
    protocol_meta: dict[str, Any] = Field(default_factory=dict)


class AssistantMessage(BaseModel):
    role: Literal["assistant"]
    content: list[TextBlock | ReasoningBlock | ToolCallBlock]
    provider_meta: dict[str, Any] = Field(default_factory=dict)
    protocol_meta: dict[str, Any] = Field(default_factory=dict)


class ToolResultMessage(BaseModel):
    role: Literal["tool"]
    tool_call_id: str
    tool_name: str
    content: list[TextBlock | ImageBlock]
    is_error: bool = False
    provider_meta: dict[str, Any] = Field(default_factory=dict)
    protocol_meta: dict[str, Any] = Field(default_factory=dict)
```

```python
UserMessage
AssistantMessage
ToolResultMessage
```

Content blocks:

- `TextBlock(text, provider_meta={})`
- `ImageBlock(data: bytes | str, mime_type: str)`
- `ReasoningBlock(text: str, signature: str | None = None, redacted: bool = False)`
- `ToolCallBlock(id: str, name: str, arguments: dict[str, Any], provider_meta={})`

Image normalization rules:

- `ImageBlock.data` is normalized as base64-encoded image bytes without a `data:` URL prefix
- `ImageBlock.mime_type` is required and must be an explicit image media type such as `image/png` or `image/jpeg`
- mixed text and image ordering inside a message is significant and must be preserved during normalization and provider serialization
- provider adapters are responsible for converting normalized image blocks into provider-native formats such as inline parts or data URLs

Image normalization rules:

- `ImageBlock.data` is normalized as base64-encoded image bytes without the `data:` URL prefix
- `ImageBlock.mime_type` is required and must be an explicit media type such as `image/png` or `image/jpeg`
- message content order is preserved exactly, so mixed text and image blocks remain in caller-specified order
- provider adapters are responsible for converting normalized images into provider-native wire representations such as inline base64 parts, data URLs, or provider content parts

Provider metadata is required because some vendors expose opaque identifiers that must be replayed in subsequent turns within the same provider session.

Every message and content block must support a generic metadata container. At minimum, the internal model must allow:

- `provider_meta`: opaque provider-specific structured data needed for same-provider replay
- `protocol_meta`: protocol-level metadata used for debugging, tracing, and protocol compatibility
- `annotations`: optional list or dict reserved for extensibility without changing the union shape

The library does not interpret most of this data generically. It exists so protocol gotchas do not force immediate breaking changes to the public model.

### Message model semantics

The message model is the durable conversation transcript.

- `UserMessage` represents caller input
- `AssistantMessage` represents one completed assistant turn
- `ToolResultMessage` represents the caller's reply to a prior assistant tool call

This model is intentionally different from the streaming event protocol.

- messages are persisted conversation state
- stream events are transient updates for one in-flight assistant response

A single assistant message may contain multiple blocks, for example reasoning plus visible text, or visible text plus one or more tool calls.

Example transcript:

```python
request = GenerateRequest(
    messages=[
        UserMessage(
            content="Summarize this file and call search if needed."
        ),
        AssistantMessage(
            content=[
                ReasoningBlock(
                    text="I should inspect the imports first.",
                    signature="provider-specific-signature",
                ),
                ToolCallBlock(
                    id="call_1",
                    name="search",
                    arguments={"query": "imports"},
                ),
            ]
        ),
        ToolResultMessage(
            tool_call_id="call_1",
            tool_name="search",
            content=[
                TextBlock(text="Found 3 imports: httpx, json, typing")
            ],
        ),
        AssistantMessage(
            content=[
                TextBlock(text="This file imports httpx, json, and typing.")
            ]
        ),
    ]
)
```

In this example:

- the first assistant message asked to call a tool
- the tool result message answered that tool call
- the second assistant message provided the visible answer after receiving the tool result

Only completed assistant turns belong in the persisted transcript. In-progress token deltas do not.

### Image inputs and outputs

Phase 1 multimodal support is defined as follows:

- `UserMessage` may contain mixed `TextBlock` and `ImageBlock` content for vision-capable models
- `ToolResultMessage` may contain mixed `TextBlock` and `ImageBlock` content so tools can return images back to the model
- `AssistantMessage` does not require normalized image output support in phase 1 and remains limited to text, reasoning, and tool-call blocks

This supports the most important initial multimodal workflows:

- user provides an image to the model
- a tool returns an image that the model must interpret in a later step

Assistant-generated image output may be added later behind `supports_image_outputs`, but it is not required for the first implementation slice.

Example user multimodal input:

```python
GenerateRequest(
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

Example tool result with text and image:

```python
ToolResultMessage(
    tool_call_id="call_1",
    tool_name="get_circle_with_description",
    content=[
        TextBlock(text="A red circle with a diameter of 100 pixels."),
        ImageBlock(data=base64_png, mime_type="image/png"),
    ],
    is_error=False,
)
```

### Image inputs and outputs

Phase 1 image support is defined as follows:

- `UserMessage` may contain mixed `TextBlock` and `ImageBlock` content for vision-capable models
- `ToolResultMessage` may contain mixed `TextBlock` and `ImageBlock` content so tool outputs can include images
- `AssistantMessage` does not require normalized image output support in phase 1 and is limited to text, reasoning, and tool-call blocks

This matches the most important multimodal workflows for the initial implementation:

- user supplies an image to the model
- a tool returns an image that the model must interpret on the next step

Assistant-generated image output may be added later behind `supports_image_outputs`, but it is not required for the first implementation slice.

Example user multimodal input:

```python
GenerateRequest(
    messages=[
        UserMessage(
            content=[
                TextBlock(text="What is in this image?"),
                ImageBlock(
                    data=base64_png,
                    mime_type="image/png",
                ),
            ]
        )
    ]
)
```

Example tool result with text and image:

```python
ToolResultMessage(
    tool_call_id="call_1",
    tool_name="get_circle_with_description",
    content=[
        TextBlock(text="A red circle with a diameter of 100 pixels."),
        ImageBlock(data=base64_png, mime_type="image/png"),
    ],
    is_error=False,
)
```

## Public API surface

### Main client

```python
class AsyncLLMClient:
    def __init__(
        self,
        *,
        http_client: aiohttp.ClientSession | None = None,
        auth_registry: AuthRegistry | None = None,
        model_registry: ModelRegistry | None = None,
        provider_registry: ProviderRegistry | None = None,
    ) -> None:
        ...

    async def generate(
        self,
        model: str | ModelSpec,
        request: GenerateRequest,
        *,
        provider: str | None = None,
        options: RequestOptions | None = None,
    ) -> AssistantResponse:
        ...

    def stream(
        self,
        model: str | ModelSpec,
        request: GenerateRequest,
        *,
        provider: str | None = None,
        options: RequestOptions | None = None,
    ) -> StreamHandle:
        ...
```

`generate()` must internally consume `stream()` and return the finalized response.

### Convenience helpers

The top-level module exposes:

```python
async def generate(...)
def stream(...)
def get_model(provider: str, model: str) -> ModelSpec
def list_models(provider: str | None = None) -> list[ModelSpec]
```

Model resolution rules:

- `provider + model` is the canonical lookup key
- if the user passes only a bare model string, the client may resolve it only when it is unambiguous in the registry
- ambiguous bare model strings must raise a model resolution error rather than silently choosing a provider
- the recommended external string form is `provider/model`
- resolution errors must include the candidate providers/models to make correction straightforward
- registry internals must be keyed by `(provider, model)` even if convenience string helpers are exposed on top
- public code must prefer explicit lookups such as `get_model(provider, model)` over implicit global resolution when correctness matters

### StreamHandle

The stream object must be both an async iterator and a collector.

```python
class StreamHandle(AsyncIterator[StreamEvent]):
    async def final_response(self) -> AssistantResponse:
        ...
```

This avoids forcing the user to manually reassemble partial deltas.

Typical usage:

```python
stream = client.stream(
    "openai/gpt-5-mini",
    GenerateRequest(
        messages=[
            UserMessage(content="What files import httpx?")
        ]
    ),
)

async for event in stream:
    if event.type == "response_start":
        print("assistant started")

    elif event.type == "text_delta":
        print(event.delta, end="", flush=True)

    elif event.type == "reasoning_delta":
        # Usually hidden from end users, but available to callers
        pass

    elif event.type == "tool_call_end":
        print("\ntool call finished")

    elif event.type == "usage":
        print("\nusage update:", event.usage)

    elif event.type == "response_end":
        print("\nfinished:", event.response.finish_reason)

    elif event.type == "error":
        print("\nerror:", event.error.message)

final_response = await stream.final_response()
```

## Streaming event protocol

Use an event model that is expressive enough for all target vendors.

Required event types:

- `response_start`
- `text_start`
- `text_delta`
- `text_end`
- `reasoning_start`
- `reasoning_delta`
- `reasoning_end`
- `tool_call_start`
- `tool_call_delta`
- `tool_call_end`
- `usage`
- `response_end`
- `error`

Required event schemas:

```python
class ResponseStartEvent(BaseModel):
    type: Literal["response_start"]
    provider: str
    model: str
    response_id: str | None = None


class TextDeltaEvent(BaseModel):
    type: Literal["text_delta"]
    index: int
    delta: str


class ReasoningDeltaEvent(BaseModel):
    type: Literal["reasoning_delta"]
    index: int
    delta: str


class ToolCallDeltaEvent(BaseModel):
    type: Literal["tool_call_delta"]
    index: int
    delta: str


class UsageEvent(BaseModel):
    type: Literal["usage"]
    usage: Usage


class ResponseEndEvent(BaseModel):
    type: Literal["response_end"]
    response: AssistantResponse


class ErrorEvent(BaseModel):
    type: Literal["error"]
    error: ErrorInfo
    partial_response: AssistantResponse | None = None
```

Implementation rules:

- `index` refers to the index in the assembled `AssistantResponse.content`
- providers may emit `text_start` and then one or more `text_delta` events before `text_end`
- tool calls must stream raw argument deltas where possible and parse final JSON only once the block is complete
- if a provider cannot stream a given content type incrementally, it must still produce a correct final response and may emit only start/end or only final block events

Design notes:

- Streaming text, reasoning, and tool arguments as separate event classes avoids lossy normalization.
- `usage` may be emitted zero or more times; the final response carries the final normalized totals.
- `response_end` must include the fully assembled `AssistantResponse`.
- Provider adapters must never raise mid-stream once iteration has started; they must emit `error` and terminate cleanly.

Usage completeness rule:

- some providers emit usage only in the final chunk, while others expose usage earlier in the stream
- the final response model must distinguish final usage from partial usage and from no usage being available
- `final` means the provider supplied authoritative end-of-request usage
- `partial` means some usage was observed, but interruption or provider behavior prevents treating it as final
- `none` means no trustworthy usage is available

### How callers detect intermediate and final states

The stream protocol represents one assistant response in progress.

- intermediate state: any `*_start`, `*_delta`, `*_end`, or `usage` event before termination
- final success: `response_end`
- final failure: `error`

There is not a separate concept of multiple complete intermediate assistant responses during a single request. Instead, one response is incrementally assembled from events.

If the caller wants a rolling partial view for UI rendering, the caller must assemble it from stream events or use a library-provided partial assembler.

Example partial assembler:

```python
blocks: list[dict | None] = []

async for event in stream:
    if event.type == "text_start":
        while len(blocks) <= event.index:
            blocks.append(None)
        blocks[event.index] = {"type": "text", "text": ""}

    elif event.type == "text_delta":
        blocks[event.index]["text"] += event.delta

    elif event.type == "tool_call_start":
        while len(blocks) <= event.index:
            blocks.append(None)
        blocks[event.index] = {
            "type": "tool_call",
            "arguments_json": "",
        }

    elif event.type == "tool_call_delta":
        blocks[event.index]["arguments_json"] += event.delta

    elif event.type == "response_end":
        final_response = event.response
```

This separation is deliberate:

- messages model durable history
- events model live generation progress
- `final_response()` returns the completed assistant response even if the caller has already consumed some or all events

## Unified response model

```python
class AssistantResponse(BaseModel):
    provider: str
    model: str
    content: list[AssistantContentBlock]
    finish_reason: FinishReason
    usage: Usage
    response_id: str | None = None
    protocol_state: dict[str, Any] = Field(default_factory=dict)
    provider_meta: dict[str, Any] = Field(default_factory=dict)
```

`response_id` must be surfaced when available. It is valuable for debugging, tracing, and some multi-turn APIs.

`protocol_state` is reserved for opaque response-scoped state that must be carried into later turns for the same provider when required, such as hidden continuation handles, encrypted reasoning artifacts, upstream message IDs, session cursors, or provider-native replay tokens.

Required additional response fields:

```python
class AssistantResponse(BaseModel):
    provider: str
    model: str
    api_family: str
    content: list[AssistantContentBlock]
    finish_reason: FinishReason
    usage: Usage
    response_id: str | None = None
    request_id: str | None = None
    protocol_state: dict[str, Any] = Field(default_factory=dict)
    provider_meta: dict[str, Any] = Field(default_factory=dict)
```

- `request_id` must capture any upstream request identifier if available from headers or payloads
- `api_family` must always be included in the final response so debugging and session replay logic are explicit

Required usage extension:

```python
class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    completeness: Literal["final", "partial", "none"] = "none"
```

`completeness` is required because stream interruption may leave usage totals incomplete or absent.

Cost interpretation rule:

- cost is authoritative only when usage completeness is `final`
- when completeness is `partial`, cost must be surfaced as best-effort only

## Authentication design

Authentication must be independent from provider implementations.

### Core auth abstraction

```python
class AuthStrategy(Protocol):
    async def apply(self, request: aiohttp.client_reqrep.ClientRequest) -> None:
        ...
```

Built-in strategies:

- `NoAuth`
- `HeaderAPIKeyAuth(header_name="Authorization" | "x-api-key" | "x-goog-api-key", prefix="Bearer " | "")`
- `BearerTokenAuth(token: str)`
- `QueryAPIKeyAuth(param_name="key")`
- `CallableTokenAuth(get_token: Callable[[], Awaitable[str] | str])`
- `RefreshingOAuthAuth(get_access_token: Callable[[], Awaitable[AccessToken]])`
- `CompositeAuth([...])`

### Auth resolution rules

The client must support auth at three levels, highest precedence first:

1. per-call `RequestOptions.auth`
2. provider default auth registered on the client
3. environment-based fallback helpers

### Environment helpers

Provide explicit helper functions. Do not rely on hidden implicit environment resolution. Example:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`

Do not hardwire all auth behavior to environment variables. Users need explicit control in servers, tests, and multi-tenant systems.

### OAuth and token refresh

The core library does not own browser-style login flows. It supports only using and refreshing tokens that are already available.

Required behavior:

- accept static bearer tokens
- accept callback-based token suppliers
- allow async refresh before request dispatch
- allow providers to request extra headers derived from auth state

This keeps the core dependency-free with respect to OAuth login UX.

The implementation must document the expected token shape and refresh behavior for providers that rely on OAuth-derived bearer tokens.

Anthropic implementation details:

- use OAuth 2.0 authorization code flow with PKCE
- open the provider authorization URL in the user browser
- run a loopback callback server on localhost for CLI login flows
- accept manual paste of the final redirect URL or code as a fallback when loopback callback is not available
- exchange the authorization code for `access_token`, `refresh_token`, and `expires_in`
- refresh tokens with `grant_type=refresh_token`
- store the access token as the effective bearer credential used by the API provider

Anthropic localhost callback server workflow:

1. generate PKCE verifier and challenge
2. generate state value and bind a localhost HTTP callback server before opening the browser
3. construct the authorization URL with `response_type=code`, `client_id`, `redirect_uri`, `scope`, `code_challenge`, `code_challenge_method=S256`, and `state`
4. open the browser to that URL
5. the local callback handler validates route, checks for provider error parameters, validates `state`, and extracts the authorization `code`
6. if automatic callback is not received, allow the user to paste either the final redirect URL or the raw code
7. exchange the code at the token endpoint
8. persist `access`, `refresh`, and `expires` with a small pre-expiry safety margin

ChatGPT/OpenAI subscription implementation details:

- use OAuth 2.0 authorization code flow with PKCE
- run a loopback callback server on localhost for CLI login flows
- accept manual paste fallback if the callback server cannot be used
- exchange the authorization code for `access_token`, `refresh_token`, and `expires_in`
- refresh tokens with `grant_type=refresh_token`
- decode the returned access token as a JWT and extract `chatgpt_account_id`
- persist both the access token and the extracted account ID because the ChatGPT backend requires both bearer auth and a separate account-scoped header

ChatGPT/OpenAI localhost callback server workflow:

1. generate PKCE verifier and challenge
2. generate a cryptographically random OAuth state value
3. bind a localhost HTTP callback server before opening the browser
4. construct the authorization URL with `response_type=code`, `client_id`, `redirect_uri`, `scope`, `code_challenge`, `code_challenge_method=S256`, and `state`
5. open the browser to that URL
6. the callback handler validates the request path, validates `state`, and extracts the authorization `code`
7. if loopback callback fails or is unavailable, allow manual paste of the redirect URL or code
8. exchange the code at the token endpoint using `application/x-www-form-urlencoded`
9. parse the returned access token as a JWT, extract `chatgpt_account_id`, and fail the flow if that claim is missing
10. persist `access`, `refresh`, `expires`, and `account_id` for later API use

The core provider library does not implement the interactive browser login flow in phase 1. The auth model must support a companion CLI module implementing these flows without changing provider interfaces.

### Vertex note

Vertex AI is a non-goal for the initial implementation.

Do not design the first version around Google service-account JWT minting, Application Default Credentials, or Vertex-specific auth flows. If Vertex is added later, it is introduced as a separate provider/backend with its own auth design.

## Transport layer design

### Shared HTTP transport

Provide a small wrapper around `aiohttp.ClientSession` so providers depend on a narrow interface.

Responsibilities:

- request building
- timeout handling
- base URL merging
- streaming responses
- converting transport exceptions into library exceptions

### Stream decoders

Implement two internal stream decoders:

- SSE decoder for OpenAI and Anthropic
- incremental JSON/chunk decoder for Gemini

If OpenAI WebSocket mode is enabled, add a third internal transport path:

- WebSocket transport for OpenAI Responses continuation mode

Do not force one decoder abstraction on all providers if the underlying wire format differs. The provider adapter must choose the correct decoder.

### WebSocket transport

Some providers offer a persistent WebSocket mode in addition to streamed HTTP. This must be integrated as a transport implementation detail, not as a separate public client API.

Design rules:

- the public `generate()` and `stream()` API remains unchanged
- the provider adapter selects HTTP SSE or WebSocket based on `provider_options`
- WebSocket support is provider-specific and optional at install time
- no generic multiplexing abstraction is required in phase 1

Required internal interface:

```python
class WebSocketTransport(Protocol):
    async def connect(self, *, url: str, headers: dict[str, str]) -> None:
        ...

    async def send_json(self, payload: dict[str, Any]) -> None:
        ...

    async def iter_json(self) -> AsyncIterator[dict[str, Any]]:
        ...

    async def close(self) -> None:
        ...
```

Providers that do not need WebSockets do not depend on this interface.

### Error classification

Define normalized exceptions:

- `AuthenticationError`
- `RateLimitError`
- `ContextLengthError`
- `ProviderProtocolError`
- `TransientProviderError`
- `PermanentProviderError`

Providers must map HTTP status codes and known vendor error payloads into these exceptions.

Required error payload shape:

```python
class ErrorInfo(BaseModel):
    code: str
    message: str
    provider: str | None = None
    api_family: str | None = None
    status_code: int | None = None
    retryable: bool = False
    raw: dict[str, Any] | None = None
```

Mapping rules:

- HTTP 401/403 -> `AuthenticationError` unless provider clearly signals quota exhaustion
- HTTP 408/429 -> `RateLimitError` or `TransientProviderError` depending on provider semantics
- known context-limit errors -> `ContextLengthError`
- malformed streaming frames or invalid payload shapes -> `ProviderProtocolError`
- 5xx before first byte -> `TransientProviderError`
- unsupported feature errors -> `PermanentProviderError`

## Provider design

### Additional provider scope notes

The initial implementation target is:

- OpenAI API
- OpenAI Codex via ChatGPT subscription
- Anthropic API
- Anthropic subscription access
- Gemini API
- OpenRouter

The architecture leaves room for nearby provider families that are operationally important:

- ChatGPT subscription-backed Codex access lives under the ChatGPT provider, not as a separate top-level provider family from ChatGPT itself
- OpenRouter is treated as an OpenAI-compatible aggregator/provider
- Google Gemini CLI and Antigravity are Google OAuth-backed alternate Gemini-family backends and are not required in phase 1
- Amazon Bedrock is a future provider/backend and is not part of the first implementation slice

## OpenAI provider

### Primary API

Use the OpenAI Responses API as the primary OpenAI integration. It has the most future-proof surface for reasoning, tool calls, multimodality, and response metadata.

Keep a provider-internal compatibility layer so an OpenAI-compatible endpoint can later use Chat Completions where required, but the public API does not expose two separate OpenAI integrations in phase 1.

### OpenAI auth

Primary auth modes:

- API key via `Authorization: Bearer <token>`
- OAuth bearer token via the same header

The transport does not care whether a bearer token came from a static key or an OAuth refresh flow.

### OpenAI request mapping

Map unified request fields roughly as follows:

- `system_prompt` -> developer or system input depending on model family and compatibility mode
- `messages` -> `input`
- `tools` -> function tools
- `tool_choice` -> provider equivalent
- `max_output_tokens` -> OpenAI output token field
- `reasoning` -> reasoning config when supported
- `response_format` -> JSON schema mode when supported

Concrete request shape for initial implementation:

- endpoint: Responses API over HTTP POST
- auth header: `Authorization: Bearer <token>`
- content type: JSON
- streaming: enabled by request flag
- system prompt placement:
  - use developer-style top-level/system input for reasoning-capable models when supported
  - otherwise fall back to a system message/input item
- tools:
  - map each tool to an OpenAI function tool definition
  - map `tool_choice` into provider-native equivalent
- reasoning:
  - map effort if requested and supported
  - request encrypted reasoning artifacts only when needed for same-provider continuation
- usage:
  - parse prompt, completion, cached, and total token counts when present

OpenAI-specific provider options are:

- `transport`: `"sse" | "websocket" | "auto"`
- `generate`: `bool | None`
- `previous_response_id`: `str | None`
- `store`: `bool | None`
- `context_management`: provider-native dict when supported

Concrete implementation rules:

- implement SSE streaming over `aiohttp.ClientSession` streamed responses
- parse `data:` frames and ignore comment/blank frames
- stop on `[DONE]` if present
- accumulate output blocks in provider assembly state and emit normalized events as frames arrive
- preserve upstream response/message IDs in `response_id` and `protocol_state`
- when serializing tool results that contain images, map them into structured OpenAI-compatible tool-output content items rather than flattening them into plain text

### OpenAI WebSocket mode integration

OpenAI Responses WebSocket mode is modeled as an alternate transport for the OpenAI provider, not as a separate provider.

Why it fits the existing design:

- event ordering matches the existing Responses streaming event model
- request payloads are nearly identical to HTTP Responses payloads
- continuation still uses `previous_response_id`
- the main difference is transport and connection-local continuation state

Implementation requirements:

- keep `openai` as one provider adapter
- select SSE or WebSocket using `provider_options["transport"]`
- keep the normalized event stream identical regardless of transport
- preserve response IDs and continuation hints in `AssistantResponse.response_id` and `AssistantResponse.protocol_state`

OpenAI transport values:

- `"sse"`: use standard HTTP streaming
- `"websocket"`: require a persistent WebSocket connection
- `"auto"`: the provider selects WebSocket only when the runtime supports it and the request pattern benefits from persistent continuation

#### Connection model

OpenAI WebSocket mode uses one persistent connection to `/v1/responses` and allows multiple `response.create` operations over that socket, but only one in-flight response at a time.

Library implications:

- treat one OpenAI WebSocket connection as a sequential provider session
- no multiplexing support is required
- use multiple connections if the caller needs parallel OpenAI runs
- the connection lifetime limit must be treated as provider behavior and surfaced through reconnect logic

Required internal session object:

```python
class ProviderSession(Protocol):
    provider: str

    async def stream(self, request: GenerateRequest, options: RequestOptions) -> AsyncIterator[StreamEvent]:
        ...

    async def close(self) -> None:
        ...
```

For most providers, sessions are thin wrappers over stateless HTTP execution. For OpenAI WebSocket mode, the session owns the persistent socket.

#### Continuation model

OpenAI WebSocket mode continues a run by sending only incremental input plus `previous_response_id`.

Map this into the library as follows:

- `AssistantResponse.response_id` stores the OpenAI response ID
- `AssistantResponse.protocol_state` stores provider-private transport details such as `{"previous_response_id": "resp_123"}` when they are required for same-provider continuation
- `SessionHints.continue_from` or `provider_options["previous_response_id"]` forces explicit continuation from a prior response

Provider behavior:

- if a caller is using an active OpenAI WebSocket session and does not override continuation, the provider continues from the most recent response automatically using the last known `response_id`
- if the caller explicitly provides `previous_response_id`, that value wins
- if the connection-local cache can no longer satisfy the requested continuation, the provider must surface `previous_response_not_found` as a structured provider error

This keeps continuation provider-bound while still using the same public request and response model.

#### Incremental inputs vs full transcript

The generic library API is transcript-oriented, but OpenAI WebSocket mode is optimized for incremental inputs.

Adapter behavior requirements:

- same-provider WebSocket continuation serializes only newly added input items plus `previous_response_id`
- if the provider cannot safely compute the incremental delta from prior session state, it must fall back to regular HTTP/SSE-style full request serialization
- callers must not be required to manually construct incremental deltas for correctness

This means the optimization is provider-owned. The public API remains stable.

#### Warmup requests

OpenAI WebSocket mode supports `generate=false` warmup requests that pre-establish request state without generating output.

Map this to provider options:

- `provider_options["generate"] = False`

Rules:

- a warmup request still returns a provider response object if the provider supplies a response ID
- the resulting `response_id` may be used as the continuation base for the next generated turn
- `generate=false` is OpenAI-specific and must not become a common top-level request field

#### Store and ZDR compatibility

OpenAI WebSocket mode is compatible with `store=false` and Zero Data Retention because the most recent previous-response state may live only in connection-local memory.

Design implications:

- `store` remains an OpenAI provider option, not a generic client field
- same-session low-latency continuation can work without persisted provider state
- if the socket reconnects and `store=false`, the in-memory continuation path may be unavailable and the provider must either fail with `previous_response_not_found` or start a new chain when requested by the caller

#### Reconnect behavior

OpenAI WebSocket connections may close due to network failures or provider-enforced duration limits.

Required behavior:

- if the socket closes, the session becomes unusable until reconnected
- on reconnect, if continuation is still valid through persisted `previous_response_id`, the provider may continue the chain
- if continuation is not available, the provider must require a new chain or a full new context

Error mapping requirements:

- `previous_response_not_found` -> `ProviderProtocolError` or `PermanentProviderError` depending on retryability context
- `websocket_connection_limit_reached` -> reconnect-required provider error, typically retryable after opening a new socket

#### Compaction support

OpenAI compaction is treated as provider-specific continuation management.

Rules:

- server-side compaction through normal response generation is transparent to the generic API
- standalone compaction that returns a new compacted window starts a new response chain
- compacted windows are provider-native payload fragments and remain in `protocol_state` or provider-internal handling, not first-class common message types

#### When to choose WebSocket mode

The provider uses WebSocket mode only when it materially improves latency, typically in long-running tool-heavy loops.

Good candidates:

- agentic coding
- repeated tool-calling workflows
- orchestration loops with many continuation steps

HTTP SSE remains the simpler default for one-shot or short exchanges.

### OpenAI gotchas

1. Reasoning data is not universally portable.
   Some reasoning payloads are opaque and only safe to replay back to the same provider family.

2. Tool call identifiers can be verbose and provider-specific.
   They only need to be stable within the same provider session.

3. Some OpenAI-compatible servers do not support the `developer` role or advanced reasoning fields.
   Compatibility flags must exist on `ModelSpec.extra` or provider config.

4. Some endpoints expose cached token counts separately.
   Usage normalization must avoid double-counting cached tokens in total input.

5. Service tiers may alter pricing.
   Cost tracking must allow a pricing multiplier or explicit override.

6. `store` and other request fields are not universally accepted by compatible proxies.
   Provider options must allow disabling incompatible fields without changing the common API.

7. Reasoning-capable OpenAI-family models may prefer `developer` instructions, while some compatible endpoints only accept `system`.
   This must be driven by compatibility flags on the model or provider configuration, not hardcoded per provider name.

8. WebSocket continuation is connection-local when relying on uncached `previous_response_id` state.
   A reconnect may lose the low-latency continuation path even though the response protocol itself remains unchanged.

9. Tool result images must remain attached to the tool-result payload as structured multimodal content.
   OpenAI-compatible providers expect image-bearing tool outputs to be serialized as content items, not stringified blobs.

### OpenAI session replay policy

- Preserve text, tool calls, tool results, and provider-native reasoning artifacts only within the same OpenAI-family session.
- Do not define or support replay into Anthropic or Gemini sessions.

OpenAI-compatible implementation details:

- OpenRouter reuses this adapter with provider-specific compatibility flags
- provider compatibility is driven by `ModelSpec.capabilities`, `ModelSpec.protocol_defaults`, and `RequestOptions.provider_options`
- fields that differ across compatible providers, such as developer-role support, reasoning-field names, or routing payloads, must be isolated in compatibility helpers rather than forked provider logic where possible

## ChatGPT subscription provider

The implementation treats ChatGPT subscription-backed access as a separate provider from direct OpenAI API access, even if both ultimately expose a Responses-like protocol.

Provider name:

- `chatgpt`

API family:

- `chatgpt-responses`

Codex placement:

- Codex is modeled as ChatGPT-backed model access within this provider, not as a completely separate provider family
- implementation-wise, Codex is a model/backend variant exposed by the ChatGPT subscription surface
- if the codebase needs naming distinction, prefer model metadata or a ChatGPT sub-family marker rather than a separate top-level provider abstraction

This separation is important because auth, base URL, headers, retry behavior, and transport capabilities differ materially from direct OpenAI API usage.

### Why ChatGPT should not be modeled as plain OpenAI

Although the request and event shapes are close to OpenAI Responses, ChatGPT subscription access differs in several ways:

- it uses a different base URL and backend namespace
- it is authenticated with ChatGPT OAuth access tokens rather than OpenAI API keys
- requests require additional account-scoped headers
- it supports SSE and may support WebSocket transport
- it relies on session identifiers for connection reuse and prompt caching behavior
- it returns subscription- and usage-limit-specific errors that are surfaced differently from standard OpenAI API errors

For implementation clarity, reuse shared OpenAI Responses message conversion and event processing where possible, but keep ChatGPT as a dedicated provider adapter.

### ChatGPT auth

Primary auth mode:

- OAuth access token obtained from ChatGPT login flow, sent as `Authorization: Bearer <token>`

Additional requirement:

- extract the ChatGPT account ID from the JWT access token and send it in a separate account header

Implementation requirement:

- do not treat a ChatGPT access token as interchangeable with an OpenAI API key
- the provider adapter must decode the JWT payload and extract `chatgpt_account_id`
- if account ID extraction fails, fail before issuing the request

Required auth model addition:

```python
class ChatGPTAccessTokenAuth(BearerTokenAuth):
    account_id: str
```

The account ID is either supplied explicitly by a higher-level auth helper or lazily derived by the provider from the token. If derived by the provider, the extracted value must be cached per token.

Implementation details for the companion auth flow:

- use authorization code plus PKCE
- generate a random OAuth state value
- open the ChatGPT authorization URL with loopback redirect URI and PKCE challenge
- receive the callback on localhost when possible
- support manual code or redirect-URL paste as fallback
- exchange the code at the token endpoint using `application/x-www-form-urlencoded`
- refresh with `grant_type=refresh_token`
- parse the JWT access token and extract `chatgpt_account_id` from the provider-specific claim namespace
- fail login or refresh if the account ID cannot be extracted

### ChatGPT endpoints and base URL

Use a provider-specific base URL, for example:

- `https://chatgpt.com/backend-api`

The concrete streaming endpoint is provider-specific, such as a `codex/responses`-style path under the ChatGPT backend namespace.

Do not route ChatGPT traffic through the standard `api.openai.com` OpenAI provider.

Codex models use this same ChatGPT backend family and inherit the same OAuth, account-header, session, and transport behavior.

Concrete HTTP behavior:

- endpoint family is modeled as a ChatGPT backend path under the configured base URL
- request auth uses bearer token plus account-scoped header
- request body carries top-level instructions, input messages, tool configuration, session hint, and reasoning controls
- request body remains close to Responses-style semantics to maximize shared parsing code

### ChatGPT request mapping

ChatGPT request construction is similar to OpenAI Responses with these notable differences:

- system prompt is sent as top-level instructions rather than injected into the input list
- session ID is passed through for cache/session affinity
- include encrypted reasoning content when supported so same-provider replay remains possible
- enable tool choice and parallel tool calls by default when tools are present
- text verbosity is a provider-specific option separate from the generic API

ChatGPT-specific provider options:

- `transport`: `"sse" | "websocket" | "auto"`
- `session_id`: string
- `text_verbosity`: `"low" | "medium" | "high"`
- `reasoning_summary`: provider-specific enum

Concrete implementation rules:

- use HTTP POST with SSE streaming in phase 1
- send system prompt as top-level instructions, not as a normal input message
- include `session_id` when provided
- always attach account-specific headers derived from auth state
- persist any provider-specific replay artifacts in `protocol_state`

Additional ChatGPT implementation rule:

- retry only before the first stream byte on transient network failures and selected 429 or 5xx responses; never retry once streaming has begun

### ChatGPT headers

In addition to `Authorization`, ChatGPT requests may require provider-specific headers such as:

- account identifier header
- beta/feature flag header
- client request ID header for WebSocket mode
- session header
- originator/client identity headers

Design rule:

- the ChatGPT provider owns these headers completely
- the generic OpenAI provider must not inherit them
- user-supplied headers may override defaults only where safe

### ChatGPT error handling

ChatGPT error handling differs from standard OpenAI API handling.

Required behavior:

- parse structured error payloads for subscription or usage-limit conditions
- surface friendly rate-limit or plan-limit messages when the backend indicates account exhaustion
- classify these as `RateLimitError` or `AuthenticationError` as appropriate
- use retry with backoff for transient 429 and 5xx failures before stream start

Retry policy for ChatGPT:

- retry network failures and transient 429/5xx responses before first stream byte
- do not retry after stream processing has begun
- use exponential backoff with a small retry cap

This is stricter than the generic default because the subscription-backed backend may exhibit temporary overloaded states that are worth retrying.

### ChatGPT prompt caching and sessions

ChatGPT uses `session_id` as a first-class session hint.

Implementation guidance:

- when `session_id` is present, send it to the provider as both a request/session hint and a prompt cache key if supported by the backend
- preserve the distinction between generic `SessionHints` and provider-specific connection reuse
- do not assume session IDs are portable to the standard OpenAI provider

### ChatGPT reasoning behavior

ChatGPT shares the unified reasoning model with OpenAI Responses, and the adapter may require additional clamping or remapping for model-specific effort values.

Implementation guidance:

- support the common `ReasoningConfig`
- map effort levels conservatively when the backend rejects certain values for specific model families
- request encrypted reasoning content for same-provider replay when available

### ChatGPT implementation strategy

The recommended design is:

- reuse OpenAI-style message conversion for inputs, tool calls, and tool results
- reuse the OpenAI Responses event normalizer for streamed response items
- implement a dedicated ChatGPT request builder
- implement dedicated ChatGPT auth/header builders
- implement a dedicated SSE parser
- keep ChatGPT-specific retry and error parsing local to this provider

This gives maximum code reuse without conflating two operationally different providers.

### ChatGPT vs OpenAI summary

Key differences from the direct OpenAI provider:

1. Auth source
   ChatGPT uses OAuth access tokens tied to a user subscription; OpenAI uses API keys or generic bearer tokens.

2. Account-scoped headers
   ChatGPT requires an account identifier header derived from the token; OpenAI does not.

3. Base URL
   ChatGPT uses a ChatGPT backend base URL; OpenAI uses the standard OpenAI API base URL or compatible endpoints.

4. System prompt placement
   ChatGPT prefers top-level instructions; OpenAI commonly places the system/developer prompt in the input list.

5. Transport options
   ChatGPT uses a distinct backend and header model even when the transport remains standard streamed HTTP.

6. Session semantics
   ChatGPT session IDs are part of backend/session behavior; OpenAI session hints are generally request-level only.

7. Error semantics
   ChatGPT can return subscription-plan and usage-limit errors tied to the interactive product; OpenAI API errors are more conventional API quota/auth errors.

8. Provider boundary
   ChatGPT is implemented as a sibling of the OpenAI provider, not as a mode flag inside it.

## OpenRouter provider

OpenRouter is treated as a separate provider name built on an OpenAI-compatible wire protocol.

Provider name:

- `openrouter`

API family:

- `openai-responses` or `openai-completions`, depending on the implemented backend surface

Design notes:

- auth is typically static API key bearer auth
- the model namespace is aggregator-style and may encode upstream vendor/model identity in the model ID
- OpenRouter-specific routing or provider-preference options remain in `provider_options`
- some pricing, usage, and reasoning behavior may reflect the upstream routed provider rather than a uniform OpenRouter-native contract

Implementation guidance:

- reuse the OpenAI-compatible provider stack where possible
- keep OpenRouter-specific headers and routing fields isolated in the OpenRouter adapter
- do not assume all OpenRouter models support the same optional OpenAI fields

Concrete implementation rules:

- default auth header: `Authorization: Bearer <OPENROUTER_API_KEY>`
- treat model IDs as opaque strings, often in `<vendor>/<model>` form
- implement OpenRouter first through the OpenAI-compatible request path
- expose routing controls through `provider_options`, for example provider ordering or restriction lists
- be conservative with reasoning/config fields because upstream routed models may differ in support

## Gemini CLI and Antigravity

Google Gemini CLI and Antigravity are not the same as the direct Gemini API.

Gemini CLI:

- an alternate Google-hosted backend associated with Google Cloud Code Assist
- authenticated via Google OAuth rather than a plain Gemini API key
- may expose Gemini models through a backend with different headers, quotas, and retry behavior from the public Gemini API

Antigravity:

- a Google-hosted backend that can expose not only Gemini-family models but also other model families such as Claude or GPT-OSS-style offerings
- authenticated via Google OAuth
- operationally distinct from the direct Gemini API even when serving Gemini-family models

Design implication:

- these are not folded into the direct `gemini` provider
- if implemented later, treat them as separate providers or separate Google-hosted backends with their own auth and header logic
- they may still reuse parts of Gemini-style content conversion, especially where the backend shape is close to Google's content-part model

For the current project, both Gemini CLI and Antigravity are out of scope for phase 1.

## Amazon Bedrock

Amazon Bedrock is a future backend/provider and is not part of the initial implementation target.

Design notes:

- Bedrock is operationally different from direct vendor APIs because it is an AWS-hosted aggregation layer
- auth is AWS-native rather than simple bearer-token API key auth
- request signing and AWS credential resolution make Bedrock substantially different from OpenAI, Anthropic, and Gemini direct integrations
- model IDs may refer to vendor models hosted through Bedrock rather than direct vendor-native endpoints

Implementation implication:

- do not shape the initial provider abstractions around Bedrock-specific signing requirements
- if added later, Bedrock is introduced as its own provider/backend with a dedicated auth and transport path
- same-provider session continuity rules would still apply at the Bedrock-provider level

## Anthropic provider

### Anthropic auth

Primary auth modes:

- static API key via `x-api-key`
- OAuth bearer token via `Authorization: Bearer <token>`

Anthropic also requires version headers. The provider must own those defaults.

Anthropic covers both:

- direct API-key access
- subscription-backed OAuth access

Implementation details for Anthropic OAuth support:

- use authorization code plus PKCE
- use a localhost callback server for CLI login flows
- allow manual paste fallback when the callback cannot be received automatically
- exchange the code for access and refresh tokens at the OAuth token endpoint
- refresh with `grant_type=refresh_token`
- apply a small expiry buffer when persisting credentials so refresh happens slightly before actual expiration
- use the resulting access token directly as the bearer credential for Anthropic API calls

Concrete OAuth server behavior:

- the localhost callback server binds before opening the browser to avoid race conditions
- the callback route explicitly validates path, `state`, and required query parameters
- provider-declared error query parameters are converted into user-facing auth failures
- when loopback bind fails, the flow degrades to manual paste rather than aborting immediately
- refresh logic is shared with non-interactive auth helpers so the provider path only receives a valid access token

### Anthropic request mapping

Target the Messages API.

Map unified inputs to:

- top-level `system`
- alternating `messages`
- `tools`
- tool choice
- thinking config
- cache control when enabled

Concrete request shape for initial implementation:

- endpoint: Messages API over HTTP POST
- auth:
  - API key mode via `x-api-key`
  - subscription mode via `Authorization: Bearer <token>`
- required provider headers:
  - Anthropic version header
  - optional beta headers when specific capabilities are needed
- system prompt is sent via top-level system field
- messages are sent in Anthropic message format with content blocks
- tools are mapped to Anthropic tool definitions
- thinking config is translated from the unified reasoning model
- cache-control hints are attached only where supported and useful

Concrete implementation rules:

- implement SSE/event-stream parsing over `aiohttp`
- map streamed text, thinking, and tool-use deltas into normalized events
- filter empty blocks before request dispatch
- preserve any signature-bearing reasoning artifacts in `protocol_state` for same-provider continuation
- when using subscription/OAuth auth, allow provider-specific headers needed by the subscription backend to be attached by the Anthropic adapter
- preserve mixed text/image ordering for user input and tool results when converting to Anthropic content blocks

### Anthropic gotchas

1. Tool call IDs are provider-constrained.
   The adapter generates or validates IDs that satisfy Anthropic requirements for Anthropic sessions.

2. Thinking content may be interleaved with visible answer content.
   The stream protocol must preserve reasoning blocks separately.

3. Some thinking modes are adaptive rather than strictly token-budgeted.
   The unified API exposes a generic reasoning config and the provider translates it.

4. Prompt caching is exposed differently than OpenAI.
   Anthropic cache control attaches to message or content segments rather than a generic cache key.

5. Anthropic requires provider-specific headers such as version and beta flags.
   These must stay inside the provider implementation.

6. Tool result ordering matters.
   A tool call without a corresponding tool result can invalidate later replay. The library validates or repairs incomplete history before sending.

7. Empty content blocks may trigger provider errors.
   The adapter aggressively filters empty text and empty reasoning blocks.

8. Disabling reasoning may require an explicit provider-native disabled value.
   For models that default to thinking behavior, omitting the thinking field is not sufficient.

9. Vision-capable turns may include images in user content and tool results.
   The adapter must preserve multimodal block ordering and avoid flattening image-bearing tool outputs into plain text.

### Anthropic session replay policy

- Same-provider replay preserves reasoning signatures and tool-use structure.
- No cross-provider replay behavior is required.

## Gemini provider

The initial design targets the direct Gemini API only.

### Gemini auth

Direct Gemini API:

- API key via `x-goog-api-key` header or query parameter

Vertex/GCP-style Gemini access is out of scope for the initial version. If added later, prefer either a distinct provider name such as `vertex-gemini` or a dedicated auth/backend configuration, not silent mode switching.

### Gemini request mapping

Target the `generateContent` streaming API shape.

Map unified inputs to:

- content parts
- function declarations
- tool choice / function calling mode
- generation config
- thinking config

Concrete request shape for initial implementation:

- endpoint: direct Gemini generateContent-style streaming endpoint
- auth: direct API key via header or query parameter, with header preferred
- system prompt: provider-native system instruction/config field where supported
- messages: convert into content/parts structure
- tools: convert into function declarations and tool-calling config
- reasoning: map into Gemini thinking configuration when supported

Concrete implementation rules:

- implement streaming JSON/chunk parsing with `aiohttp`
- assemble content parts into normalized text, reasoning, and tool-call blocks
- synthesize tool-call IDs if the provider omits them
- preserve thought signatures and related replay metadata only in same-provider `protocol_state`

- if a streamed Gemini tool call omits arguments, normalize them to an empty object rather than failing generic tool-call assembly
- preserve inline image parts in user messages and image-bearing tool results when converting to Gemini content parts

### Gemini gotchas

1. Streaming format differs from SSE-centric providers.
   The provider must decode incremental JSON chunks rather than assume SSE.

2. Thinking and visible text can arrive as distinct parts.
   Preserve them as separate block types.

3. Gemini thought signatures are provider-specific replay artifacts.
   They are replayed back only to compatible Gemini requests in the same provider session.

4. Function calls may not always carry stable IDs.
   The adapter must synthesize deterministic IDs when absent.

5. Older Gemini variants and some backends may not support multimodal tool results the same way newer variants do.
   The adapter is conservative when replaying tool outputs with images.

6. Google tool schemas can be stricter than generic JSON Schema in practice.
   The public tool schema format stays simple and serializable.

7. Thought signatures are not portable.
   Reuse only within the same provider family and model line where supported.

8. Image-bearing tool results may require provider-specific routing to the correct content-part type.
   The adapter must not collapse these results into plain text if the target Gemini model supports image input.

### Gemini session replay policy

- Same Gemini family: preserve thought signatures and tool call metadata.
- No cross-provider replay behavior is required.

## Unified reasoning model

Use a provider-neutral reasoning config:

```python
class ReasoningConfig(BaseModel):
    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    summary: Literal["auto", "concise", "detailed"] | None = None
```

Provider mapping:

- OpenAI: `effort` and `summary` map directly where supported
- Anthropic: map to provider-native thinking controls conservatively and document any dropped fields
- Gemini: map to thinking config or provider equivalent

Important policy:

- reasoning is best-effort
- unsupported reasoning settings are ignored only if the provider cannot support them cleanly and the downgrade is documented
- if a provider cannot preserve reasoning replay artifacts safely, it must omit replay support rather than invent synthetic artifacts
- if a user requests strict reasoning behavior, allow a future strict mode that raises instead of silently degrading

Reasoning confidence requirements:

- the adapter must document whether reasoning output is visible text, summary text, encrypted replay material, or some combination
- the adapter must preserve provider-native reasoning artifacts only when they are valid for same-provider replay
- the adapter must not synthesize opaque reasoning artifacts unless the provider explicitly allows deterministic regeneration
- reasoning stream assembly must handle both incremental deltas and terminal reconciliation events without duplicating text
- if reasoning usage is reported separately, the adapter must expose it through normalized usage fields

## Tool calling design

### Tool schema

Keep tool definitions simple and JSON-native.

```python
class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
```

Use Pydantic for the library's typed surface, but keep tool schemas themselves JSON-native. Let callers provide JSON Schema-like dictionaries rather than forcing a Pydantic model-to-schema workflow for every tool.

### Tool choice

```python
ToolChoice = (
    Literal["auto", "none", "required"]
    | SpecificToolChoice
)
```

Provider mapping differs:

- OpenAI supports richer tool choice semantics
- Anthropic has its own tool choice shape
- Gemini often treats function-calling mode separately

### Tool call IDs

Define library policy:

- always carry a `tool_call_id`
- if provider omits it, synthesize one
- keep the identifier stable within a provider session so tool results can be correlated correctly
- provider adapters may normalize or constrain IDs to satisfy provider-native requirements, but must preserve correlation with subsequent tool results in the same session

Streaming assembly rule:

- if a provider emits a tool call without arguments, normalize the arguments to `{}` rather than failing generic assembly
- if a provider omits an ID, synthesize one that is stable for the lifetime of that response and usable by later tool results in the same session

### Orphaned tool calls

Before dispatch, validate conversation history:

- if an assistant message contains tool calls with no matching tool results before the next user turn, either
  - raise a validation error, or
  - inject synthetic tool error results if permissive mode is enabled

Default recommendation: raise by default, permissive repair as an opt-in compatibility mode.

Implementation note:

- even in strict mode, provider adapters have an internal normalization pass that filters structurally invalid empty blocks and other provider-known malformed artifacts that do not change conversation semantics
- permissive mode may omit orphaned tool calls from the outbound provider transcript rather than inventing synthetic tool results
- synthetic tool error results remain an opt-in compatibility feature, not the default repair path

Normalization modes:

- `strict`: raise on invalid semantic history
- `permissive`: repair provider-rejectable but non-semantic structural issues

Permissive normalization may:

- drop empty text or reasoning blocks
- omit orphaned tool calls when they would invalidate the next outbound request
- preserve valid user content, visible assistant text, and valid tool results
- avoid inventing synthetic semantic content by default

Default behavior: perform strict semantic validation plus conservative structural normalization before provider serialization.

## Session continuity rules

Sessions are provider-bound.

- A session started with OpenAI stays on OpenAI.
- A session started with Anthropic stays on Anthropic.
- A session started with Gemini stays on Gemini.
- Switching providers starts a new session and does not reuse prior assistant state.

This simplifies the implementation substantially:

- no cross-provider message transformation layer is required
- no foreign tool-call ID normalization is required
- no provider-to-provider reasoning conversion is required
- no cross-provider replay policy matrix is required
- provider adapters only need to support same-provider continuation

Within the same provider, replay preserves:

- response IDs
- reasoning signatures
- thought signatures
- provider-native structured blocks
- provider-native tool call identifiers

This same-provider continuity must be controlled by the provider adapter, not generic client code.

## Usage and cost tracking

### Usage model

```python
class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    completeness: Literal["final", "partial", "none"] = "none"
```

Reasoning tokens may overlap with output on some providers. The provider adapter must document whether `reasoning_tokens` is informational or additive.

### Cost model

```python
class ModelPricing(BaseModel):
    input_per_million: Decimal | None = None
    output_per_million: Decimal | None = None
    cache_read_per_million: Decimal | None = None
    cache_write_per_million: Decimal | None = None
```

```python
class CostBreakdown(BaseModel):
    input_cost: Decimal | None = None
    output_cost: Decimal | None = None
    cache_read_cost: Decimal | None = None
    cache_write_cost: Decimal | None = None
    total_cost: Decimal | None = None
```

### Cost rules

- compute cost only when model pricing is known
- keep pricing in a registry, not scattered in providers
- allow provider-specific multipliers such as premium service tiers
- do not round until final presentation

Additional rule:

- if usage completeness is `partial`, cost is treated as partial as well and surfaced as best-effort rather than authoritative

### Session aggregation

The implementation provides helper utilities:

- `accumulate_usage(existing, new)`
- `estimate_cost(model, usage)`
- `ConversationUsageTracker`

## Model registry design

The registry is data-driven.

Required capabilities:

- register built-in models
- register custom models at runtime
- override base URLs for compatible endpoints
- override pricing
- override compatibility flags

Built-in model metadata, including pricing, is maintained as generated registry data rather than hardcoded ad hoc inside provider adapters.

Registry maintenance requirements:

- keep a checked-in generated model registry used at runtime
- refresh it with a project script, for example `scripts/generate-models.py`
- allow that script to pull from external model catalogs and provider model-list endpoints
- ensure runtime operation does not depend on live pricing fetches

Required upstream source for base metadata:

- `https://models.dev/api.json`

Other useful optional sources may include:

- provider-native model listing endpoints
- aggregator catalogs such as OpenRouter
- internal curated override files for corrections, aliases, and provider-specific capability flags

The generation pipeline normalizes upstream differences in:

- pricing units
- cache read/write pricing
- capability naming
- modality declarations
- context and output limits
- provider/model aliases

Local override files always win over fetched data so the library can correct bad upstream metadata without waiting for third-party updates.

Example:

```python
registry.register(
    ModelSpec(
        provider="openai",
        model="gpt-4.1-mini",
        api_family="openai-responses",
        supports_reasoning=True,
        supports_tools=True,
        supports_images=True,
        pricing=ModelPricing(...),
    )
)
```

## Provider-specific option bags

The common API remains narrow. Provider extras go through `provider_options`.

Examples:

- OpenAI: `{"service_tier": "priority"}`
- Anthropic: `{"interleaved_thinking": True}`
- Gemini: `{"project": "my-project", "location": "us-central1"}`

Do not expose these provider-specific knobs as first-class parameters on `generate()` or `stream()`.

## Validation rules

Validate before network I/O:

- tool names must be non-empty and provider-safe
- tool result references must match a preceding tool call
- image blocks must include mime type and data
- response format requests must be compatible with the target model
- unsupported modalities fail early

Validation happens in two stages:

- common validation in client layer
- provider validation inside adapter for vendor-specific constraints

Validation policy:

- perform strict semantic validation first
- then run a conservative structural normalization pass before provider serialization
- reserve permissive semantic repair for explicit opt-in modes only

Examples of structural normalization that may run even outside permissive semantic repair:

- removing empty content blocks known to trigger provider errors
- applying provider-safe tool-call ID normalization
- normalizing missing tool-call argument payloads to `{}`

Image validation rules:

- image blocks must include both `mime_type` and non-empty base64 `data`
- only image media types supported by the target provider are accepted
- mixed text and image block ordering must be preserved during serialization
- if the target model does not support image input, validation fails early rather than silently dropping image blocks
- tool result images must be preserved when the target provider supports vision-capable tool-result replay

Pydantic is the primary mechanism for common validation. Use model validators and field validators for:

- message/content union discrimination
- token and temperature range checks
- tool schema shape checks
- provider option normalization where safe

Provider adapters still perform vendor-specific validation that cannot be expressed cleanly at the shared model layer.

## Pydantic guidance

Use Pydantic for all public request and response models.

Required rules:

- public surface models inherit from `BaseModel`
- immutable registry metadata uses `ConfigDict(frozen=True)`
- request option models that hold non-Pydantic runtime objects use `arbitrary_types_allowed=True`
- message and content block hierarchies use discriminated unions
- serialization uses `model_dump()` and `model_dump_json()`
- parsing from persisted conversation state uses `model_validate()` and `model_validate_json()`

Internal hot-path streaming assemblers may use plain dicts or lightweight internal objects if profiling shows Pydantic overhead in token-by-token assembly. The finalized public objects returned to callers remain Pydantic models.

## Retry and cancellation

### Cancellation

Rely on task cancellation plus `aiohttp` stream closure. Ensure provider streams stop promptly when the caller cancels iteration.

### Retry policy

Do not retry streaming requests automatically by default. Streaming retries are often not safe.

A future retry helper may be added only for:

- connection errors before first byte
- 429 and 5xx for non-streaming requests

## Observability

Add lightweight hooks:

- request start
- response headers received
- first token latency
- request end
- error

These can be callbacks or a simple event sink. Do not add a logging framework dependency in the core.

## Security considerations

- never log auth headers or raw tokens
- never serialize provider-private reasoning signatures unless the caller explicitly persists full responses
- redact image payloads and large tool outputs in debug traces
- keep auth state outside model objects where possible

## Compatibility and extensibility

The design supports future providers if they can map onto the common abstractions:

- OpenAI-compatible endpoints
- Azure-hosted OpenAI-style APIs
- Bedrock-hosted vendor models
- local model servers

The common model is driven by the initial three providers, not by every edge case from day one.

## Minimal implementation plan

Development workflow:

- use `uv sync` to install dependencies
- use `uv run ...` for project commands
- use `uv run pytest` for validation, preferring targeted test paths while iterating

Phase 1:

- core types
- `aiohttp` transport
- SSE parser
- OpenAI Responses provider
- ChatGPT/Codex subscription provider
- Anthropic Messages provider
- Gemini provider
- OpenRouter provider
- unified streaming events
- usage and cost calculation

Phase 2:

- runtime model registry
- provider capability flags
- stricter validation
- auth registry and token refresh callbacks
- conversation usage tracker

Phase 3:

- custom compatible endpoints
- optional permissive history repair
- richer JSON mode support
- optional extras for advanced auth flows

## Near-term improvement plan

The current implementation direction is correct, but provider confidence should be raised systematically rather than provider by provider in an ad hoc way.

Priority 1: provider confidence baseline

- define a shared adapter checklist and require each provider to satisfy it before being treated as production-ready
- add a provider test matrix covering request construction, streaming success, streaming failure, incomplete termination, tool-call round trips, usage extraction, and multimodal replay where supported
- document provider-specific replay requirements in one place so continuation behavior is intentional rather than incidental

Priority 2: OpenAI-family parity and cleanup

- align OpenAI request defaults intentionally, especially tool strictness, prompt caching behavior, reasoning defaults, and storage behavior
- expand OpenAI Responses stream coverage to include all reasoning-summary and terminal event variants used by the upstream protocol
- ensure finish-reason normalization reflects tool-use, incomplete, content-filter, cancelled, and error outcomes consistently
- tighten OpenAI same-provider replay rules for assistant text IDs, reasoning items, and tool-call metadata so replay depends on explicit preserved metadata rather than synthesized placeholders where possible

Priority 3: cross-provider reasoning confidence

- document exactly which reasoning controls each provider supports and what is silently downgraded versus rejected
- standardize how reasoning summaries, hidden reasoning artifacts, and encrypted replay material are stored in `provider_meta` and `protocol_state`
- require reasoning-specific tests for providers that claim `supports_reasoning`

Priority 4: compatibility and observability

- move important provider option defaults out of incidental adapter logic into documented compatibility helpers or model capability data
- add structured debug metadata for request id, response id, replay handles, and provider-specific stream anomalies
- document which provider options are stable API and which are experimental escape hatches

Completion criteria for a provider adapter:

- deterministic request serialization for supported features
- explicit validation or rejection of unsupported common features
- complete terminal stream handling for the selected upstream protocol
- documented replay policy and preserved metadata requirements
- passing targeted tests for the provider confidence matrix

## Required implementation defaults

- The implementation uses `aiohttp.ClientSession` with connection pooling and a reusable client instance.
- The implementation prefers streaming APIs for all providers.
- The implementation uses OpenAI Responses, ChatGPT/Codex subscription-backed access, Anthropic Messages, Gemini generateContent, and OpenRouter as the first-class initial provider set.
- The implementation keeps provider-native replay metadata only within the same provider session.
- The implementation keeps auth pluggable and request-scoped.
- The implementation tracks cost in the model registry, not in provider adapters.

## Final recommendation

Implement the library as a small layered system:

- a stable provider-agnostic request and response model
- a narrow async transport built on `aiohttp`
- one adapter per provider
- a pluggable auth layer
- a model registry with pricing and capability metadata

This structure is sufficient for the initial provider set without vendor SDKs, and it leaves room for future compatible providers without forcing major API redesign.
For phase 1, that specifically means OpenAI API, OpenAI Codex via ChatGPT subscription, Anthropic API, Anthropic subscription access, Gemini API, and OpenRouter.


## Deferred and unresolved items

This section categorizes items that are intentionally deferred, implementation-defined, or pending future expansion. These items are not part of the phase 1 required behavior unless explicitly stated elsewhere.

### Deferred to a later phase

- normalized assistant-generated image output
- Vertex AI support
- Amazon Bedrock support
- custom compatible endpoint families beyond the initial provider set
- richer JSON mode beyond the required structured-output support described in this document
- interactive OAuth login UX in the core package

### Provider-specific and implementation-defined

- the exact OpenAI WebSocket `auto` transport selection heuristic
- internal connection pooling and reuse strategy for provider sessions
- the exact storage format of generated model registry metadata
- the exact local override file format used by model registry generation

### Must be clarified during implementation if encountered

- provider-native image MIME subtype restrictions not documented in upstream APIs
- provider-native limits for image size, count, or combined multimodal payload size
- any provider response shape that introduces new opaque replay artifacts not representable by current `provider_meta` or `protocol_state`

When an unresolved behavior is encountered during implementation, the implementation must preserve correctness first, document the issue, and add the provider-specific handling to this specification before broadening the public API.
