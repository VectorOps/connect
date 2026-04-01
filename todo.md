# Implementation Plan

## Status snapshot

- The core package is implemented under `src/connect/` with typed public models, transport helpers, registry/model loading, and the async client API.
- Implemented providers: OpenAI, OpenRouter, ChatGPT, Anthropic, and Gemini.
- Implemented auth layers include API-key auth, bearer/query auth helpers, dynamic auth routing, environment resolution, and OAuth2 credential storage/refresh flows for ChatGPT.
- Built-in generated model metadata already includes OpenAI, ChatGPT, Anthropic, Gemini, and OpenRouter model records in `src/connect/data/models.json`.
- Existing tests cover transport helpers, OpenAI, Gemini, and Anthropic provider behavior, live integration paths for OpenAI/Gemini/Anthropic/ChatGPT, and ChatGPT credentials.
- The initial provider set is now implemented for direct API access. The main remaining work is usage helpers, broader provider-confidence coverage, observability hooks, and model-registry generation tooling.

## Note

- This file began as a forward-looking implementation plan. The architectural sections below are still useful, but the repository is now materially ahead of the original bootstrap-state notes.

## Current repository state

- `docs/design.md` remains the implementation spec.
- The package already contains the shared client/runtime layers: `client.py`, `types.py`, `exceptions.py`, `registry.py`, `models.py`, `auth.py`, `auth_router.py`, transport modules, and credential helpers.
- Provider modules currently implemented are `openai.py`, `openrouter.py`, `chatgpt.py`, `anthropic.py`, and `gemini.py`.
- OAuth2 credential management is present under `src/connect/credentials/`, including ChatGPT login/refresh support.
- The checked-in generated model registry already ships data for the initial provider set, including Gemini and Anthropic model metadata.
- The test suite already includes transport coverage, provider coverage for OpenAI/Gemini/Anthropic, ChatGPT credential coverage, and live integration coverage for OpenAI/Gemini/Anthropic/ChatGPT.
- Not yet implemented from the target layout/spec are `src/connect/usage.py`, observability hooks, Anthropic OAuth credential management, and a checked-in model-registry generation script.

## Guidelines

- Keep provider registration data-driven instead of hardcoding dispatch logic throughout the codebase.
- Maintain generated model metadata in a checked-in artifact plus a refresh script rather than querying model catalogs at runtime.
- Treat streaming, tool-call assembly, reasoning blocks, image handling, and provider quirks as first-class test areas.
- Isolate provider-specific request building and event normalization behind per-provider adapters.
- Add or update automated tests for every code change when there is a reasonable place to cover the behavior.
- Use `uv` for dependency management and command execution.

## Implementation goals

- Build a pure-Python, async-first LLM client under `src/connect/`.
- Implement the initial provider set from the design: OpenAI, ChatGPT, Anthropic, Gemini, and OpenRouter.
- Keep transport, auth, provider mapping, model registry, usage accounting, and public models separated.
- Make streaming the canonical execution path and derive non-streaming generation from it.

## Proposed delivery strategy

Deliver the library in tightly scoped slices so the package is usable early and expands safely:

1. Package skeleton and shared public models
2. Transport, exceptions, and streaming assembly
3. Registry and model metadata
4. Provider adapters one by one
5. Client API and top-level helpers
6. Validation, cost tracking, and observability hooks
7. Test coverage and packaging polish

## Target package layout

Create the layout required by the design:

```text
src/connect/
    __init__.py
    client.py
    auth.py
    types.py
    models.py
    usage.py
    exceptions.py
    registry.py
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
```

If OpenAI WebSocket support is included in the first pass, add an internal optional transport module under `transport/` rather than exposing a separate public API.

## Workstream 1: package and dependency setup

1. Update `pyproject.toml` dependencies:
   - required: `aiohttp`, `pydantic`
   - optional extra: `websockets`
   - dev dependencies if this repo adopts them: `pytest`, `pytest-asyncio`, and optionally `anyio`
2. Decide the supported Pydantic major version and code to it consistently.
3. Add package exports in `src/connect/__init__.py` for the public client, request/response models, auth helpers, and registry helpers.
4. Keep the package typed with the existing `py.typed` marker.

## Workstream 2: public data model foundation

Implement `src/connect/types.py` first because every other layer depends on it.

### Required model groups

- content blocks: `TextBlock`, `ImageBlock`, `ReasoningBlock`, `ToolCallBlock`
- messages: `UserMessage`, `AssistantMessage`, `ToolResultMessage`
- request models: `GenerateRequest`, `RequestOptions`, `ReasoningConfig`, `ToolSpec`, `SessionHints`, `ResponseFormat`, `SpecificToolChoice`
- response and event models: `AssistantResponse`, `Usage`, `ErrorInfo`, all stream event types, `FinishReason`
- model metadata: `ModelPricing`, `ModelSpec`

### Key rules to encode immediately

- use discriminated unions for messages and content blocks
- freeze registry metadata models where required
- make request option models accept runtime objects like `aiohttp.ClientTimeout`
- preserve `provider_meta`, `protocol_meta`, `annotations`, and `protocol_state`
- normalize `ImageBlock.data` as base64 text without a data URL prefix
- include `Usage.completeness`

### Validation to include in phase 1

- reject empty or malformed image blocks
- validate tool names and schema presence
- ensure tool results reference a prior tool call when validation has enough context
- reject unsupported multimodal usage before dispatch when model capabilities are known

## Workstream 3: exceptions, transport, and streaming primitives

Implement the shared runtime pieces next.

### `exceptions.py`

Add the normalized exception hierarchy from the design:

- `AuthenticationError`
- `RateLimitError`
- `ContextLengthError`
- `ProviderProtocolError`
- `TransientProviderError`
- `PermanentProviderError`

Also define a small common base exception that stores normalized `ErrorInfo`.

### `transport/http.py`

Build a narrow wrapper around `aiohttp.ClientSession` responsible for:

- client ownership versus caller-provided client reuse
- request creation
- base URL handling
- header merging
- timeout normalization
- async streaming helpers
- conversion of transport-layer failures into library exceptions

### `transport/sse.py`

Implement an internal SSE parser over line iteration that:

- handles multi-line `data:` frames
- ignores comments and blank keepalive lines
- yields decoded payload strings or structured frame objects
- recognizes terminal `[DONE]`

### `transport/json_stream.py`

Implement incremental JSON decoding for Gemini streaming responses.

### Stream assembly support

Add internal helpers that let providers assemble partial text, reasoning, and tool-call argument streams into the final `AssistantResponse` while emitting normalized events in order.

## Workstream 4: authentication layer

Implement `src/connect/auth.py` independently from providers.

### Required auth primitives

- `AuthStrategy` protocol
- `NoAuth`
- `HeaderAPIKeyAuth`
- `BearerTokenAuth`
- `QueryAPIKeyAuth`
- `CallableTokenAuth`
- `RefreshingOAuthAuth`
- `CompositeAuth`

### Additional helpers

- `AuthRegistry` for provider-level defaults
- explicit environment helper functions for OpenAI, Anthropic, Gemini/Google, and OpenRouter
- token/result container type for refreshing OAuth flows
- ChatGPT-specific bearer auth wrapper carrying account ID or exposing a way to derive it safely

### Scope control

- implement token application and refresh support in phase 1
- do not implement interactive browser login in the core package yet
- design the module so a companion CLI login package can plug into it later

## Workstream 5: registry and model metadata

Implement `registry.py` and `models.py` as data-driven infrastructure.

### `registry.py`

Provide:

- `ProviderRegistry`
- `ModelRegistry`
- registration and lookup by `(provider, model)`
- safe bare-model resolution with ambiguity errors
- `get_model()` and `list_models()` helpers
- use `provider/model` as the canonical combined identifier form when a single string is needed

### `models.py`

Provide:

- built-in `ModelSpec` instances or a generated data load path
- pricing metadata structures
- capability flags and protocol defaults
- room for runtime overrides

### Registry generation plan

Add a future script such as `scripts/generate-models.py` that:

- fetches base metadata from `https://models.dev/api.json`
- normalizes provider/model capability and pricing data
- applies local overrides
- writes a checked-in generated module or JSON asset consumed at runtime

For the first usable cut, seed a hand-maintained built-in registry for the required providers, then replace it with generated data once the runtime shape is stable.

## Workstream 6: provider abstraction layer

Implement `providers/base.py` before any concrete provider.

### Base abstractions

- provider protocol / adapter interface from the design
- shared request normalization helpers
- shared event assembly helpers
- shared stop-reason and usage normalization helpers
- common provider-side validation hooks

### Common provider responsibilities

- serialize validated unified requests into provider-native payloads
- apply auth and required headers
- decode provider stream formats
- emit exactly one terminal `response_end` or `error`
- preserve same-provider replay metadata in `protocol_state` and `provider_meta`

## Workstream 7: provider implementation order

Implement providers in an order that maximizes reuse.

### 7.1 OpenAI provider

Build `providers/openai.py` first because it defines much of the common streaming and tool-call behavior.

Required scope:

- Responses API over HTTP SSE in phase 1
- request mapping for messages, tools, tool choice, reasoning, response format, and provider options
- usage parsing including cached token fields when available
- preservation of response IDs and continuation state
- image-aware tool result serialization

Defer WebSocket transport until the SSE path and provider session model are stable, unless there is a hard requirement to ship it immediately.

### 7.2 OpenRouter provider

Build `providers/openrouter.py` next by reusing the OpenAI-compatible mapping path while isolating:

- base URL differences
- auth defaults
- routing-specific provider options
- conservative compatibility flags

### 7.3 ChatGPT provider

Build `providers/chatgpt.py` after OpenAI so it can reuse message conversion and event normalization but keep separate:

- base URL
- OAuth/account-header logic
- session-specific provider options
- provider-specific retries before first byte
- subscription-specific error classification

### 7.4 Anthropic provider

Build `providers/anthropic.py` with its own SSE event mapping and content conversion.

Required scope:

- Messages API
- top-level system prompt handling
- tool definitions and tool choice mapping
- thinking/reasoning translation
- mixed text/image ordering preservation for user and tool result content
- reasoning signature preservation for same-provider replay

### 7.5 Gemini provider

Build `providers/gemini.py` last among the initial set because it uses a distinct chunked JSON stream shape.

Required scope:

- generateContent-style streaming
- content/parts mapping
- tool declaration mapping
- thought signature preservation
- synthesized tool-call IDs when the provider omits them
- image-preserving serialization for user messages and tool results

## Workstream 8: client API and stream handle

Implement `src/connect/client.py` once at least one provider works end to end.

### Public client scope

- `AsyncLLMClient`
- `generate()` that consumes `stream()` internally
- `stream()` returning a `StreamHandle`
- lifecycle support for internally owned `aiohttp.ClientSession`

### `StreamHandle` responsibilities

- async iteration over normalized `StreamEvent`s
- assembly of the final `AssistantResponse`
- `final_response()` support even after partial or full iteration
- correct terminal behavior on success and error

### Top-level module helpers

Expose from `__init__.py`:

- `generate`
- `stream`
- `get_model`
- `list_models`
- key public models and auth strategies

## Workstream 9: usage, pricing, and observability

Implement `usage.py` after the core request path exists.

### Required features

- `accumulate_usage(existing, new)`
- `estimate_cost(model, usage)`
- `ConversationUsageTracker`
- handling of `Usage.completeness` when estimating cost

### Observability hooks

Add lightweight optional callbacks or an event sink for:

- request start
- response headers received
- first token latency
- request end
- error

Do not add a logging framework dependency.

## Workstream 10: validation and normalization policy

Implement validation in two layers.

### Client-layer validation

- request shape validation through Pydantic
- tool result to tool call correlation checks
- modality checks against `ModelSpec`
- response-format compatibility checks when capability flags exist

### Provider-layer validation and normalization

- provider-safe tool-call ID normalization
- empty-block filtering where upstream APIs reject empty content
- provider-specific reasoning disable semantics
- normalization of missing streamed tool arguments to `{}`

Default mode should stay strict for semantic errors, with only conservative structural normalization enabled automatically.

## Workstream 11: tests

Because the main repo currently has no tests, establish test structure early once the first provider path is in place.

### Suggested test layout

```text
tests/
    test_types.py
    test_registry.py
    test_usage.py
    transport/
        test_sse.py
        test_json_stream.py
    providers/
        test_openai.py
        test_openrouter.py
        test_chatgpt.py
        test_anthropic.py
        test_gemini.py
    test_client.py
```

### Minimum test categories

- Pydantic model validation and serialization
- SSE frame decoding
- incremental JSON chunk decoding
- stream assembly into final responses
- tool-call delta assembly and final argument parsing
- text, reasoning, and tool-call event ordering
- usage normalization and completeness handling
- registry ambiguity errors
- provider request serialization for multimodal input and tool-result images
- provider error mapping for auth, rate limits, context limits, protocol failures, and retryable failures

### Testing strategy

- use transport mocks rather than live API calls for the default suite
- build representative recorded provider payload fixtures
- add focused tests around known edge cases from the design, especially orphaned tool calls, empty blocks, mixed image/text ordering, and partial usage

## Workstream 12: incremental milestones

Use these milestones to keep the implementation shippable:

### Milestone A: package foundation

- dependencies configured
- public types implemented
- exceptions and HTTP transport implemented
- SSE and JSON stream decoders implemented

### Milestone B: first usable provider

- OpenAI SSE streaming works end to end
- `AsyncLLMClient.stream()` and `generate()` work with one provider
- usage parsing and final response assembly are in place
- base tests pass

### Milestone C: provider family expansion

- OpenRouter and ChatGPT implemented on top of shared OpenAI-compatible behavior
- Anthropic implemented with SSE mapping
- Gemini implemented with JSON streaming

Status: complete for the direct API provider set.

### Milestone D: production hardening

- registry/model metadata refined
- validation expanded
- usage/cost helpers finalized
- observability hooks added
- test coverage broadened across error cases and multimodal flows

Status: in progress. Validation and provider coverage have advanced, but usage helpers, observability hooks, dedicated ChatGPT/OpenRouter adapter tests, and registry-generation tooling are still outstanding.

## Recommended implementation order by file

1. `pyproject.toml`
2. `src/connect/types.py`
3. `src/connect/exceptions.py`
4. `src/connect/transport/http.py`
5. `src/connect/transport/sse.py`
6. `src/connect/transport/json_stream.py`
7. `src/connect/auth.py`
8. `src/connect/registry.py`
9. `src/connect/models.py`
10. `src/connect/providers/base.py`
11. `src/connect/providers/openai.py`
12. `src/connect/client.py`
13. `src/connect/providers/openrouter.py`
14. `src/connect/providers/chatgpt.py`
15. `src/connect/providers/anthropic.py`
16. `src/connect/providers/gemini.py`
17. `src/connect/usage.py`
18. `src/connect/__init__.py`
19. `tests/...`
20. `scripts/generate-models.py`

## Risks and design checkpoints

- The design is broad; avoid implementing all provider edge cases before the shared abstractions are proven by one provider.
- ChatGPT and OpenAI should share internals where possible, but remain separate adapters because auth, headers, and retry rules differ.
- Gemini should not be forced into the SSE abstraction; keep its chunked JSON parser separate.
- Usage totals and reasoning token semantics vary by provider; store normalized fields without overpromising perfect additivity.
- Same-provider replay metadata must be preserved without making cross-provider continuation a goal.
- Optional WebSocket support should remain an additive feature behind an extra dependency.

## Definition of done for the first full implementation pass

- The required package layout exists under `src/connect/`.
- The public client can stream and generate against OpenAI, ChatGPT, Anthropic, Gemini, and OpenRouter through one normalized API.
- Public request, response, message, content, usage, and stream event models are implemented with Pydantic.
- Auth is request-scoped and pluggable.
- Usage and cost helpers exist and respect usage completeness.
- Provider-specific replay metadata is preserved for same-provider continuation.
- Automated tests cover the core transport, registry, client, and provider serialization/streaming logic.
- Model metadata is registry-driven, with a path toward generated checked-in metadata.

Current gap to this bar: provider implementation is largely complete, but the repository still needs `usage.py`, fuller provider-confidence coverage for ChatGPT/OpenRouter and broader client/registry tests, observability hooks, and model-registry generation tooling.

## Nice-to-have items after the first implementation pass

- OpenAI WebSocket transport
- generated model registry refresh script wired to checked-in output
- permissive semantic repair mode
- richer JSON mode support
- companion CLI auth/login module for OAuth-backed providers