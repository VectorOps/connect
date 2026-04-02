# Tool schema

This project uses a canonical tool schema for custom function tools.

The canonical schema is not any provider's wire format. It is an internal,
vendor-neutral JSON Schema representation for tool input parameters.

## Canonical schema

Each tool declares:

- `name`
- `description`
- `input_schema`

By default, `ToolSpec` treats `input_schema` as canonical and normalizes it.
This is `schema_mode="canonical"`.

The canonical `input_schema` must be a JSON Schema object-root schema.

Example:

```json
{
  "type": "object",
  "properties": {
    "query": { "type": "string" },
    "limit": { "type": "integer" }
  },
  "required": ["query"],
  "additionalProperties": false
}
```

## Normalization rules

Canonical schemas are normalized before being stored on `ToolSpec`.

Rules:

- root schema must be an object schema
- `properties` must be an object when present
- `required` must be an array of declared property names
- `required` defaults to `[]`
- `additionalProperties` defaults to `false`
- nested schemas are normalized recursively through common schema-bearing
  keywords including:
  - `properties`
  - `items`, `prefixItems`, `additionalItems`, `unevaluatedItems`
  - `anyOf`, `oneOf`, `allOf`, `not`, `if`, `then`, `else`, `contains`
  - `additionalProperties`, `unevaluatedProperties`, `propertyNames`
  - `$defs`, `definitions`, `patternProperties`, `dependentSchemas`
  - schema-valued entries inside `dependencies`

This gives the project one internal format independent of provider syntax.

The canonical validator is intentionally practical rather than a full JSON
Schema implementation. It preserves unsupported keywords as-is when they do not
require canonical object normalization, while normalizing the nested schemas it
does understand.

## External schema mode

`ToolSpec` also supports `schema_mode="external"` for cases where the caller
needs to preserve an original provider-oriented or draft-specific JSON Schema
document.

In external mode:

- the schema is stored as provided
- canonical normalization is skipped
- object-root canonical restrictions are not applied
- advanced constructs such as `$schema`, top-level `$ref`, and arbitrary draft
  features are preserved verbatim

Use external mode when you already have a complete JSON Schema document and do
not want this project to rewrite it into the internal canonical form.

## Why this is needed

Providers do not accept the same tool definition format.

Examples:

- OpenAI Responses uses `tools[].parameters`
- Anthropic uses `tools[].input_schema`
- Gemini uses `tools[].functionDeclarations[].parametersJsonSchema`

Even when providers all accept JSON-Schema-like structures, they differ in:

- wrapper field names
- strictness requirements
- optional keywords
- tool choice encoding
- response and streaming event shapes

Because of that, this project separates:

- canonical schema for custom function inputs
- provider-specific request encoding
- provider-specific response decoding

## Provider mappings

### OpenAI Responses

Canonical tool definitions are converted into:

```json
{
  "type": "function",
  "name": "tool_name",
  "description": "Tool description",
  "parameters": { "type": "object", "properties": {}, "required": [], "additionalProperties": false },
  "strict": true
}
```

When `strict=true`, the OpenAI transport applies an additional provider-specific
conversion step. In this repo, object schemas are tightened so that:

- `required` contains every property name
- `additionalProperties` is `false`

This is intentionally transport-specific. It is not part of the canonical
schema itself.

### Anthropic

Canonical tool definitions are converted into:

```json
{
  "name": "tool_name",
  "description": "Tool description",
  "input_schema": { "type": "object", "properties": {}, "required": [], "additionalProperties": false }
}
```

Anthropic uses its own request syntax and its own tool-use content blocks in
responses, but the input parameter schema comes from the same canonical schema.

### Gemini

Canonical tool definitions are converted into:

```json
{
  "functionDeclarations": [
    {
      "name": "tool_name",
      "description": "Tool description",
      "parametersJsonSchema": { "type": "object", "properties": {}, "required": [], "additionalProperties": false }
    }
  ]
}
```

Gemini has its own tool calling configuration and response parts, but the input
schema still originates from the same canonical definition.

## What the canonical schema does not cover

The canonical schema is only for custom function tool input parameters.

It does not define:

- provider-native built-in tools
- provider-specific tool choice flags
- provider-specific streaming events
- provider-specific tool call or tool result wire formats

Examples outside the canonical schema:

- OpenAI built-in tools
- Anthropic agent/tool extensions
- Gemini built-in tools like search or code execution
- provider-native parallel tool call controls

Those should be modeled separately and handled by the relevant transport.

## Recommended mental model

Think of tool calling as three layers:

1. Internal tool definition
   - name
   - description
   - canonical input schema

2. Provider request encoding
   - OpenAI `parameters`
   - Anthropic `input_schema`
   - Gemini `parametersJsonSchema`

3. Provider response decoding
   - OpenAI function-call items/events
   - Anthropic `tool_use` blocks
   - Gemini `functionCall` / `functionResponse` parts

Only layer 1 is canonical. Layers 2 and 3 are transport-specific.