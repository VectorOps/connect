from __future__ import annotations

import copy
import typing


class ToolSchemaError(ValueError):
    pass


def normalize_canonical_tool_schema(schema: dict[str, typing.Any]) -> dict[str, typing.Any]:
    if not isinstance(schema, dict) or not schema:
        raise ToolSchemaError("Tool schemas must be non-empty JSON schema objects")

    normalized = _normalize_tool_schema_node(copy.deepcopy(schema), path="input_schema")
    if normalized.get("type") != "object":
        raise ToolSchemaError("Tool schemas must use an object root schema")
    return normalized


def _normalize_tool_schema_node(node: typing.Any, *, path: str) -> typing.Any:
    if isinstance(node, list):
        return [
            _normalize_tool_schema_node(item, path=f"{path}[{index}]")
            for index, item in enumerate(node)
        ]

    if not isinstance(node, dict):
        return node

    normalized: dict[str, typing.Any] = {}
    for key, value in node.items():
        if key == "properties":
            if not isinstance(value, dict):
                raise ToolSchemaError(f"{path}.properties must be an object")
            normalized[key] = {
                str(property_name): _normalize_tool_schema_node(
                    property_schema,
                    path=f"{path}.properties.{property_name}",
                )
                for property_name, property_schema in value.items()
            }
            continue

        if key == "items":
            if isinstance(value, list):
                normalized[key] = [
                    _normalize_tool_schema_node(item_schema, path=f"{path}.items[{index}]")
                    for index, item_schema in enumerate(value)
                ]
            else:
                normalized[key] = _normalize_tool_schema_node(value, path=f"{path}.items")
            continue

        if key in {"anyOf", "oneOf", "allOf"}:
            if not isinstance(value, list):
                raise ToolSchemaError(f"{path}.{key} must be an array")
            normalized[key] = [
                _normalize_tool_schema_node(item, path=f"{path}.{key}[{index}]")
                for index, item in enumerate(value)
            ]
            continue

        if key in {"additionalProperties", "not", "if", "then", "else", "contains", "propertyNames"}:
            if isinstance(value, dict):
                normalized[key] = _normalize_tool_schema_node(value, path=f"{path}.{key}")
            else:
                normalized[key] = value
            continue

        normalized[key] = value

    is_object_schema = (
        normalized.get("type") == "object"
        or "properties" in normalized
        or "required" in normalized
        or "additionalProperties" in normalized
    )
    if not is_object_schema:
        return normalized

    normalized["type"] = "object"
    properties = normalized.get("properties")
    if properties is None:
        properties = {}
        normalized["properties"] = properties
    elif not isinstance(properties, dict):
        raise ToolSchemaError(f"{path}.properties must be an object")

    required = normalized.get("required")
    if required is None:
        normalized["required"] = []
    else:
        if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
            raise ToolSchemaError(f"{path}.required must be an array of property names")
        deduped_required: list[str] = []
        known_properties = set(properties)
        for item in required:
            if item not in known_properties:
                raise ToolSchemaError(f"{path}.required references unknown property '{item}'")
            if item not in deduped_required:
                deduped_required.append(item)
        normalized["required"] = deduped_required

    if "additionalProperties" not in normalized:
        normalized["additionalProperties"] = False

    return normalized