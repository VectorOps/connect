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


def _normalize_schema_mapping(
    value: typing.Any,
    *,
    path: str,
) -> dict[str, typing.Any]:
    if not isinstance(value, dict):
        raise ToolSchemaError(f"{path} must be an object")

    return {
        str(name): _normalize_tool_schema_node(item, path=f"{path}.{name}")
        for name, item in value.items()
    }


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
            normalized[key] = _normalize_schema_mapping(value, path=f"{path}.properties")
            continue

        if key in {"$defs", "definitions", "patternProperties", "dependentSchemas"}:
            normalized[key] = _normalize_schema_mapping(value, path=f"{path}.{key}")
            continue

        if key in {"items", "additionalItems", "unevaluatedItems"}:
            if isinstance(value, list):
                normalized[key] = [
                    _normalize_tool_schema_node(item_schema, path=f"{path}.items[{index}]")
                    for index, item_schema in enumerate(value)
                ]
            else:
                normalized[key] = _normalize_tool_schema_node(value, path=f"{path}.{key}")
            continue

        if key == "prefixItems":
            if not isinstance(value, list):
                raise ToolSchemaError(f"{path}.prefixItems must be an array")
            normalized[key] = [
                _normalize_tool_schema_node(item, path=f"{path}.prefixItems[{index}]")
                for index, item in enumerate(value)
            ]
            continue

        if key in {"anyOf", "oneOf", "allOf"}:
            if not isinstance(value, list):
                raise ToolSchemaError(f"{path}.{key} must be an array")
            normalized[key] = [
                _normalize_tool_schema_node(item, path=f"{path}.{key}[{index}]")
                for index, item in enumerate(value)
            ]
            continue

        if key == "dependencies":
            if not isinstance(value, dict):
                raise ToolSchemaError(f"{path}.dependencies must be an object")
            normalized[key] = {
                str(name): (
                    _normalize_tool_schema_node(item, path=f"{path}.dependencies.{name}")
                    if isinstance(item, dict)
                    else item
                )
                for name, item in value.items()
            }
            continue

        if key in {
            "additionalProperties",
            "unevaluatedProperties",
            "not",
            "if",
            "then",
            "else",
            "contains",
            "propertyNames",
        }:
            if isinstance(value, dict):
                normalized[key] = _normalize_tool_schema_node(value, path=f"{path}.{key}")
            else:
                normalized[key] = value
            continue

        normalized[key] = value

    is_object_schema = (
        normalized.get("type") == "object"
        or "properties" in normalized
        or "patternProperties" in normalized
        or "required" in normalized
        or "additionalProperties" in normalized
        or "unevaluatedProperties" in normalized
        or "propertyNames" in normalized
        or "dependentSchemas" in normalized
        or "dependencies" in normalized
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