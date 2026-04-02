from __future__ import annotations

import pytest
from pydantic import ValidationError

from connect.tool_schema import ToolSchemaError, normalize_canonical_tool_schema
from connect.types import ToolSpec


def test_normalize_canonical_tool_schema_defaults_required_and_additional_properties() -> None:
    schema = normalize_canonical_tool_schema(
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
        }
    )

    assert schema == {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
        },
        "required": [],
        "additionalProperties": False,
    }


def test_normalize_canonical_tool_schema_recursively_normalizes_nested_objects() -> None:
    schema = normalize_canonical_tool_schema(
        {
            "type": "object",
            "properties": {
                "filters": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                    },
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                        },
                    },
                },
            },
        }
    )

    assert schema == {
        "type": "object",
        "properties": {
            "filters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                },
                "required": [],
                "additionalProperties": False,
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
        "required": [],
        "additionalProperties": False,
    }


def test_normalize_canonical_tool_schema_supports_common_json_schema_constructs() -> None:
    schema = normalize_canonical_tool_schema(
        {
            "type": "object",
            "$defs": {
                "filter": {
                    "type": "object",
                    "patternProperties": {
                        "^tag:": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "string"},
                            },
                        }
                    },
                }
            },
            "properties": {
                "filters": {"$ref": "#/$defs/filter"},
                "tuple": {
                    "type": "array",
                    "prefixItems": [
                        {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                            },
                        },
                        {"type": "string"},
                    ],
                },
            },
            "dependentSchemas": {
                "filters": {
                    "properties": {
                        "mode": {"type": "string"},
                    },
                }
            },
        }
    )

    assert schema == {
        "type": "object",
        "$defs": {
            "filter": {
                "type": "object",
                "patternProperties": {
                    "^tag:": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                        },
                        "required": [],
                        "additionalProperties": False,
                    }
                },
                "properties": {},
                "required": [],
                "additionalProperties": False,
            }
        },
        "properties": {
            "filters": {"$ref": "#/$defs/filter"},
            "tuple": {
                "type": "array",
                "prefixItems": [
                    {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                    {"type": "string"},
                ],
            },
        },
        "dependentSchemas": {
            "filters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string"},
                },
                "required": [],
                "additionalProperties": False,
            }
        },
        "required": [],
        "additionalProperties": False,
    }


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ({"type": "string"}, "object root schema"),
        ({"type": "object", "properties": []}, "properties must be an object"),
        ({"type": "object", "properties": {"id": {"type": "string"}}, "required": "id"}, "required must be an array"),
        ({"type": "object", "$defs": []}, r"\$defs must be an object"),
        ({"type": "object", "prefixItems": {}}, "prefixItems must be an array"),
        ({"type": "object", "dependentSchemas": []}, "dependentSchemas must be an object"),
        (
            {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["missing"]},
            "required references unknown property 'missing'",
        ),
    ],
)
def test_normalize_canonical_tool_schema_rejects_invalid_schemas(
    schema: dict,
    message: str,
) -> None:
    with pytest.raises(ToolSchemaError, match=message):
        normalize_canonical_tool_schema(schema)


def test_tool_spec_validates_and_stores_normalized_canonical_schema() -> None:
    spec = ToolSpec(
        name="lookup_status",
        description="Lookup a status string.",
        input_schema={
            "type": "object",
            "properties": {
                "id": {"type": "string"},
            },
        },
    )

    assert spec.input_schema == {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
        },
        "required": [],
        "additionalProperties": False,
    }


def test_tool_spec_external_schema_preserves_original_document() -> None:
    spec = ToolSpec.external(
        name="lookup_status",
        description="Lookup a status string.",
        input_schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": {
                "input": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                }
            },
            "$ref": "#/$defs/input",
        },
    )

    assert spec.schema_mode == "external"
    assert spec.input_schema == {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": {
            "input": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
            }
        },
        "$ref": "#/$defs/input",
    }


def test_tool_spec_rejects_invalid_canonical_schema() -> None:
    with pytest.raises(ValidationError, match="object root schema"):
        ToolSpec(
            name="lookup_status",
            description="Lookup a status string.",
            input_schema={"type": "string"},
        )