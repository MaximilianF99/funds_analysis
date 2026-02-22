from __future__ import annotations

from typing import Any


def resolve_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline JSON Schema $defs/$ref so the schema is self-contained.

    Anthropic's tool_use API does not follow $ref pointers, so Pydantic's
    generated JSON schema must be flattened before sending it as input_schema.
    """
    if "$defs" not in schema:
        return schema

    defs = schema.pop("$defs")

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_name = node["$ref"].split("/")[-1]
                return _walk(defs[ref_name])
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(item) for item in node]
        return node

    return _walk(schema)
