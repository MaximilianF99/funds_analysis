from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolResult:
    """Unified result from a forced tool-use LLM call."""

    tool_name: str
    tool_input: dict[str, Any]


class LLMClient(Protocol):
    """Provider-agnostic interface for LLM calls with forced tool use."""

    @property
    def model(self) -> str: ...

    def call_with_tool(
        self,
        *,
        system: str,
        user_message: str,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
        max_tokens: int = 8192,
    ) -> ToolResult:
        """Send a single-turn request and force the model to call the specified tool."""
        ...


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------


class AnthropicClient:
    """LLMClient implementation backed by the Anthropic Messages API."""

    def __init__(self, model: str):
        from anthropic import Anthropic

        self._model = model
        self._client = Anthropic()

    @property
    def model(self) -> str:
        return self._model

    def call_with_tool(
        self,
        *,
        system: str,
        user_message: str,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
        max_tokens: int = 8192,
    ) -> ToolResult:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
            tools=[
                {
                    "name": tool_name,
                    "description": tool_description,
                    "input_schema": input_schema,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return ToolResult(tool_name=block.name, tool_input=block.input)

        raise ValueError(
            f"No '{tool_name}' tool call in Anthropic response "
            f"(stop_reason={response.stop_reason})"
        )


# ---------------------------------------------------------------------------
# Gemini adapter
# ---------------------------------------------------------------------------

_GEMINI_MAX_RETRIES = 3


class GeminiClient:
    """LLMClient implementation backed by the Google Gemini API."""

    _UNSUPPORTED_SCHEMA_KEYS = frozenset({
        "additionalProperties",
        "title",
        "$schema",
    })

    def __init__(self, model: str):
        from google import genai

        self._model = model
        self._client = genai.Client()

    @property
    def model(self) -> str:
        return self._model

    @classmethod
    def _sanitize_schema(
        cls, schema: Any, *, _is_properties_dict: bool = False,
    ) -> Any:
        """Strip JSON Schema keys that Gemini's function calling API does not support.

        Preserves property *names* inside ``"properties"`` dicts (e.g. a
        Pydantic field literally called ``title``) while still stripping
        the JSON Schema *keyword* ``title`` everywhere else.
        """
        if isinstance(schema, dict):
            result: dict[str, Any] = {}
            for k, v in schema.items():
                if not _is_properties_dict and k in cls._UNSUPPORTED_SCHEMA_KEYS:
                    continue
                result[k] = cls._sanitize_schema(
                    v, _is_properties_dict=(k == "properties"),
                )
            return result
        if isinstance(schema, list):
            return [cls._sanitize_schema(item) for item in schema]
        return schema

    def call_with_tool(
        self,
        *,
        system: str,
        user_message: str,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
        max_tokens: int = 8192,
    ) -> ToolResult:
        from google.genai import types

        clean_schema = self._sanitize_schema(input_schema)

        tool = types.Tool(function_declarations=[
            {
                "name": tool_name,
                "description": tool_description,
                "parameters": clean_schema,
            }
        ])

        config = types.GenerateContentConfig(
            tools=[tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=[tool_name],
                )
            ),
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=0.0,
        )

        last_error: Exception | None = None
        for attempt in range(1, _GEMINI_MAX_RETRIES + 1):
            response = self._client.models.generate_content(
                model=self._model,
                contents=user_message,
                config=config,
            )

            candidates = response.candidates or []
            if not candidates:
                last_error = ValueError(
                    f"Gemini returned no candidates for '{tool_name}'"
                )
                logger.warning(
                    "Attempt %d/%d: no candidates for '%s'",
                    attempt, _GEMINI_MAX_RETRIES, tool_name,
                )
                continue

            candidate = candidates[0]
            parts = (candidate.content and candidate.content.parts) or []

            for part in parts:
                if part.function_call:
                    return ToolResult(
                        tool_name=part.function_call.name,
                        tool_input=dict(part.function_call.args),
                    )

            finish = candidate.finish_reason
            last_error = ValueError(
                f"No '{tool_name}' function call in Gemini response "
                f"(finish_reason={finish})"
            )
            logger.warning(
                "Attempt %d/%d: no function call for '%s' (finish_reason=%s)",
                attempt, _GEMINI_MAX_RETRIES, tool_name, finish,
            )

        raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDER_PREFIXES: dict[str, type] = {
    "claude-": AnthropicClient,
    "gemini-": GeminiClient,
}


def create_client(model: str) -> LLMClient:
    """Instantiate the correct LLMClient based on the model name prefix."""
    for prefix, cls in _PROVIDER_PREFIXES.items():
        if model.startswith(prefix):
            logger.info("Using %s for model '%s'", cls.__name__, model)
            return cls(model)

    supported = ", ".join(f"{p}*" for p in _PROVIDER_PREFIXES)
    raise ValueError(
        f"Unknown model '{model}'. Supported prefixes: {supported}"
    )


# ---------------------------------------------------------------------------
# Schema utilities
# ---------------------------------------------------------------------------


def resolve_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline JSON Schema $defs/$ref so the schema is self-contained.

    LLM tool-use APIs typically do not follow $ref pointers, so Pydantic's
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
