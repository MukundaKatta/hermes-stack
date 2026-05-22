"""Layer 4: structured output enforcement.

Models hallucinate JSON. This layer takes the raw model output, locates
the JSON block (handles ```json fences, leading prose, trailing prose),
validates it against an optional JSON schema, and raises `OutputInvalid`
when it cannot recover. Patterned after MukundaKatta/agentcast-py.

If `jsonschema` is installed, full schema validation runs. Otherwise the
type and required-keys checks fall back to a small built-in that handles
the common cases used in the demo.
"""

from __future__ import annotations

import json
import re
from typing import Any

try:
    import jsonschema  # type: ignore[import-not-found]

    _HAS_JSONSCHEMA = True
except ImportError:  # pragma: no cover - exercised when extra is not installed
    _HAS_JSONSCHEMA = False


class OutputInvalid(Exception):
    """Raised when a model output cannot be cast to the requested shape."""

    def __init__(self, message: str, *, raw: str, reason: str) -> None:
        super().__init__(message)
        self.raw = raw
        self.reason = reason


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def _extract_json_block(raw: str) -> str:
    """Pull a JSON object/array out of a possibly chatty model response."""
    text = raw.strip()
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fall back: first balanced {...} or [...] in the text.
    for opener, closer in (("{", "}"), ("[", "]")):
        first = text.find(opener)
        last = text.rfind(closer)
        if first != -1 and last != -1 and last > first:
            return text[first : last + 1]
    return text


def _minimal_validate(parsed: Any, schema: dict[str, Any]) -> None:
    """Tiny fallback validator used when jsonschema is not installed.

    Handles the common shapes used in the demo:
    - top-level type "object" with `required` and `properties`
    - top-level type "array" with `minItems`
    """
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(parsed, dict):
            raise ValueError(f"expected object, got {type(parsed).__name__}")
        for key in schema.get("required", []):
            if key not in parsed:
                raise ValueError(f"missing required key: {key}")
        for key, sub in schema.get("properties", {}).items():
            if key in parsed and "type" in sub:
                _check_primitive(parsed[key], sub["type"], key)
    elif expected_type == "array":
        if not isinstance(parsed, list):
            raise ValueError(f"expected array, got {type(parsed).__name__}")
        if "minItems" in schema and len(parsed) < schema["minItems"]:
            raise ValueError(f"array too short: {len(parsed)} < {schema['minItems']}")


def _check_primitive(value: Any, type_name: str, key: str) -> None:
    expected = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }.get(type_name)
    if expected is None:
        return
    if not isinstance(value, expected):
        raise ValueError(f"key '{key}' expected {type_name}, got {type(value).__name__}")


def cast_json(raw: str, schema: dict[str, Any] | None = None) -> Any:
    """Parse a JSON block out of `raw` and optionally validate against schema.

    Raises OutputInvalid with `reason` set to one of:
      - "no-json"         (could not locate a JSON block)
      - "parse-error"     (JSON parser rejected it)
      - "schema-error"    (validator rejected it)
    """
    block = _extract_json_block(raw)
    try:
        parsed = json.loads(block)
    except json.JSONDecodeError as exc:
        if not block:
            raise OutputInvalid(
                "no JSON block found in model output",
                raw=raw,
                reason="no-json",
            ) from exc
        raise OutputInvalid(
            f"JSON parse failed: {exc.msg}",
            raw=raw,
            reason="parse-error",
        ) from exc

    if schema is not None:
        try:
            if _HAS_JSONSCHEMA:
                jsonschema.validate(parsed, schema)
            else:
                _minimal_validate(parsed, schema)
        except Exception as exc:  # noqa: BLE001 - re-raise as our typed error
            raise OutputInvalid(
                f"schema validation failed: {exc}",
                raw=raw,
                reason="schema-error",
            ) from exc

    return parsed
