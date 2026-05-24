"""Layer 5: tool-arg validation.

Before the agent invokes a tool with arguments the model produced, we
validate those arguments against a registered schema and either:

  * raise ``ToolArgError`` with an LLM-friendly retry hint (the model
    can read the hint and try again, without burning a tool execution
    cycle on broken input), or
  * return the (optionally coerced) args ready for the tool call.

Schemas are dict shapes — same vocabulary as JSON Schema "required" +
"properties.type", deliberately small so the layer has zero dependency
on the optional jsonschema package.  When ``jsonschema`` IS installed
(the ``[schema]`` extra), it is used as a deeper second pass after the
cheap checks succeed.

Patterned after MukundaKatta/agentvet, MukundaKatta/agentvet-rs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

try:
    import jsonschema as _jsonschema  # type: ignore
except ImportError:  # pragma: no cover — exercised when the extra isn't installed
    _jsonschema = None


# Map of JSON Schema type names → Python types we accept.  ``int`` is also
# valid for ``"number"``, because that is what models tend to emit and we
# don't want to fail a number constraint on a perfectly fine integer.
_PY_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
    "null": (type(None),),
}


class ToolArgError(Exception):
    """Raised when tool args do not match the registered schema.

    Carries a ``hint`` field meant to be fed back to the model verbatim:
    naming the offending field and what was expected.  Holding it as a
    structured attribute (not just the message string) keeps the audit
    log usable and lets the agent inject the hint into a repair prompt
    without re-parsing it.
    """

    def __init__(
        self,
        message: str,
        *,
        tool: str,
        field: str | None = None,
        expected: str | None = None,
        got: str | None = None,
    ) -> None:
        super().__init__(message)
        self.tool = tool
        self.field = field
        self.expected = expected
        self.got = got
        self.hint = self._build_hint()

    def _build_hint(self) -> str:
        # Short, model-readable.  No periods at the end so the agent can
        # interpolate it into a longer sentence without double punctuation.
        if self.field and self.expected:
            return (
                f"Tool '{self.tool}' rejected arguments: field '{self.field}' "
                f"expected {self.expected}, got {self.got or '(missing)'}"
            )
        if self.expected:
            # No field (e.g. unknown-tool case) — still surface the expected
            # value so the model sees the registry options.
            return (
                f"Tool '{self.tool}' rejected: {self} (expected {self.expected})"
            )
        return f"Tool '{self.tool}' rejected arguments: {self}"


@dataclass
class ToolVet:
    """Registry of (tool_name → schema) used to gatekeep tool calls.

    Usage:

        vet = ToolVet()
        vet.register("fetch_url", {
            "type": "object",
            "required": ["url"],
            "properties": {"url": {"type": "string"}, "timeout": {"type": "number"}},
        })

        try:
            args = vet.check("fetch_url", model_produced_args)
        except ToolArgError as exc:
            audit("tool.args.invalid", {"hint": exc.hint})
            raise

    Unregistered tool names raise immediately so an attacker prompt
    cannot smuggle a new tool name past the layer.
    """

    schemas: dict[str, dict] = field(default_factory=dict)
    coerce: Callable[[str, Any], Any] | None = None

    def register(self, tool: str, schema: dict) -> None:
        if not isinstance(schema, dict):
            raise TypeError("schema must be a dict")
        self.schemas[tool] = schema

    def known(self) -> list[str]:
        return sorted(self.schemas.keys())

    def check(self, tool: str, args: Any) -> Any:
        schema = self.schemas.get(tool)
        if schema is None:
            raise ToolArgError(
                f"unknown tool: {tool}",
                tool=tool,
                field=None,
                expected=f"one of {self.known()}",
                got=tool,
            )
        # Cheap top-level checks first.
        if schema.get("type") == "object":
            if not isinstance(args, dict):
                raise ToolArgError(
                    "tool args must be an object",
                    tool=tool,
                    field=None,
                    expected="object",
                    got=type(args).__name__,
                )
            self._check_required(tool, args, schema)
            self._check_properties(tool, args, schema)
        # Optional deeper pass with jsonschema if installed.
        if _jsonschema is not None:
            try:
                _jsonschema.validate(args, schema)
            except _jsonschema.ValidationError as exc:  # pragma: no cover - deps
                path = ".".join(str(p) for p in exc.absolute_path) or None
                raise ToolArgError(
                    str(exc.message),
                    tool=tool,
                    field=path,
                    expected=str(exc.validator_value),
                    got=type(exc.instance).__name__,
                ) from None
        return args

    @staticmethod
    def _check_required(tool: str, args: dict, schema: dict) -> None:
        for required in schema.get("required", []):
            if required not in args:
                raise ToolArgError(
                    f"missing required field: {required}",
                    tool=tool,
                    field=required,
                    expected="present",
                    got="absent",
                )

    @staticmethod
    def _check_properties(tool: str, args: dict, schema: dict) -> None:
        props = schema.get("properties", {})
        for name, spec in props.items():
            if name not in args:
                continue
            expected_type = spec.get("type")
            if expected_type is None:
                continue
            types = _PY_TYPES.get(expected_type)
            if types is None:
                # Schema uses an unknown type — let jsonschema (or the call
                # site) catch it; we don't pretend to handle it cheaply.
                continue
            value = args[name]
            # bool is a subclass of int; if the schema asks for integer/
            # number we explicitly reject True/False so a tool that needs
            # an int doesn't silently accept a flag.
            if expected_type in ("number", "integer") and isinstance(value, bool):
                raise ToolArgError(
                    f"field {name!r} must be {expected_type}, not boolean",
                    tool=tool,
                    field=name,
                    expected=expected_type,
                    got="boolean",
                )
            if not isinstance(value, types):
                raise ToolArgError(
                    f"field {name!r} has wrong type",
                    tool=tool,
                    field=name,
                    expected=expected_type,
                    got=type(value).__name__,
                )
