"""Structured output cast tests."""

import pytest

from hermes_stack.cast import OutputInvalid, cast_json


def test_extracts_json_from_fenced_block() -> None:
    raw = 'Sure! Here is the answer:\n```json\n{"a": 1, "b": "two"}\n```\nLet me know.'
    out = cast_json(raw)
    assert out == {"a": 1, "b": "two"}


def test_extracts_json_when_no_fence() -> None:
    raw = 'okay {"a": 1} thanks'
    out = cast_json(raw)
    assert out == {"a": 1}


def test_raises_when_no_json_block() -> None:
    with pytest.raises(OutputInvalid) as info:
        cast_json("just prose without any json")
    assert info.value.reason in {"no-json", "parse-error"}


def test_raises_on_invalid_json() -> None:
    raw = '```json\n{"a": 1,\n```'
    with pytest.raises(OutputInvalid) as info:
        cast_json(raw)
    assert info.value.reason == "parse-error"


def test_schema_accepts_well_formed_object() -> None:
    schema = {
        "type": "object",
        "required": ["title", "key_points"],
        "properties": {
            "title": {"type": "string"},
            "key_points": {"type": "array"},
        },
    }
    raw = '```json\n{"title": "Hi", "key_points": ["a", "b"]}\n```'
    out = cast_json(raw, schema=schema)
    assert out["title"] == "Hi"


def test_schema_rejects_missing_required_key() -> None:
    schema = {
        "type": "object",
        "required": ["title", "key_points"],
    }
    raw = '```json\n{"title": "Hi"}\n```'
    with pytest.raises(OutputInvalid) as info:
        cast_json(raw, schema=schema)
    assert info.value.reason == "schema-error"
