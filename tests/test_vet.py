"""Tests for the tool-arg vet layer (layer 5)."""

from __future__ import annotations

import pytest

from hermes_stack import ToolArgError, ToolVet
from hermes_stack.agent import HermesAgent


# ---------------------------------------------------------------------------
# ToolVet primitive
# ---------------------------------------------------------------------------


def _schema_fetch_url() -> dict:
    return {
        "type": "object",
        "required": ["url"],
        "properties": {
            "url": {"type": "string"},
            "timeout": {"type": "number"},
            "follow_redirects": {"type": "boolean"},
        },
    }


def test_vet_accepts_valid_args():
    vet = ToolVet()
    vet.register("fetch_url", _schema_fetch_url())
    out = vet.check("fetch_url", {"url": "https://example.com", "timeout": 5.0})
    assert out == {"url": "https://example.com", "timeout": 5.0}


def test_vet_rejects_unknown_tool():
    vet = ToolVet()
    vet.register("fetch_url", _schema_fetch_url())
    with pytest.raises(ToolArgError) as exc:
        vet.check("delete_db", {})
    assert "unknown tool" in str(exc.value).lower()
    assert "fetch_url" in exc.value.hint  # hint suggests known tools


def test_vet_rejects_missing_required_field():
    vet = ToolVet()
    vet.register("fetch_url", _schema_fetch_url())
    with pytest.raises(ToolArgError) as exc:
        vet.check("fetch_url", {})
    assert exc.value.field == "url"
    assert exc.value.expected == "present"
    assert "url" in exc.value.hint


def test_vet_rejects_wrong_type():
    vet = ToolVet()
    vet.register("fetch_url", _schema_fetch_url())
    with pytest.raises(ToolArgError) as exc:
        vet.check("fetch_url", {"url": 42})
    assert exc.value.field == "url"
    assert exc.value.expected == "string"
    assert exc.value.got == "int"


def test_vet_rejects_bool_when_number_expected():
    """JSON Schema number must not silently swallow a True/False flag."""
    vet = ToolVet()
    vet.register("fetch_url", _schema_fetch_url())
    with pytest.raises(ToolArgError) as exc:
        vet.check("fetch_url", {"url": "https://example.com", "timeout": True})
    assert exc.value.field == "timeout"
    assert exc.value.got == "boolean"


def test_vet_accepts_int_for_number():
    vet = ToolVet()
    vet.register("fetch_url", _schema_fetch_url())
    out = vet.check("fetch_url", {"url": "https://example.com", "timeout": 5})
    assert out["timeout"] == 5


def test_vet_rejects_non_object_args():
    vet = ToolVet()
    vet.register("fetch_url", _schema_fetch_url())
    with pytest.raises(ToolArgError) as exc:
        vet.check("fetch_url", "https://example.com")
    assert exc.value.expected == "object"


def test_vet_known_lists_registered_tools():
    vet = ToolVet()
    vet.register("fetch_url", _schema_fetch_url())
    vet.register("send_email", {"type": "object", "required": ["to"]})
    assert vet.known() == ["fetch_url", "send_email"]


def test_vet_register_rejects_non_dict_schema():
    vet = ToolVet()
    with pytest.raises(TypeError):
        vet.register("fetch_url", "not a dict")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# HermesAgent.call_tool
# ---------------------------------------------------------------------------


def test_agent_call_tool_runs_when_args_valid():
    vet = ToolVet()
    vet.register("add", {
        "type": "object",
        "required": ["a", "b"],
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
    })
    agent = HermesAgent(vet=vet)

    result = agent.call_tool("add", {"a": 2, "b": 3}, lambda *, a, b: a + b)
    assert result == 5


def test_agent_call_tool_rejects_bad_args():
    vet = ToolVet()
    vet.register("add", {
        "type": "object",
        "required": ["a", "b"],
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
    })
    agent = HermesAgent(vet=vet)

    def boom(**_):
        raise AssertionError("tool fn must not run when args are invalid")

    with pytest.raises(ToolArgError):
        agent.call_tool("add", {"a": 1}, boom)


def test_agent_call_tool_requires_vet():
    """Failing closed: no registered vet → no tool execution."""
    agent = HermesAgent(vet=None)
    with pytest.raises(RuntimeError):
        agent.call_tool("anything", {}, lambda **_: None)
