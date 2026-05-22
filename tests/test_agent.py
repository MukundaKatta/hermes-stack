"""End-to-end HermesAgent tests using the stub client."""

import json
from pathlib import Path

import pytest

from hermes_stack import (
    BudgetCap,
    BudgetExceeded,
    ChatMessage,
    EgressDenied,
    EgressGuard,
    HermesAgent,
    HermesStub,
    Tracer,
)


SUMMARY_SCHEMA = {
    "type": "object",
    "required": ["title", "key_points", "sentiment", "confidence"],
    "properties": {
        "title": {"type": "string"},
        "key_points": {"type": "array", "minItems": 1},
        "sentiment": {"type": "string"},
        "confidence": {"type": "number"},
    },
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_stub_run_structured_returns_well_formed_json(tmp_path: Path) -> None:
    trace_file = tmp_path / "run.jsonl"
    with Tracer(trace_file) as tracer:
        agent = HermesAgent(
            client=HermesStub(),
            budget=BudgetCap(usd_cap=1.0, call_cap=5),
            guard=EgressGuard.from_iterable(["example.com"]),
            tracer=tracer,
        )
        result = agent.run_structured(
            [
                ChatMessage(role="system", content="Reply with JSON only."),
                ChatMessage(role="user", content="SUMMARIZE_URL test"),
            ],
            schema=SUMMARY_SCHEMA,
        )
    assert result.structured["title"] == "Stub Summary"
    assert len(result.structured["key_points"]) >= 1
    rows = _read_jsonl(trace_file)
    events = [r["event"] for r in rows]
    assert "hermes.call" in events
    assert "hermes.ok" in events
    assert "cast.ok" in events


def test_egress_denied_to_unlisted_host(tmp_path: Path) -> None:
    trace_file = tmp_path / "run.jsonl"
    with Tracer(trace_file) as tracer:
        agent = HermesAgent(
            client=HermesStub(),
            guard=EgressGuard.from_iterable(["example.com"]),
            tracer=tracer,
        )
        with pytest.raises(EgressDenied):
            agent.fetch_for_tool("https://evil.example.com/steal")
    rows = _read_jsonl(trace_file)
    denied = [r for r in rows if r["event"] == "egress.denied"]
    assert len(denied) == 1
    assert denied[0]["payload"]["host"] == "evil.example.com"


def test_budget_cap_fires_on_cumulative_spend(tmp_path: Path) -> None:
    trace_file = tmp_path / "run.jsonl"
    # The stub reports prompt=120, completion=140 tokens per call.
    # At the rates in hermes.py, each stub call costs ~0.0001960 USD.
    # Two calls fit under $0.0005 cap; the third pushes over.
    cap = BudgetCap(usd_cap=0.0005, call_cap=10)
    raised = False
    with Tracer(trace_file) as tracer:
        agent = HermesAgent(client=HermesStub(), budget=cap, tracer=tracer)
        try:
            for _ in range(5):
                agent.chat([ChatMessage(role="user", content="hello")])
        except BudgetExceeded as exc:
            raised = True
            assert exc.kind == "usd"
    assert raised, "expected BudgetExceeded to fire"
    rows = _read_jsonl(trace_file)
    events = [r["event"] for r in rows]
    assert "budget.exceeded" in events


def test_repair_retry_on_unparsable_first_response(tmp_path: Path) -> None:
    """The agent should retry once when the first reply does not parse."""

    class FlakyClient:
        model = "flaky-stub"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages):
            from hermes_stack.hermes import HermesResponse, estimate_cost

            self.calls += 1
            if self.calls == 1:
                text = "I think the answer is 42 maybe"
            else:
                text = '```json\n{"answer": 42}\n```'
            return HermesResponse(
                text=text,
                prompt_tokens=10,
                completion_tokens=10,
                usd_cost=estimate_cost(10, 10),
                raw={"stub": True},
            )

    client = FlakyClient()
    trace_file = tmp_path / "run.jsonl"
    with Tracer(trace_file) as tracer:
        agent = HermesAgent(client=client, tracer=tracer)
        result = agent.run_structured(
            [ChatMessage(role="user", content="give me JSON")],
            schema={"type": "object", "required": ["answer"]},
        )
    assert client.calls == 2
    assert result.structured == {"answer": 42}
    rows = _read_jsonl(trace_file)
    events = [r["event"] for r in rows]
    assert events.count("hermes.call") == 2
    assert "cast.repair" in events
