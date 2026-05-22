"""Audit trace tests."""

import json
from pathlib import Path

from hermes_stack.trace import Tracer


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_tracer_writes_start_and_end_events(tmp_path: Path) -> None:
    trace_file = tmp_path / "run.jsonl"
    with Tracer(trace_file) as t:
        t.event("custom.event", {"hello": "world"})
    rows = _read_jsonl(trace_file)
    assert [r["event"] for r in rows] == ["run.start", "custom.event", "run.end"]
    assert rows[-1]["payload"]["ok"] is True
    assert rows[1]["payload"]["hello"] == "world"


def test_tracer_records_exception_on_failure(tmp_path: Path) -> None:
    trace_file = tmp_path / "run.jsonl"
    try:
        with Tracer(trace_file) as t:
            t.event("midway", {})
            raise RuntimeError("kaboom")
    except RuntimeError:
        pass
    rows = _read_jsonl(trace_file)
    last = rows[-1]
    assert last["event"] == "run.end"
    assert last["payload"]["ok"] is False
    assert last["payload"]["exception_type"] == "RuntimeError"
    assert "kaboom" in last["payload"]["exception_message"]


def test_tracer_run_id_appears_on_every_row(tmp_path: Path) -> None:
    trace_file = tmp_path / "run.jsonl"
    with Tracer(trace_file, run_id="fixed-id") as t:
        t.event("a")
        t.event("b")
    rows = _read_jsonl(trace_file)
    for row in rows:
        assert row["run_id"] == "fixed-id"
