"""Tests for the multi-process SharedBudgetCap."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import pytest

from hermes_stack import BudgetExceeded, SharedBudgetCap


def test_shared_budget_creates_file_on_first_use(tmp_path):
    p = tmp_path / "budget.json"
    cap = SharedBudgetCap(path=p, usd_cap=5.00, call_cap=10)
    assert p.exists()
    assert json.loads(p.read_text()) == {"spent_usd": 0.0, "calls": 0}
    snap = cap.snapshot()
    assert snap["spent_usd"] == 0.0
    assert snap["usd_cap"] == 5.00
    assert snap["calls"] == 0


def test_shared_budget_record_spend(tmp_path):
    cap = SharedBudgetCap(path=tmp_path / "b.json", usd_cap=1.00)
    cap.record_spend(0.25)
    cap.record_spend(0.30)
    assert cap.snapshot()["spent_usd"] == pytest.approx(0.55, abs=1e-9)
    assert cap.remaining_usd() == pytest.approx(0.45, abs=1e-9)


def test_shared_budget_raises_when_usd_cap_exceeded(tmp_path):
    cap = SharedBudgetCap(path=tmp_path / "b.json", usd_cap=0.50)
    cap.record_spend(0.40)
    with pytest.raises(BudgetExceeded) as exc:
        cap.record_spend(0.20)
    assert exc.value.kind == "usd"
    # First spend stuck, second was rejected — file reflects only the first.
    assert cap.snapshot()["spent_usd"] == pytest.approx(0.40, abs=1e-9)


def test_shared_budget_reserve_call_caps(tmp_path):
    cap = SharedBudgetCap(path=tmp_path / "b.json", usd_cap=10.0, call_cap=2)
    cap.reserve_call()
    cap.reserve_call()
    with pytest.raises(BudgetExceeded) as exc:
        cap.reserve_call()
    assert exc.value.kind == "calls"


def test_shared_budget_rejects_negative_spend(tmp_path):
    cap = SharedBudgetCap(path=tmp_path / "b.json")
    with pytest.raises(ValueError):
        cap.record_spend(-0.01)


def test_shared_budget_reset(tmp_path):
    cap = SharedBudgetCap(path=tmp_path / "b.json", usd_cap=5.0)
    cap.record_spend(2.0)
    cap.reserve_call()
    cap.reset()
    snap = cap.snapshot()
    assert snap["spent_usd"] == 0.0
    assert snap["calls"] == 0


def test_shared_budget_requires_existing_parent_dir(tmp_path):
    missing = tmp_path / "does-not-exist" / "b.json"
    with pytest.raises(FileNotFoundError):
        SharedBudgetCap(path=missing)


def test_two_handles_see_same_state(tmp_path):
    """The whole point of the layer: two SharedBudgetCap instances pointing
    at the same file coordinate through it."""
    p = tmp_path / "b.json"
    a = SharedBudgetCap(path=p, usd_cap=1.00)
    b = SharedBudgetCap(path=p, usd_cap=1.00)

    a.record_spend(0.60)
    b.record_spend(0.30)
    # Either handle reading the file sees the combined total.
    assert a.snapshot()["spent_usd"] == pytest.approx(0.90, abs=1e-9)
    assert b.snapshot()["spent_usd"] == pytest.approx(0.90, abs=1e-9)
    # And the third call (over the cap) raises through either handle.
    with pytest.raises(BudgetExceeded):
        a.record_spend(0.20)


# --- cross-process: spawn workers that all hit the same JSON file ----------


def _worker_spend(path: str, usd_cap: float, per_call_usd: float, calls: int) -> int:
    cap = SharedBudgetCap(path=Path(path), usd_cap=usd_cap)
    ok = 0
    for _ in range(calls):
        try:
            cap.record_spend(per_call_usd)
            ok += 1
        except BudgetExceeded:
            break
    return ok


def test_shared_budget_caps_total_across_processes(tmp_path):
    """Three workers, each thinking it can spend $1.00 per call, all sharing
    a $5.00 ceiling.  The file-level lock must ensure they collectively
    don't break past the cap."""
    p = tmp_path / "shared.json"
    SharedBudgetCap(path=p, usd_cap=5.00).snapshot()  # ensure file exists

    # If we let each worker try 10 spends of $1.00, naive non-coordinated
    # counters would burn $30; the shared file should hold the total at
    # exactly $5.00 (5 successful spends total, distributed across workers).
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=3) as pool:
        results = pool.starmap(
            _worker_spend,
            [(str(p), 5.00, 1.00, 10) for _ in range(3)],
        )
    successes = sum(results)
    assert successes == 5, f"expected 5 successes across processes, got {successes}"
    final = SharedBudgetCap(path=p, usd_cap=5.00).snapshot()
    assert final["spent_usd"] == pytest.approx(5.00, abs=1e-9)
