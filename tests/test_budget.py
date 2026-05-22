"""Budget cap tests."""

import pytest

from hermes_stack.budget import BudgetCap, BudgetExceeded


def test_budget_starts_at_zero_spend() -> None:
    cap = BudgetCap(usd_cap=1.0, call_cap=5)
    snap = cap.snapshot()
    assert snap["spent_usd"] == 0.0
    assert snap["calls"] == 0
    assert cap.remaining_usd() == 1.0
    assert cap.remaining_calls() == 5


def test_record_spend_accumulates() -> None:
    cap = BudgetCap(usd_cap=1.0, call_cap=10)
    cap.record_spend(0.2)
    cap.record_spend(0.3)
    assert cap.snapshot()["spent_usd"] == 0.5
    assert cap.remaining_usd() == pytest.approx(0.5)


def test_record_spend_raises_at_cap() -> None:
    cap = BudgetCap(usd_cap=0.10, call_cap=10)
    cap.record_spend(0.05)
    with pytest.raises(BudgetExceeded) as info:
        cap.record_spend(0.06)
    err = info.value
    assert err.kind == "usd"
    assert err.cap == 0.10
    assert err.requested > 0.10


def test_reserve_call_raises_at_call_cap() -> None:
    cap = BudgetCap(usd_cap=10.0, call_cap=2)
    cap.reserve_call()
    cap.reserve_call()
    with pytest.raises(BudgetExceeded) as info:
        cap.reserve_call()
    assert info.value.kind == "calls"


def test_record_spend_rejects_negative() -> None:
    cap = BudgetCap()
    with pytest.raises(ValueError):
        cap.record_spend(-0.01)
