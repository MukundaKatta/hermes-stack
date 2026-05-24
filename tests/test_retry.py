"""Tests for the retry layer (layer 6)."""

from __future__ import annotations

import asyncio

import pytest

from hermes_stack import (
    BudgetCap,
    BudgetExceeded,
    HermesAgent,
    RetryPolicy,
    aretry,
    is_hermes_retryable,
    retry,
)
from hermes_stack.hermes import HermesResponse


class _ProviderError(Exception):
    """Local exception class — ruff B017 forbids pytest.raises(Exception)."""


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


def test_policy_defaults_are_sane():
    p = RetryPolicy()
    assert p.max_attempts == 4
    assert p.base_delay_s == 0.5
    assert p.max_delay_s == 30.0
    assert p.jitter == "full"


@pytest.mark.parametrize(
    "max_attempts,base,max_d,jit",
    [
        (0, 0.5, 30.0, "full"),
        (4, -0.1, 30.0, "full"),
        (4, 0.5, -1.0, "full"),
        (4, 0.5, 30.0, "weird"),
    ],
)
def test_policy_validates_args(max_attempts, base, max_d, jit):
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=max_attempts, base_delay_s=base, max_delay_s=max_d, jitter=jit)


def test_policy_delay_none_jitter_is_deterministic():
    p = RetryPolicy(base_delay_s=1.0, max_delay_s=10.0, jitter="none")
    assert p.delay_for(1) == 1.0
    assert p.delay_for(2) == 2.0
    assert p.delay_for(3) == 4.0
    assert p.delay_for(4) == 8.0
    assert p.delay_for(5) == 10.0  # capped at max_delay_s


def test_policy_full_jitter_within_bounds():
    p = RetryPolicy(base_delay_s=1.0, max_delay_s=10.0, jitter="full")
    for _ in range(20):
        d = p.delay_for(3)
        assert 0.0 <= d <= 4.0  # base*2^(attempt-1) = 4.0


def test_policy_equal_jitter_within_bounds():
    p = RetryPolicy(base_delay_s=1.0, max_delay_s=10.0, jitter="equal")
    for _ in range(20):
        d = p.delay_for(3)
        # equal jitter is in [exp/2, exp]
        assert 2.0 <= d <= 4.0


# ---------------------------------------------------------------------------
# is_hermes_retryable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("rate_limit_error: too many", True),
        ("overloaded_error", True),
        ("api_error", True),
        ("server_error: 500", True),
        ("ThrottlingException", True),
        ("ServiceUnavailableException", True),
        ("timeout while reading", True),
        ("authentication_error", False),
        ("invalid_request_error", False),
        ("validation failed", False),
    ],
)
def test_is_hermes_retryable(msg, expected):
    assert is_hermes_retryable(_ProviderError(msg)) is expected


# ---------------------------------------------------------------------------
# retry (sync)
# ---------------------------------------------------------------------------


def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _ProviderError("rate_limit_error")
        return "ok"

    out = retry(
        flaky,
        policy=RetryPolicy(max_attempts=5, base_delay_s=0.0, jitter="none"),
        sleep=lambda _: None,
    )
    assert out == "ok"
    assert calls["n"] == 3


def test_retry_gives_up_after_max_attempts():
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise _ProviderError("rate_limit_error")

    with pytest.raises(_ProviderError):
        retry(
            boom,
            policy=RetryPolicy(max_attempts=3, base_delay_s=0.0, jitter="none"),
            sleep=lambda _: None,
        )
    assert calls["n"] == 3


def test_retry_propagates_non_retryable_immediately():
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise _ProviderError("authentication_error: bad key")

    with pytest.raises(_ProviderError):
        retry(
            boom,
            policy=RetryPolicy(max_attempts=5, base_delay_s=0.0, jitter="none"),
            sleep=lambda _: None,
        )
    # Only one attempt — auth errors don't retry.
    assert calls["n"] == 1


def test_retry_calls_on_retry_hook():
    events: list = []

    def flaky():
        if len(events) < 2:
            raise _ProviderError("overloaded_error")
        return 42

    out = retry(
        flaky,
        policy=RetryPolicy(max_attempts=5, base_delay_s=0.1, jitter="none"),
        sleep=lambda _: None,
        on_retry=lambda exc, n, d: events.append((type(exc).__name__, n, d)),
    )
    assert out == 42
    assert len(events) == 2
    assert events[0][1] == 1  # first retry attempt number
    assert events[1][1] == 2


def test_retry_passes_args_kwargs():
    def add(a: int, *, b: int) -> int:
        return a + b

    assert retry(add, 2, b=3, policy=RetryPolicy(max_attempts=1)) == 5


# ---------------------------------------------------------------------------
# aretry (async)
# ---------------------------------------------------------------------------


def test_aretry_succeeds_after_transient_failures():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise _ProviderError("rate_limit_error")
        return "ok"

    async def fake_sleep(_d: float) -> None:
        return None

    out = asyncio.run(
        aretry(
            flaky,
            policy=RetryPolicy(max_attempts=3, base_delay_s=0.0, jitter="none"),
            sleep=fake_sleep,
        )
    )
    assert out == "ok"
    assert calls["n"] == 2


def test_aretry_propagates_non_retryable():
    async def boom():
        raise _ProviderError("invalid_request_error")

    async def fake_sleep(_d: float) -> None:
        return None

    with pytest.raises(_ProviderError):
        asyncio.run(
            aretry(
                boom,
                policy=RetryPolicy(max_attempts=5, base_delay_s=0.0, jitter="none"),
                sleep=fake_sleep,
            )
        )


def test_aretry_accepts_sync_fn():
    """aretry should work with a sync callable that returns a value directly."""

    def add(a: int, b: int) -> int:
        return a + b

    out = asyncio.run(aretry(add, 2, 3, policy=RetryPolicy(max_attempts=1)))
    assert out == 5


# ---------------------------------------------------------------------------
# HermesAgent integration
# ---------------------------------------------------------------------------


class FlakeyClient:
    """Stub Hermes client that raises overloaded N times then returns a response."""

    model = "stub"

    def __init__(self, fail_n: int, exc: Exception):
        self.fail_n = fail_n
        self.exc = exc
        self.calls = 0

    def complete(self, _msgs):
        self.calls += 1
        if self.calls <= self.fail_n:
            raise self.exc
        return HermesResponse(
            text='{"ok":true}',
            prompt_tokens=10,
            completion_tokens=10,
            usd_cost=0.001,
            raw={"stub": True},
        )


def test_agent_retries_transient_failures():
    client = FlakeyClient(fail_n=2, exc=_ProviderError("overloaded_error"))
    agent = HermesAgent(
        client=client,
        budget=BudgetCap(usd_cap=1.0, call_cap=10),
        # base_delay_s=0 + jitter="none" → delay_for returns 0.0 for every
        # attempt, so time.sleep(0) doesn't slow the test down.
        retry=RetryPolicy(max_attempts=4, base_delay_s=0.0, jitter="none"),
    )
    resp = agent.chat([])
    assert resp.text == '{"ok":true}'
    assert client.calls == 3
    # Each attempt reserved a call against the cap.
    assert agent.budget.snapshot()["calls"] == 3


def test_agent_without_retry_policy_propagates_first_failure():
    client = FlakeyClient(fail_n=2, exc=_ProviderError("overloaded_error"))
    agent = HermesAgent(client=client, budget=BudgetCap(usd_cap=1.0, call_cap=10))
    with pytest.raises(_ProviderError):
        agent.chat([])
    assert client.calls == 1


def test_agent_does_not_retry_non_retryable_errors():
    client = FlakeyClient(fail_n=5, exc=_ProviderError("authentication_error"))
    agent = HermesAgent(
        client=client,
        budget=BudgetCap(usd_cap=1.0, call_cap=10),
        retry=RetryPolicy(max_attempts=4, base_delay_s=0.0, jitter="none"),
    )
    with pytest.raises(_ProviderError):
        agent.chat([])
    # Only one attempt — auth errors aren't retryable.
    assert client.calls == 1


def test_agent_does_not_retry_budget_exceeded():
    """A BudgetExceeded raised inside the call must NOT be treated as
    transient — the cap is the user's intentional limit, not a transient
    upstream blip."""
    client = FlakeyClient(fail_n=0, exc=Exception("never"))
    agent = HermesAgent(
        client=client,
        budget=BudgetCap(usd_cap=0.0005, call_cap=10),  # tiny cap
        retry=RetryPolicy(max_attempts=4, base_delay_s=0.0, jitter="none"),
    )
    with pytest.raises(BudgetExceeded):
        agent.chat([])
    # First call ran, hit the budget cap; no retry attempted.
    assert client.calls == 1
