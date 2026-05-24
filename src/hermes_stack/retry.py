"""Layer 6: transient-failure retry.

A long-running personal agent will see brief 429s, 5xxs, and "overloaded"
responses from the upstream Hermes inference host that resolve on a
retry a few hundred milliseconds later.  Without backoff, those
transient blips either propagate as visible failures or chew through
the budget cap with hot-loop retries.

This module gives ``HermesAgent`` a small retry primitive:

* ``RetryPolicy`` — configurable max attempts, base + max delay, jitter
  mode (``"full"`` is the default; matches AWS / aiohttp recommendations).
* ``retry(fn, ...)`` / ``aretry(fn, ...)`` — runtime-agnostic wrappers
  that re-invoke ``fn`` on retryable failures and re-raise after the
  budget is spent.  Sync and async share the same policy + predicate.
* ``is_hermes_retryable(exc)`` — Hermes-3 errors look Anthropic-shaped
  (the OpenRouter ``nousresearch/hermes-3-llama-3.1-405b`` route returns
  the same code vocabulary), so this is a curated alias of the
  Anthropic transient-failure list.

Wiring into ``HermesAgent`` is opt-in: pass ``retry=RetryPolicy()`` to
the constructor.  Each retry attempt reserves a call against the
budget cap, so a runaway retry loop cannot quietly burn through
``call_cap``.

Patterned after MukundaKatta/llm-retry-py + MukundaKatta/llm-retry (Rust).
"""

from __future__ import annotations

import asyncio
import inspect
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar


T = TypeVar("T")


# Same vocabulary as predicates in llm-circuit-breaker-py / llm-retry-py.
# Hermes-3 surfaces upstream provider errors verbatim through OpenRouter,
# so we treat the Anthropic-shape codes as the canonical retryable set.
HERMES_RETRYABLE_CODES: tuple[str, ...] = (
    "rate_limit_error",
    "rate_limit_exceeded",
    "overloaded_error",
    "api_error",
    "server_error",
    "timeout",
    "ServiceUnavailableException",
    "ThrottlingException",
)


def is_hermes_retryable(exc: BaseException) -> bool:
    """Default `should_retry` for Hermes calls."""
    msg = f"{type(exc).__name__}: {exc}"
    return any(code in msg for code in HERMES_RETRYABLE_CODES)


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with optional jitter.

    Attributes:
        max_attempts: total tries (including the first).  Defaults to 4
            so a transient 429 sees three retries before propagating.
        base_delay_s: base sleep before the second attempt.  Subsequent
            attempts sleep ``base_delay_s * 2 ** (attempt-1)`` capped
            at ``max_delay_s``.  Defaults to 0.5s.
        max_delay_s: absolute cap on per-attempt sleep.  Defaults to 30s.
        jitter: ``"full"`` (default) uniformly randomizes the sleep in
            ``[0, computed_delay]``; ``"equal"`` does
            ``computed_delay/2 + random(0..computed_delay/2)``; ``"none"``
            disables jitter (deterministic, useful for tests).
    """

    max_attempts: int = 4
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0
    jitter: str = "full"

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay_s < 0:
            raise ValueError("base_delay_s must be >= 0")
        if self.max_delay_s < 0:
            raise ValueError("max_delay_s must be >= 0")
        if self.jitter not in ("full", "equal", "none"):
            raise ValueError("jitter must be 'full' | 'equal' | 'none'")

    def delay_for(self, attempt: int) -> float:
        """Sleep before ``attempt`` (1-indexed; attempt=1 is the first
        retry, i.e. comes AFTER the first failure)."""
        if attempt < 1:
            return 0.0
        exp = self.base_delay_s * (2 ** (attempt - 1))
        capped = min(exp, self.max_delay_s)
        if self.jitter == "none":
            return capped
        if self.jitter == "equal":
            return capped / 2 + random.uniform(0, capped / 2)
        return random.uniform(0, capped)


def retry(
    fn: Callable[..., T],
    *args: Any,
    policy: RetryPolicy | None = None,
    should_retry: Callable[[BaseException], bool] | None = None,
    on_retry: Callable[[BaseException, int, float], None] | None = None,
    sleep: Callable[[float], None] | None = None,
    **kwargs: Any,
) -> T:
    """Run ``fn`` with backoff retries.

    Re-raises the last exception once ``policy.max_attempts`` is reached
    OR ``should_retry(exc)`` returns False.  ``on_retry(exc, attempt,
    delay_s)`` is invoked just before each sleep so callers can plug in
    audit hooks.  ``sleep`` lets tests inject a fake sleeper.
    """
    pol = policy or RetryPolicy()
    check = should_retry or is_hermes_retryable
    sleep_fn = sleep or time.sleep
    last_exc: BaseException | None = None
    for attempt in range(1, pol.max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 — exception-policy gate handles classification
            last_exc = exc
            if attempt >= pol.max_attempts or not check(exc):
                raise
            delay = pol.delay_for(attempt)
            if on_retry is not None:
                on_retry(exc, attempt, delay)
            if delay > 0:
                sleep_fn(delay)
    # Unreachable — the loop either returns or re-raises.
    raise last_exc  # type: ignore[misc]


async def aretry(
    fn: Callable[..., T | Awaitable[T]],
    *args: Any,
    policy: RetryPolicy | None = None,
    should_retry: Callable[[BaseException], bool] | None = None,
    on_retry: Callable[[BaseException, int, float], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    **kwargs: Any,
) -> T:
    """Async retry wrapper.  Awaits ``fn`` if it returns a coroutine."""
    pol = policy or RetryPolicy()
    check = should_retry or is_hermes_retryable
    sleep_fn = sleep or asyncio.sleep
    last_exc: BaseException | None = None
    for attempt in range(1, pol.max_attempts + 1):
        try:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result  # type: ignore[return-value]
        except BaseException as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= pol.max_attempts or not check(exc):
                raise
            delay = pol.delay_for(attempt)
            if on_retry is not None:
                on_retry(exc, attempt, delay)
            if delay > 0:
                await sleep_fn(delay)
    raise last_exc  # type: ignore[misc]


__all__ = [
    "HERMES_RETRYABLE_CODES",
    "RetryPolicy",
    "aretry",
    "is_hermes_retryable",
    "retry",
]
