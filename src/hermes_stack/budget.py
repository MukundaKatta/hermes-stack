"""Layer 1: budget cap.

USD ceiling + per-call ceiling for a single Hermes session. Thread-safe
on the spend update path so concurrent tool calls cannot race past the
cap. Patterned after MukundaKatta/token-budget-py and MukundaKatta/agentleash.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    """Raised when a recorded spend or call would push past a cap.

    The session that raises this exception stops cooperating. The agent
    must not retry the same call without raising the cap or starting a
    fresh session.
    """

    def __init__(self, message: str, *, kind: str, requested: float, cap: float) -> None:
        super().__init__(message)
        self.kind = kind
        self.requested = requested
        self.cap = cap


@dataclass
class BudgetCap:
    """USD + call-count cap for one Hermes session.

    All amounts are USD. `record_spend` is the only mutating call; it
    blocks on the internal lock briefly so two tools in parallel can
    cooperate around the cap.
    """

    usd_cap: float = 1.00
    call_cap: int = 50
    _spent_usd: float = field(default=0.0, init=False)
    _calls: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def remaining_usd(self) -> float:
        with self._lock:
            return max(self.usd_cap - self._spent_usd, 0.0)

    def remaining_calls(self) -> int:
        with self._lock:
            return max(self.call_cap - self._calls, 0)

    def reserve_call(self) -> None:
        """Increment the call counter or raise.

        Call this before issuing a Hermes API call. Pair with
        `record_spend` after the call returns so the cap can settle.
        """
        with self._lock:
            if self._calls + 1 > self.call_cap:
                raise BudgetExceeded(
                    f"call cap reached: {self._calls + 1} > {self.call_cap}",
                    kind="calls",
                    requested=float(self._calls + 1),
                    cap=float(self.call_cap),
                )
            self._calls += 1

    def record_spend(self, usd: float) -> None:
        """Add USD to the running spend or raise."""
        if usd < 0:
            raise ValueError("usd must be non-negative")
        with self._lock:
            new_total = self._spent_usd + usd
            if new_total > self.usd_cap:
                raise BudgetExceeded(
                    f"USD cap reached: ${new_total:.4f} > ${self.usd_cap:.4f}",
                    kind="usd",
                    requested=new_total,
                    cap=self.usd_cap,
                )
            self._spent_usd = new_total

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "spent_usd": round(self._spent_usd, 6),
                "usd_cap": self.usd_cap,
                "calls": self._calls,
                "call_cap": self.call_cap,
            }
