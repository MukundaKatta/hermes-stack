"""Persistent shared budget — multi-process USD ceiling.

The in-memory ``BudgetCap`` works inside one process. The moment a user
spawns multiple worker processes (e.g. three parallel agent runs, one
LaunchAgent + one CLI ad-hoc, etc.) each one carries its own counter, so
``usd_cap=$5`` enforced three ways = $15 total.

``SharedBudgetCap`` fixes that by persisting the running spend in a JSON
file under an OS-level advisory lock (``fcntl.flock``).  Every reserve /
record acquires the lock, reads the file, writes the new total, then
releases — so two workers' updates cannot interleave.

Drop-in compatible with ``BudgetCap`` on the spend-recording side so
``HermesAgent`` can take either.

Patterned after the lesson in MukundaKatta/hermes-budget-skin:
"three workers running in parallel burned through $40 of Claude budget
in 18 minutes because each one had its own $5 cap and there was no
shared counter."
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .budget import BudgetExceeded


_DEFAULT_STATE = {"spent_usd": 0.0, "calls": 0}


@dataclass
class SharedBudgetCap:
    """Cross-process USD + call-count cap.

    ``path`` points at a JSON file we own — the parent directory must
    exist.  We create the file on first use with the default zero state.
    Concurrent processes coordinate through ``fcntl.flock`` on that same
    file, so reading + writing the new total is atomic against any other
    process that also wraps its updates in this class.

    The in-process ``threading.Lock`` keeps a single Python process from
    racing itself when multiple threads share the same SharedBudgetCap.
    """

    path: Path | str
    usd_cap: float = 1.00
    call_cap: int = 50
    _thread_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.path.parent.exists():
            raise FileNotFoundError(
                f"shared-budget parent dir does not exist: {self.path.parent}"
            )
        if not self.path.exists():
            self._write(_DEFAULT_STATE)

    # ---- File I/O under flock -------------------------------------------
    @contextlib.contextmanager
    def _locked_state(self):
        # Open in r+ so the same fd serves both reads and writes; create
        # if missing.  fcntl.flock is an OS-level advisory lock — every
        # cooperating process must also use it; an attacker can ignore it.
        # Good enough for the multi-worker coordination problem we have;
        # not a security boundary.
        with self._thread_lock:
            mode = "r+" if self.path.exists() else "w+"
            with open(self.path, mode) as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    fh.seek(0)
                    raw = fh.read().strip()
                    state = json.loads(raw) if raw else dict(_DEFAULT_STATE)
                    yield state, fh
                    fh.seek(0)
                    fh.truncate()
                    json.dump(state, fh)
                    fh.flush()
                    os.fsync(fh.fileno())
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def _write(self, state: dict[str, Any]) -> None:
        with open(self.path, "w") as fh:
            json.dump(state, fh)
            fh.flush()
            os.fsync(fh.fileno())

    # ---- Public API ------------------------------------------------------
    def snapshot(self) -> dict:
        with self._locked_state() as (state, _):
            return {
                "spent_usd": round(state.get("spent_usd", 0.0), 6),
                "usd_cap": self.usd_cap,
                "calls": int(state.get("calls", 0)),
                "call_cap": self.call_cap,
                "path": str(self.path),
            }

    def remaining_usd(self) -> float:
        with self._locked_state() as (state, _):
            return max(self.usd_cap - float(state.get("spent_usd", 0.0)), 0.0)

    def remaining_calls(self) -> int:
        with self._locked_state() as (state, _):
            return max(self.call_cap - int(state.get("calls", 0)), 0)

    def reserve_call(self) -> None:
        with self._locked_state() as (state, _):
            calls = int(state.get("calls", 0)) + 1
            if calls > self.call_cap:
                raise BudgetExceeded(
                    f"shared call cap reached: {calls} > {self.call_cap}",
                    kind="calls",
                    requested=float(calls),
                    cap=float(self.call_cap),
                )
            state["calls"] = calls

    def record_spend(self, usd: float) -> None:
        if usd < 0:
            raise ValueError("usd must be non-negative")
        with self._locked_state() as (state, _):
            new_total = float(state.get("spent_usd", 0.0)) + usd
            if new_total > self.usd_cap:
                raise BudgetExceeded(
                    f"shared USD cap reached: ${new_total:.4f} > ${self.usd_cap:.4f}",
                    kind="usd",
                    requested=new_total,
                    cap=self.usd_cap,
                )
            state["spent_usd"] = new_total

    def reset(self) -> None:
        """Zero the shared counter.  Intended for tests / daily-reset crons."""
        with self._locked_state() as (state, _):
            state["spent_usd"] = 0.0
            state["calls"] = 0
