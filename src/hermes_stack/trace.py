"""Layer 3: audit trace.

Append-only JSONL log of every Hermes call, denial, and exception in a
session. Patterned after MukundaKatta/agenttrace-rs.

One run = one JSONL file. Each line is a dict with `event` and `ts` plus
the event-specific payload. The file is written line by line and flushed
after each event, so a crashed agent still leaves a partial trail.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any


@dataclass
class AuditEvent:
    """One row in the JSONL audit log."""

    ts: float
    event: str
    run_id: str
    payload: dict[str, Any] = field(default_factory=dict)


class Tracer:
    """JSONL audit logger.

    Use as a context manager so the file handle closes deterministically.
    Each `event(...)` call appends one line.
    """

    def __init__(self, path: str | os.PathLike, run_id: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._fh = None

    def __enter__(self) -> "Tracer":
        self._fh = open(self.path, "a", encoding="utf-8")
        self.event("run.start", {"run_id": self.run_id})
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.event("run.end", {"ok": True})
        else:
            self.event(
                "run.end",
                {
                    "ok": False,
                    "exception_type": exc_type.__name__,
                    "exception_message": str(exc),
                },
            )
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def event(self, name: str, payload: dict[str, Any] | None = None) -> AuditEvent:
        ev = AuditEvent(
            ts=time.time(),
            event=name,
            run_id=self.run_id,
            payload=payload or {},
        )
        if self._fh is None:
            # Allow standalone usage outside `with`, by opening lazily.
            self._fh = open(self.path, "a", encoding="utf-8")
        self._fh.write(json.dumps(asdict(ev), default=str))
        self._fh.write("\n")
        self._fh.flush()
        return ev
