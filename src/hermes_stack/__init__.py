"""hermes-stack: four-layer governance harness for Hermes Agent calls.

The four layers, each enforceable on its own:

1. Budget cap         - USD ceiling + per-call ceiling. Raises BudgetExceeded.
2. Egress allowlist   - Hostname allowlist on every outbound fetch. Raises EgressDenied.
3. Audit trace        - Append-only JSONL of every call, denial, exception.
4. Structured output  - JSON schema check + one repair retry. Raises OutputInvalid.

Built for the DEV Community Hermes Agent Challenge (May 2026).
Patterned after the @mukundakatta agent-stack:
  - agentguard-py    https://github.com/MukundaKatta/agentguard-py
  - agentleash       https://github.com/MukundaKatta/agentleash
  - agenttrace-rs    https://github.com/MukundaKatta/agenttrace-rs
  - agentcast-py     https://github.com/MukundaKatta/agentcast-py
"""

from .agent import HermesAgent, HermesResult
from .budget import BudgetCap, BudgetExceeded
from .cast import OutputInvalid, cast_json
from .guard import EgressDenied, EgressGuard
from .hermes import HermesClient, HermesStub, ChatMessage
from .trace import AuditEvent, Tracer

__all__ = [
    "AuditEvent",
    "BudgetCap",
    "BudgetExceeded",
    "ChatMessage",
    "EgressDenied",
    "EgressGuard",
    "HermesAgent",
    "HermesClient",
    "HermesResult",
    "HermesStub",
    "OutputInvalid",
    "Tracer",
    "cast_json",
]

__version__ = "0.1.0"
