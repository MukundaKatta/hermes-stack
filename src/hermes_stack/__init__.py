"""hermes-stack: five-layer governance harness for Hermes Agent calls.

The five layers, each enforceable on its own:

1. Budget cap         - USD ceiling + per-call ceiling. Raises BudgetExceeded.
                        SharedBudgetCap variant gives the same cap across
                        multiple processes via fcntl.flock + JSON state file.
2. Egress allowlist   - Hostname allowlist on every outbound fetch. Raises EgressDenied.
3. Audit trace        - Append-only JSONL of every call, denial, exception.
4. Structured output  - JSON schema check + one repair retry. Raises OutputInvalid.
5. Tool-arg vet       - Validate model-produced tool arguments before the
                        tool runs. Raises ToolArgError with an LLM-readable hint.

Built for the DEV Community Hermes Agent Challenge (May 2026).
Patterned after the @mukundakatta agent-stack:
  - agentguard-py    https://github.com/MukundaKatta/agentguard-py
  - agentleash       https://github.com/MukundaKatta/agentleash
  - agenttrace-rs    https://github.com/MukundaKatta/agenttrace-rs
  - agentcast-py     https://github.com/MukundaKatta/agentcast-py
  - agentvet         https://github.com/MukundaKatta/agentvet
  - token-budget-py  https://github.com/MukundaKatta/token-budget-py
"""

from .agent import HermesAgent, HermesResult
from .budget import BudgetCap, BudgetExceeded
from .cast import OutputInvalid, cast_json
from .guard import EgressDenied, EgressGuard
from .hermes import HermesClient, HermesStub, ChatMessage
from .shared_budget import SharedBudgetCap
from .trace import AuditEvent, Tracer
from .vet import ToolArgError, ToolVet

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
    "SharedBudgetCap",
    "ToolArgError",
    "ToolVet",
    "Tracer",
    "cast_json",
]

__version__ = "0.2.0"
