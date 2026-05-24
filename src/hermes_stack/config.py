"""Env-driven HermesAgent factory.

Wiring six layers by hand is fine for tests and demos but tedious for a
long-running agent.  ``HermesConfig.from_env()`` reads a small set of
documented env vars, constructs each layer with sensible defaults, and
hands back a fully-configured ``HermesAgent``:

  * ``OPENROUTER_API_KEY``        — when set, use the real HermesClient;
                                    otherwise fall back to HermesStub.
  * ``HERMES_MODEL``              — override the default model id.
  * ``HERMES_BUDGET_USD_CAP``     — dollars; default 1.00.
  * ``HERMES_BUDGET_CALL_CAP``    — int; default 50.
  * ``HERMES_BUDGET_PATH``        — when set, use SharedBudgetCap with
                                    this file path instead of in-process
                                    BudgetCap; multi-process safe.
  * ``HERMES_ALLOW_HOSTS``        — comma-separated hostnames to allow
                                    through EgressGuard.  HERMES_API_HOST
                                    is always added automatically.
  * ``HERMES_TRACE_PATH``         — when set, Tracer writes to this path.
  * ``HERMES_RETRY_MAX_ATTEMPTS`` — int; when > 1, attach a RetryPolicy.
  * ``HERMES_RETRY_BASE_DELAY``   — seconds; default 0.5.

Skipping a var is the no-op default — a minimal env yields a stub agent
with a $1 USD cap and no retries, exactly what the demo needs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .agent import HermesAgent
from .budget import BudgetCap
from .guard import EgressGuard
from .hermes import HermesClient, HermesStub
from .retry import RetryPolicy
from .shared_budget import SharedBudgetCap
from .trace import Tracer
from .vet import ToolVet


def _read_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _read_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _read_list(name: str) -> list[str]:
    raw = os.environ.get(name) or ""
    return [h.strip() for h in raw.split(",") if h.strip()]


@dataclass
class HermesConfig:
    """Settings resolved from env at agent-build time.

    Exposed as a dataclass (not just a function call) so callers can also
    construct it manually for tests, override individual fields, and
    introspect what was picked up from the environment.
    """

    model: str | None = None
    budget_usd_cap: float = 1.00
    budget_call_cap: int = 50
    budget_path: str | None = None
    allow_hosts: list[str] | None = None
    trace_path: str | None = None
    retry_max_attempts: int = 1   # 1 = no retries
    retry_base_delay_s: float = 0.5
    use_stub: bool = True         # set False when OPENROUTER_API_KEY present

    @classmethod
    def from_env(cls) -> HermesConfig:
        api_key = os.environ.get("OPENROUTER_API_KEY") or ""
        return cls(
            model=os.environ.get("HERMES_MODEL") or None,
            budget_usd_cap=_read_float("HERMES_BUDGET_USD_CAP", 1.00),
            budget_call_cap=_read_int("HERMES_BUDGET_CALL_CAP", 50),
            budget_path=os.environ.get("HERMES_BUDGET_PATH") or None,
            allow_hosts=_read_list("HERMES_ALLOW_HOSTS") or None,
            trace_path=os.environ.get("HERMES_TRACE_PATH") or None,
            retry_max_attempts=_read_int("HERMES_RETRY_MAX_ATTEMPTS", 1),
            retry_base_delay_s=_read_float("HERMES_RETRY_BASE_DELAY", 0.5),
            use_stub=not bool(api_key),
        )

    # -----------------------------------------------------------------------
    # Build the agent.  Each layer construction is one branch so callers can
    # subclass and override `_build_budget` / `_build_guard` / etc. in tests.
    # -----------------------------------------------------------------------
    def build(self, *, vet: ToolVet | None = None) -> HermesAgent:
        """Construct a HermesAgent from this config.

        ``vet`` stays a constructor argument rather than an env-driven
        thing because tool schemas are inherently code-defined — putting
        them in env would be more painful than useful.
        """
        client = self._build_client()
        budget = self._build_budget()
        guard = self._build_guard()
        tracer = self._build_tracer()
        retry_policy = self._build_retry()
        return HermesAgent(
            client=client,
            budget=budget,
            guard=guard,
            tracer=tracer,
            vet=vet,
            retry=retry_policy,
        )

    # ---- builders ----
    def _build_client(self):
        if self.use_stub:
            # HermesStub doesn't take a model kwarg — its model is a class
            # attribute. The `model` field on HermesConfig is only meaningful
            # when use_stub is False.
            return HermesStub()
        return HermesClient(model=self.model) if self.model else HermesClient()

    def _build_budget(self):
        if self.budget_path:
            path = Path(self.budget_path)
            # SharedBudgetCap requires the parent dir to exist; create it
            # eagerly so first-run setup doesn't fail on a missing /var/run.
            path.parent.mkdir(parents=True, exist_ok=True)
            return SharedBudgetCap(
                path=path,
                usd_cap=self.budget_usd_cap,
                call_cap=self.budget_call_cap,
            )
        return BudgetCap(usd_cap=self.budget_usd_cap, call_cap=self.budget_call_cap)

    def _build_guard(self) -> EgressGuard:
        guard = EgressGuard()
        for host in self.allow_hosts or ():
            guard.allow(host)
        return guard

    def _build_tracer(self) -> Tracer | None:
        if not self.trace_path:
            return None
        return Tracer(self.trace_path)

    def _build_retry(self) -> RetryPolicy | None:
        if self.retry_max_attempts <= 1:
            return None
        return RetryPolicy(
            max_attempts=self.retry_max_attempts,
            base_delay_s=self.retry_base_delay_s,
        )


def agent_from_env(*, vet: ToolVet | None = None) -> HermesAgent:
    """One-line factory: ``agent = agent_from_env()`` builds the whole stack."""
    return HermesConfig.from_env().build(vet=vet)


__all__ = ["HermesConfig", "agent_from_env"]
