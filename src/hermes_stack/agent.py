"""HermesAgent: compose the four governance layers around a Hermes call.

A single class so the demo and tests can show the whole flow in one
place. The agent owns one BudgetCap, one EgressGuard, one Tracer, and
one Hermes client. Every call goes through the same path:

  1. reserve_call    (budget)
  2. emit pre-event   (trace)
  3. hermes.complete  (real or stub)
  4. record_spend     (budget)
  5. cast_json        (output)
  6. emit post-event  (trace)

Any layer can fail closed. The exception name and message land in the
audit log before it propagates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .budget import BudgetCap, BudgetExceeded
from .cast import OutputInvalid, cast_json
from .guard import EgressDenied, EgressGuard
from .hermes import HERMES_API_HOST, ChatMessage, HermesResponse, HermesStub
from .trace import Tracer
from .vet import ToolArgError, ToolVet


@dataclass
class HermesResult:
    """Outcome of one HermesAgent.run call."""

    structured: Any
    response: HermesResponse
    fetched_url: str | None
    fetched_chars: int


class HermesAgent:
    """Wraps a Hermes client with budget, egress, trace, cast, and vet layers.

    `client` defaults to HermesStub so the smoke test runs without
    network or API keys. Swap to HermesClient when OPENROUTER_API_KEY is
    set.

    `budget` accepts either an in-process ``BudgetCap`` or a multi-process
    ``SharedBudgetCap`` — both satisfy the duck-typed interface
    (``reserve_call``, ``record_spend``, ``snapshot``,
    ``remaining_calls``).

    `vet` is optional. When passed, any ``call_tool`` invocation runs the
    args through the registered schema first; the call is rejected with
    a ``ToolArgError`` carrying an LLM-readable hint if the schema fails.
    """

    def __init__(
        self,
        client=None,
        *,
        budget=None,
        guard: EgressGuard | None = None,
        tracer: Tracer | None = None,
        vet: ToolVet | None = None,
    ) -> None:
        self.client = client or HermesStub()
        self.budget = budget or BudgetCap()
        self.guard = guard or EgressGuard()
        # The Hermes API host is allowlisted automatically so callers
        # only have to think about tool fetch targets.
        self.guard.allow(HERMES_API_HOST)
        self.tracer = tracer  # may be None; methods no-op when so
        self.vet = vet

    # ---- Tracing helpers -------------------------------------------------
    def _trace(self, name: str, payload: dict | None = None) -> None:
        if self.tracer is not None:
            self.tracer.event(name, payload)

    # ---- Public API ------------------------------------------------------
    def fetch_for_tool(self, url: str) -> str:
        """Guarded fetch the agent can use as a tool action."""
        try:
            host = self.guard.check(url)
        except EgressDenied as exc:
            self._trace(
                "egress.denied",
                {"host": exc.host, "url": exc.url},
            )
            raise
        self._trace("egress.allowed", {"host": host, "url": url})
        body = self.guard.fetch(url)
        self._trace(
            "tool.fetch.ok",
            {"host": host, "url": url, "chars": len(body)},
        )
        return body

    def call_tool(self, name: str, args: Any, fn: Any) -> Any:
        """Validate model-produced tool args, then invoke ``fn(**args)``.

        Requires a ``ToolVet`` registered on the agent.  If no vet is
        registered we refuse rather than silently letting the model run
        arbitrary tools — failing closed is the whole point of the layer.
        Returns whatever ``fn`` returns; traces the gate decision either
        way so an audit log can answer "did the model try a bad call
        and get caught?" later.
        """
        if self.vet is None:
            raise RuntimeError(
                "call_tool requires a ToolVet registered on the agent"
            )
        try:
            checked = self.vet.check(name, args)
        except ToolArgError as exc:
            self._trace(
                "tool.args.invalid",
                {
                    "tool": exc.tool,
                    "field": exc.field,
                    "expected": exc.expected,
                    "got": exc.got,
                    "hint": exc.hint,
                },
            )
            raise
        self._trace("tool.args.ok", {"tool": name})
        return fn(**checked) if isinstance(checked, dict) else fn(checked)

    def chat(self, messages: Iterable[ChatMessage]) -> HermesResponse:
        """Single Hermes call wrapped in budget + trace."""
        msgs = list(messages)
        self.budget.reserve_call()
        self._trace(
            "hermes.call",
            {
                "model": self.client.model,
                "messages": len(msgs),
                "calls_left": self.budget.remaining_calls(),
            },
        )
        try:
            resp = self.client.complete(msgs)
        except Exception as exc:  # noqa: BLE001 - capture and re-raise
            self._trace(
                "hermes.error",
                {"exception_type": type(exc).__name__, "message": str(exc)},
            )
            raise
        # Spend recording can itself raise BudgetExceeded; trace either way.
        try:
            self.budget.record_spend(resp.usd_cost)
        except BudgetExceeded as exc:
            self._trace(
                "budget.exceeded",
                {
                    "kind": exc.kind,
                    "requested": exc.requested,
                    "cap": exc.cap,
                    "usd_cost_of_this_call": resp.usd_cost,
                },
            )
            raise
        self._trace(
            "hermes.ok",
            {
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "usd_cost": resp.usd_cost,
                "snapshot": self.budget.snapshot(),
            },
        )
        return resp

    def run_structured(
        self,
        messages: Iterable[ChatMessage],
        schema: dict[str, Any] | None = None,
    ) -> HermesResult:
        """Chat + cast in one shot. Returns the parsed structured output.

        On OutputInvalid, the agent issues one repair call where it
        appends a system reminder asking for valid JSON and gives the
        model another shot. Mirrors the agentcast retry-with-LLM pattern
        from MukundaKatta/agentcast-py.
        """
        msgs = list(messages)
        resp = self.chat(msgs)
        try:
            structured = cast_json(resp.text, schema)
        except OutputInvalid as exc:
            self._trace(
                "cast.invalid",
                {"reason": exc.reason, "raw_excerpt": exc.raw[:200]},
            )
            # One repair retry. The repaired call counts against the cap.
            repair_msgs = list(msgs) + [
                ChatMessage(role="assistant", content=resp.text),
                ChatMessage(
                    role="user",
                    content=(
                        "Your previous reply did not parse as JSON matching the schema. "
                        "Reply again with ONLY a JSON object in a ```json fenced block. "
                        f"reason={exc.reason}"
                    ),
                ),
            ]
            self._trace("cast.repair", {"reason": exc.reason})
            resp2 = self.chat(repair_msgs)
            structured = cast_json(resp2.text, schema)
            resp = resp2
        self._trace("cast.ok", {"keys": _summary_keys(structured)})
        return HermesResult(
            structured=structured,
            response=resp,
            fetched_url=None,
            fetched_chars=0,
        )

    def summarize_url(
        self,
        url: str,
        schema: dict[str, Any] | None = None,
        max_chars: int = 4000,
    ) -> HermesResult:
        """End-to-end: fetch under allowlist, then ask Hermes for JSON."""
        body = self.fetch_for_tool(url)
        snippet = body[:max_chars]
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are a careful summarizer. Reply with ONLY a JSON object "
                    "in a ```json fenced block. No prose outside the fence."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    "SUMMARIZE_URL\n\n"
                    f"URL: {url}\n\n"
                    "Document:\n"
                    f"{snippet}\n\n"
                    "Return a JSON object with keys: title (string), key_points (array of 3-5 strings), "
                    "sentiment (one of positive, neutral, negative), confidence (number between 0 and 1)."
                ),
            ),
        ]
        result = self.run_structured(messages, schema=schema)
        return HermesResult(
            structured=result.structured,
            response=result.response,
            fetched_url=url,
            fetched_chars=len(body),
        )


def _summary_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return sorted(value.keys())
    if isinstance(value, list):
        return [f"<list len={len(value)}>"]
    return [type(value).__name__]
