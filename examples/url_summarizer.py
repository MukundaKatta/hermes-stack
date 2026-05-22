"""URL summarizer demo for hermes-stack.

Runs a single Hermes Agent call that:
  1. Tries to fetch a denied URL (proves egress denial works).
  2. Fetches an allowlisted URL and asks Hermes to return a structured summary.
  3. Hits the budget cap on a second tight-budget agent and surfaces the error.

If OPENROUTER_API_KEY is set, the real Hermes-3-Llama-3.1-405B model on
OpenRouter handles the call. Otherwise the deterministic HermesStub
plays the role and the demo still proves all four layers.

Run:
    python3 examples/url_summarizer.py

Optional env:
    OPENROUTER_API_KEY=...     # use the real Hermes-3-405B free tier
    HERMES_STACK_URL=...       # override the target URL
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make ./src importable when running the file directly from the repo root.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from hermes_stack import (  # noqa: E402  (after sys.path tweak)
    BudgetCap,
    BudgetExceeded,
    ChatMessage,
    EgressDenied,
    EgressGuard,
    HermesAgent,
    HermesClient,
    HermesStub,
    Tracer,
)


SUMMARY_SCHEMA = {
    "type": "object",
    "required": ["title", "key_points", "sentiment", "confidence"],
    "properties": {
        "title": {"type": "string"},
        "key_points": {"type": "array", "minItems": 1},
        "sentiment": {"type": "string"},
        "confidence": {"type": "number"},
    },
}

DEFAULT_URL = "https://example.com/"
DENIED_URL = "https://evil.example.com/steal-secrets"


def make_client():
    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            client = HermesClient()
            print(f"[hermes-stack] using real Hermes via OpenRouter: model={client.model}")
            return client, "real"
        except Exception as exc:  # noqa: BLE001
            print(f"[hermes-stack] real client init failed ({exc}); falling back to stub.")
    print("[hermes-stack] OPENROUTER_API_KEY not set; using HermesStub (offline).")
    return HermesStub(), "stub"


def banner(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main() -> int:
    url = os.environ.get("HERMES_STACK_URL", DEFAULT_URL)
    trace_path = HERE.parent / "traces" / "run.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path.exists():
        trace_path.unlink()

    client, mode = make_client()
    target_host = url.split("://", 1)[-1].split("/", 1)[0]

    with Tracer(trace_path, run_id="demo") as tracer:
        agent = HermesAgent(
            client=client,
            budget=BudgetCap(usd_cap=0.50, call_cap=5),
            guard=EgressGuard.from_iterable([target_host]),
            tracer=tracer,
        )

        # ---- Step 1: prove egress denial fires ---------------------------
        banner("STEP 1 / 3   Egress allowlist denies an unlisted host")
        try:
            agent.fetch_for_tool(DENIED_URL)
        except EgressDenied as exc:
            print(f"egress denied: host={exc.host} url={exc.url}")
        else:
            print("WARNING: expected EgressDenied was not raised")

        # ---- Step 2: real (or stub) summary -----------------------------
        banner("STEP 2 / 3   Hermes call returns structured JSON")
        try:
            result = agent.summarize_url(url, schema=SUMMARY_SCHEMA)
        except Exception as exc:  # noqa: BLE001
            print(f"summarize_url failed: {type(exc).__name__}: {exc}")
            return 2

        snap = agent.budget.snapshot()
        print(f"fetched {result.fetched_chars} chars from {result.fetched_url}")
        print(
            "tokens: "
            f"prompt={result.response.prompt_tokens} "
            f"completion={result.response.completion_tokens}"
        )
        print(f"call cost: ${result.response.usd_cost:.6f}")
        print(f"budget snapshot: {snap}")
        print("structured output:")
        print(json.dumps(result.structured, indent=2))

        # ---- Step 3: prove budget cap fires -----------------------------
        banner("STEP 3 / 3   Budget cap fires when spend pushes over")
        tight = HermesAgent(
            client=HermesStub(),  # use the stub here so we know the per-call cost
            budget=BudgetCap(usd_cap=0.0002, call_cap=20),
            guard=EgressGuard(),
            tracer=tracer,
        )
        caught = False
        for i in range(1, 10):
            try:
                tight.chat([ChatMessage(role="user", content=f"ping {i}")])
            except BudgetExceeded as exc:
                caught = True
                print(
                    f"budget caught at call {i}: kind={exc.kind} "
                    f"requested=${exc.requested:.6f} cap=${exc.cap:.6f}"
                )
                break
        if not caught:
            print("WARNING: budget cap did not fire as expected")

    banner("DONE")
    print(f"mode: {mode}")
    print(f"trace saved to {trace_path}")
    print(f"trace lines: {sum(1 for _ in trace_path.open())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
