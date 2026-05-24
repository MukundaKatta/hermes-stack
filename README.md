# hermes-stack

A six-layer governance harness for Hermes Agent calls.
Drop one `HermesAgent` around the model and you get a budget cap, an egress allowlist, an audit trace, structured-output enforcement, tool-arg validation, and exponential-backoff retries on transient failures.
Built for the [DEV Community Hermes Agent Challenge](https://dev.to/challenges/hermes-agent-2026-05-15).

## What it gives you

| Layer | What it does | Library it mirrors |
| --- | --- | --- |
| `BudgetCap` | USD ceiling + per-call ceiling. Raises `BudgetExceeded` on overflow. | [token-budget-py](https://github.com/MukundaKatta/token-budget-py), [agentleash](https://github.com/MukundaKatta/agentleash) |
| `SharedBudgetCap` | Same USD + call ceiling **across multiple processes**, coordinated via `fcntl.flock` on a JSON state file. Fixes the "three workers, $5 cap each, $15 effective ceiling" footgun. | [token-budget-py](https://github.com/MukundaKatta/token-budget-py), [hermes-budget-skin](https://github.com/MukundaKatta/hermes-budget-skin) |
| `EgressGuard` | Hostname allowlist on every outbound fetch. Raises `EgressDenied`. | [agentguard-py](https://github.com/MukundaKatta/agentguard-py) |
| `Tracer` | Append-only JSONL of every call, denial, exception. | [agenttrace-rs](https://github.com/MukundaKatta/agenttrace-rs) |
| `cast_json` | Pulls JSON out of a chatty model reply, validates against a schema, retries once. | [agentcast-py](https://github.com/MukundaKatta/agentcast-py) |
| `ToolVet` | Validates tool args **before** the tool runs. Raises `ToolArgError` with an LLM-readable hint so the model can repair the call without burning a tool execution cycle. | [agentvet](https://github.com/MukundaKatta/agentvet), [agentvet-rs](https://github.com/MukundaKatta/agentvet-rs) |
| `RetryPolicy` | Exponential backoff (full jitter) on transient failures: `rate_limit_error`, `overloaded_error`, `api_error`, `timeout`. Each attempt counts against the budget cap so a runaway retry loop cannot quietly exceed `call_cap`. | [llm-retry-py](https://github.com/MukundaKatta/llm-retry-py), [llm-retry](https://crates.io/crates/llm-retry) |

Each layer fails closed. The audit trail captures the failure before it propagates.

### Retry usage

```python
from hermes_stack import HermesAgent, BudgetCap, RetryPolicy

agent = HermesAgent(
    client=HermesClient(),
    budget=BudgetCap(usd_cap=1.00, call_cap=20),
    retry=RetryPolicy(max_attempts=4, base_delay_s=0.5, jitter="full"),
)

# A transient 429 / overloaded_error retries automatically up to 4 times
# with backoff; auth or validation errors propagate immediately.
resp = agent.chat([ChatMessage(role="user", content="hi")])
```

`RetryPolicy` exposes `delay_for(attempt)` so you can preview the backoff
curve, and supports `jitter="full" | "equal" | "none"` (the last is for
deterministic tests). Pair with a `ToolVet` and `Tracer` for full audit
of every retried attempt (`hermes.retry` events land in the JSONL).

### One-line wiring from env

Wiring six layers by hand is fine for tests but tedious for a long-running
agent.  `HermesConfig.from_env()` reads a small env-var surface and hands
back a fully configured `HermesAgent`:

```bash
export OPENROUTER_API_KEY=sk-or-...
export HERMES_BUDGET_USD_CAP=5.00
export HERMES_BUDGET_CALL_CAP=200
export HERMES_BUDGET_PATH=/var/run/hermes/budget.json   # multi-process safe
export HERMES_ALLOW_HOSTS=api.anthropic.com,docs.python.org
export HERMES_TRACE_PATH=/var/log/hermes-audit.jsonl
export HERMES_RETRY_MAX_ATTEMPTS=4
export HERMES_RETRY_BASE_DELAY=0.5
```

```python
from hermes_stack import agent_from_env

agent = agent_from_env()   # full stack ready
resp = agent.chat([ChatMessage(role="user", content="hi")])
```

Skip any env var to fall back to its default. With nothing set you get a
`HermesStub` client, a $1 in-process budget cap, no retries — exactly
what the smoke demo needs. `HermesConfig` is also exposed as a dataclass
if you want to build manually for tests or introspect what was picked up
from the environment.

## Quickstart

```bash
git clone https://github.com/MukundaKatta/hermes-stack.git
cd hermes-stack
python3 -m pip install -e ".[dev,schema]"
python3 -m pytest tests/ -v
python3 examples/url_summarizer.py
```

The demo runs offline by default with `HermesStub`.
Set `OPENROUTER_API_KEY` to hit the real `nousresearch/hermes-3-llama-3.1-405b` free tier on [OpenRouter](https://openrouter.ai/).

```bash
export OPENROUTER_API_KEY=sk-or-...
python3 examples/url_summarizer.py
```

## Minimal usage

```python
from hermes_stack import (
    BudgetCap, ChatMessage, EgressGuard, HermesAgent, HermesStub, Tracer,
)

SCHEMA = {
    "type": "object",
    "required": ["title", "key_points", "sentiment", "confidence"],
}

with Tracer("traces/run.jsonl") as tracer:
    agent = HermesAgent(
        client=HermesStub(),                              # swap for HermesClient()
        budget=BudgetCap(usd_cap=0.50, call_cap=5),
        guard=EgressGuard.from_iterable(["example.com"]),
        tracer=tracer,
    )
    result = agent.summarize_url("https://example.com/", schema=SCHEMA)
    print(result.structured)
```

## Demo output (stub mode)

```
[hermes-stack] OPENROUTER_API_KEY not set; using HermesStub (offline).

============================================================
STEP 1 / 3   Egress allowlist denies an unlisted host
============================================================
egress denied: host=evil.example.com url=https://evil.example.com/steal-secrets

============================================================
STEP 2 / 3   Hermes call returns structured JSON
============================================================
fetched 528 chars from https://example.com/
tokens: prompt=120 completion=140
call cost: $0.000196
budget snapshot: {'spent_usd': 0.000196, 'usd_cap': 0.5, 'calls': 1, 'call_cap': 5}
structured output:
{
  "title": "Stub Summary",
  "key_points": [
    "The hermes-stack wraps Hermes Agent calls with four governance layers.",
    "Each layer fails closed and writes to the audit trail.",
    "The structured-output check rejects model JSON that misses a required key."
  ],
  "sentiment": "neutral",
  "confidence": 0.82
}

============================================================
STEP 3 / 3   Budget cap fires when spend pushes over
============================================================
budget caught at call 2: kind=usd requested=$0.000392 cap=$0.000200

============================================================
DONE
============================================================
mode: stub
trace saved to /Users/ubl/hermes-stack/traces/run.jsonl
trace lines: 12
```

The audit JSONL contains:

```text
run.start
egress.denied
egress.allowed
tool.fetch.ok
hermes.call
hermes.ok
cast.ok
hermes.call
hermes.ok
hermes.call
budget.exceeded
run.end
```

One line per event. Every entry has `ts`, `event`, `run_id`, and a payload. A crashed run still leaves a `run.end` row with the exception type and message.

## The five layers in plain words

**Budget cap.** Pick a dollar number you can live with as the worst case. Pick a call-count number too. If a tool call would push past either, the agent stops. The cap is thread-safe, so two tools running in parallel cannot race past it.

**Shared budget cap (multi-process).** The plain `BudgetCap` lives in one process. The moment you spawn a worker pool, a LaunchAgent, or a CLI ad-hoc next to a daemon, each one carries its own counter — three $5 caps = $15 effective ceiling. `SharedBudgetCap` persists the running spend in a JSON file under `fcntl.flock`, so every cooperating process reads and writes the same total. Drop-in compatible with `BudgetCap` on the spend-recording side.

```python
from hermes_stack import HermesAgent, SharedBudgetCap

agent = HermesAgent(
    budget=SharedBudgetCap(path="/var/run/hermes/budget.json", usd_cap=5.00),
)
```

**Egress allowlist.** Tools fetch URLs. Models hallucinate URLs. The allowlist is the safety net. The Hermes inference host is added automatically; you only think about the URLs your tools actually need.

**Audit trace.** Every call, every denial, every exception gets one JSONL row. You can replay a session, attach it to a bug report, or feed it into a regression test. The file is flushed after each row so a crashed agent still leaves a trail.

**Structured output.** Hermes-3 follows instructions, but real responses still drift. `cast_json` strips chat prose, extracts the JSON block (fenced or inline), and validates against your schema. If the first reply fails, the agent issues one repair prompt and casts again.

**Tool-arg vet.** Models hallucinate tool arguments the way they hallucinate URLs. Register each tool's schema with `ToolVet`; the agent then validates model-produced args before invoking the tool. Failures raise `ToolArgError` carrying a one-line, LLM-readable hint suitable for direct injection into a repair prompt.

```python
from hermes_stack import HermesAgent, ToolVet

vet = ToolVet()
vet.register("fetch_url", {
    "type": "object",
    "required": ["url"],
    "properties": {"url": {"type": "string"}, "timeout": {"type": "number"}},
})

agent = HermesAgent(vet=vet)
agent.call_tool("fetch_url", model_produced_args, real_fetch_fn)
```

`call_tool` fails closed if no vet is registered — the agent refuses to run an arbitrary tool without a schema.

## Tests

```bash
python3 -m pytest tests/ -v
```

44 tests cover budget (in-process + shared multi-process), guard, trace, cast, vet, and an end-to-end agent run with both happy-path and failure-path cases (egress denial, budget overflow, repair retry, bad tool args). The shared-budget suite includes a real `multiprocessing.Pool` scenario that proves three workers cannot collectively breach a shared cap.

## How it composes with the agent-stack

`hermes-stack` is the Hermes Agent-specific composition of the four layers above. The standalone libraries live in their own repos so you can pull them into other agents (Claude, GPT, Bedrock, local) without dragging the Hermes wrapper along.

- Python: [agentguard-py](https://github.com/MukundaKatta/agentguard-py), [agentleash](https://github.com/MukundaKatta/agentleash), [agentcast-py](https://github.com/MukundaKatta/agentcast-py), [token-budget-py](https://github.com/MukundaKatta/token-budget-py)
- Rust: [agentguard-rs](https://github.com/MukundaKatta/agentguard-rs), [agentleash-rs](https://github.com/MukundaKatta/agentleash-rs), [agentcast-rs](https://github.com/MukundaKatta/agentcast-rs), [agenttrace-rs](https://github.com/MukundaKatta/agenttrace-rs), [agentsnap-rs](https://github.com/MukundaKatta/agentsnap-rs)

Companion write-up: [Wrapping Hermes Agent with agent-stack](https://dev.to/mukundakatta/wrapping-hermes-agent-with-agent-stack-six-tiny-libs-for-the-boring-parts-c3a).

## License

MIT.
