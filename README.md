# hermes-stack

A four-layer governance harness for Hermes Agent calls.
Drop one `HermesAgent` around the model and you get a budget cap, an egress allowlist, an audit trace, and structured-output enforcement on every call.
Built for the [DEV Community Hermes Agent Challenge](https://dev.to/challenges/hermes-agent-2026-05-15).

## What it gives you

| Layer | What it does | Library it mirrors |
| --- | --- | --- |
| `BudgetCap` | USD ceiling + per-call ceiling. Raises `BudgetExceeded` on overflow. | [token-budget-py](https://github.com/MukundaKatta/token-budget-py), [agentleash](https://github.com/MukundaKatta/agentleash) |
| `EgressGuard` | Hostname allowlist on every outbound fetch. Raises `EgressDenied`. | [agentguard-py](https://github.com/MukundaKatta/agentguard-py) |
| `Tracer` | Append-only JSONL of every call, denial, exception. | [agenttrace-rs](https://github.com/MukundaKatta/agenttrace-rs) |
| `cast_json` | Pulls JSON out of a chatty model reply, validates against a schema, retries once. | [agentcast-py](https://github.com/MukundaKatta/agentcast-py) |

Each layer fails closed. The audit trail captures the failure before it propagates.

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

## The four layers in plain words

**Budget cap.** Pick a dollar number you can live with as the worst case. Pick a call-count number too. If a tool call would push past either, the agent stops. The cap is thread-safe, so two tools running in parallel cannot race past it.

**Egress allowlist.** Tools fetch URLs. Models hallucinate URLs. The allowlist is the safety net. The Hermes inference host is added automatically; you only think about the URLs your tools actually need.

**Audit trace.** Every call, every denial, every exception gets one JSONL row. You can replay a session, attach it to a bug report, or feed it into a regression test. The file is flushed after each row so a crashed agent still leaves a trail.

**Structured output.** Hermes-3 follows instructions, but real responses still drift. `cast_json` strips chat prose, extracts the JSON block (fenced or inline), and validates against your schema. If the first reply fails, the agent issues one repair prompt and casts again.

## Tests

```bash
python3 -m pytest tests/ -v
```

23 tests cover budget, guard, trace, cast, and an end-to-end agent run with both happy-path and failure-path cases (egress denial, budget overflow, repair retry).

## How it composes with the agent-stack

`hermes-stack` is the Hermes Agent-specific composition of the four layers above. The standalone libraries live in their own repos so you can pull them into other agents (Claude, GPT, Bedrock, local) without dragging the Hermes wrapper along.

- Python: [agentguard-py](https://github.com/MukundaKatta/agentguard-py), [agentleash](https://github.com/MukundaKatta/agentleash), [agentcast-py](https://github.com/MukundaKatta/agentcast-py), [token-budget-py](https://github.com/MukundaKatta/token-budget-py)
- Rust: [agentguard-rs](https://github.com/MukundaKatta/agentguard-rs), [agentleash-rs](https://github.com/MukundaKatta/agentleash-rs), [agentcast-rs](https://github.com/MukundaKatta/agentcast-rs), [agenttrace-rs](https://github.com/MukundaKatta/agenttrace-rs), [agentsnap-rs](https://github.com/MukundaKatta/agentsnap-rs)

Companion write-up: [Wrapping Hermes Agent with agent-stack](https://dev.to/mukundakatta/wrapping-hermes-agent-with-agent-stack-six-tiny-libs-for-the-boring-parts-c3a).

## License

MIT.
