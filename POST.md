---
title: "hermes-stack: four governance layers for Hermes Agent in 200 lines"
published: false
tags: hermesagentchallenge, devchallenge, agents, python
---

*This is a submission for the [Hermes Agent Challenge](https://dev.to/challenges/hermes-agent-2026-05-15).*

## What I Built

[hermes-stack](https://github.com/MukundaKatta/hermes-stack) is a tiny Python harness that wraps a Hermes Agent call with four governance layers:

1. A budget cap that stops the agent when it would spend past a dollar amount or a call count.
2. An egress allowlist that blocks any outbound HTTP fetch the agent did not declare up front.
3. An audit trace that writes every call, denial, and exception to a JSONL file.
4. A structured output check that pulls JSON out of the model reply, validates it against a schema, and retries once if the first reply does not parse.

Each layer fails closed. The audit trail captures the failure before it propagates. The whole thing is a single `HermesAgent` you wrap around a Hermes call.

I wrote a companion piece for the Write prompt: [Wrapping Hermes Agent with agent-stack](https://dev.to/mukundakatta/wrapping-hermes-agent-with-agent-stack-six-tiny-libs-for-the-boring-parts-c3a). This post is the working repo version, with the layers actually wired up, the demo running, and the tests green.

## Demo

A 60-second demo lives at `examples/url_summarizer.py`. It does three things in order, so each layer gets a chance to prove itself.

```bash
git clone https://github.com/MukundaKatta/hermes-stack.git
cd hermes-stack
python3 -m pip install -e ".[dev,schema]"
python3 -m pytest tests/ -v
python3 examples/url_summarizer.py
```

The demo runs offline by default with `HermesStub`. Set `OPENROUTER_API_KEY` and it switches to the real `nousresearch/hermes-3-llama-3.1-405b` free tier on OpenRouter.

Output, trimmed:

```
[hermes-stack] OPENROUTER_API_KEY not set; using HermesStub (offline).

STEP 1 / 3   Egress allowlist denies an unlisted host
egress denied: host=evil.example.com url=https://evil.example.com/steal-secrets

STEP 2 / 3   Hermes call returns structured JSON
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

STEP 3 / 3   Budget cap fires when spend pushes over
budget caught at call 2: kind=usd requested=$0.000392 cap=$0.000200

DONE
mode: stub
trace saved to /Users/ubl/hermes-stack/traces/run.jsonl
trace lines: 12
```

The trace JSONL has one line per event:

```
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

That ordering is the whole point. The egress check fires before any network call. The budget check sits between the model call and the next one. The cast check sits between the model reply and your code. The trace sits across all of them.

## Code

Repository: [github.com/MukundaKatta/hermes-stack](https://github.com/MukundaKatta/hermes-stack)

The interesting part is the agent class. Every Hermes call goes through the same path: reserve a call slot, emit a pre-event, call the model, record the spend, cast the output, emit a post-event. Any step can raise and the audit trail catches it.

```python
class HermesAgent:
    def chat(self, messages):
        self.budget.reserve_call()
        self._trace("hermes.call", {...})
        try:
            resp = self.client.complete(messages)
        except Exception as exc:
            self._trace("hermes.error", {...})
            raise
        try:
            self.budget.record_spend(resp.usd_cost)
        except BudgetExceeded as exc:
            self._trace("budget.exceeded", {...})
            raise
        self._trace("hermes.ok", {...})
        return resp
```

`run_structured` is the cast layer on top of `chat`. It tries to parse JSON out of the reply, and if the parse or schema check fails, it sends one repair prompt and tries again. Mirrors the pattern in [agentcast-py](https://github.com/MukundaKatta/agentcast-py).

```python
def run_structured(self, messages, schema=None):
    resp = self.chat(messages)
    try:
        structured = cast_json(resp.text, schema)
    except OutputInvalid as exc:
        self._trace("cast.invalid", {"reason": exc.reason})
        repair = messages + [
            ChatMessage(role="assistant", content=resp.text),
            ChatMessage(role="user", content=(
                "Reply again with ONLY a JSON object in a fenced block. "
                f"reason={exc.reason}"
            )),
        ]
        resp = self.chat(repair)
        structured = cast_json(resp.text, schema)
    return HermesResult(structured=structured, response=resp, ...)
```

The repair call counts against the cap. There is no infinite retry. One repair, then up.

The budget layer is the smallest piece worth showing. It is a dataclass with a lock, a USD ceiling, and a call count.

```python
@dataclass
class BudgetCap:
    usd_cap: float = 1.00
    call_cap: int = 50
    ...
    def record_spend(self, usd):
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
```

The lock matters when an agent kicks off two tool calls in parallel. Without it, both threads can read the current spend, both think there is room, and you blow past the cap by one call.

### My Tech Stack

- Python 3.10+ with `requests` and an optional `jsonschema` extra.
- Hermes-3-Llama-3.1-405B via the OpenRouter free tier. Hosted on `openrouter.ai`.
- A deterministic offline `HermesStub` so the demo runs without a key.
- Pytest for the 23-test suite.

The full stack is intentionally tiny. The four governance modules together are about 300 lines, including type hints and docstrings. The harness should be small enough that you read the whole thing before depending on it.

## How I Used Hermes Agent

Hermes-3-Llama-3.1-405B is the agentic model the challenge is built around. It is instruction-following enough that you can ask for JSON and usually get JSON. It does well on multi-step prompting where you walk it through a structured task.

What it is not is governed. Out of the box, a Hermes call has no per-session budget, no allowlist on tool fetches, no audit trail, and no contract on the reply shape. None of those are model problems. They are wrapper problems. So I wrote a wrapper.

The agentic capability I leaned on most is structured output with tool use. The `summarize_url` flow asks Hermes to act as a summarizer over a fetched document and return a JSON object with `title`, `key_points`, `sentiment`, and `confidence`. The structured output layer catches the cases where Hermes drifts into prose or skips a required key, and the repair prompt walks it back.

I picked Hermes for two reasons. First, the free tier on OpenRouter is enough to test the full path end-to-end, including the budget cap, without a paid key. Second, it runs locally if you want it to. The same wrapper works against `vllm` or `llama.cpp` serving the same Hermes-3 checkpoint. You only swap the URL inside `HermesClient`.

## What I learned

**Per-call cost matters more than total cost.** A cap of one dollar is easy to think about. A cap of $0.000196 per call is the number that actually catches a runaway loop.

**Repair prompts cost more than first prompts.** The cap has to leave room for the repair call. Size it to exactly one call's worth and the repair will never fire because the budget catches it first.

**Egress allowlists are smaller than people expect.** For the URL summarizer demo, the whole allowlist is `{example.com, openrouter.ai}`. Two hosts is the entire attack surface for outbound HTTP.

The repo is public and MIT-licensed: [github.com/MukundaKatta/hermes-stack](https://github.com/MukundaKatta/hermes-stack). Issues and PRs welcome.

---

Thanks to the DEV team and Nous Research for running the challenge. The four-layer pattern was waiting for an excuse to land in one place, and Hermes was a good excuse.
