"""Hermes inference client.

Two implementations behind one Protocol:

- HermesClient: real OpenRouter call against
  `nousresearch/hermes-3-llama-3.1-405b` (free tier). Requires
  OPENROUTER_API_KEY in the environment.

- HermesStub: deterministic local stub for offline runs, demos, and
  tests. Emits a canned JSON-friendly response that matches the
  structured-output schema used by the URL summarizer.

The agent layer treats both the same way. Swap the client without
changing the rest of the stack.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

import requests


HERMES_API_HOST = "openrouter.ai"
HERMES_DEFAULT_MODEL = "nousresearch/hermes-3-llama-3.1-405b"

# Rough public OpenRouter pricing for the Hermes-3-405B free tier as of
# May 2026 (free models still report a cost-per-token for accounting).
# Numbers are USD per 1K tokens. They are deliberately conservative so
# the budget cap fires earlier rather than later.
PROMPT_USD_PER_1K = 0.0007
COMPLETION_USD_PER_1K = 0.0008


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class HermesResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    usd_cost: float
    raw: dict[str, Any]


class _HermesProto(Protocol):
    model: str

    def complete(self, messages: Iterable[ChatMessage]) -> HermesResponse:
        ...


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """USD cost for a call, using the public per-1K rates above."""
    return round(
        (prompt_tokens / 1000.0) * PROMPT_USD_PER_1K
        + (completion_tokens / 1000.0) * COMPLETION_USD_PER_1K,
        6,
    )


class HermesClient:
    """Real Hermes call via the OpenRouter chat-completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = HERMES_DEFAULT_MODEL,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or ""
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Use HermesStub for offline runs."
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, messages: Iterable[ChatMessage]) -> HermesResponse:
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "max_tokens": 800,
            "temperature": 0.1,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/MukundaKatta/hermes-stack",
            "X-Title": "hermes-stack",
        }
        resp = requests.post(url, json=body, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        usd_cost = estimate_cost(prompt_tokens, completion_tokens)
        return HermesResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usd_cost=usd_cost,
            raw=data,
        )


class HermesStub:
    """Deterministic offline Hermes for demos and tests.

    The stub inspects the most recent user message. If the message
    contains the cue `SUMMARIZE_URL`, the stub returns a fenced JSON
    block matching the URL-summarizer schema. Otherwise it echoes back
    a tiny acknowledgement so callers can still exercise the rest of
    the stack.
    """

    model = "hermes-stub-offline"

    def __init__(self, prompt_tokens: int = 120, completion_tokens: int = 140) -> None:
        self._pt = prompt_tokens
        self._ct = completion_tokens

    def complete(self, messages: Iterable[ChatMessage]) -> HermesResponse:
        messages = list(messages)
        last = next(
            (m for m in reversed(messages) if m.role == "user"),
            ChatMessage(role="user", content=""),
        )
        content = last.content
        if "SUMMARIZE_URL" in content:
            text = (
                "Here is the structured summary you asked for:\n"
                "```json\n"
                + json.dumps(
                    {
                        "title": "Stub Summary",
                        "key_points": [
                            "The hermes-stack wraps Hermes Agent calls with four governance layers.",
                            "Each layer fails closed and writes to the audit trail.",
                            "The structured-output check rejects model JSON that misses a required key.",
                        ],
                        "sentiment": "neutral",
                        "confidence": 0.82,
                    },
                    indent=2,
                )
                + "\n```\n"
            )
        else:
            text = "stub: " + (content[:80] if content else "(empty)")
        return HermesResponse(
            text=text,
            prompt_tokens=self._pt,
            completion_tokens=self._ct,
            usd_cost=estimate_cost(self._pt, self._ct),
            raw={"model": self.model, "stub": True},
        )
