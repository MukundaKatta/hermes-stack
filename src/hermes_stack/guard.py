"""Layer 2: egress allowlist.

Outbound HTTP hostname check. Wraps `requests.get` so the agent can only
reach hosts you explicitly allow. Patterned after MukundaKatta/agentguard-py.

The Hermes inference host is added to the allowlist automatically inside
HermesAgent, so callers only have to think about tool fetch targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import urlparse

import requests


class EgressDenied(Exception):
    """Raised when a fetch targets a host outside the allowlist."""

    def __init__(self, host: str, url: str) -> None:
        super().__init__(f"egress denied: host '{host}' not in allowlist (url={url})")
        self.host = host
        self.url = url


@dataclass
class EgressGuard:
    """Hostname allowlist for outbound HTTP fetches.

    Hosts match exactly. A future version could add suffix matching; for
    the demo, exact match keeps the rule obvious.
    """

    allowed_hosts: set[str] = field(default_factory=set)

    @classmethod
    def from_iterable(cls, hosts: Iterable[str]) -> "EgressGuard":
        return cls(allowed_hosts={h.lower() for h in hosts})

    def allow(self, host: str) -> None:
        self.allowed_hosts.add(host.lower())

    def check(self, url: str) -> str:
        """Verify the URL targets an allowed host. Returns the host on pass."""
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if not host:
            raise EgressDenied(host="<empty>", url=url)
        if host not in self.allowed_hosts:
            raise EgressDenied(host=host, url=url)
        return host

    def fetch(self, url: str, *, timeout: float = 10.0) -> str:
        """Allow-checked HTTP GET that returns the response body as text."""
        self.check(url)
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.text
