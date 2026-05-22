"""Egress guard tests."""

import pytest

from hermes_stack.guard import EgressDenied, EgressGuard


def test_allowlist_accepts_listed_host() -> None:
    guard = EgressGuard.from_iterable(["example.com"])
    assert guard.check("https://example.com/path?x=1") == "example.com"


def test_allowlist_denies_unlisted_host() -> None:
    guard = EgressGuard.from_iterable(["example.com"])
    with pytest.raises(EgressDenied) as info:
        guard.check("https://evil.example.com/steal")
    assert info.value.host == "evil.example.com"


def test_allowlist_is_case_insensitive() -> None:
    guard = EgressGuard.from_iterable(["Example.COM"])
    assert guard.check("https://EXAMPLE.com/") == "example.com"


def test_allowlist_rejects_empty_host() -> None:
    guard = EgressGuard.from_iterable(["example.com"])
    with pytest.raises(EgressDenied):
        guard.check("not-a-url")


def test_allow_adds_runtime_host() -> None:
    guard = EgressGuard()
    with pytest.raises(EgressDenied):
        guard.check("https://example.com/")
    guard.allow("example.com")
    assert guard.check("https://example.com/") == "example.com"
