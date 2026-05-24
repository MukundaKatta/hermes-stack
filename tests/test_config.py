"""Tests for HermesConfig.from_env() — opinionated env-driven setup."""

from __future__ import annotations

import pytest

from hermes_stack import (
    BudgetCap,
    HermesAgent,
    HermesConfig,
    HermesStub,
    SharedBudgetCap,
    Tracer,
    agent_from_env,
)
from hermes_stack.hermes import HermesClient


# Every test starts with a clean env to avoid bleed from one test poisoning
# the next.  We can't just `monkeypatch.setenv` everything we care about and
# call it good — fields we don't set need to be ABSENT, not stale.
ENV_KEYS = (
    "OPENROUTER_API_KEY",
    "HERMES_MODEL",
    "HERMES_BUDGET_USD_CAP",
    "HERMES_BUDGET_CALL_CAP",
    "HERMES_BUDGET_PATH",
    "HERMES_ALLOW_HOSTS",
    "HERMES_TRACE_PATH",
    "HERMES_RETRY_MAX_ATTEMPTS",
    "HERMES_RETRY_BASE_DELAY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    yield


# ---------------------------------------------------------------------------
# from_env defaults
# ---------------------------------------------------------------------------


def test_from_env_with_empty_env_returns_defaults():
    cfg = HermesConfig.from_env()
    assert cfg.use_stub is True              # no OPENROUTER_API_KEY → stub
    assert cfg.budget_usd_cap == 1.00
    assert cfg.budget_call_cap == 50
    assert cfg.budget_path is None
    assert cfg.allow_hosts is None
    assert cfg.trace_path is None
    assert cfg.retry_max_attempts == 1       # 1 = no retries
    assert cfg.model is None


def test_from_env_picks_up_openrouter_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-something")
    cfg = HermesConfig.from_env()
    assert cfg.use_stub is False


def test_from_env_picks_up_budget_caps(monkeypatch):
    monkeypatch.setenv("HERMES_BUDGET_USD_CAP", "5.50")
    monkeypatch.setenv("HERMES_BUDGET_CALL_CAP", "120")
    cfg = HermesConfig.from_env()
    assert cfg.budget_usd_cap == 5.50
    assert cfg.budget_call_cap == 120


def test_from_env_picks_up_allow_hosts(monkeypatch):
    monkeypatch.setenv(
        "HERMES_ALLOW_HOSTS",
        "api.anthropic.com, example.com, host.with-port.io",
    )
    cfg = HermesConfig.from_env()
    assert cfg.allow_hosts == [
        "api.anthropic.com",
        "example.com",
        "host.with-port.io",
    ]


def test_from_env_empty_allow_hosts_is_none(monkeypatch):
    monkeypatch.setenv("HERMES_ALLOW_HOSTS", "  ,  ,  ")
    cfg = HermesConfig.from_env()
    assert cfg.allow_hosts is None


def test_from_env_picks_up_retry_attempts(monkeypatch):
    monkeypatch.setenv("HERMES_RETRY_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("HERMES_RETRY_BASE_DELAY", "0.25")
    cfg = HermesConfig.from_env()
    assert cfg.retry_max_attempts == 5
    assert cfg.retry_base_delay_s == 0.25


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------


def test_build_with_defaults_uses_stub_and_inprocess_budget(tmp_path):
    cfg = HermesConfig.from_env()
    agent = cfg.build()
    assert isinstance(agent, HermesAgent)
    assert isinstance(agent.client, HermesStub)
    assert isinstance(agent.budget, BudgetCap)
    assert agent.tracer is None
    assert agent.retry_policy is None


def test_build_with_real_client_when_api_key_present(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    agent = HermesConfig.from_env().build()
    assert isinstance(agent.client, HermesClient)


def test_build_with_budget_path_uses_shared_budget(tmp_path, monkeypatch):
    p = tmp_path / "shared.json"
    monkeypatch.setenv("HERMES_BUDGET_PATH", str(p))
    monkeypatch.setenv("HERMES_BUDGET_USD_CAP", "2.50")
    agent = HermesConfig.from_env().build()
    assert isinstance(agent.budget, SharedBudgetCap)
    assert p.exists()
    snap = agent.budget.snapshot()
    assert snap["usd_cap"] == 2.50


def test_build_creates_missing_parent_dir_for_budget_path(tmp_path, monkeypatch):
    """SharedBudgetCap requires the parent dir; from_env should create it."""
    p = tmp_path / "subdir" / "shared.json"
    monkeypatch.setenv("HERMES_BUDGET_PATH", str(p))
    HermesConfig.from_env().build()
    assert p.parent.is_dir()
    assert p.exists()


def test_build_attaches_tracer_when_trace_path_set(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    agent = HermesConfig.from_env().build()
    assert isinstance(agent.tracer, Tracer)


def test_build_attaches_retry_policy_when_attempts_gt_1(monkeypatch):
    monkeypatch.setenv("HERMES_RETRY_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("HERMES_RETRY_BASE_DELAY", "0.1")
    agent = HermesConfig.from_env().build()
    assert agent.retry_policy is not None
    assert agent.retry_policy.max_attempts == 4
    assert agent.retry_policy.base_delay_s == 0.1


def test_build_skips_retry_when_attempts_is_1(monkeypatch):
    """1 attempt = the default = no retry layer attached."""
    monkeypatch.setenv("HERMES_RETRY_MAX_ATTEMPTS", "1")
    agent = HermesConfig.from_env().build()
    assert agent.retry_policy is None


def test_build_adds_allow_hosts_to_guard(monkeypatch):
    monkeypatch.setenv("HERMES_ALLOW_HOSTS", "example.com, other.example")
    agent = HermesConfig.from_env().build()
    # HERMES_API_HOST is always added by the agent constructor; our hosts
    # land on top.
    host = agent.guard.check("https://example.com/page")
    assert host == "example.com"
    host = agent.guard.check("https://other.example/path")
    assert host == "other.example"


# ---------------------------------------------------------------------------
# agent_from_env top-level helper
# ---------------------------------------------------------------------------


def test_agent_from_env_one_liner():
    """The full one-line factory builds a usable agent."""
    agent = agent_from_env()
    assert isinstance(agent, HermesAgent)
    assert isinstance(agent.client, HermesStub)


def test_agent_from_env_accepts_vet():
    """ToolVet stays a constructor arg, not an env var."""
    from hermes_stack import ToolVet

    vet = ToolVet()
    vet.register("noop", {"type": "object"})
    agent = agent_from_env(vet=vet)
    assert agent.vet is vet
