"""
LLM router tests — budget enforcement, disable guard, extract_text parsing.
These tests run without an LLM API key using mock clients.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from leantrade.llm.router import BudgetExceededError, BudgetTracker, LLMRouter


# ── BudgetTracker ─────────────────────────────────────────────────────────────

def test_budget_check_passes_under_limit():
    bt = BudgetTracker(per_run_limit=5.0, per_day_limit=20.0)
    bt.check(4.99)  # should not raise


def test_budget_check_raises_over_run_limit():
    bt = BudgetTracker(per_run_limit=1.0, per_day_limit=100.0)
    bt.record(0.95)
    with pytest.raises(BudgetExceededError, match="Run budget"):
        bt.check(0.10)


def test_budget_check_raises_over_day_limit():
    bt = BudgetTracker(per_run_limit=100.0, per_day_limit=1.0)
    bt.record(0.95)
    with pytest.raises(BudgetExceededError, match="Daily budget"):
        bt.check(0.10)


def test_budget_record_accumulates():
    bt = BudgetTracker(per_run_limit=10.0, per_day_limit=10.0)
    bt.record(1.0)
    bt.record(2.0)
    assert bt.run_spent == pytest.approx(3.0)


# ── extract_text ──────────────────────────────────────────────────────────────

def _make_router(tmp_path: Path) -> LLMRouter:
    """Create a router with LLM disabled (safe for tests without API keys)."""
    with patch("leantrade.llm.router.get_config") as mock_cfg, \
         patch("leantrade.llm.router.get_secrets"):
        cfg = MagicMock()
        cfg.llm.enabled = False
        cfg.llm.budget_usd_per_run = 5.0
        cfg.llm.budget_usd_per_day = 20.0
        cfg.llm.cache = True
        cfg.llm.models = MagicMock()
        cfg.workspace.data_dir = tmp_path
        mock_cfg.return_value = cfg
        return LLMRouter(cache_db_path=tmp_path / "cache.duckdb")


def test_router_raises_when_disabled(tmp_path: Path):
    router = _make_router(tmp_path)
    with pytest.raises(RuntimeError, match="disabled"):
        asyncio.run(router.complete("extract", [{"role": "user", "content": "hi"}]))


def test_extract_text_openai_format(tmp_path: Path):
    router = _make_router(tmp_path)
    response = {"choices": [{"message": {"content": "hello world"}}]}
    assert router.extract_text(response) == "hello world"


def test_extract_text_anthropic_format(tmp_path: Path):
    router = _make_router(tmp_path)
    response = {"content": [{"text": "anthropic response"}]}
    assert router.extract_text(response) == "anthropic response"


def test_extract_text_ollama_format(tmp_path: Path):
    router = _make_router(tmp_path)
    response = {"message": {"content": "ollama response"}}
    assert router.extract_text(response) == "ollama response"


def test_extract_text_unknown_format_returns_empty(tmp_path: Path):
    router = _make_router(tmp_path)
    assert router.extract_text({"unexpected": "format"}) == ""


# ── Cache integration with router ─────────────────────────────────────────────

def test_router_cache_stores_responses(tmp_path: Path):
    """Router must write to cache; second identical call must be a hit."""
    with patch("leantrade.llm.router.get_config") as mock_cfg, \
         patch("leantrade.llm.router.get_secrets"):
        cfg = MagicMock()
        cfg.llm.enabled = True
        cfg.llm.budget_usd_per_run = 5.0
        cfg.llm.budget_usd_per_day = 20.0
        cfg.llm.cache = True
        cfg.llm.models = MagicMock()
        cfg.workspace.data_dir = tmp_path
        mock_cfg.return_value = cfg

        router = LLMRouter(cache_db_path=tmp_path / "cache.duckdb")
        # Inject a mock client that returns a fixed response
        mock_client = MagicMock()
        mock_client.model = "mock-model"
        mock_client.complete = AsyncMock(return_value={
            "choices": [{"message": {"content": "cached response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })
        mock_client.extract_usage = MagicMock(return_value={
            "prompt_tokens": 10, "completion_tokens": 5
        })
        router._clients["extract"] = mock_client

        messages = [{"role": "user", "content": "test prompt"}]
        # First call → hits the LLM
        r1 = asyncio.run(router.complete("extract", messages))
        assert mock_client.complete.call_count == 1

        # Second identical call → should hit cache, NOT call client again
        r2 = asyncio.run(router.complete("extract", messages))
        assert mock_client.complete.call_count == 1  # still 1
        assert router.extract_text(r1) == router.extract_text(r2)
