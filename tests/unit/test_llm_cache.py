"""
LLM cache tests — DuckDB persistence, hit/miss, stats.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from learntrade.llm.cache import LLMCache, _cache_key


def test_cache_key_stable():
    """Same inputs must always produce the same key."""
    k1 = _cache_key("gpt-4o", [{"role": "user", "content": "hi"}], {"temperature": 0.0})
    k2 = _cache_key("gpt-4o", [{"role": "user", "content": "hi"}], {"temperature": 0.0})
    assert k1 == k2


def test_cache_key_differs_on_model():
    k1 = _cache_key("gpt-4o", [{"role": "user", "content": "hi"}], {})
    k2 = _cache_key("gpt-4o-mini", [{"role": "user", "content": "hi"}], {})
    assert k1 != k2


def test_cache_key_differs_on_content():
    k1 = _cache_key("m", [{"role": "user", "content": "a"}], {})
    k2 = _cache_key("m", [{"role": "user", "content": "b"}], {})
    assert k1 != k2


def test_cache_miss_returns_none(tmp_path: Path):
    cache = LLMCache(tmp_path / "cache.duckdb")
    assert cache.get("nonexistent_key") is None
    cache.close()


def test_cache_set_and_get(tmp_path: Path):
    cache = LLMCache(tmp_path / "cache.duckdb")
    response = {"choices": [{"message": {"content": "hello"}}]}
    cache.set("key1", "gpt-4o", "extract", response)
    result = cache.get("key1")
    assert result is not None
    assert result["choices"][0]["message"]["content"] == "hello"
    cache.close()


def test_cache_hit_increments_count(tmp_path: Path):
    cache = LLMCache(tmp_path / "cache.duckdb")
    cache.set("k", "m", "t", {"data": 1})
    cache.get("k")
    cache.get("k")
    stats = cache.stats()
    assert stats["total_hits"] >= 2
    cache.close()


def test_cache_idempotent_set(tmp_path: Path):
    cache = LLMCache(tmp_path / "cache.duckdb")
    cache.set("k", "m", "t", {"v": 1})
    cache.set("k", "m", "t", {"v": 2})  # second write (INSERT OR REPLACE)
    result = cache.get("k")
    assert result["v"] == 2
    cache.close()


def test_cache_invalidate(tmp_path: Path):
    cache = LLMCache(tmp_path / "cache.duckdb")
    cache.set("k", "m", "t", {"v": 1})
    cache.invalidate("k")
    assert cache.get("k") is None
    cache.close()


def test_cache_stats_empty(tmp_path: Path):
    cache = LLMCache(tmp_path / "cache.duckdb")
    stats = cache.stats()
    assert stats["total_entries"] == 0
    assert stats["total_hits"] == 0
    cache.close()


def test_cache_persists_across_instances(tmp_path: Path):
    path = tmp_path / "cache.duckdb"
    cache1 = LLMCache(path)
    cache1.set("k", "m", "t", {"persisted": True})
    cache1.close()

    cache2 = LLMCache(path)
    result = cache2.get("k")
    assert result is not None
    assert result["persisted"] is True
    cache2.close()
