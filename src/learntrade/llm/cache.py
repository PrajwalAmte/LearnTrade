"""
LLM response cache — DuckDB-backed, keyed on (prompt_hash, model, parameters).

Cache hits return immediately at zero cost.
The cache makes runs reproducible: re-running a backtest with the same
prompts, model, and temperature always returns the same responses.
TTL is None by default (permanent cache); pass ttl_days to expire old entries.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

logger = logging.getLogger(__name__)


def _cache_key(model: str, messages: list[dict], params: dict[str, Any]) -> str:
    """Stable SHA-256 key over (model, messages, params)."""
    blob = json.dumps(
        {"model": model, "messages": messages, "params": params},
        sort_keys=True,
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(blob).hexdigest()


class LLMCache:
    """
    Persistent response cache stored in a single DuckDB table.

    Thread-safety: DuckDB file connections are not thread-safe; create one
    instance per process (or use a connection pool in future phases).
    """

    def __init__(self, db_path: Path, ttl_days: int | None = None) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(db_path))
        self._ttl_days = ttl_days
        self._init_schema()

    def _init_schema(self) -> None:
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key    VARCHAR PRIMARY KEY,
                model        VARCHAR NOT NULL,
                task         VARCHAR NOT NULL,
                prompt_hash  VARCHAR NOT NULL,
                response_json JSON NOT NULL,
                created_at   TIMESTAMP NOT NULL,
                hit_count    INTEGER NOT NULL DEFAULT 0
            )
        """)

    def get(self, key: str) -> dict[str, Any] | None:
        rows = self._con.execute(
            "SELECT response_json FROM llm_cache WHERE cache_key = ?", [key]
        ).fetchone()
        if rows is None:
            return None
        self._con.execute(
            "UPDATE llm_cache SET hit_count = hit_count + 1 WHERE cache_key = ?", [key]
        )
        raw = rows[0]
        return json.loads(raw) if isinstance(raw, str) else raw

    def set(self, key: str, model: str, task: str, response: dict[str, Any]) -> None:
        prompt_hash = key[:16]
        self._con.execute(
            """
            INSERT OR REPLACE INTO llm_cache
            VALUES (?, ?, ?, ?, ?, ?, 0)
            """,
            [key, model, task, prompt_hash, json.dumps(response),
             datetime.now(tz=timezone.utc).isoformat()],
        )

    def invalidate(self, key: str) -> None:
        self._con.execute("DELETE FROM llm_cache WHERE cache_key = ?", [key])

    def purge_expired(self) -> int:
        if self._ttl_days is None:
            return 0
        cutoff = datetime.now(tz=timezone.utc)
        from datetime import timedelta
        cutoff -= timedelta(days=self._ttl_days)
        result = self._con.execute(
            "DELETE FROM llm_cache WHERE created_at < ? RETURNING cache_key",
            [cutoff.isoformat()],
        ).fetchall()
        return len(result)

    def stats(self) -> dict[str, Any]:
        total, hits = self._con.execute(
            "SELECT COUNT(*), SUM(hit_count) FROM llm_cache"
        ).fetchone()
        return {"total_entries": total or 0, "total_hits": hits or 0}

    def close(self) -> None:
        self._con.close()

    def cache_key(
        self,
        model: str,
        messages: list[dict[str, str]],
        params: dict[str, Any],
    ) -> str:
        return _cache_key(model, messages, params)
