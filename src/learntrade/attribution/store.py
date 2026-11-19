"""
Attribution store — writes DecisionRecord after each decision,
updates realised returns when the horizon elapses, and supports queries.

Schema: one row per decision (instrument × date × horizon).
LLMs explain allocations in thesis/counter_thesis; weights come from the optimiser.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from learntrade.core.types import DecisionRecord, InstrumentId


class AttributionStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id          VARCHAR PRIMARY KEY,
                instrument_id        VARCHAR NOT NULL,
                horizon              VARCHAR NOT NULL,
                as_of                TIMESTAMP NOT NULL,
                expected_return      DOUBLE,
                p_positive           DOUBLE,
                cross_sectional_rank DOUBLE,
                target_weight        DOUBLE,
                current_weight       DOUBLE,
                trade                VARCHAR,
                thesis               VARCHAR,
                counter_thesis       VARCHAR,
                invalidation_conds   JSON,
                agreement_with_quant VARCHAR,
                model_contributions  JSON,
                llm_cost_usd         DOUBLE DEFAULT 0.0,
                pit_level            VARCHAR,
                realised_return      DOUBLE,
                created_at           TIMESTAMP NOT NULL
            )
        """)

    # ── Write ──────────────────────────────────────────────────────────────

    def write(self, record: DecisionRecord) -> None:
        fc = record.forecast
        attr = record.attribution
        self._con.execute(
            """
            INSERT OR REPLACE INTO decisions VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                record.decision_id,
                str(record.instrument_id),
                record.horizon,
                record.as_of.isoformat(),
                float(fc.expected_return) if fc else None,
                float(fc.p_positive) if fc else None,
                float(fc.rank) if fc and fc.rank is not None else None,
                record.target_weight,
                record.current_weight,
                record.trade,
                record.thesis,
                record.counter_thesis,
                json.dumps(record.invalidation_conditions),
                record.agreement_with_quant,
                json.dumps(attr.models) if attr else None,
                float(attr.llm_cost_usd) if attr else 0.0,
                fc.pit_level if fc else None,
                record.realised_return,
                datetime.now(tz=timezone.utc).isoformat(),
            ],
        )

    def update_realised_return(self, decision_id: str, realised_return: float) -> None:
        self._con.execute(
            "UPDATE decisions SET realised_return = ? WHERE decision_id = ?",
            [realised_return, decision_id],
        )

    # ── Query ──────────────────────────────────────────────────────────────

    def query(
        self,
        instrument_id: str | None = None,
        market: str | None = None,
        horizon: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> pl.DataFrame:
        """
        ``instrument_id`` matches one instrument exactly (e.g. "XNYS:IBM").
        ``market`` matches every instrument on that exchange (a MIC prefix,
        e.g. "XNYS" matches "XNYS:IBM", "XNYS:AAPL", ...) — these are
        different filters and were previously conflated, so passing a bare
        MIC as ``instrument_id`` silently matched nothing.
        """
        conditions = []
        params: list[Any] = []
        if instrument_id:
            conditions.append("instrument_id = ?")
            params.append(instrument_id)
        if market:
            conditions.append("instrument_id LIKE ?")
            params.append(f"{market}:%")
        if horizon:
            conditions.append("horizon = ?")
            params.append(horizon)
        if since:
            conditions.append("as_of >= ?")
            params.append(since.isoformat())
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        result = self._con.execute(
            f"SELECT * FROM decisions {where} ORDER BY as_of DESC LIMIT ?",
            params + [limit],
        )
        return pl.from_arrow(result.to_arrow_table())

    def stats(self) -> dict[str, Any]:
        total = self._con.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        with_outcome = self._con.execute(
            "SELECT COUNT(*) FROM decisions WHERE realised_return IS NOT NULL"
        ).fetchone()[0]
        return {"total_decisions": total, "with_realised_return": with_outcome}

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "AttributionStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
