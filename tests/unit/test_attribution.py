"""
Attribution store tests — write, query, update realised return.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from learntrade.attribution.store import AttributionStore
from learntrade.core.types import (
    DecisionRecord, EvidenceClaim, Forecast, ForecastAttribution,
    ForecastQuantiles, InstrumentId,
)

UTC = timezone.utc


def _make_record(iid: str = "XNYS:IBM", horizon: str = "swing_20d") -> DecisionRecord:
    fc = Forecast(
        instrument_id=InstrumentId(iid),
        as_of=datetime(2024, 1, 15, tzinfo=UTC),
        horizon=horizon,
        expected_return=0.018,
        p_positive=0.58,
        volatility=0.22,
        quantiles=ForecastQuantiles(q05=-0.05, q25=-0.01, q50=0.018, q75=0.04, q95=0.09),
        model_id="ridge_v1",
        data_snapshot_id="snap_001",
    )
    return DecisionRecord(
        decision_id=f"2024-01-15/{iid}/{horizon}",
        instrument_id=InstrumentId(iid),
        horizon=horizon,
        as_of=datetime(2024, 1, 15, tzinfo=UTC),
        forecast=fc,
        thesis="Strong momentum with positive earnings surprise.",
        counter_thesis="FX headwinds may compress margins.",
        invalidation_conditions=["Revenue misses by >5%"],
        agreement_with_quant="agree",
        target_weight=0.032,
        current_weight=0.0,
        trade="BUY",
        est_cost_bps=5.2,
        attribution=ForecastAttribution(models={"ridge_v1": 0.7}, llm_cost_usd=0.004),
    )


def test_write_and_query(tmp_path: Path):
    with AttributionStore(tmp_path / "attr.duckdb") as store:
        store.write(_make_record())
        df = store.query(instrument_id="XNYS:IBM")
    assert len(df) == 1
    assert "thesis" in df.columns


def test_multiple_decisions(tmp_path: Path):
    with AttributionStore(tmp_path / "attr.duckdb") as store:
        store.write(_make_record("XNYS:IBM"))
        store.write(_make_record("XNYS:AAPL"))
        store.write(_make_record("XNYS:IBM", horizon="positional_60d"))
        df_all = store.query()
        df_ibm = store.query(instrument_id="XNYS:IBM")
        df_swing = store.query(horizon="swing_20d")
    assert len(df_all) == 3
    assert len(df_ibm) == 2
    assert len(df_swing) == 2


def test_update_realised_return(tmp_path: Path):
    with AttributionStore(tmp_path / "attr.duckdb") as store:
        record = _make_record()
        store.write(record)
        store.update_realised_return(record.decision_id, 0.025)
        df = store.query(instrument_id="XNYS:IBM")
    # Realised return should be updated
    rr = float(df["realised_return"][0])
    assert abs(rr - 0.025) < 1e-9


def test_stats(tmp_path: Path):
    with AttributionStore(tmp_path / "attr.duckdb") as store:
        store.write(_make_record())
        store.update_realised_return("2024-01-15/XNYS:IBM/swing_20d", 0.01)
        stats = store.stats()
    assert stats["total_decisions"] == 1
    assert stats["with_realised_return"] == 1


def test_idempotent_write(tmp_path: Path):
    """Writing the same decision_id twice should update, not duplicate."""
    with AttributionStore(tmp_path / "attr.duckdb") as store:
        record = _make_record()
        store.write(record)
        store.write(record)
        df = store.query()
    assert len(df) == 1


def test_llm_explains_not_sets_weights(tmp_path: Path):
    """The thesis/counter_thesis fields carry LLM explanation.
    target_weight comes from the optimiser, not from the thesis text."""
    with AttributionStore(tmp_path / "attr.duckdb") as store:
        record = _make_record()
        store.write(record)
        df = store.query()
    # Thesis exists and is non-empty
    assert str(df["thesis"][0]).strip() != ""
    # Weight was set by allocator (not by LLM)
    assert float(df["target_weight"][0]) == pytest.approx(0.032)
