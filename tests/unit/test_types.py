"""
Core domain type tests.
"""

from __future__ import annotations

import pytest

from leantrade.core.types import (
    Forecast,
    ForecastQuantiles,
    Instrument,
    InstrumentId,
    Observation,
)
from datetime import datetime, timezone


def test_instrument_id_normalises_to_uppercase():
    iid = InstrumentId("xnys:ibm")
    assert iid == "XNYS:IBM"
    assert iid.mic == "XNYS"
    assert iid.symbol == "IBM"


def test_instrument_id_requires_colon():
    with pytest.raises(ValueError):
        InstrumentId("XNYSIIBM")


def test_instrument_mic_symbol_mismatch_raises():
    with pytest.raises(ValueError):
        Instrument(
            id=InstrumentId("XNYS:IBM"),
            mic="XNAS",
            symbol="IBM",
            currency="USD",
        )


def test_instrument_valid():
    inst = Instrument(
        id=InstrumentId("XNYS:IBM"),
        mic="XNYS",
        symbol="IBM",
        currency="USD",
    )
    assert inst.asset_class == "equity"


def test_forecast_model():
    fc = Forecast(
        instrument_id=InstrumentId("XNYS:IBM"),
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        horizon="swing_20d",
        expected_return=0.02,
        p_positive=0.58,
        volatility=0.04,
        quantiles=ForecastQuantiles(q05=-0.05, q25=-0.01, q50=0.02, q75=0.05, q95=0.09),
        model_id="lgbm_rank_v1",
        data_snapshot_id="snap_001",
    )
    assert fc.p_positive == 0.58
