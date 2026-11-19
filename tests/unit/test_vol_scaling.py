"""Forecast volatility/range must reflect each stock's own risk, not one global number."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl

from learntrade.core.types import InstrumentId
from learntrade.quant.models.ridge import RidgeReturnModel


def test_riskier_stock_gets_wider_range():
    rng = np.random.default_rng(0)
    n = 400
    X = pl.DataFrame({"ret_20d": rng.normal(0, 0.05, n), "ewma_vol_20d": rng.uniform(0.15, 0.35, n)})
    y = pl.Series(0.2 * X["ret_20d"].to_numpy() + rng.normal(0, 0.03, n))
    m = RidgeReturnModel("swing_20d", "t")
    m.fit(X, y)
    Xp = pl.DataFrame({"ret_20d": [0.0, 0.0], "ewma_vol_20d": [0.15, 0.45]})
    ids = [InstrumentId("XNYS:CALM"), InstrumentId("XNYS:WILD")]
    calm, wild = m.predict_distribution(Xp, ids, datetime.now(tz=timezone.utc))
    assert wild.volatility > 2 * calm.volatility
    assert (wild.quantiles.q95 - wild.quantiles.q05) > (calm.quantiles.q95 - calm.quantiles.q05)


def test_missing_vol_feature_falls_back_to_global():
    rng = np.random.default_rng(1)
    X = pl.DataFrame({"ret_20d": rng.normal(0, 0.05, 200)})
    y = pl.Series(rng.normal(0, 0.03, 200))
    m = RidgeReturnModel("swing_20d", "t")
    m.fit(X, y)
    a, b = m.predict_distribution(
        pl.DataFrame({"ret_20d": [0.0, 0.01]}),
        [InstrumentId("XNYS:A"), InstrumentId("XNYS:B")], datetime.now(tz=timezone.utc))
    assert a.volatility == b.volatility
