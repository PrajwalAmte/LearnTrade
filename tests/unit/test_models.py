"""
Model tests — Ridge, EWMA vol, fit/predict/save/load roundtrip.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from leantrade.core.types import InstrumentId
from leantrade.quant.models.ewma_vol import EWMAVolModel
from leantrade.quant.models.ridge import RidgeReturnModel

UTC = timezone.utc


def _make_Xy(n: int = 200, n_features: int = 8, seed: int = 42) -> tuple[pl.DataFrame, pl.Series]:
    rng = np.random.default_rng(seed)
    X_np = rng.normal(size=(n, n_features))
    # Label has a noisy linear relationship with the first two features
    y_np = X_np[:, 0] * 0.3 + X_np[:, 1] * 0.2 + rng.normal(0, 0.1, n)
    cols = [f"feat_{i}" for i in range(n_features)]
    return pl.DataFrame(X_np, schema=cols), pl.Series(y_np)


def test_ridge_fit_predict():
    X, y = _make_Xy()
    model = RidgeReturnModel("swing_20d", "v1")
    model.fit(X, y)
    iids = [InstrumentId(f"XNYS:SYM{i:02d}") for i in range(len(X))]
    forecasts = model.predict_distribution(X, iids, datetime.now(tz=UTC))
    assert len(forecasts) == len(X)
    for f in forecasts:
        assert not np.isnan(f.expected_return)
        assert 0 < f.p_positive < 1
        assert f.volatility > 0


def test_ridge_raises_before_fit():
    model = RidgeReturnModel("swing_20d", "v1")
    X, _ = _make_Xy(n=10)
    iids = [InstrumentId(f"XNYS:SYM{i}") for i in range(10)]
    with pytest.raises(RuntimeError):
        model.predict_distribution(X, iids, datetime.now(tz=UTC))


def test_ridge_save_load_roundtrip(tmp_path: Path):
    X, y = _make_Xy()
    model = RidgeReturnModel("swing_20d", "v1")
    model.fit(X, y)
    path = str(tmp_path / "ridge.pkl")
    model.save(path)

    loaded = RidgeReturnModel.load(path)
    iids = [InstrumentId(f"XNYS:SYM{i}") for i in range(len(X))]
    orig_preds = [f.expected_return for f in model.predict_distribution(X, iids, datetime.now(tz=UTC))]
    load_preds = [f.expected_return for f in loaded.predict_distribution(X, iids, datetime.now(tz=UTC))]
    assert orig_preds == pytest.approx(load_preds, rel=1e-5)


def test_ridge_requires_min_samples():
    X, y = _make_Xy(n=10)
    model = RidgeReturnModel("swing_20d", "v1")
    with pytest.raises(ValueError, match="Too few"):
        model.fit(X, y)


def test_ewma_vol_known_value():
    """With constant returns, EWMA vol should equal the constant * sqrt(252)."""
    daily_ret = 0.01
    r = np.full(100, daily_ret)
    model = EWMAVolModel(span=20)
    vol = model.predict_vol(r, annualise=True)
    expected = daily_ret * np.sqrt(252)
    # EWMA with exponentially decaying weights converges to daily_ret * sqrt(252)
    assert abs(vol - expected) < expected * 0.05  # within 5%


def test_ewma_vol_zero_returns():
    r = np.zeros(50)
    model = EWMAVolModel(span=20)
    vol = model.predict_vol(r, annualise=False)
    assert vol == pytest.approx(0.0, abs=1e-9)


def test_ewma_vol_save_load(tmp_path: Path):
    model = EWMAVolModel(span=30)
    path = str(tmp_path / "ewma.pkl")
    model.save(path)
    loaded = EWMAVolModel.load(path)
    assert loaded.span == 30
