"""
Feature engine tests — price/volume features and cross-sectional transforms.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest

from learntrade.features.cross_sectional import cross_sectional_rank, cross_sectional_zscore
from learntrade.features.price_volume import compute_price_volume_features, feature_names

UTC = timezone.utc


def _make_bars(n_instruments: int = 3, n_days: int = 100) -> pl.DataFrame:
    """Generate synthetic OHLCV bars."""
    rng = np.random.default_rng(42)
    rows = []
    for sym_i in range(n_instruments):
        price = 100.0 + sym_i * 10
        for d in range(n_days):
            dt = datetime(2024, 1, 1, tzinfo=UTC)
            import datetime as _dt
            dt = datetime(2024, 1, 1, tzinfo=UTC) + _dt.timedelta(days=d)
            ret = rng.normal(0, 0.01)
            price *= (1 + ret)
            rows.append({
                "instrument_id": f"XNYS:SYM{sym_i:02d}",
                "event_time": dt,
                "open": price * (1 - 0.002),
                "high": price * (1 + 0.005),
                "low": price * (1 - 0.005),
                "close": price,
                "adj_close": price,
                "volume": float(rng.integers(100_000, 1_000_000)),
            })
    return pl.DataFrame(rows)


def test_feature_names_matches_output():
    expected = feature_names()
    bars = _make_bars()
    featured = compute_price_volume_features(bars)
    for name in expected:
        assert name in featured.columns, f"Missing feature column: {name}"


def test_return_features_are_finite():
    bars = _make_bars(n_instruments=2, n_days=80)
    featured = compute_price_volume_features(bars, return_windows=[1, 5, 20])
    # Drop the first few rows (NaN from shift) and check the rest
    tail = featured.tail(50)
    assert tail["ret_1d"].drop_nulls().is_finite().all()
    assert tail["ret_5d"].drop_nulls().is_finite().all()


def test_ewma_vol_is_non_negative():
    bars = _make_bars(n_instruments=2, n_days=80)
    featured = compute_price_volume_features(bars)
    vol_col = featured["ewma_vol_20d"].drop_nulls()
    assert (vol_col >= 0).all()


def test_volume_surprise_centred_near_zero():
    bars = _make_bars(n_instruments=1, n_days=100)
    featured = compute_price_volume_features(bars)
    # With roughly constant volume, vol_surprise should be near zero on average
    surprise = featured["vol_surprise_20d"].drop_nulls().to_numpy()
    assert abs(float(np.mean(surprise))) < 1.0


def test_52w_high_ratio_in_zero_one():
    bars = _make_bars(n_instruments=2, n_days=100)
    featured = compute_price_volume_features(bars)
    ratio = featured["high_52w_ratio"].drop_nulls()
    assert (ratio > 0).all()
    assert (ratio <= 1.001).all()  # slight tolerance for float precision


def test_cross_sectional_rank_in_zero_one():
    bars = _make_bars(n_instruments=5, n_days=30)
    featured = compute_price_volume_features(bars, return_windows=[1, 5, 20])
    ranked = cross_sectional_rank(featured, ["ret_1d", "ret_5d"])
    for col in ["ret_1d_rank", "ret_5d_rank"]:
        vals = ranked[col].drop_nulls()
        assert (vals >= 0).all()
        assert (vals <= 1.001).all()


def test_cross_sectional_zscore_centred():
    bars = _make_bars(n_instruments=10, n_days=30)
    featured = compute_price_volume_features(bars, return_windows=[1])
    zscored = cross_sectional_zscore(featured, ["ret_1d"])
    # Mean of z-scores within each date should be ~0
    daily_means = (
        zscored.drop_nulls(subset=["ret_1d_cs"])
        .group_by("event_time")
        .agg(pl.col("ret_1d_cs").mean().alias("mean_z"))
    )
    assert (daily_means["mean_z"].abs() < 0.5).mean() > 0.9  # 90% of days within tolerance


def test_no_lookahead_in_shift_features():
    """Verify that shift(w) over groups doesn't use future values."""
    bars = _make_bars(n_instruments=1, n_days=30)
    featured = compute_price_volume_features(bars, return_windows=[5])
    # Row i should not contain information from rows i+1 to i+5
    # We verify by checking that ret_5d at day t uses the close from day t-5
    sym = featured.filter(pl.col("instrument_id") == "XNYS:SYM00").sort("event_time")
    closes = sym["close"].to_list()
    expected_ret5 = math.log(closes[6] / closes[1])  # day[6] / day[1]
    actual_ret5 = sym["ret_5d"][6]
    if actual_ret5 is not None:
        assert abs(float(actual_ret5) - expected_ret5) < 1e-9
