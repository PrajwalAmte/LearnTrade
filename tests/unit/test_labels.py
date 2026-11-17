"""
Label generation tests — forward returns, PIT enforcement, benchmark excess returns.
"""

from __future__ import annotations

import datetime as _dt
from datetime import datetime, timezone

import polars as pl
import pytest

from leantrade.quant.targets.labels import (
    compute_forward_returns,
    merge_features_labels,
)

UTC = timezone.utc


def _bars(n_days: int = 30, n_instruments: int = 2) -> pl.DataFrame:
    import numpy as np
    rng = np.random.default_rng(99)
    rows = []
    for i in range(n_instruments):
        price = 100.0 + i * 20
        for d in range(n_days):
            dt = datetime(2024, 1, 1, tzinfo=UTC) + _dt.timedelta(days=d)
            price *= 1 + rng.normal(0, 0.01)
            rows.append({
                "instrument_id": f"XNYS:SYM{i}",
                "event_time": dt,
                "close": price,
                "adj_close": price,
            })
    return pl.DataFrame(rows)


def test_forward_returns_shape():
    bars = _bars(n_days=30, n_instruments=2)
    labels = compute_forward_returns(bars, forward_days=5)
    assert "instrument_id" in labels.columns
    assert "event_time" in labels.columns
    assert "label" in labels.columns
    assert "label_available_at" in labels.columns


def test_forward_returns_no_nan_in_middle():
    bars = _bars(n_days=50, n_instruments=1)
    labels = compute_forward_returns(bars, forward_days=5)
    # Last 5 rows must be dropped (no forward value available)
    assert len(labels) <= 50 - 5


def test_forward_returns_values_correct():
    """The 5-day forward return at row i must equal log(close[i+5] / close[i])."""
    import numpy as np
    bars = _bars(n_days=20, n_instruments=1)
    labels = compute_forward_returns(bars, forward_days=5)
    bars_sorted = bars.sort("event_time")
    closes = bars_sorted["adj_close"].to_list()
    # Row 0 → forward return = log(closes[5] / closes[0])
    expected = np.log(closes[5] / closes[0])
    actual = labels.sort("event_time")["label"][0]
    assert abs(float(actual) - expected) < 1e-9


def test_label_available_at_is_future():
    """label_available_at must be strictly after event_time."""
    bars = _bars(n_days=30)
    labels = compute_forward_returns(bars, forward_days=5)
    # For every row: label_available_at > event_time
    # (bars are daily, so available_at = event_time + 5 bars)
    for row in labels.iter_rows(named=True):
        et = row["event_time"]
        lat = row["label_available_at"]
        assert lat > et, f"label_available_at {lat} <= event_time {et}"


def test_forward_returns_requires_positive_forward_days():
    bars = _bars()
    with pytest.raises(ValueError):
        compute_forward_returns(bars, forward_days=0)


def test_merge_features_labels_inner_join():
    bars = _bars(n_days=30, n_instruments=2)
    labels = compute_forward_returns(bars, forward_days=5)

    # Create a simple features DataFrame (same schema as labels)
    features = bars.select([
        pl.col("instrument_id"),
        pl.col("event_time"),
        (pl.col("close") / pl.col("close").shift(1).over("instrument_id")).log().alias("ret_1d"),
    ]).drop_nulls()

    merged = merge_features_labels(features, labels)
    assert len(merged) > 0
    assert "label" in merged.columns
    assert "ret_1d" in merged.columns
    # No nulls in label
    assert merged["label"].is_not_null().all()
