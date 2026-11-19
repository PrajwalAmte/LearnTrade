"""Cross-sectional feature normalisation: 'stronger than other stocks today', consistently."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from learntrade.features.price_volume import PriceVolumeFeatureSet, add_cross_sectional
from learntrade.data.store.pit import PITStore
from learntrade.setup.demo import generate_demo_data

UTC = timezone.utc


def test_zscore_is_per_date_and_neutral_when_too_few_stocks():
    d1, d2 = datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC)
    df = pl.DataFrame({
        "event_time": [d1] * 6 + [d2] * 3,
        "x": [1.0, 2, 3, 4, 5, 6, 10.0, 20.0, 30.0],
    })
    out = add_cross_sectional(df, ["x"], "event_time")
    day1 = out.filter(pl.col("event_time") == d1)["x_cs"].to_numpy()
    assert abs(day1.mean()) < 1e-9 and day1[-1] > day1[0]
    assert (out.filter(pl.col("event_time") == d2)["x_cs"] == 0.0).all()   # only 3 names -> neutral


def test_feature_set_keeps_raw_columns_and_models_use_cs_plus_raw_vol(tmp_path):
    with PITStore(tmp_path) as store:
        generate_demo_data(store, mic="XNYS", n_days=300, seed=5)
        fs = PriceVolumeFeatureSet()
        ids = store.get_universe("XNYS", datetime.now(tz=UTC))
        latest = fs.compute(ids, datetime.now(tz=UTC), store)
        hist = fs.compute_history(ids, datetime.now(tz=UTC) - timedelta(days=200),
                                  datetime.now(tz=UTC), store)
    assert "ret_20d" in latest.columns and "ret_20d_cs" in latest.columns
    assert fs.feature_names[-1] == "ewma_vol_20d"
    assert all(n.endswith("_cs") for n in fs.feature_names[:-1])
    assert set(fs.feature_names) <= set(hist.columns)
    # per-date mean of every normalised feature is ~0
    one_day = hist.filter(pl.col("event_time") == hist["event_time"].max())
    assert abs(float(one_day["ret_20d_cs"].mean())) < 1e-6


def test_raw_mode_still_available():
    fs = PriceVolumeFeatureSet(cross_sectional=False)
    assert fs.feature_names == fs.raw_feature_names
