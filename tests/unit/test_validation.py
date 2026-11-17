"""
Validation metric tests — IC, t-stat, block bootstrap, portfolio metrics, walk-forward CV.
"""

from __future__ import annotations

import datetime as _dt
from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest

from leantrade.quant.validation.metrics import (
    block_bootstrap_ci,
    compute_portfolio_metrics,
    ic_tstat,
    information_coefficient,
)
from leantrade.quant.validation.walk_forward import PurgedWalkForwardCV

UTC = timezone.utc


def test_ic_perfect_rank_correlation():
    x = np.arange(10, dtype=float)
    assert abs(information_coefficient(x, x) - 1.0) < 1e-9


def test_ic_perfect_negative_correlation():
    x = np.arange(10, dtype=float)
    y = -x
    assert abs(information_coefficient(x, y) + 1.0) < 1e-9


def test_ic_uncorrelated_near_zero():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    ic = information_coefficient(x, y)
    assert abs(ic) < 0.15


def test_ic_too_few_samples_returns_nan():
    x = np.array([1.0, 2.0, 3.0])
    assert np.isnan(information_coefficient(x, x[::-1]))


def test_ic_tstat_positive_series():
    ic_arr = np.ones(20) * 0.08  # constant positive IC
    t = ic_tstat(ic_arr)
    # t = 0.08 / (0 / sqrt(20)) → inf, but std is 0 → nan
    # Let's use a realistic series instead
    ic_arr = np.ones(20) * 0.05 + np.random.default_rng(2).normal(0, 0.02, 20)
    t = ic_tstat(ic_arr)
    assert not np.isnan(t)
    assert t > 0  # positive IC series → positive t


def test_block_bootstrap_ci_symmetry():
    # For a constant series, CI should be degenerate
    arr = np.ones(50) * 0.05
    lo, hi = block_bootstrap_ci(arr)
    assert abs(lo - 0.05) < 1e-9
    assert abs(hi - 0.05) < 1e-9


def test_block_bootstrap_ci_ordering():
    rng = np.random.default_rng(42)
    arr = rng.normal(0.04, 0.03, 100)
    lo, hi = block_bootstrap_ci(arr, ci=0.95, n_bootstrap=200)
    assert lo < hi
    assert lo < float(np.mean(arr)) < hi


def test_portfolio_metrics_all_zero_returns():
    r = np.zeros(252)
    m = compute_portfolio_metrics(r)
    assert m["cagr"] == pytest.approx(0.0, abs=1e-6)
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-6)


def test_portfolio_metrics_monotone_growth():
    r = np.ones(252) * 0.001  # ~28% CAGR
    m = compute_portfolio_metrics(r)
    assert m["cagr"] > 0
    assert m["sharpe"] > 0
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-4)


def test_portfolio_metrics_negative_returns():
    r = np.ones(252) * -0.001
    m = compute_portfolio_metrics(r)
    assert m["cagr"] < 0


def _make_dates_series(n: int) -> pl.Series:
    base = datetime(2020, 1, 1, tzinfo=UTC)
    return pl.Series([base + _dt.timedelta(days=i) for i in range(n)])


def test_walk_forward_yields_folds():
    dates = _make_dates_series(600)
    # Duplicate each date 5 times (simulating 5 instruments per day)
    dates = pl.concat([dates] * 5)
    cv = PurgedWalkForwardCV(n_splits=3, embargo_bars=5, min_train_bars=50)
    folds = list(cv.split(dates, horizon_bars=10))
    assert len(folds) >= 1


def test_walk_forward_no_overlap():
    """Train and test indices must not overlap."""
    dates = _make_dates_series(600)
    dates = pl.concat([dates] * 3)
    cv = PurgedWalkForwardCV(n_splits=3, embargo_bars=20, min_train_bars=50)
    for fold in cv.split(dates, horizon_bars=20):
        overlap = set(fold.train_indices) & set(fold.test_indices)
        assert len(overlap) == 0, f"Fold {fold.fold_id} has overlapping indices"


def test_walk_forward_test_after_train():
    """Max train date must be strictly before min test date."""
    import datetime as _dt
    dates = _make_dates_series(600)
    dates = pl.concat([dates] * 2)
    all_dates_list = dates.to_list()
    cv = PurgedWalkForwardCV(n_splits=3, embargo_bars=20, min_train_bars=50)
    for fold in cv.split(dates, horizon_bars=20):
        max_train = str(max(all_dates_list[i] for i in fold.train_indices))
        min_test = str(min(all_dates_list[i] for i in fold.test_indices))
        assert max_train < min_test, f"Fold {fold.fold_id}: train bleeds into test"
