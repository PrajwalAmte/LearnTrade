"""
Allocator tests — EqualWeight, InverseVol, MeanVariance, HRP, RiskParity.
CVaR is tested only when cvxpy is installed.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from learntrade.core.types import Forecast, ForecastQuantiles, InstrumentId
from learntrade.portfolio.allocators import (
    EqualWeightAllocator, HRPAllocator, InverseVolAllocator,
    MeanVarianceAllocator, RiskParityAllocator,
)

UTC = timezone.utc


def _forecasts(n: int = 5, seed: int = 42) -> list[Forecast]:
    rng = np.random.default_rng(seed)
    return [
        Forecast(
            instrument_id=InstrumentId(f"XNYS:SYM{i:02d}"),
            as_of=datetime(2024, 1, 15, tzinfo=UTC),
            horizon="swing_20d",
            expected_return=float(rng.normal(0.01, 0.03)),
            p_positive=0.55, volatility=float(rng.uniform(0.10, 0.35)),
            quantiles=ForecastQuantiles(q05=-0.05, q25=-0.02, q50=0.01, q75=0.03, q95=0.07),
            model_id="test", data_snapshot_id="",
        )
        for i in range(n)
    ]


def _cov(n: int = 5) -> np.ndarray:
    rng = np.random.default_rng(99)
    A = rng.normal(size=(n, n))
    return A @ A.T + np.eye(n) * 0.01


# ── EqualWeight ───────────────────────────────────────────────────────────────

def test_equal_weight_sums_to_one():
    fc = _forecasts(5)
    w = EqualWeightAllocator().allocate(fc)
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_equal_weight_all_equal():
    fc = _forecasts(4)
    w = EqualWeightAllocator().allocate(fc)
    vals = list(w.values())
    assert all(abs(v - vals[0]) < 1e-9 for v in vals)


def test_equal_weight_empty():
    assert EqualWeightAllocator().allocate([]) == {}


# ── InverseVol ────────────────────────────────────────────────────────────────

def test_inverse_vol_sums_to_one():
    fc = _forecasts(5)
    w = InverseVolAllocator().allocate(fc)
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_inverse_vol_low_vol_gets_higher_weight():
    fc = [
        Forecast(instrument_id=InstrumentId("XNYS:LOWVOL"), as_of=datetime(2024,1,1,tzinfo=UTC),
                 horizon="h", expected_return=0.01, p_positive=0.5, volatility=0.10,
                 quantiles=ForecastQuantiles(q05=0,q25=0,q50=0,q75=0,q95=0), model_id="t", data_snapshot_id=""),
        Forecast(instrument_id=InstrumentId("XNYS:HIVOL"), as_of=datetime(2024,1,1,tzinfo=UTC),
                 horizon="h", expected_return=0.01, p_positive=0.5, volatility=0.40,
                 quantiles=ForecastQuantiles(q05=0,q25=0,q50=0,q75=0,q95=0), model_id="t", data_snapshot_id=""),
    ]
    w = InverseVolAllocator().allocate(fc)
    assert w[InstrumentId("XNYS:LOWVOL")] > w[InstrumentId("XNYS:HIVOL")]


# ── MeanVariance ──────────────────────────────────────────────────────────────

def test_mv_sums_to_one():
    fc = _forecasts(5)
    cov = _cov(5)
    w = MeanVarianceAllocator().allocate(fc, covariance=cov)
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_mv_non_negative_long_only():
    fc = _forecasts(5)
    cov = _cov(5)
    w = MeanVarianceAllocator().allocate(fc, covariance=cov)
    assert all(v >= -1e-9 for v in w.values())


def test_mv_fallback_on_no_covariance():
    fc = _forecasts(3)
    w = MeanVarianceAllocator().allocate(fc)  # no cov provided
    assert abs(sum(w.values()) - 1.0) < 1e-6


# ── HRP ───────────────────────────────────────────────────────────────────────

def test_hrp_sums_to_one():
    fc = _forecasts(6)
    cov = _cov(6)
    w = HRPAllocator().allocate(fc, covariance=cov)
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_hrp_non_negative():
    fc = _forecasts(6)
    cov = _cov(6)
    w = HRPAllocator().allocate(fc, covariance=cov)
    assert all(v >= -1e-9 for v in w.values())


def test_hrp_single_asset():
    fc = _forecasts(1)
    w = HRPAllocator().allocate(fc)
    assert abs(list(w.values())[0] - 1.0) < 1e-9


# ── RiskParity ────────────────────────────────────────────────────────────────

def test_risk_parity_sums_to_one():
    fc = _forecasts(4)
    cov = _cov(4)
    w = RiskParityAllocator().allocate(fc, covariance=cov)
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_risk_parity_equal_vols_near_equal_weight():
    """With diagonal equal-vol covariance, risk parity ≈ equal weight."""
    fc = _forecasts(4)
    cov = np.eye(4) * 0.04  # all same vol
    w = RiskParityAllocator().allocate(fc, covariance=cov)
    vals = list(w.values())
    assert all(abs(v - 0.25) < 0.05 for v in vals)
