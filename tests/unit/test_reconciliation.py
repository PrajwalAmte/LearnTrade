"""
Reconciliation tests — event-driven vs vectorised results.
"""

from __future__ import annotations

import datetime as _dt
from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest

from learntrade.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    EventDrivenBacktest,
    VectorisedBacktest,
)
from learntrade.backtest.reconciliation import reconcile_results
from learntrade.markets.loader import get_pack

UTC = timezone.utc


def _make_test_data(n_instruments: int = 5, n_days: int = 60) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Generate a simple multiplicative random walk per instrument.

    NOTE: an earlier version of this fixture computed
    ``price = 100 * (1 + ret) ** (d + 1)`` with a FRESH ``ret`` drawn every
    day — i.e. it re-exponentiated a brand new one-day return by a growing
    exponent instead of compounding the walk day over day. That produces
    day-to-day price RATIOS of 7x-12x by day ~48 (verified: not a "fat
    tail", an artifact of the formula), which made both backtest engines
    report portfolio returns of roughly -100% and pass this test's
    tolerance only because they were both equally nonsensical. Compounding
    each day's return into the previous close, as below, is what an actual
    random walk requires.
    """
    rng = np.random.default_rng(42)
    bars, sigs = [], []
    prices = {f"XNYS:SYM{i:02d}": 100.0 for i in range(n_instruments)}
    for d in range(n_days):
        dt = datetime(2024, 1, 2, tzinfo=UTC) + _dt.timedelta(days=d)
        for i in range(n_instruments):
            iid = f"XNYS:SYM{i:02d}"
            ret = rng.normal(0.0003, 0.012)
            price = prices[iid] * (1 + ret)
            prices[iid] = price
            bars.append({
                "instrument_id": iid, "event_time": dt,
                "open": price * 0.999, "high": price * 1.004,
                "low": price * 0.996, "close": price,
                "adj_close": price, "volume": float(rng.integers(50_000, 500_000)),
            })
            sigs.append({"date": dt, "instrument_id": iid, "signal": rng.normal(ret, 0.01)})
    return pl.DataFrame(bars), pl.DataFrame(sigs)


def test_reconciliation_passes_within_tolerance():
    # Realistic cadence: trade every 20 days like a 20-day model would. (Trading DAILY on
    # this noisy fixture amplifies fill-timing differences between the engines; see
    # test_costs_agree_between_engines for the daily case.)
    bars, signals = _make_test_data(n_instruments=10, n_days=250)
    pack = get_pack("XNYS")
    cfg = BacktestConfig(pit_level="none", top_n=5, rebalance_every=20)

    vec_result = VectorisedBacktest().run(signals, bars, pack, cfg)
    ed_result = EventDrivenBacktest().run(signals, bars, pack, cfg, initial_cash=100_000.0)
    # Both engines charge trades with the SAME cost function
    # (execution.costs.compute_trade_cost), so the remaining gap is fill mechanics only:
    # EventDriven fills at the next bar's OPEN with lot/tick rounding and a participation
    # cap; Vectorised uses close-to-close returns. Measured gap here is ~240 bps CAGR
    # (daily-return correlation ~0.98). 500 bps still catches an engine disagreeing by an
    # order of magnitude (see test_reconciliation_fails_on_large_gap).
    report = reconcile_results(ed_result, vec_result, tolerance_bps=500.0)
    assert report.passed, f"Reconciliation failed: {report.message}"
    n = min(len(vec_result.daily_returns), len(ed_result.daily_returns))
    corr = np.corrcoef(vec_result.daily_returns[:n], ed_result.daily_returns[:n])[0, 1]
    assert corr > 0.9, f"daily returns should track each other, correlation={corr:.2f}"


def _per_turnover(res) -> float:
    return float(np.sum(res.cost_bps) / max(float(np.sum(res.turnover)), 1e-9))


def test_costs_agree_between_engines():
    """Same cost function in both engines: cost per unit of trading matches (NYSE and NSE),
    and at a realistic monthly cadence total cost matches too. (Day to day the event-driven
    engine trades a bit less because of cash limits and lot rounding, so only the price of a
    trade is compared when trading daily.)"""
    bars, signals = _make_test_data(n_instruments=10, n_days=250)
    for mic in ("XNYS", "XNSE"):
        pack = get_pack(mic)
        b = bars.with_columns(pl.col("instrument_id").str.replace("XNYS", mic))
        s = signals.with_columns(pl.col("instrument_id").str.replace("XNYS", mic))
        for every in (1, 20):
            cfg = BacktestConfig(pit_level="none", top_n=5, rebalance_every=every)
            vec = VectorisedBacktest().run(s, b, pack, cfg)
            ed = EventDrivenBacktest().run(s, b, pack, cfg, initial_cash=100_000.0)
            v, e = _per_turnover(vec), _per_turnover(ed)
            assert abs(v - e) / e < 0.10, f"{mic} every={every}: {v:.2f} vs {e:.2f} bps/turnover"
            if every == 20:
                tv, te = float(np.sum(vec.cost_bps)), float(np.sum(ed.cost_bps))
                assert abs(tv - te) / te < 0.10, f"{mic}: total cost {tv:.1f} vs {te:.1f} bps"


def test_reconciliation_fails_on_large_gap():
    """Fabricate a large gap to verify the failure path."""
    rng = np.random.default_rng(1)
    ed = BacktestResult(
        daily_returns=np.ones(252) * 0.002,   # ~65% CAGR
        portfolio_weights=pl.DataFrame(),
        turnover=np.zeros(252), cost_bps=np.zeros(252),
        signal_ic=[], config=BacktestConfig(engine="event_driven"),
    )
    vec = BacktestResult(
        daily_returns=np.ones(252) * -0.002,  # ~-40% CAGR
        portfolio_weights=pl.DataFrame(),
        turnover=np.zeros(252), cost_bps=np.zeros(252),
        signal_ic=[], config=BacktestConfig(engine="vectorised"),
    )
    report = reconcile_results(ed, vec, tolerance_bps=5.0)
    assert not report.passed
    assert report.cagr_gap_bps > 5.0
