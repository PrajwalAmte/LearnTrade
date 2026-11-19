"""Evaluation honesty: overlap-corrected stats, holdout purge, reliability verdict, trading cadence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from learntrade.backtest.engine import BacktestConfig, EventDrivenBacktest, VectorisedBacktest
from learntrade.markets.loader import get_pack
from learntrade.quant.reliability import assess_reliability
from learntrade.quant.validation.metrics import ic_tstat, ic_tstat_overlap
from tests.unit.test_reconciliation import _make_test_data

UTC = timezone.utc


def test_overlap_adjusted_tstat_is_smaller_by_sqrt_horizon():
    rng = np.random.default_rng(0)
    ic = rng.normal(0.05, 0.2, 400)
    naive, adj = ic_tstat(ic), ic_tstat_overlap(ic, 20)
    assert adj < naive
    assert abs(adj - naive / np.sqrt(20)) < 1e-9


def test_trading_only_on_horizon_cuts_turnover_and_cost():
    bars, signals = _make_test_data(n_instruments=10, n_days=120)
    pack = get_pack("XNYS")
    daily = VectorisedBacktest().run(signals, bars, pack, BacktestConfig(top_n=5, rebalance_every=1))
    monthly = VectorisedBacktest().run(signals, bars, pack, BacktestConfig(top_n=5, rebalance_every=20))
    assert monthly.turnover.sum() < 0.5 * daily.turnover.sum()
    assert monthly.cost_bps.sum() < 0.5 * daily.cost_bps.sum()
    ed_daily = EventDrivenBacktest().run(signals, bars, pack, BacktestConfig(top_n=5, rebalance_every=1))
    ed_month = EventDrivenBacktest().run(signals, bars, pack, BacktestConfig(top_n=5, rebalance_every=20))
    assert len(ed_month.daily_returns) == len(ed_daily.daily_returns)


def test_vectorised_costs_use_full_market_pack_model():
    """Zero-cost vs real pack: the real pack must charge something beyond nothing."""
    bars, signals = _make_test_data(n_instruments=10, n_days=60)
    res = VectorisedBacktest().run(signals, bars, get_pack("XNSE"), BacktestConfig(top_n=5))
    assert res.cost_bps.sum() > 0


def _report(**kw):
    base = {
        "model_id": "ridge_v1", "label_type": "excess", "n_instruments": 40, "n_days": 700,
        "benchmark_id": "XNSE:NIFTY50",
        "cv": {"rank_ic_mean": 0.06, "rank_ic_ci": [0.01, 0.11], "hit_rate": 0.6,
               "tstat_overlap_adjusted": 2.1},
        "calibration": {"roc_auc": 0.57, "brier_score": 0.245},
        "holdout": {"rank_ic_mean": 0.05, "n_days": 120},
    }
    base.update(kw)
    return base


def test_reliability_levels():
    assert assess_reliability(_report())["level"] == "modest"
    unproven = assess_reliability(_report(cv={"rank_ic_mean": 0.06, "rank_ic_ci": [-0.01, 0.12]}))
    assert unproven["level"] == "unproven"
    weak_t = assess_reliability(_report(cv={"rank_ic_mean": 0.06, "rank_ic_ci": [0.01, 0.11],
                                            "tstat_overlap_adjusted": 1.5}))
    assert weak_t["level"] == "unproven"   # range clears zero but t-stat does not
    tiny = assess_reliability(_report(cv={"rank_ic_mean": 0.004, "rank_ic_ci": [-0.1, 0.1]}))
    assert tiny["level"] == "no_evidence"     # +0.004 is not "leaning the right way"
    none = assess_reliability(_report(cv={"rank_ic_mean": -0.02, "rank_ic_ci": [-0.1, 0.06]}))
    assert none["level"] == "no_evidence"


def test_failed_holdout_downgrades_and_is_explained():
    r = assess_reliability(_report(holdout={"rank_ic_mean": -0.04, "n_days": 120}))
    assert r["level"] == "unproven"
    assert any("untouched" in x for x in r["reasons"])


def test_reliability_flags_small_raw_and_stale():
    r = assess_reliability(
        _report(n_instruments=9, n_days=300, label_type="raw", holdout=None),
        last_bar="2026-01-01", today=datetime(2026, 11, 19, tzinfo=UTC).date())
    text = " ".join(r["reasons"])
    assert "9 stocks" in text and "raw returns" in text and "No untouched" in text
    assert "days old" in text


def test_holdout_split_purges_forward_window(tmp_path):
    from learntrade.data.store.pit import PITStore
    from learntrade.quant.trainer import QuantTrainer

    t = QuantTrainer(PITStore(tmp_path), "swing_20d", holdout_days=50, min_train_bars=50)
    start = datetime(2022, 1, 3, tzinfo=UTC)
    dates = [start + timedelta(days=i) for i in range(500)]
    full = pl.DataFrame({"event_time": dates * 3, "x": [1.0] * 1500})
    dev, hold = t._split_holdout(full)
    assert hold is not None
    # No development row may share a date with, or sit within the label window of, the holdout.
    assert dev["event_time"].max() < hold["event_time"].min()
    gap = len([d for d in dates if dev["event_time"].max() < d < hold["event_time"].min()])
    assert gap >= 20
    assert hold["event_time"].n_unique() == 50
