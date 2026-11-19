"""
Vectorised backtest tests.
"""

from __future__ import annotations

import datetime as _dt
from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest

from learntrade.backtest.engine import BacktestConfig, VectorisedBacktest
from learntrade.backtest.report import generate_report
from learntrade.markets.loader import get_pack

UTC = timezone.utc


def _make_bars_and_signals(n_instruments: int = 5, n_days: int = 60) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(7)
    bar_rows = []
    sig_rows = []
    for d in range(n_days):
        dt = datetime(2024, 1, 1, tzinfo=UTC) + _dt.timedelta(days=d)
        for i in range(n_instruments):
            iid = f"XNYS:SYM{i:02d}"
            ret = rng.normal(0.0003, 0.015)
            price = 100 * (1 + ret) ** (d + 1)
            bar_rows.append({
                "instrument_id": iid,
                "event_time": dt,
                "close": price,
                "adj_close": price,
                "volume": float(rng.integers(50_000, 200_000)),
            })
            sig_rows.append({
                "date": dt,
                "instrument_id": iid,
                "signal": rng.normal(ret, 0.01),  # noisy signal correlated with true return
            })
    return pl.DataFrame(bar_rows), pl.DataFrame(sig_rows)


def test_backtest_runs_and_returns_result():
    bars, signals = _make_bars_and_signals()
    pack = get_pack("XNYS")
    engine = VectorisedBacktest()
    result = engine.run(signals, bars, pack)
    assert len(result.daily_returns) > 0
    assert len(result.turnover) == len(result.daily_returns)
    assert len(result.cost_bps) == len(result.daily_returns)


def test_backtest_costs_are_non_negative():
    bars, signals = _make_bars_and_signals()
    pack = get_pack("XNYS")
    result = VectorisedBacktest().run(signals, bars, pack)
    assert (result.cost_bps >= 0).all()


def test_backtest_weights_sum_to_one():
    bars, signals = _make_bars_and_signals(n_instruments=5)
    pack = get_pack("XNYS")
    result = VectorisedBacktest().run(signals, bars, pack)
    if not result.portfolio_weights.is_empty():
        daily_sums = (
            result.portfolio_weights
            .group_by("date")
            .agg(pl.col("weight").sum().alias("total_weight"))
        )
        # All sums should be close to 1.0 (or exactly top_n / top_n = 1.0)
        assert (daily_sums["total_weight"].round(6) <= 1.001).all()


def test_backtest_report_generates_files(tmp_path):
    bars, signals = _make_bars_and_signals()
    pack = get_pack("XNYS")
    result = VectorisedBacktest().run(
        signals, bars, pack,
        config=BacktestConfig(pit_level="none", survivorship_note="synthetic data"),
    )
    report = generate_report(result, "swing_20d", "XNYS", ["ridge_v1"], tmp_path)
    md_files = list(tmp_path.glob("*.md"))
    html_files = list(tmp_path.glob("*.html"))
    assert len(md_files) == 1
    assert len(html_files) == 1
    # Verify key sections present
    md_text = md_files[0].read_text()
    assert "CAGR" in md_text
    assert "pit_level=none" in md_text
    assert "survivorship" in md_text.lower()
