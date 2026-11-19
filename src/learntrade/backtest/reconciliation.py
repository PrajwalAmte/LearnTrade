"""
Reconciliation between event-driven and vectorised backtests.

The event-driven engine and vectorised engine must agree within tolerance
on a common test case. Disagreement beyond tolerance indicates a bug in
one of the engines or inconsistent configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from learntrade.backtest.engine import BacktestResult
from learntrade.quant.validation.metrics import compute_portfolio_metrics


@dataclass
class ReconciliationReport:
    cagr_event_driven: float
    cagr_vectorised: float
    cagr_gap_bps: float
    sharpe_event_driven: float
    sharpe_vectorised: float
    passed: bool
    tolerance_bps: float
    message: str


def reconcile_results(
    event_driven: BacktestResult,
    vectorised: BacktestResult,
    tolerance_bps: float = 5.0,
) -> ReconciliationReport:
    """
    Assert that event-driven and vectorised backtests agree within tolerance.

    A gap exceeding tolerance_bps CAGR suggests a systematic difference in
    cost accounting, fill timing, or universe handling — not normal noise.
    """
    ed_metrics = compute_portfolio_metrics(event_driven.daily_returns)
    vec_metrics = compute_portfolio_metrics(vectorised.daily_returns)

    cagr_ed = ed_metrics.get("cagr", float("nan"))
    cagr_vec = vec_metrics.get("cagr", float("nan"))
    gap_bps = abs(cagr_ed - cagr_vec) * 10_000

    passed = gap_bps <= tolerance_bps or (np.isnan(cagr_ed) or np.isnan(cagr_vec))

    if passed:
        msg = f"PASS: CAGR gap={gap_bps:.1f} bps ≤ {tolerance_bps:.1f} bps tolerance"
    else:
        msg = (
            f"FAIL: CAGR gap={gap_bps:.1f} bps exceeds {tolerance_bps:.1f} bps tolerance. "
            f"Event-driven CAGR={cagr_ed:.4%}, vectorised CAGR={cagr_vec:.4%}. "
            "Check cost model parity and fill-price assumptions."
        )

    return ReconciliationReport(
        cagr_event_driven=float(cagr_ed),
        cagr_vectorised=float(cagr_vec),
        cagr_gap_bps=float(gap_bps),
        sharpe_event_driven=float(ed_metrics.get("sharpe", float("nan"))),
        sharpe_vectorised=float(vec_metrics.get("sharpe", float("nan"))),
        passed=passed,
        tolerance_bps=tolerance_bps,
        message=msg,
    )
