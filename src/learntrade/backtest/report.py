"""
Backtest report generation — Markdown and HTML output.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from learntrade.backtest.engine import BacktestResult
from learntrade.quant.calibration.metrics import CalibrationReport
from learntrade.quant.validation.metrics import block_bootstrap_ci, compute_portfolio_metrics, ic_tstat

if TYPE_CHECKING:
    from learntrade.backtest.reconciliation import ReconciliationReport


@dataclass
class BacktestReport:
    run_id: str
    horizon: str
    market: str
    as_of: str
    pit_level: str
    survivorship_note: str
    portfolio_metrics: dict
    ic_metrics: dict
    calibration: dict | None
    cost_drag_bps_annual: float
    model_ids: list[str]


def generate_report(
    result: BacktestResult,
    horizon: str,
    market: str,
    model_ids: list[str],
    output_dir: Path,
    calibration_report: CalibrationReport | None = None,
    fmt: Literal["html", "markdown", "both"] = "both",
) -> BacktestReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    pm = compute_portfolio_metrics(result.daily_returns)

    ic_arr = np.array(result.signal_ic)
    ic_metrics: dict = {}
    if len(ic_arr) > 0:
        ic_metrics = {
            "ic_mean": float(np.mean(ic_arr)),
            "ic_std": float(np.std(ic_arr, ddof=1)) if len(ic_arr) > 1 else 0.0,
            "ic_tstat": ic_tstat(ic_arr) if len(ic_arr) > 1 else 0.0,
            "hit_rate": float(np.mean(ic_arr > 0)),
            "n_periods": len(ic_arr),
        }
        if len(ic_arr) > 5:
            lo, hi = block_bootstrap_ci(ic_arr)
            ic_metrics["ic_ci_95_low"] = lo
            ic_metrics["ic_ci_95_high"] = hi

    cost_drag = float(np.mean(result.cost_bps) * 252) if len(result.cost_bps) > 0 else 0.0

    report = BacktestReport(
        run_id=run_id,
        horizon=horizon,
        market=market,
        as_of=datetime.now(tz=timezone.utc).isoformat()[:10],
        pit_level=result.pit_level,
        survivorship_note=result.config.survivorship_note,
        portfolio_metrics=pm,
        ic_metrics=ic_metrics,
        calibration=_cal_dict(calibration_report),
        cost_drag_bps_annual=cost_drag,
        model_ids=model_ids,
    )

    if fmt in ("markdown", "both"):
        (output_dir / f"{run_id}_{market}_{horizon}.md").write_text(_md(report))
    if fmt in ("html", "both"):
        (output_dir / f"{run_id}_{market}_{horizon}.html").write_text(_html(report))

    return report


def generate_reconciliation_report(
    event_driven: "BacktestResult",
    vectorised: "BacktestResult",
    output_dir: Path,
) -> "ReconciliationReport":
    """Generate a reconciliation report comparing the two engines."""
    from learntrade.backtest.reconciliation import reconcile_results
    report = reconcile_results(event_driven, vectorised)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{run_id}_reconciliation.md"
    path.write_text(
        f"# Reconciliation Report\n\n"
        f"- **Result:** {'PASS' if report.passed else 'FAIL'}\n"
        f"- **CAGR gap:** {report.cagr_gap_bps:.1f} bps (tolerance: {report.tolerance_bps:.1f} bps)\n"
        f"- **Event-driven CAGR:** {report.cagr_event_driven:.4%}\n"
        f"- **Vectorised CAGR:** {report.cagr_vectorised:.4%}\n"
        f"- **Message:** {report.message}\n"
    )
    return report


def _cal_dict(cal: CalibrationReport | None) -> dict | None:
    if cal is None:
        return None
    return {
        "brier_score": cal.brier_score,
        "roc_auc": cal.roc_auc,
        "directional_accuracy": cal.directional_accuracy,
        "balanced_accuracy": cal.balanced_accuracy,
        "n_samples": cal.n_samples,
    }


def _fmt(v: float, pct: bool = False) -> str:
    if np.isnan(v):
        return "—"
    return f"{v:.2%}" if pct else f"{v:.4f}"


def _md(r: BacktestReport) -> str:
    pm, ic = r.portfolio_metrics, r.ic_metrics
    lines = [
        f"# Backtest Report: {r.market} {r.horizon}",
        f"",
        f"**Run ID:** `{r.run_id}` | **As of:** {r.as_of} | **PIT level:** `{r.pit_level}`",
        f"**Models:** {', '.join(r.model_ids)}",
        "",
    ]
    if r.survivorship_note:
        lines += [f"> **Survivorship note:** {r.survivorship_note}", ""]
    if r.pit_level == "none":
        lines += [
            "> ⚠ **Data has `pit_level=none`** — results are unvalidated and may contain lookahead bias.",
            "",
        ]
    lines += ["## Portfolio Performance", "",
              "| Metric | Value |", "|---|---|",
              f"| CAGR | {_fmt(pm.get('cagr', float('nan')), True)} |",
              f"| Volatility | {_fmt(pm.get('volatility', float('nan')), True)} |",
              f"| Sharpe | {_fmt(pm.get('sharpe', float('nan')))} |",
              f"| Sortino | {_fmt(pm.get('sortino', float('nan')))} |",
              f"| Max Drawdown | {_fmt(pm.get('max_drawdown', float('nan')), True)} |",
              f"| Calmar | {_fmt(pm.get('calmar', float('nan')))} |",
              f"| Cost Drag (bps/yr) | {r.cost_drag_bps_annual:.1f} |",
              f"| Trading Days | {int(pm.get('n_days', 0))} |",
              ""]
    if ic:
        ci_str = ""
        if "ic_ci_95_low" in ic:
            ci_str = f"| IC 95% CI | [{ic['ic_ci_95_low']:.4f}, {ic['ic_ci_95_high']:.4f}] |"
        lines += ["## Forecast Quality (IC)", "",
                  "| Metric | Value |", "|---|---|",
                  f"| IC Mean | {_fmt(ic.get('ic_mean', float('nan')))} |",
                  f"| IC Std | {_fmt(ic.get('ic_std', float('nan')))} |",
                  f"| IC t-stat | {_fmt(ic.get('ic_tstat', float('nan')))} |",
                  f"| Hit Rate | {_fmt(ic.get('hit_rate', float('nan')), True)} |",
                  f"| N Periods | {ic.get('n_periods', 0)} |"]
        if ci_str:
            lines.append(ci_str)
        lines.append("")
    if r.calibration:
        c = r.calibration
        lines += ["## Calibration", "",
                  "| Metric | Value |", "|---|---|",
                  f"| Brier Score | {_fmt(c.get('brier_score', float('nan')))} |",
                  f"| ROC-AUC | {_fmt(c.get('roc_auc', float('nan')))} |",
                  f"| Directional Accuracy | {_fmt(c.get('directional_accuracy', float('nan')), True)} |",
                  f"| Balanced Accuracy | {_fmt(c.get('balanced_accuracy', float('nan')), True)} |",
                  ""]
    return "\n".join(lines)


def _html(r: BacktestReport) -> str:
    md_content = _md(r)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Backtest: {r.market} {r.horizon}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px;text-align:left}}
th{{background:#f5f5f5}}blockquote{{border-left:4px solid #f0ad4e;padding-left:12px;color:#555}}</style>
</head><body><pre style="white-space:pre-wrap">{md_content}</pre></body></html>"""
