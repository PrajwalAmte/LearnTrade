"""
Out-of-sample evaluation ladder.

Every candidate (momentum baseline, ridge, LightGBM, LightGBM ranker) is scored on the
SAME stock list, the SAME final untouched dates, the SAME cost model and the SAME
trading cadence. Models are fitted on the development period only, so nothing here is
in-sample. This is the table to read before believing any single model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from learntrade.core.errors import friendly_error
from learntrade.core.types import InstrumentId
from learntrade.backtest.engine import BacktestConfig, EventDrivenBacktest, VectorisedBacktest
from learntrade.quant.validation.metrics import (
    block_bootstrap_ci, compute_portfolio_metrics, ic_by_date, ic_tstat_overlap,
)

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    name: str
    rank_ic: float
    rank_ic_ci: tuple[float, float]
    hit_rate: float
    t_adj: float
    total_return: float
    cagr: float
    sharpe: float
    max_dd: float
    cost_drag_bps: float
    n_days: int


@dataclass
class EvaluationResult:
    market: str
    horizon: str
    start: str
    end: str
    n_instruments: int
    rebalance_every: int
    candidates: list[Candidate] = field(default_factory=list)
    equal_weight_return: float | None = None
    benchmark_id: str | None = None
    benchmark_return: float | None = None
    notes: list[str] = field(default_factory=list)


def _period_return(bars: pl.DataFrame, iid: str, start, end) -> float | None:
    b = bars.filter(pl.col("instrument_id") == iid).sort("event_time")
    col = "adj_close" if "adj_close" in b.columns else "close"
    b = b.filter((pl.col("event_time") >= start) & (pl.col("event_time") <= end))
    if b.height < 2:
        return None
    first, last = float(b[col][0]), float(b[col][-1])
    return last / first - 1 if first > 0 else None


def run_evaluation(
    store: Any, market: str, horizon: str, since: datetime, holdout_days: int,
    models: dict[str, type], top_n: int = 10, engine: str = "vectorised",
) -> EvaluationResult:
    from learntrade.markets.loader import get_pack
    from learntrade.quant.targets.horizon import get_horizon
    from learntrade.quant.trainer import QuantTrainer

    now = datetime.now(tz=timezone.utc)
    fwd = get_horizon(horizon).forward_days
    pack = get_pack(market)
    out = EvaluationResult(market=market, horizon=horizon, start="", end="",
                           n_instruments=0, rebalance_every=fwd)

    candidates: dict[str, tuple[type | None, str]] = {"momentum (baseline)": (None, "momentum")}
    candidates.update({n: (c, "model") for n, c in models.items()})

    hold_ref: pl.DataFrame | None = None
    bars: pl.DataFrame | None = None
    mode_note = ""
    dataset = None   # features + labels, built once and shared by every candidate
    for name, (cls, kind) in candidates.items():
        try:
            trainer = QuantTrainer(store, horizon, model_class=cls or next(iter(models.values())),
                                   holdout_days=holdout_days)
            if dataset is None:
                dataset = trainer.build_dataset(market, since, now)
            sigs, hold, n_inst = trainer.holdout_signals(market, since, now, signal=kind,
                                                          dataset=dataset)
            mode_note = membership_note(trainer._membership)
        except Exception as exc:
            out.notes.append(f"{name}: could not be evaluated. {friendly_error(exc)}")
            continue
        if hold_ref is None:
            hold_ref = hold
            out.n_instruments = n_inst
            out.start = str(hold["event_time"].min())[:10]
            out.end = str(hold["event_time"].max())[:10]
            instruments = store.get_universe_union(market, now)
            bars = store.query_bars(
                instruments, as_of=now, since=hold["event_time"].min() - timedelta(days=45))

        per_date = ic_by_date(
            sigs["date"].to_numpy(), sigs["signal"].to_numpy(), _aligned_labels(sigs, hold)
        )
        r = per_date["rank_ic"]
        if len(r) < 3:
            out.notes.append(f"{name}: too few usable dates for an IC")
            continue
        ci = block_bootstrap_ci(r, min_block=fwd)
        cfg = BacktestConfig(top_n=top_n, rebalance_every=fwd, ic_horizon=fwd,
                             survivorship_note="Stock list is today's list (survivorship bias).")
        runner = EventDrivenBacktest() if engine == "event_driven" else VectorisedBacktest()
        res = runner.run(signals=sigs, bars=bars, market_pack=pack, config=cfg)
        pm = compute_portfolio_metrics(res.daily_returns)
        drag = float(np.mean(res.cost_bps) * 252) if len(res.cost_bps) else 0.0
        out.candidates.append(Candidate(
            name=name, rank_ic=float(np.mean(r)), rank_ic_ci=(ci[0], ci[1]),
            hit_rate=float(np.mean(r > 0)), t_adj=float(ic_tstat_overlap(r, fwd)),
            total_return=float(np.prod(1.0 + res.daily_returns) - 1.0),
            cagr=float(pm.get("cagr", np.nan)), sharpe=float(pm.get("sharpe", np.nan)),
            max_dd=float(pm.get("max_drawdown", np.nan)), cost_drag_bps=drag, n_days=len(r),
        ))

    if hold_ref is not None and bars is not None:
        start, end = hold_ref["event_time"].min(), hold_ref["event_time"].max()
        evaluated = set(hold_ref["instrument_id"].unique().to_list())
        rets = [x for x in (_period_return(bars, str(i), start, end)
                            for i in bars["instrument_id"].unique().to_list()
                            if i in evaluated) if x is not None]
        out.equal_weight_return = float(np.mean(rets)) if rets else None
        bm = pack.benchmark
        if bm:
            bbars = store.query_bars([InstrumentId(bm)], as_of=now, since=start - timedelta(days=5))
            r_bm = _period_return(bbars, bm, start, end) if not bbars.is_empty() else None
            out.benchmark_id, out.benchmark_return = bm, r_bm
            if r_bm is None:
                out.notes.append(f"Benchmark {bm} has no data; run `learntrade data pull`.")
    out.notes.append(
        "One test window only. A different recent period could rank these differently; "
        "treat gaps between models smaller than their uncertainty ranges as ties."
    )
    if mode_note:
        out.notes.append(mode_note)
    return out


def membership_note(mem: dict) -> str:
    if mem.get("mode") == "point_in_time":
        note = (f"Stock membership is applied date by date ({mem.get('snapshots')} snapshots), "
                f"so each date only uses stocks that were in the list then.")
        cov = mem.get("coverage")
        if cov is not None and cov < 0.95:
            note += (f" {100 * (1 - cov):.0f}% of past members have no price data "
                     f"(usually delisted), so the picture is not complete.")
        return note
    return ("The stock list is today's list. Companies that dropped out in the past are not "
            "included, which tends to make past results look a little better than they were.")


def _aligned_labels(sigs: pl.DataFrame, hold: pl.DataFrame) -> np.ndarray:
    """Labels in the same row order as ``sigs`` (signals were built from ``hold`` rows)."""
    joined = sigs.join(
        hold.select(pl.col("event_time").alias("date"), "instrument_id", "label"),
        on=["date", "instrument_id"], how="left",
    )
    return joined["label"].to_numpy()


def to_markdown(r: EvaluationResult) -> str:
    L = [f"# Out-of-sample evaluation: {r.market} {r.horizon}", "",
         f"Test window: **{r.start} to {r.end}** (never used for fitting). "
         f"{r.n_instruments} stocks, trading every {r.rebalance_every} days, "
         f"full {r.market} cost model, top stocks held equal-weight.", "",
         "| Candidate | Rank IC | 95% range | Hit rate | t (adjusted) | Return over window (net) | Sharpe | Max drawdown | Cost drag (bps/yr) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in r.candidates:
        L.append(f"| {c.name} | {c.rank_ic:+.3f} | [{c.rank_ic_ci[0]:+.3f}, {c.rank_ic_ci[1]:+.3f}] | "
                 f"{c.hit_rate:.0%} | {c.t_adj:.2f} | {c.total_return:+.1%} | {c.sharpe:.2f} | "
                 f"{c.max_dd:.1%} | {c.cost_drag_bps:.0f} |")
    L.append("")
    if r.equal_weight_return is not None:
        L.append(f"- Equal-weight all {r.n_instruments} stocks over the window (no costs): "
                 f"**{r.equal_weight_return:+.1%}**")
    if r.benchmark_return is not None:
        L.append(f"- Benchmark {r.benchmark_id} over the window: **{r.benchmark_return:+.1%}**")
    L.append("")
    L += [f"- {n}" for n in r.notes]
    return "\n".join(L) + "\n"
