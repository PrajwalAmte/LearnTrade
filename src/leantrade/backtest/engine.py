"""
Backtesting engines — vectorised (fast) and event-driven (accurate).

VectorisedBacktest:  Phase 2 engine, equal-weight top-N, close-price fills.
EventDrivenBacktest: Phase 3 engine, full cost model, next-bar open fills.
Both produce a BacktestResult; use reconcile_results() to compare them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from leantrade.markets.schema import MarketPack


@dataclass
class BacktestConfig:
    max_position: float = 0.05
    long_only: bool = True
    top_n: int = 20
    pit_level: str = "none"
    survivorship_note: str = ""
    engine: Literal["vectorised", "event_driven"] = "vectorised"


@dataclass
class BacktestResult:
    daily_returns: np.ndarray
    portfolio_weights: pl.DataFrame
    turnover: np.ndarray
    cost_bps: np.ndarray
    signal_ic: list[float]
    config: BacktestConfig

    @property
    def pit_level(self) -> str:
        return self.config.pit_level


class VectorisedBacktest:
    """
    Vectorised backtester: signals → equal-weight top-N → next-bar return - cost.
    """

    def run(
        self,
        signals: pl.DataFrame,
        bars: pl.DataFrame,
        market_pack: MarketPack,
        config: BacktestConfig | None = None,
    ) -> BacktestResult:
        """
        Args:
            signals: [date, instrument_id, signal]  — higher signal = better rank
            bars:    [instrument_id, event_time, close, (adj_close), volume]
            market_pack: used for cost model
            config: backtest parameters
        """
        cfg = config or BacktestConfig()

        close_col = "adj_close" if "adj_close" in bars.columns else "close"
        bars_sorted = bars.sort(["instrument_id", "event_time"]).with_columns(
            pl.col(close_col).cast(pl.Float64, strict=False)
        )

        # Compute next-bar returns (uses shift(-1) which is "next" return)
        bars_with_next = bars_sorted.with_columns(
            (pl.col(close_col).shift(-1).over("instrument_id") / pl.col(close_col) - 1)
            .alias("next_ret")
        )

        signal_dates = sorted(signals["date"].unique().to_list())
        daily_returns: list[float] = []
        turnover_list: list[float] = []
        cost_list: list[float] = []
        ic_list: list[float] = []
        weight_records: list[dict] = []
        prev_weights: dict[str, float] = {}

        # Commission in bps from Market Pack
        commission_bps = market_pack.costs.commission.bps

        for date in signal_dates:
            date_sigs = (
                signals.filter(pl.col("date") == date)
                .sort("signal", descending=True)
            )
            if date_sigs.is_empty():
                daily_returns.append(0.0)
                turnover_list.append(0.0)
                cost_list.append(0.0)
                continue

            top = date_sigs.head(cfg.top_n)
            top_ids = top["instrument_id"].to_list()
            w = 1.0 / max(len(top_ids), 1)
            new_weights = {iid: w for iid in top_ids}

            # Turnover vs previous
            all_ids = set(prev_weights) | set(new_weights)
            to = sum(abs(new_weights.get(i, 0.0) - prev_weights.get(i, 0.0)) for i in all_ids)
            turnover_list.append(to)

            cost = to * commission_bps / 10_000
            cost_list.append(cost * 10_000)  # store as bps

            # Portfolio return using next-bar returns
            next_bars = bars_with_next.filter(pl.col("event_time") == date)
            port_ret = 0.0
            sig_vals: list[float] = []
            ret_vals: list[float] = []

            for iid, w_i in new_weights.items():
                row = next_bars.filter(pl.col("instrument_id") == iid)
                if not row.is_empty():
                    ret = row["next_ret"][0]
                    if ret is not None and not np.isnan(float(ret)):
                        port_ret += w_i * float(ret)

            # IC for this period
            for row in date_sigs.iter_rows(named=True):
                iid = row["instrument_id"]
                sig = float(row["signal"])
                r_row = next_bars.filter(pl.col("instrument_id") == iid)
                if not r_row.is_empty():
                    ret = r_row["next_ret"][0]
                    if ret is not None and not np.isnan(float(ret)):
                        sig_vals.append(sig)
                        ret_vals.append(float(ret))

            if len(sig_vals) >= 5:
                ic, _ = spearmanr(sig_vals, ret_vals)
                ic_list.append(float(ic) if not np.isnan(ic) else 0.0)

            port_ret -= cost
            daily_returns.append(port_ret)

            for iid, w_i in new_weights.items():
                weight_records.append({"date": date, "instrument_id": iid, "weight": w_i})
            prev_weights = new_weights

        return BacktestResult(
            daily_returns=np.array(daily_returns),
            portfolio_weights=pl.DataFrame(weight_records) if weight_records else pl.DataFrame(),
            turnover=np.array(turnover_list),
            cost_bps=np.array(cost_list),
            signal_ic=ic_list,
            config=cfg,
        )


class EventDrivenBacktest:
    """
    Event-driven backtest using the full simulator pipeline.

    Applies next-bar open fills, lot/tick rounding, participation limits,
    and the complete Market Pack cost model (commission, taxes, fees, spread, impact).
    """

    def run(
        self,
        signals: pl.DataFrame,
        bars: pl.DataFrame,
        market_pack: MarketPack,
        config: BacktestConfig | None = None,
        initial_cash: float = 100_000.0,
    ) -> BacktestResult:
        """
        Args:
            signals: [date, instrument_id, signal]  — higher signal = better rank
            bars:    [instrument_id, event_time, open, close, (adj_close), volume]
            market_pack: used for cost model, lot/tick sizes
            config: backtest parameters
            initial_cash: starting portfolio value
        """
        from leantrade.core.types import InstrumentId, Order
        from leantrade.execution.simulator.engine import EventDrivenSimulator, SimulatorConfig

        cfg = config or BacktestConfig(engine="event_driven")

        sim = EventDrivenSimulator(
            pack=market_pack,
            config=SimulatorConfig(
                initial_cash=initial_cash,
                base_currency=market_pack.currency,
                max_participation=0.10,
                long_only=cfg.long_only,
            ),
        )

        signal_dates = sorted(signals["date"].unique().to_list())
        ic_list: list[float] = []

        # Precompute each instrument's next-bar return per date so IC can be
        # measured against what the signal is actually supposed to predict
        # (a forward RETURN), not against the raw close PRICE level — a
        # signal that is merely correlated with "which stocks are
        # expensive" previously showed up as a large, meaningless "IC".
        close_col = "adj_close" if "adj_close" in bars.columns else "close"
        bars_with_next = (
            bars.sort(["instrument_id", "event_time"])
            .with_columns(pl.col(close_col).cast(pl.Float64, strict=False))
            .with_columns(
                (pl.col(close_col).shift(-1).over("instrument_id") / pl.col(close_col) - 1)
                .alias("next_ret")
            )
        )

        for date in signal_dates:
            date_sigs = signals.filter(pl.col("date") == date).sort("signal", descending=True)
            today_bars = bars.filter(pl.col("event_time") == date)
            today_next_rets = bars_with_next.filter(pl.col("event_time") == date)

            # Build target equal-weight portfolio from top-N signals
            top = date_sigs.head(cfg.top_n)
            top_ids = set(top["instrument_id"].to_list())
            target_w = 1.0 / max(len(top_ids), 1)
            current_nav = sim.nav

            # Generate orders to move toward target
            orders: list[Order] = []
            current_pos = {str(iid): pos.quantity for iid, pos in sim.positions.items()}

            # Buys: instruments in target but not (or under-weight) in current
            for iid_str in top_ids:
                target_notional = current_nav * target_w
                bar_row = today_bars.filter(pl.col("instrument_id") == iid_str)
                if bar_row.is_empty():
                    continue
                price_col = "open" if "open" in today_bars.columns else "close"
                price = bar_row[price_col][0]
                if price is None or float(price) <= 0:
                    continue
                target_qty = target_notional / float(price)
                current_qty = current_pos.get(iid_str, 0.0)
                delta = target_qty - current_qty
                if delta > 1:
                    orders.append(Order(
                        instrument_id=InstrumentId(iid_str),
                        side="buy",
                        quantity=delta,
                    ))

            # Sells: instruments in current portfolio but not in target
            for iid_str, qty in current_pos.items():
                if iid_str not in top_ids and qty > 0:
                    orders.append(Order(
                        instrument_id=InstrumentId(iid_str),
                        side="sell",
                        quantity=qty,
                    ))

            sim.step(date, today_bars, orders)

            # Compute IC for this step: signal vs. each instrument's
            # NEXT-BAR RETURN (not its price level — see comment above).
            if not date_sigs.is_empty() and not today_next_rets.is_empty():
                sig_vals, ret_vals = [], []
                for row in date_sigs.iter_rows(named=True):
                    iid = row["instrument_id"]
                    bar = today_next_rets.filter(pl.col("instrument_id") == iid)
                    if not bar.is_empty():
                        r = bar["next_ret"][0]
                        if r is not None and not (isinstance(r, float) and np.isnan(r)):
                            sig_vals.append(float(row["signal"]))
                            ret_vals.append(float(r))
                if len(sig_vals) >= 5:
                    ic, _ = spearmanr(sig_vals, ret_vals)
                    if not np.isnan(ic):
                        ic_list.append(float(ic))

        result = sim.get_result()

        # Build weight records from last known portfolio
        weight_records = [
            {"date": signal_dates[-1] if signal_dates else None,
             "instrument_id": iid, "weight": pos.quantity * pos.avg_cost / max(sim.nav, 1)}
            for iid, pos in sim.positions.items()
        ]

        mean_cost = result.total_cost_bps

        return BacktestResult(
            daily_returns=result.daily_returns,
            portfolio_weights=pl.DataFrame(weight_records) if weight_records else pl.DataFrame(),
            turnover=np.full(len(result.daily_returns), result.turnover),
            cost_bps=np.full(len(result.daily_returns), mean_cost),
            signal_ic=ic_list,
            config=cfg,
        )
