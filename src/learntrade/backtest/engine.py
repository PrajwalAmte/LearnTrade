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

from learntrade.markets.schema import MarketPack


@dataclass
class BacktestConfig:
    max_position: float = 0.05
    long_only: bool = True
    top_n: int = 20
    pit_level: str = "none"
    survivorship_note: str = ""
    engine: Literal["vectorised", "event_driven"] = "vectorised"
    # Trade only every N signal dates (hold in between). Set this to the forecast
    # horizon (e.g. 20 for a 20-day model); daily trading on a 20-day forecast just
    # pays costs to churn a signal that has not changed.
    rebalance_every: int = 1
    initial_nav: float = 100_000.0     # sizes costs; flat fees depend on trade size
    # Measure IC against the return over this many bars (set to the forecast horizon so the
    # backtest IC is comparable to the training IC). 1 = next-bar return.
    ic_horizon: int = 1


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

        # Next-bar return plus the volume/volatility inputs the shared cost model needs
        # (same inputs the event-driven simulator feeds it).
        has_vol = "volume" in bars_sorted.columns
        if has_vol:
            bars_sorted = bars_sorted.with_columns(pl.col("volume").cast(pl.Float64, strict=False))
        bars_with_next = bars_sorted.with_columns(
            (pl.col(close_col).shift(-1).over("instrument_id") / pl.col(close_col) - 1)
            .alias("next_ret"),
            (pl.col(close_col).shift(-max(1, cfg.ic_horizon)).over("instrument_id") / pl.col(close_col) - 1)
            .alias("ic_ret"),
            (pl.col(close_col) / pl.col(close_col).shift(1).over("instrument_id") - 1)
            .rolling_std(window_size=20, min_samples=5).over("instrument_id")
            .mul(np.sqrt(252)).alias("_vol_ann"),
        )
        if has_vol:
            bars_with_next = bars_with_next.with_columns(
                pl.col("volume").rolling_mean(window_size=20, min_samples=1)
                .over("instrument_id").alias("_adv")
            )

        signal_dates = sorted(signals["date"].unique().to_list())
        daily_returns: list[float] = []
        turnover_list: list[float] = []
        cost_list: list[float] = []
        ic_list: list[float] = []
        weight_records: list[dict] = []
        prev_weights: dict[str, float] = {}

        nav = cfg.initial_nav
        every = max(1, cfg.rebalance_every)

        for step_i, date in enumerate(signal_dates):
            date_sigs = (
                signals.filter(pl.col("date") == date)
                .sort("signal", descending=True)
            )
            if date_sigs.is_empty():
                daily_returns.append(0.0)
                turnover_list.append(0.0)
                cost_list.append(0.0)
                continue

            next_bars = bars_with_next.filter(pl.col("event_time") == date)

            if step_i % every == 0 or not prev_weights:
                top = date_sigs.head(cfg.top_n)
                top_ids = top["instrument_id"].to_list()
                w = 1.0 / max(len(top_ids), 1)
                new_weights = {iid: w for iid in top_ids}
            else:
                new_weights = prev_weights   # hold: no trades, no cost

            # Turnover vs previous
            all_ids = set(prev_weights) | set(new_weights)
            to = sum(abs(new_weights.get(i, 0.0) - prev_weights.get(i, 0.0)) for i in all_ids)
            turnover_list.append(to)

            # Full Market Pack cost model (commission, taxes, fees, spread, impact) —
            # the SAME function the event-driven simulator uses, so the two engines
            # differ only in fill mechanics, not in what a trade costs.
            cost = _vectorised_rebalance_cost(
                all_ids, new_weights, prev_weights, next_bars, close_col, nav, market_pack
            )
            cost_list.append(cost * 10_000)  # store as bps of NAV

            # Portfolio return using next-bar returns
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
                    ret = r_row["ic_ret"][0]
                    if ret is not None and not np.isnan(float(ret)):
                        sig_vals.append(sig)
                        ret_vals.append(float(ret))

            if len(sig_vals) >= 5:
                ic, _ = spearmanr(sig_vals, ret_vals)
                ic_list.append(float(ic) if not np.isnan(ic) else 0.0)

            port_ret -= cost
            daily_returns.append(port_ret)
            nav *= (1.0 + port_ret)

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


def _vectorised_rebalance_cost(
    all_ids: set[str],
    new_w: dict[str, float],
    prev_w: dict[str, float],
    day_bars: pl.DataFrame,
    close_col: str,
    nav: float,
    pack: MarketPack,
) -> float:
    """Total trading cost of one rebalance as a fraction of NAV, via compute_trade_cost()."""
    from learntrade.execution.costs import compute_trade_cost

    total = 0.0
    for iid in all_ids:
        dw = new_w.get(iid, 0.0) - prev_w.get(iid, 0.0)
        if abs(dw) < 1e-12:
            continue
        row = day_bars.filter(pl.col("instrument_id") == iid)
        if row.is_empty():
            continue
        price = row[close_col][0]
        if price is None or float(price) <= 0:
            continue
        adv = row["_adv"][0] if "_adv" in row.columns else None
        vol = row["_vol_ann"][0] if "_vol_ann" in row.columns else None
        notional = abs(dw) * nav
        cost = compute_trade_cost(
            side="buy" if dw > 0 else "sell",
            quantity=notional / float(price),
            price=float(price),
            adv=max(float(adv), 1.0) if adv is not None and not np.isnan(float(adv)) else 1e9,
            volatility_annual=float(vol) if vol is not None and not np.isnan(float(vol)) else 0.20,
            pack=pack,
        )
        total += cost.total
    return total / nav if nav > 0 else 0.0


class EventDrivenBacktest:
    """
    Event-driven backtest using the full simulator pipeline.

    Applies next-bar open fills, lot/tick rounding, participation limits,
    and the complete Market Pack cost model (commission, taxes, fees, spread, impact).
    """

    @staticmethod
    def _record_day(fills: list, nav_before: float, cost_list: list[float],
                    turnover_list: list[float]) -> None:
        """Cost (bps of NAV) and turnover (fraction of NAV) for the fills of one day."""
        cost = sum(f.cost.total for f in fills)
        traded = sum(abs(f.quantity * f.fill_price) for f in fills)
        cost_list.append(cost / nav_before * 10_000)
        turnover_list.append(traded / nav_before)

    @staticmethod
    def _ic_step(ic_list: list[float], date_sigs: pl.DataFrame, today_next_rets: pl.DataFrame) -> None:
        """Rank IC of signal vs each instrument's NEXT-BAR RETURN (not its price level)."""
        if date_sigs.is_empty() or today_next_rets.is_empty():
            return
        sig_vals, ret_vals = [], []
        for row in date_sigs.iter_rows(named=True):
            bar = today_next_rets.filter(pl.col("instrument_id") == row["instrument_id"])
            if not bar.is_empty():
                r = bar["next_ret"][0]
                if r is not None and not (isinstance(r, float) and np.isnan(r)):
                    sig_vals.append(float(row["signal"]))
                    ret_vals.append(float(r))
        if len(sig_vals) >= 5:
            ic, _ = spearmanr(sig_vals, ret_vals)
            if not np.isnan(ic):
                ic_list.append(float(ic))

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
        from learntrade.core.types import InstrumentId, Order
        from learntrade.execution.simulator.engine import EventDrivenSimulator, SimulatorConfig

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
                (pl.col(close_col).shift(-max(1, cfg.ic_horizon)).over("instrument_id")
                 / pl.col(close_col) - 1).alias("next_ret")
            )
        )
        cost_list: list[float] = []
        turnover_list: list[float] = []

        for step_i, date in enumerate(signal_dates):
            date_sigs = signals.filter(pl.col("date") == date).sort("signal", descending=True)
            today_bars = bars.filter(pl.col("event_time") == date)
            today_next_rets = bars_with_next.filter(pl.col("event_time") == date)

            # Build target equal-weight portfolio from top-N signals
            top = date_sigs.head(cfg.top_n)
            top_ids = set(top["instrument_id"].to_list())
            target_w = 1.0 / max(len(top_ids), 1)
            current_nav = sim.nav

            # Generate orders to move toward target (only on rebalance dates)
            orders: list[Order] = []
            if step_i % max(1, cfg.rebalance_every) != 0:
                nav_before = max(sim.nav, 1.0)
                self._record_day(sim.step(date, today_bars, orders), nav_before, cost_list, turnover_list)
                self._ic_step(ic_list, date_sigs, today_next_rets)
                continue
            current_pos = {str(iid): pos.quantity for iid, pos in sim.positions.items()}

            # Sells first, so their proceeds are available to fund the buys. Both loops run in
            # sorted order so a run is reproducible: set order varies between processes, and with
            # a cash limit the order decides which buys get funded.
            for iid_str, qty in sorted(current_pos.items()):
                if iid_str not in top_ids and qty > 0:
                    orders.append(Order(
                        instrument_id=InstrumentId(iid_str),
                        side="sell",
                        quantity=qty,
                    ))

            # Buys: instruments in target but not (or under-weight) in current
            for iid_str in sorted(top_ids):
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

            nav_before = max(sim.nav, 1.0)
            self._record_day(sim.step(date, today_bars, orders), nav_before, cost_list, turnover_list)

            self._ic_step(ic_list, date_sigs, today_next_rets)

        result = sim.get_result()

        # Build weight records from last known portfolio
        weight_records = [
            {"date": signal_dates[-1] if signal_dates else None,
             "instrument_id": iid, "weight": pos.quantity * pos.avg_cost / max(sim.nav, 1)}
            for iid, pos in sim.positions.items()
        ]

        return BacktestResult(
            daily_returns=result.daily_returns,
            portfolio_weights=pl.DataFrame(weight_records) if weight_records else pl.DataFrame(),
            # Real per-day figures from the fills, not one flat average repeated.
            turnover=np.array(turnover_list),
            cost_bps=np.array(cost_list),
            signal_ic=ic_list,
            config=cfg,
        )
