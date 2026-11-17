"""
Event-driven simulation engine.

Design contract:
  - Orders submitted at step t are filled at the OPEN of step t+1 (next-bar fill).
  - Same-bar look-ahead is structurally impossible: orders arrive after the bar closes.
  - Full cost model from the Market Pack is applied on every fill.
  - Participation limit: orders capped at max_participation × that day's volume.
  - Price-limit days: if the bar's high == low (halt proxy), no fill occurs.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import polars as pl

from leantrade.core.types import InstrumentId, Order
from leantrade.execution.costs import TradeCost, compute_trade_cost
from leantrade.execution.microstructure import (
    is_trading_day,
    participation_limit_qty,
    round_to_lot,
    round_to_tick,
)
from leantrade.execution.simulator.state import Fill, PortfolioState, Position
from leantrade.markets.schema import MarketPack


@dataclass
class SimulatorConfig:
    initial_cash: float = 100_000.0
    base_currency: str = "USD"
    max_participation: float = 0.10   # cap at 10% of daily volume
    ewma_vol_span: int = 20            # for volatility estimate in cost model
    long_only: bool = True


@dataclass
class SimulatorResult:
    nav_history: list[dict]           # [{"date": ..., "nav": ...}]
    fill_history: list[Fill]
    daily_returns: np.ndarray
    total_cost_bps: float             # mean cost per trade in bps
    turnover: float                   # mean daily turnover fraction
    engine_type: str = "event_driven"

    @property
    def daily_returns_arr(self) -> np.ndarray:
        return self.daily_returns


class EventDrivenSimulator:
    """
    Step-by-step event-driven backtester.

    Usage:
        sim = EventDrivenSimulator(pack, config)
        for date in dates:
            bars = get_bars(date)        # today's OHLCV
            fills = sim.step(date, bars, orders)   # orders from last step are filled
        result = sim.get_result()
    """

    def __init__(self, pack: MarketPack, config: SimulatorConfig | None = None) -> None:
        self._pack = pack
        self._cfg = config or SimulatorConfig()
        self._state = PortfolioState(cash=self._cfg.initial_cash,
                                     base_currency=self._cfg.base_currency)
        self._pending_orders: list[Order] = []
        self._vol_history: dict[str, list[float]] = {}  # instrument_id → daily returns
        self._prev_prices: dict[str, float] = {}
        self._all_costs: list[float] = []
        self._all_turnovers: list[float] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def step(
        self,
        date: datetime,
        bars: pl.DataFrame,
        new_orders: list[Order] | None = None,
    ) -> list[Fill]:
        """
        Process one bar.

        1. Fill pending orders using today's prices.
        2. Update volatility estimates.
        3. Record NAV.
        4. Queue new_orders for next step.
        Returns fills executed this step.
        """
        # 1. Build price lookup for this day
        prices = self._extract_prices(bars)

        # 2. Fill pending orders
        fills = self._execute_pending(bars, prices, date)

        # 3. Update vol estimates
        self._update_vols(prices)

        # 4. Record NAV
        nav = self._state.nav(prices)
        self._state.nav_history.append({"date": date, "nav": nav})

        # 5. Queue new orders
        self._pending_orders = new_orders or []

        return fills

    def get_result(self) -> SimulatorResult:
        navs = [h["nav"] for h in self._state.nav_history]
        daily_rets = np.diff(navs) / np.array(navs[:-1]) if len(navs) > 1 else np.array([])

        mean_cost_bps = float(np.mean(self._all_costs)) if self._all_costs else 0.0
        mean_turnover = float(np.mean(self._all_turnovers)) if self._all_turnovers else 0.0

        return SimulatorResult(
            nav_history=self._state.nav_history,
            fill_history=self._state.fill_history,
            daily_returns=daily_rets,
            total_cost_bps=mean_cost_bps,
            turnover=mean_turnover,
        )

    @property
    def nav(self) -> float:
        return self._state.cash + sum(
            p.quantity * self._prev_prices.get(str(p.instrument_id), p.avg_cost)
            for p in self._state.positions.values()
        )

    @property
    def positions(self) -> dict[str, Position]:
        return self._state.positions

    @property
    def cash(self) -> float:
        return self._state.cash

    # ── Internals ──────────────────────────────────────────────────────────

    def _extract_prices(self, bars: pl.DataFrame) -> dict[str, float]:
        """Build {instrument_id: open_price} from today's bars."""
        if bars.is_empty():
            return {}
        price_col = "open" if "open" in bars.columns else "close"
        return {
            str(row["instrument_id"]): float(row[price_col])
            for row in bars.iter_rows(named=True)
            if row.get(price_col) is not None
        }

    def _execute_pending(
        self,
        bars: pl.DataFrame,
        prices: dict[str, float],
        date: datetime,
    ) -> list[Fill]:
        fills: list[Fill] = []
        prev_nav = self.nav

        for order in self._pending_orders:
            iid = str(order.instrument_id)
            price = prices.get(iid)
            if price is None or price <= 0:
                continue  # instrument not trading today

            # Check for price-limit day (halt proxy: very narrow range)
            bar_row = bars.filter(pl.col("instrument_id") == iid)
            if not bar_row.is_empty():
                hi = bar_row["high"][0] if "high" in bars.columns else price
                lo = bar_row["low"][0] if "low" in bars.columns else price
                if hi is not None and lo is not None and float(hi) == float(lo) and float(hi) > 0:
                    continue  # halt / price-limit day: no fill

            volume = 0.0
            if not bar_row.is_empty() and "volume" in bars.columns:
                v = bar_row["volume"][0]
                volume = float(v) if v is not None else 0.0

            # Apply lot rounding
            qty = round_to_lot(order.quantity, self._pack.lot_size)
            if qty <= 0:
                continue

            # Apply participation limit
            qty = participation_limit_qty(qty, volume, self._cfg.max_participation)
            if qty <= 0:
                continue

            partial = qty < order.quantity

            # Tick-round the fill price
            fill_price = round_to_tick(price, self._pack.tick_size)
            if fill_price <= 0:
                continue

            # Long-only guard
            if self._cfg.long_only and order.side == "sell":
                current_qty = self._state.positions.get(iid, Position(
                    instrument_id=order.instrument_id, quantity=0,
                    avg_cost=0, currency="USD"
                )).quantity
                qty = min(qty, current_qty)
                if qty <= 0:
                    continue

            # Volatility for cost model
            vol_ann = self._get_vol_annual(iid)

            cost = compute_trade_cost(
                side=order.side,
                quantity=qty,
                price=fill_price,
                adv=max(volume, 1.0),
                volatility_annual=vol_ann,
                pack=self._pack,
            )
            # NOTE: spread is charged exactly once, as `cost.spread_cost`
            # (part of `cost.total`, deducted from cash via
            # Fill.net_cash_impact). It must not also be baked into
            # `fill_price` — doing both would charge the half-spread twice
            # for every trade.

            # Cash check: a buy can never be funded by more cash than the
            # portfolio has. Scale the quantity down to what's affordable
            # (respecting lot size) instead of letting cash go negative, and
            # skip the order entirely if not even one lot is affordable.
            if order.side == "buy":
                required_cash = qty * fill_price + cost.total
                if required_cash > self._state.cash:
                    if self._state.cash <= 0:
                        continue
                    # Leave a small buffer for the cost recomputed below.
                    affordable_qty = self._state.cash / (fill_price * 1.02)
                    qty = round_to_lot(min(qty, affordable_qty), self._pack.lot_size)
                    if qty <= 0:
                        continue
                    partial = True
                    cost = compute_trade_cost(
                        side=order.side, quantity=qty, price=fill_price,
                        adv=max(volume, 1.0), volatility_annual=vol_ann, pack=self._pack,
                    )
                    if qty * fill_price + cost.total > self._state.cash:
                        continue  # still unaffordable even after scaling down

            fill = Fill(
                order_id=str(uuid.uuid4()),
                instrument_id=order.instrument_id,
                side=order.side,
                quantity=qty,
                fill_price=fill_price,
                cost=cost,
                date=date,
                partial=partial,
                currency=self._pack.currency,
            )
            self._state.apply_fill(fill)
            fills.append(fill)
            self._all_costs.append(cost.total_bps)

        # Compute turnover
        new_nav = self.nav
        base = (prev_nav + new_nav) / 2 if (prev_nav + new_nav) > 0 else 1
        traded_notional = sum(f.quantity * f.fill_price for f in fills)
        self._all_turnovers.append(traded_notional / base)

        return fills

    def _update_vols(self, prices: dict[str, float]) -> None:
        for iid, px in prices.items():
            prev = self._prev_prices.get(iid)
            if prev and prev > 0 and px > 0:
                ret = math.log(px / prev)
                self._vol_history.setdefault(iid, []).append(ret)
                if len(self._vol_history[iid]) > self._cfg.ewma_vol_span * 5:
                    self._vol_history[iid] = self._vol_history[iid][-self._cfg.ewma_vol_span * 5:]
        self._prev_prices.update(prices)

    def _get_vol_annual(self, iid: str) -> float:
        rets = self._vol_history.get(iid, [])
        if len(rets) < 5:
            return 0.20  # default 20% vol when no history available
        span = self._cfg.ewma_vol_span
        decay = 1.0 - 2.0 / (span + 1)
        weights = decay ** np.arange(len(rets) - 1, -1, -1)
        weights /= weights.sum()
        ewma_var = float(np.dot(weights, np.array(rets) ** 2))
        return float(math.sqrt(ewma_var * 252))
