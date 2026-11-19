"""
Paper trading session — runs the event-driven simulator on live data.

Paper trading generates proposed orders (which are NOT submitted to any broker)
so users can track how suggestions would have performed with no real money at risk.
Portfolio state is persisted in the app-state database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from learntrade.core.types import InstrumentId, Order
from learntrade.data.store.app_state import AppStateDB
from learntrade.data.store.pit import PITStore
from learntrade.execution.simulator.engine import EventDrivenSimulator, SimulatorConfig
from learntrade.execution.simulator.state import Fill
from learntrade.markets.schema import MarketPack

logger = logging.getLogger(__name__)

PAPER_PORTFOLIO_NAME = "paper_trading"


@dataclass
class PaperTradingSnapshot:
    as_of: datetime
    nav: float
    positions: dict[str, Any]
    fills: list[Fill]
    n_positions: int


class PaperTradingSession:
    """
    Paper trading: runs the event-driven simulator on fresh data each refresh.

    No real orders are submitted. Portfolio state persists across restarts
    via AppStateDB.
    """

    def __init__(
        self,
        pit_store: PITStore,
        app_state_db: AppStateDB,
        market_pack: MarketPack,
        initial_cash: float = 100_000.0,
        base_currency: str = "USD",
        max_participation: float = 0.10,
    ) -> None:
        self._store = pit_store
        self._app_state = app_state_db
        self._pack = market_pack
        self._sim = EventDrivenSimulator(
            pack=market_pack,
            config=SimulatorConfig(
                initial_cash=initial_cash,
                base_currency=base_currency,
                max_participation=max_participation,
            ),
        )
        self._last_refresh: datetime | None = None

    def step(
        self,
        as_of: datetime,
        bars: pl.DataFrame,
        orders: list[Order],
    ) -> PaperTradingSnapshot:
        """
        Process one refresh step: fill pending orders, then queue new ones.
        Call this once per data-refresh cycle (e.g. end of trading day).
        """
        fills = self._sim.step(as_of, bars, orders)
        nav = self._sim.nav
        positions = {
            iid: {"quantity": pos.quantity, "avg_cost": pos.avg_cost, "currency": pos.currency}
            for iid, pos in self._sim.positions.items()
        }

        self._persist(as_of, nav, positions, fills)
        self._last_refresh = as_of

        logger.info(
            "Paper trading step %s: NAV=%.2f positions=%d fills=%d",
            as_of.date(), nav, len(positions), len(fills),
        )

        return PaperTradingSnapshot(
            as_of=as_of,
            nav=nav,
            positions=positions,
            fills=fills,
            n_positions=len(positions),
        )

    def _persist(
        self,
        as_of: datetime,
        nav: float,
        positions: dict[str, Any],
        fills: list[Fill],
    ) -> None:
        from learntrade.core.types import Holding
        holdings = [
            Holding(
                instrument_id=InstrumentId(iid),
                quantity=data["quantity"],
                average_cost=data["avg_cost"],
                currency=data["currency"],
            )
            for iid, data in positions.items()
        ]
        self._app_state.save_portfolio(PAPER_PORTFOLIO_NAME, holdings)

    @property
    def current_nav(self) -> float:
        return self._sim.nav

    @property
    def n_positions(self) -> int:
        return len(self._sim.positions)
