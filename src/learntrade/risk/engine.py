"""
Risk engine — hard limits enforced after portfolio optimisation.

The risk engine never sets weights; it only blocks orders that would
violate hard constraints. Engaging the kill switch stops ALL order
generation immediately and irreversibly until it is explicitly disengaged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from learntrade.core.types import Holding, InstrumentId, Order

logger = logging.getLogger(__name__)


@dataclass
class RiskEngineConfig:
    max_drawdown_stop: float = 0.20     # halt if drawdown from peak exceeds this
    daily_loss_stop: float = 0.03       # halt if today's P&L < -this fraction
    kill_switch_active: bool = False
    max_position: float = 0.10          # max single-instrument weight
    max_sector: float = 0.35            # max sector weight
    max_leverage: float = 1.0           # gross exposure / NAV (1.0 = long-only)
    min_cash_pct: float = 0.0           # minimum cash fraction


class KillSwitchError(RuntimeError):
    """Raised when order generation is attempted while kill switch is active."""


class RiskEngine:
    """
    Stateful risk engine that tracks NAV high-water mark and daily P&L.

    Call ``start_of_day(nav)`` at the start of each trading day.
    Call ``check_orders(orders, holdings, nav)`` before submitting any orders.
    """

    def __init__(self, config: RiskEngineConfig | None = None) -> None:
        self._cfg = config or RiskEngineConfig()
        self._peak_nav: float | None = None
        self._start_of_day_nav: float | None = None
        self._daily_check_log: list[dict[str, Any]] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def engage_kill_switch(self) -> None:
        self._cfg.kill_switch_active = True
        logger.critical("Kill switch ENGAGED — all order generation halted")

    def disengage_kill_switch(self) -> None:
        self._cfg.kill_switch_active = False
        logger.warning("Kill switch disengaged")

    @property
    def kill_switch_active(self) -> bool:
        return self._cfg.kill_switch_active

    def start_of_day(self, nav: float) -> None:
        """Call at market open with current NAV to initialise daily tracking."""
        self._start_of_day_nav = nav
        if self._peak_nav is None or nav > self._peak_nav:
            self._peak_nav = nav

    def check_orders(
        self,
        orders: list[Order],
        holdings: list[Holding],
        nav: float,
        prices: dict[str, float] | None = None,
    ) -> list[Order]:
        """
        Filter orders that would breach hard limits.

        Returns the approved order list. May return empty list.
        Raises KillSwitchError if kill switch is active.
        """
        if self._cfg.kill_switch_active:
            raise KillSwitchError(
                "Kill switch is active — no orders will be submitted. "
                "Call disengage_kill_switch() to resume."
            )

        # Check drawdown stop
        if self._peak_nav and self._peak_nav > 0:
            drawdown = (nav - self._peak_nav) / self._peak_nav
            if drawdown < -self._cfg.max_drawdown_stop:
                logger.warning(
                    "Max drawdown stop triggered: %.1f%% drawdown (limit %.1f%%)",
                    abs(drawdown) * 100, self._cfg.max_drawdown_stop * 100,
                )
                return []

        # Check daily loss stop
        if self._start_of_day_nav and self._start_of_day_nav > 0:
            daily_pnl = (nav - self._start_of_day_nav) / self._start_of_day_nav
            if daily_pnl < -self._cfg.daily_loss_stop:
                logger.warning(
                    "Daily loss stop triggered: %.1f%% loss (limit %.1f%%)",
                    abs(daily_pnl) * 100, self._cfg.daily_loss_stop * 100,
                )
                return []

        # Check per-order position limits
        holdings_map = {str(h.instrument_id): h.quantity for h in holdings}
        prices = prices or {}
        approved: list[Order] = []

        for order in orders:
            iid = str(order.instrument_id)
            price = prices.get(iid, 1.0)

            # Existing position + new order
            existing_qty = holdings_map.get(iid, 0.0)
            if order.side == "buy":
                total_notional = (existing_qty + order.quantity) * price
            else:
                total_notional = max(existing_qty - order.quantity, 0) * price

            if nav > 0 and total_notional / nav > self._cfg.max_position:
                logger.warning(
                    "Order for %s rejected: position limit %.1f%% would be exceeded",
                    iid, self._cfg.max_position * 100,
                )
                continue

            approved.append(order)

        return approved

    def compute_exposure(
        self, holdings: list[Holding], prices: dict[str, float], nav: float
    ) -> dict[str, float]:
        """Compute gross and net exposure fractions."""
        long_notional = sum(
            h.quantity * prices.get(str(h.instrument_id), 0.0)
            for h in holdings if h.quantity > 0
        )
        short_notional = sum(
            abs(h.quantity) * prices.get(str(h.instrument_id), 0.0)
            for h in holdings if h.quantity < 0
        )
        return {
            "gross": (long_notional + short_notional) / max(nav, 1e-9),
            "net": (long_notional - short_notional) / max(nav, 1e-9),
            "long": long_notional / max(nav, 1e-9),
            "short": short_notional / max(nav, 1e-9),
        }
