"""
Portfolio state types shared by the event-driven simulator and paper trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from leantrade.core.types import InstrumentId, Order
from leantrade.execution.costs import TradeCost


@dataclass
class Position:
    instrument_id: InstrumentId
    quantity: float            # positive = long
    avg_cost: float            # average fill price (in position currency)
    currency: str
    realized_pnl: float = 0.0  # cumulative realised P&L (closes and partial closes)

    @property
    def market_value(self) -> float:
        return self.quantity * self.avg_cost  # updated externally each step

    def close_pnl(self, exit_price: float) -> float:
        """Realised P&L from closing this position at exit_price."""
        return self.quantity * (exit_price - self.avg_cost)


@dataclass
class Fill:
    order_id: str
    instrument_id: InstrumentId
    side: Literal["buy", "sell"]
    quantity: float
    fill_price: float
    cost: TradeCost
    date: datetime
    partial: bool = False
    currency: str = "USD"   # the instrument's Market Pack currency

    @property
    def net_cash_impact(self) -> float:
        """Cash change from this fill (negative for buys, positive for sells)."""
        sign = -1 if self.side == "buy" else 1
        return sign * self.quantity * self.fill_price - self.cost.total


@dataclass
class PortfolioState:
    positions: dict[str, Position] = field(default_factory=dict)
    cash: float = 0.0
    base_currency: str = "USD"
    nav_history: list[dict] = field(default_factory=list)
    fill_history: list[Fill] = field(default_factory=list)

    def nav(self, prices: dict[str, float] | None = None) -> float:
        """Net asset value: cash + sum of position market values."""
        pos_value = sum(
            p.quantity * (prices.get(str(p.instrument_id), p.avg_cost) if prices else p.avg_cost)
            for p in self.positions.values()
        )
        return self.cash + pos_value

    def apply_fill(self, fill: Fill) -> None:
        """Update positions and cash after a fill."""
        iid = str(fill.instrument_id)
        if fill.side == "buy":
            if iid in self.positions:
                pos = self.positions[iid]
                total_qty = pos.quantity + fill.quantity
                pos.avg_cost = (
                    (pos.quantity * pos.avg_cost + fill.quantity * fill.fill_price)
                    / total_qty
                )
                pos.quantity = total_qty
            else:
                self.positions[iid] = Position(
                    instrument_id=fill.instrument_id,
                    quantity=fill.quantity,
                    avg_cost=fill.fill_price,
                    currency=fill.currency,
                    realized_pnl=0.0,
                )
        else:  # sell
            if iid in self.positions:
                pos = self.positions[iid]
                realized = fill.quantity * (fill.fill_price - pos.avg_cost)
                pos.realized_pnl += realized
                pos.quantity -= fill.quantity
                if pos.quantity <= 1e-6:
                    del self.positions[iid]

        self.cash += fill.net_cash_impact
        self.fill_history.append(fill)
