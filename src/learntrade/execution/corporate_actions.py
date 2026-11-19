"""
Corporate action processor — splits, dividends, mergers, delistings.

Each corporate action is an Observation with kind="corporate_action"
and payload encoding the action type and parameters.
Applied to PortfolioState before bar processing on the ex-date.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from leantrade.execution.simulator.state import PortfolioState

logger = logging.getLogger(__name__)


@dataclass
class CorporateAction:
    instrument_id: str
    ex_date: datetime
    action_type: Literal["split", "dividend", "delisting", "merger"]
    factor: float | None = None     # split ratio (e.g. 2.0 = 2-for-1 split)
    amount: float | None = None     # dividend per share
    acquirer_id: str | None = None  # for mergers


class CorporateActionProcessor:
    """
    Applies corporate actions to a PortfolioState on each step.
    Actions are sourced from provider observations (kind="corporate_action").
    """

    def __init__(self) -> None:
        self._pending: list[CorporateAction] = []

    def add(self, action: CorporateAction) -> None:
        self._pending.append(action)

    def apply_pending(self, state: PortfolioState, as_of: datetime) -> list[str]:
        """
        Apply all actions whose ex_date <= as_of.
        Returns list of event descriptions for the attribution log.
        """
        events: list[str] = []
        remaining: list[CorporateAction] = []

        for action in self._pending:
            if action.ex_date > as_of:
                remaining.append(action)
                continue

            iid = action.instrument_id
            pos = state.positions.get(iid)
            if pos is None:
                remaining.append(action)
                continue

            if action.action_type == "split" and action.factor:
                events.append(self._apply_split(state, iid, action.factor))
            elif action.action_type == "dividend" and action.amount:
                events.append(self._apply_dividend(state, iid, action.amount))
            elif action.action_type == "delisting":
                events.append(self._apply_delisting(state, iid))
            elif action.action_type == "merger" and action.factor and action.acquirer_id:
                events.append(self._apply_merger(state, iid, action.factor, action.acquirer_id))

        self._pending = remaining
        return [e for e in events if e]

    def _apply_split(self, state: PortfolioState, iid: str, factor: float) -> str:
        pos = state.positions[iid]
        pos.quantity = round(pos.quantity * factor)
        pos.avg_cost /= factor
        return f"Split {iid} {factor}x: qty={pos.quantity:.0f} avg_cost={pos.avg_cost:.4f}"

    def _apply_dividend(self, state: PortfolioState, iid: str, amount_per_share: float) -> str:
        pos = state.positions[iid]
        cash_received = pos.quantity * amount_per_share
        state.cash += cash_received
        return f"Dividend {iid} ${amount_per_share:.4f}/share: +${cash_received:.2f} cash"

    def _apply_delisting(self, state: PortfolioState, iid: str) -> str:
        pos = state.positions[iid]
        # Delist at last known cost (zero recovery is conservative)
        proceeds = pos.quantity * pos.avg_cost
        state.cash += proceeds
        logger.warning("Delisting %s: closed at avg_cost $%.4f, proceeds $%.2f", iid, pos.avg_cost, proceeds)
        del state.positions[iid]
        return f"Delisting {iid}: +${proceeds:.2f} cash (at avg cost)"

    def _apply_merger(
        self, state: PortfolioState, iid: str, exchange_ratio: float, acquirer_id: str
    ) -> str:
        pos = state.positions[iid]
        new_qty = round(pos.quantity * exchange_ratio)
        from leantrade.core.types import InstrumentId
        from leantrade.execution.simulator.state import Position
        new_iid = acquirer_id
        if new_iid in state.positions:
            existing = state.positions[new_iid]
            total = existing.quantity + new_qty
            existing.avg_cost = (
                (existing.quantity * existing.avg_cost + new_qty * pos.avg_cost / exchange_ratio)
                / total
            )
            existing.quantity = total
        else:
            state.positions[new_iid] = Position(
                instrument_id=InstrumentId(new_iid),
                quantity=new_qty,
                avg_cost=pos.avg_cost / exchange_ratio,
                currency=pos.currency,
            )
        del state.positions[iid]
        return f"Merger {iid}→{acquirer_id} at ratio {exchange_ratio}: {new_qty:.0f} shares"
