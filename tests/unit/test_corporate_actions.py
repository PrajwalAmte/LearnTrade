"""
Corporate action tests — splits, dividends, delistings, mergers.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from leantrade.core.types import InstrumentId
from leantrade.execution.corporate_actions import CorporateAction, CorporateActionProcessor
from leantrade.execution.simulator.state import PortfolioState, Position

UTC = timezone.utc


def _state_with_position(iid: str = "XNYS:IBM", qty: float = 100.0, avg_cost: float = 50.0) -> PortfolioState:
    state = PortfolioState(cash=10_000.0)
    state.positions[iid] = Position(
        instrument_id=InstrumentId(iid),
        quantity=qty,
        avg_cost=avg_cost,
        currency="USD",
    )
    return state


def test_split_doubles_shares():
    state = _state_with_position(qty=100.0, avg_cost=50.0)
    proc = CorporateActionProcessor()
    proc.add(CorporateAction("XNYS:IBM", datetime(2024, 1, 5, tzinfo=UTC), "split", factor=2.0))
    events = proc.apply_pending(state, datetime(2024, 1, 5, tzinfo=UTC))
    pos = state.positions["XNYS:IBM"]
    assert pos.quantity == pytest.approx(200.0, abs=1)
    assert pos.avg_cost == pytest.approx(25.0, rel=1e-6)
    assert len(events) == 1


def test_split_future_not_applied():
    state = _state_with_position()
    proc = CorporateActionProcessor()
    proc.add(CorporateAction("XNYS:IBM", datetime(2024, 6, 1, tzinfo=UTC), "split", factor=2.0))
    events = proc.apply_pending(state, datetime(2024, 1, 1, tzinfo=UTC))
    # Action is in the future; must not be applied
    assert state.positions["XNYS:IBM"].quantity == pytest.approx(100.0)
    assert len(events) == 0


def test_dividend_increases_cash():
    state = _state_with_position(qty=100.0)
    initial_cash = state.cash
    proc = CorporateActionProcessor()
    proc.add(CorporateAction("XNYS:IBM", datetime(2024, 1, 5, tzinfo=UTC), "dividend", amount=1.5))
    proc.apply_pending(state, datetime(2024, 1, 5, tzinfo=UTC))
    assert state.cash == pytest.approx(initial_cash + 100.0 * 1.5)


def test_delisting_converts_to_cash():
    state = _state_with_position(qty=100.0, avg_cost=60.0)
    proc = CorporateActionProcessor()
    proc.add(CorporateAction("XNYS:IBM", datetime(2024, 1, 5, tzinfo=UTC), "delisting"))
    proc.apply_pending(state, datetime(2024, 1, 5, tzinfo=UTC))
    assert "XNYS:IBM" not in state.positions
    assert state.cash > 10_000.0  # cash increased by position value


def test_merger_converts_shares():
    state = _state_with_position("XNYS:TGT", qty=100.0, avg_cost=40.0)
    proc = CorporateActionProcessor()
    proc.add(CorporateAction(
        "XNYS:TGT", datetime(2024, 1, 5, tzinfo=UTC), "merger",
        factor=0.5, acquirer_id="XNYS:ACQ",
    ))
    proc.apply_pending(state, datetime(2024, 1, 5, tzinfo=UTC))
    assert "XNYS:TGT" not in state.positions
    assert "XNYS:ACQ" in state.positions
    assert state.positions["XNYS:ACQ"].quantity == pytest.approx(50.0, abs=1)
