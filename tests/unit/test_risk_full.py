"""
Risk engine tests — kill switch, drawdown stop, daily-loss stop, position limits.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from learntrade.core.types import Holding, InstrumentId, Order
from learntrade.risk.engine import KillSwitchError, RiskEngine, RiskEngineConfig

UTC = timezone.utc


def _order(iid: str = "XNYS:IBM", qty: float = 100.0, side: str = "buy") -> Order:
    return Order(instrument_id=InstrumentId(iid), side=side, quantity=qty)


def test_kill_switch_raises():
    engine = RiskEngine(RiskEngineConfig(kill_switch_active=True))
    with pytest.raises(KillSwitchError):
        engine.check_orders([_order()], [], 10_000.0)


def test_kill_switch_engage_disengages():
    engine = RiskEngine()
    engine.engage_kill_switch()
    assert engine.kill_switch_active
    engine.disengage_kill_switch()
    assert not engine.kill_switch_active
    # After disengaging, orders pass through
    result = engine.check_orders([_order()], [], 10_000.0)
    assert len(result) == 1


def test_drawdown_stop_blocks_orders():
    engine = RiskEngine(RiskEngineConfig(max_drawdown_stop=0.10))
    engine.start_of_day(100_000.0)  # peak = 100k
    # NAV dropped to 85k → drawdown = 15% > 10% limit
    result = engine.check_orders([_order()], [], 85_000.0)
    assert len(result) == 0


def test_drawdown_stop_passes_small_drawdown():
    # Only drawdown active; daily loss stop set high so it doesn't interfere
    engine = RiskEngine(RiskEngineConfig(max_drawdown_stop=0.20, daily_loss_stop=1.0))
    engine.start_of_day(100_000.0)  # peak = start = 100k
    # NAV dropped to 95k → drawdown = 5% < 20% limit → should pass
    result = engine.check_orders([_order()], [], 95_000.0)
    assert len(result) == 1


def test_daily_loss_stop():
    engine = RiskEngine(RiskEngineConfig(daily_loss_stop=0.02))
    engine.start_of_day(100_000.0)
    # Day P&L = -5k = -5% > daily_loss_stop of 2%
    result = engine.check_orders([_order()], [], 95_000.0)
    assert len(result) == 0


def test_position_limit_rejects_oversized_order():
    engine = RiskEngine(RiskEngineConfig(max_position=0.05))
    engine.start_of_day(100_000.0)
    # Buying 200 shares at $100 = $20k = 20% of $100k NAV > 5% limit
    result = engine.check_orders(
        [_order("XNYS:IBM", qty=200.0)], [],
        nav=100_000.0,
        prices={"XNYS:IBM": 100.0},
    )
    assert len(result) == 0


def test_position_limit_accepts_small_order():
    engine = RiskEngine(RiskEngineConfig(max_position=0.10))
    engine.start_of_day(100_000.0)
    # Buying 50 shares at $100 = $5k = 5% of $100k NAV < 10% limit
    result = engine.check_orders(
        [_order("XNYS:IBM", qty=50.0)], [],
        nav=100_000.0,
        prices={"XNYS:IBM": 100.0},
    )
    assert len(result) == 1


def test_zero_orders_still_passes():
    engine = RiskEngine()
    engine.start_of_day(100_000.0)
    result = engine.check_orders([], [], 100_000.0)
    assert result == []


def test_kill_switch_zero_emissions():
    """Exit criterion: kill switch produces zero order emissions."""
    engine = RiskEngine()
    engine.start_of_day(100_000.0)
    engine.engage_kill_switch()

    orders = [_order("XNYS:IBM"), _order("XNYS:AAPL"), _order("XNYS:MSFT")]
    with pytest.raises(KillSwitchError):
        engine.check_orders(orders, [], 100_000.0)
    # Verify no orders slipped through
    assert engine.kill_switch_active
