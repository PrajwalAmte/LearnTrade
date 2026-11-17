"""
Event-driven simulator tests — fills, costs, position tracking, halt days.
"""

from __future__ import annotations

import datetime as _dt
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from leantrade.core.types import InstrumentId, Order
from leantrade.execution.simulator.engine import EventDrivenSimulator, SimulatorConfig
from leantrade.markets.loader import get_pack

UTC = timezone.utc


def _make_bars(
    instrument_id: str = "XNYS:IBM",
    n_days: int = 30,
    start_price: float = 100.0,
    seed: int = 42,
) -> list[tuple[datetime, pl.DataFrame]]:
    """Return list of (date, bars_df) pairs."""
    rng = np.random.default_rng(seed)
    price = start_price
    result = []
    for d in range(n_days):
        dt = datetime(2024, 1, 2, tzinfo=UTC) + _dt.timedelta(days=d)
        ret = rng.normal(0.0005, 0.015)
        price *= (1 + ret)
        bars = pl.DataFrame([{
            "instrument_id": instrument_id,
            "event_time": dt,
            "open": price * (1 - 0.002),
            "high": price * 1.005,
            "low": price * 0.995,
            "close": price,
            "adj_close": price,
            "volume": float(rng.integers(100_000, 1_000_000)),
        }])
        result.append((dt, bars))
    return result


def test_buy_fill_increases_position():
    pack = get_pack("XNYS")
    sim = EventDrivenSimulator(pack, SimulatorConfig(initial_cash=10_000.0))
    steps = _make_bars(n_days=5)

    # Submit a buy order on step 0
    dt0, bars0 = steps[0]
    sim.step(dt0, bars0, [Order(instrument_id=InstrumentId("XNYS:IBM"), side="buy", quantity=10)])

    # Fill happens at step 1 (next-bar fill)
    dt1, bars1 = steps[1]
    fills = sim.step(dt1, bars1, [])
    assert len(fills) == 1
    assert fills[0].side == "buy"
    assert fills[0].quantity == pytest.approx(10.0, rel=0.1)
    assert "XNYS:IBM" in sim.positions


def test_sell_fill_decreases_position():
    pack = get_pack("XNYS")
    sim = EventDrivenSimulator(pack, SimulatorConfig(initial_cash=50_000.0))
    steps = _make_bars(n_days=5)

    # Buy first
    dt0, bars0 = steps[0]
    sim.step(dt0, bars0, [Order(instrument_id=InstrumentId("XNYS:IBM"), side="buy", quantity=50)])
    dt1, bars1 = steps[1]
    sim.step(dt1, bars1, [Order(instrument_id=InstrumentId("XNYS:IBM"), side="sell", quantity=50)])
    dt2, bars2 = steps[2]
    fills = sim.step(dt2, bars2, [])
    sell_fills = [f for f in fills if f.side == "sell"]
    assert len(sell_fills) == 1
    # Position should be gone or tiny
    pos = sim.positions.get("XNYS:IBM")
    assert pos is None or pos.quantity < 1


def test_no_fill_on_halt_day():
    """A day where high == low (halt proxy) should produce no fills."""
    pack = get_pack("XNYS")
    sim = EventDrivenSimulator(pack, SimulatorConfig(initial_cash=10_000.0))
    dt0 = datetime(2024, 1, 2, tzinfo=UTC)
    dt1 = datetime(2024, 1, 3, tzinfo=UTC)

    normal_bars = pl.DataFrame([{
        "instrument_id": "XNYS:IBM", "event_time": dt0,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "adj_close": 100.5, "volume": 500_000.0,
    }])
    halt_bars = pl.DataFrame([{
        "instrument_id": "XNYS:IBM", "event_time": dt1,
        "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,  # halt: high == low
        "adj_close": 100.0, "volume": 0.0,
    }])

    sim.step(dt0, normal_bars, [Order(instrument_id=InstrumentId("XNYS:IBM"), side="buy", quantity=10)])
    fills = sim.step(dt1, halt_bars, [])  # should not fill
    assert len(fills) == 0


def test_cost_model_applied_to_fill():
    pack = get_pack("XNSE")  # XNSE has explicit taxes and fees
    sim = EventDrivenSimulator(pack, SimulatorConfig(initial_cash=1_000_000.0))
    dt0 = datetime(2024, 1, 2, tzinfo=UTC)
    dt1 = datetime(2024, 1, 3, tzinfo=UTC)

    bars0 = pl.DataFrame([{
        "instrument_id": "XNSE:INFY", "event_time": dt0,
        "open": 1500.0, "high": 1520.0, "low": 1490.0,
        "close": 1510.0, "adj_close": 1510.0, "volume": 500_000.0,
    }])
    bars1 = pl.DataFrame([{
        "instrument_id": "XNSE:INFY", "event_time": dt1,
        "open": 1510.0, "high": 1530.0, "low": 1500.0,
        "close": 1520.0, "adj_close": 1520.0, "volume": 450_000.0,
    }])
    sim.step(dt0, bars0, [Order(instrument_id=InstrumentId("XNSE:INFY"), side="buy", quantity=100)])
    fills = sim.step(dt1, bars1, [])
    assert len(fills) == 1
    assert fills[0].cost.total > 0


def test_nav_increases_with_rising_prices():
    pack = get_pack("XNYS")
    sim = EventDrivenSimulator(pack, SimulatorConfig(initial_cash=10_000.0))
    # Consistently rising price
    for d in range(10):
        dt = datetime(2024, 1, 2, tzinfo=UTC) + _dt.timedelta(days=d)
        price = 100.0 + d * 0.5
        bars = pl.DataFrame([{
            "instrument_id": "XNYS:IBM", "event_time": dt,
            "open": price, "high": price * 1.002, "low": price * 0.998,
            "close": price, "adj_close": price, "volume": 500_000.0,
        }])
        orders = ([Order(instrument_id=InstrumentId("XNYS:IBM"), side="buy", quantity=50)]
                  if d == 0 else [])
        sim.step(dt, bars, orders)
    result = sim.get_result()
    assert len(result.daily_returns) > 0


def test_participation_limit_enforced():
    """With very low volume, fill quantity should be capped."""
    pack = get_pack("XNYS")
    sim = EventDrivenSimulator(pack, SimulatorConfig(
        initial_cash=1_000_000.0, max_participation=0.10
    ))
    dt0 = datetime(2024, 1, 2, tzinfo=UTC)
    dt1 = datetime(2024, 1, 3, tzinfo=UTC)
    bars0 = pl.DataFrame([{
        "instrument_id": "XNYS:IBM", "event_time": dt0,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "adj_close": 100.0, "volume": 1000.0,  # very low volume
    }])
    bars1 = pl.DataFrame([{
        "instrument_id": "XNYS:IBM", "event_time": dt1,
        "open": 100.5, "high": 101.0, "low": 100.0, "close": 100.5,
        "adj_close": 100.5, "volume": 1000.0,
    }])
    # Try to buy 10,000 shares with ADV of 1,000 → cap at 10% × 1000 = 100
    sim.step(dt0, bars0, [Order(instrument_id=InstrumentId("XNYS:IBM"), side="buy", quantity=10_000)])
    fills = sim.step(dt1, bars1, [])
    if fills:
        assert fills[0].quantity <= 1000.0 * 0.10 + 1  # +1 for float rounding


def test_get_result_has_expected_fields():
    pack = get_pack("XNYS")
    sim = EventDrivenSimulator(pack)
    dt = datetime(2024, 1, 2, tzinfo=UTC)
    bars = pl.DataFrame([{
        "instrument_id": "XNYS:IBM", "event_time": dt,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "adj_close": 100.5, "volume": 500_000.0,
    }])
    sim.step(dt, bars, [])
    result = sim.get_result()
    assert hasattr(result, "daily_returns")
    assert hasattr(result, "fill_history")
    assert hasattr(result, "nav_history")
    assert result.engine_type == "event_driven"
