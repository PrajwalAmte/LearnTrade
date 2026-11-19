"""
Microstructure tests — lot rounding, tick rounding, settlement.
"""

from __future__ import annotations

import pytest

from learntrade.execution.microstructure import (
    add_settlement_days,
    get_tick_size,
    parse_settlement_days,
    participation_limit_qty,
    round_to_lot,
    round_to_tick,
)
from learntrade.markets.loader import get_pack
from learntrade.markets.schema import TickBand


BANDS = [
    TickBand(below=1.0, tick=0.0001),
    TickBand(below=float("inf"), tick=0.01),
]


def test_tick_size_low_price():
    assert get_tick_size(0.50, BANDS) == pytest.approx(0.0001)


def test_tick_size_high_price():
    assert get_tick_size(100.0, BANDS) == pytest.approx(0.01)


def test_round_to_tick_exact():
    assert round_to_tick(10.00, BANDS) == pytest.approx(10.00)


def test_round_to_tick_rounds_up():
    # 10.005 → rounds to 10.01 with tick=0.01
    result = round_to_tick(10.005, BANDS)
    assert result == pytest.approx(10.01, abs=1e-9)


def test_round_to_tick_rounds_down():
    result = round_to_tick(10.004, BANDS)
    assert result == pytest.approx(10.00, abs=1e-6)


def test_round_to_lot_exact():
    assert round_to_lot(100.0, 1) == 100


def test_round_to_lot_down():
    assert round_to_lot(107.0, 10) == 100


def test_round_to_lot_fractional():
    assert round_to_lot(99.9, 100) == 0


def test_parse_settlement_t2():
    assert parse_settlement_days("T+2") == 2


def test_parse_settlement_t1():
    assert parse_settlement_days("T+1") == 1


def test_parse_settlement_t0():
    assert parse_settlement_days("T+0") == 0


def test_parse_settlement_fallback():
    assert parse_settlement_days("unknown") == 2


def test_participation_limit_caps():
    result = participation_limit_qty(10_000.0, adv=50_000.0, max_participation=0.10)
    assert result == pytest.approx(5_000.0)


def test_participation_limit_no_cap_below_threshold():
    result = participation_limit_qty(1_000.0, adv=100_000.0, max_participation=0.10)
    assert result == pytest.approx(1_000.0)


def test_add_settlement_days_xnys():
    from datetime import date
    pack = get_pack("XNYS")
    # 2024-01-02 (Tuesday) + T+1 should be 2024-01-03 (Wednesday)
    result = add_settlement_days(date(2024, 1, 2), pack)
    assert result > date(2024, 1, 2)
