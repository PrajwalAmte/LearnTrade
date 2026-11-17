"""
Clock tests.
"""

from __future__ import annotations

from datetime import datetime, timezone

from leantrade.core.clock import FixedClock, WallClock, install_clock, now


def test_wall_clock_returns_utc():
    t = WallClock().now()
    assert t.tzinfo is not None


def test_fixed_clock_pins_time():
    dt = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    clock = FixedClock(dt)
    assert clock.now() == dt


def test_fixed_clock_advance():
    dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clock = FixedClock(dt)
    clock.advance(days=1)
    assert clock.now().day == 2


def test_install_clock_replaces_global():
    dt = datetime(2020, 6, 15, tzinfo=timezone.utc)
    old = install_clock(FixedClock(dt))
    try:
        assert now() == dt
    finally:
        install_clock(old)
