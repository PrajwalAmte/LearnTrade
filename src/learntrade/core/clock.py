"""
Clocks — injectable time source so backtests and tests can pin "now".

Production code always calls ``now()``; it never imports ``datetime.utcnow``
directly. The backtest engine installs a ``FixedClock`` before each step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone


class Clock(ABC):
    @abstractmethod
    def now(self) -> datetime:
        ...


class WallClock(Clock):
    """Real-time clock; the default in production."""

    def now(self) -> datetime:
        return datetime.now(tz=timezone.utc)


class FixedClock(Clock):
    """Frozen at a specific UTC datetime; used by backtests and tests."""

    def __init__(self, dt: datetime) -> None:
        if dt.tzinfo is None:
            raise ValueError("FixedClock requires a timezone-aware datetime")
        self._dt = dt

    def now(self) -> datetime:
        return self._dt

    def advance(self, **kwargs: float) -> None:
        from datetime import timedelta
        self._dt = self._dt + timedelta(**kwargs)


_clock: Clock = WallClock()


def now() -> datetime:
    """Return the current UTC time according to the active clock."""
    return _clock.now()


def install_clock(clock: Clock) -> Clock:
    """Replace the global clock; returns the previous one."""
    global _clock
    previous = _clock
    _clock = clock
    return previous
