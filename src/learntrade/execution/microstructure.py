"""
Market microstructure utilities — lot rounding, tick rounding,
settlement cycles, and exchange calendar lookup.
"""

from __future__ import annotations

import math
from datetime import date, datetime

from leantrade.markets.schema import MarketPack, TickBand


def get_tick_size(price: float, bands: list[TickBand]) -> float:
    """Return the tick size for a given price from the pack's tick bands."""
    for band in sorted(bands, key=lambda b: b.below):
        if price < band.below:
            return band.tick
    return bands[-1].tick if bands else 0.01


def round_to_tick(price: float, bands: list[TickBand]) -> float:
    """Round price to the nearest valid tick (round half-up)."""
    tick = get_tick_size(price, bands)
    if tick <= 0:
        return price
    return round(price / tick) * tick


def round_to_lot(quantity: float, lot_size: int) -> int:
    """Round quantity DOWN to the nearest whole lot."""
    if lot_size <= 0:
        return max(0, int(quantity))
    return int(quantity // lot_size) * lot_size


def parse_settlement_days(settlement: str) -> int:
    """Parse "T+2" → 2, "T+0" → 0, "T+1" → 1."""
    try:
        return int(settlement.split("+")[-1])
    except (ValueError, IndexError):
        return 2  # conservative default


def get_calendar(pack: MarketPack):
    """
    Return an exchange_calendars calendar for the given Market Pack.
    Returns None gracefully if exchange_calendars doesn't know the calendar.
    """
    try:
        import exchange_calendars as xcals
        cal_name = pack.calendar
        if ":" in cal_name:
            cal_name = cal_name.split(":")[1]
        return xcals.get_calendar(cal_name)
    except Exception:
        return None


def is_trading_day(dt: date | datetime, pack: MarketPack) -> bool:
    """Return True if dt is a regular trading day for the market."""
    cal = get_calendar(pack)
    if cal is None:
        d = dt.date() if isinstance(dt, datetime) else dt
        return d.weekday() < 5  # weekday fallback
    d = dt.date() if isinstance(dt, datetime) else dt
    try:
        return bool(cal.is_session(d))
    except Exception:
        return d.weekday() < 5


def add_settlement_days(trade_date: date, pack: MarketPack) -> date:
    """Return the settlement date for a trade made on trade_date."""
    n = parse_settlement_days(pack.settlement)
    cal = get_calendar(pack)
    d = trade_date
    added = 0
    while added < n:
        import datetime as _dt
        d = d + _dt.timedelta(days=1)
        if cal is None:
            if d.weekday() < 5:
                added += 1
        else:
            try:
                if cal.is_session(d):
                    added += 1
            except Exception:
                if d.weekday() < 5:
                    added += 1
    return d


def participation_limit_qty(
    desired_qty: float,
    adv: float,
    max_participation: float = 0.10,
) -> float:
    """Cap order size at max_participation × ADV."""
    if adv <= 0:
        return desired_qty
    return min(desired_qty, adv * max_participation)
