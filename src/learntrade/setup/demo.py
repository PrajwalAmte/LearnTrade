"""
Demo data generator — creates synthetic OHLCV data for 50 instruments.

The demo is watermarked "demo" in source_id and is never confused with real data.
Instruments follow correlated GBM paths with realistic sector structure.
Used by: `learntrade serve` on first launch (shows ideas before data pull finishes).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from learntrade.core.types import InstrumentId, Observation
from learntrade.data.store.pit import PITStore
from learntrade.data.universe import UniverseManager

logger = logging.getLogger(__name__)

UTC = timezone.utc

# 5 synthetic sectors × 10 instruments each
_SECTORS = {
    "tech":     [f"SYM{i:02d}" for i in range(10)],
    "finance":  [f"SYM{i:02d}" for i in range(10, 20)],
    "health":   [f"SYM{i:02d}" for i in range(20, 30)],
    "energy":   [f"SYM{i:02d}" for i in range(30, 40)],
    "consumer": [f"SYM{i:02d}" for i in range(40, 50)],
}

_SECTOR_DRIFT = {"tech": 0.0003, "finance": 0.0002, "health": 0.00025,
                 "energy": 0.0001, "consumer": 0.00015}

_SECTOR_VOL = {"tech": 0.022, "finance": 0.018, "health": 0.020,
               "energy": 0.025, "consumer": 0.016}


def generate_demo_data(
    store: PITStore,
    mic: str = "XNYS",
    n_days: int = 252,
    start_date: datetime | None = None,
    seed: int = 42,
) -> int:
    """
    Generate synthetic OHLCV data and register instruments in the PIT store.

    Returns the number of observations ingested.
    """
    rng = np.random.default_rng(seed)
    # 5 trading days per 7 calendar days, so the history ends ~today (not in the future).
    start = start_date or (datetime.now(tz=UTC) - timedelta(days=int(n_days * 7 / 5) + 2))

    all_symbols: list[str] = [sym for syms in _SECTORS.values() for sym in syms]
    n = len(all_symbols)

    # Build correlated price paths (sector correlation = 0.5, cross-sector = 0.1)
    corr_matrix = _build_correlation_matrix()
    chol = np.linalg.cholesky(corr_matrix + 1e-9 * np.eye(n))

    # Starting prices: uniform $20–$200
    prices = rng.uniform(20.0, 200.0, n)

    obs_list: list[Observation] = []
    trading_day = 0
    current_date = start

    while trading_day < n_days:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue

        # Correlated daily shocks
        z = rng.standard_normal(n)
        correlated_z = chol @ z

        for i, symbol in enumerate(all_symbols):
            sector = _symbol_sector(symbol)
            drift = _SECTOR_DRIFT[sector]
            vol = _SECTOR_VOL[sector]

            ret = drift + vol * correlated_z[i]
            prices[i] *= math.exp(ret)
            price = prices[i]

            volume = float(rng.lognormal(mean=12.5, sigma=0.5))
            high = price * (1 + abs(rng.normal(0, 0.004)))
            low = price * (1 - abs(rng.normal(0, 0.004)))

            obs_list.append(Observation(
                instrument_id=InstrumentId(f"{mic}:{symbol}"),
                source_id="demo",
                kind="bar",
                event_time=current_date,
                available_at=current_date,
                payload={
                    "open": round(price * rng.uniform(0.998, 1.002), 4),
                    "high": round(high, 4),
                    "low": round(low, 4),
                    "close": round(price, 4),
                    "adj_close": round(price, 4),
                    "volume": round(volume),
                    "demo": True,  # watermark
                },
                pit_level="none",
            ))

        trading_day += 1
        current_date += timedelta(days=1)

    total = store.ingest(obs_list)

    # Register universe
    mgr = UniverseManager(store)
    mgr.import_explicit_list(mic, all_symbols, available_at=start)

    logger.info("Demo data: %d observations for %d instruments (%d trading days)",
                total, n, n_days)
    return total


def _symbol_sector(symbol: str) -> str:
    for sector, symbols in _SECTORS.items():
        if symbol in symbols:
            return sector
    return "consumer"


def _build_correlation_matrix() -> np.ndarray:
    all_symbols = [sym for syms in _SECTORS.values() for sym in syms]
    n = len(all_symbols)
    corr = np.eye(n)
    sector_of = {sym: _symbol_sector(sym) for sym in all_symbols}

    for i in range(n):
        for j in range(i + 1, n):
            if sector_of[all_symbols[i]] == sector_of[all_symbols[j]]:
                corr[i, j] = corr[j, i] = 0.5   # same sector
            else:
                corr[i, j] = corr[j, i] = 0.1   # cross-sector

    return corr
