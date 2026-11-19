"""
Horizon target definitions.

Each horizon specifies:
- forward_days: number of trading days to look forward
- target_type: how the return is computed
- benchmark: instrument against which excess return is measured
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TargetType = Literal["excess_return", "raw_return", "cross_sectional_rank"]


@dataclass(frozen=True)
class HorizonSpec:
    id: str                     # e.g. "swing_5d"
    forward_days: int
    target_type: TargetType = "excess_return"
    benchmark: str | None = None   # instrument_id of the benchmark, or None for market average


BUILTIN_HORIZONS: dict[str, HorizonSpec] = {
    "intraday_1h":     HorizonSpec("intraday_1h",     forward_days=0),
    "swing_5d":        HorizonSpec("swing_5d",        forward_days=5),
    "swing_20d":       HorizonSpec("swing_20d",       forward_days=20),
    "positional_60d":  HorizonSpec("positional_60d",  forward_days=60),
    "positional_120d": HorizonSpec("positional_120d", forward_days=120),
}


def get_horizon(horizon_id: str) -> HorizonSpec:
    if horizon_id not in BUILTIN_HORIZONS:
        raise ValueError(
            f"Unknown horizon {horizon_id!r}. "
            f"Available: {list(BUILTIN_HORIZONS.keys())}"
        )
    return BUILTIN_HORIZONS[horizon_id]
