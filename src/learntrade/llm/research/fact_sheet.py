"""
Fact sheet builder — deterministic Python summary of one instrument's state.

LLMs receive ONLY this sheet, never raw data. This prevents:
- Token waste from large DataFrames
- Accidental arithmetic by the LLM (numbers come pre-computed)
- Leakage of future data (the sheet is PIT-safe)

The sheet is intentionally compact: ~200–400 tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from learntrade.core.types import Forecast


@dataclass
class FactSheet:
    instrument_id: str
    as_of: str                          # ISO date string, UTC
    horizon: str
    price: float
    returns: dict[str, float]           # {"1d": 0.01, "5d": 0.03, ...}
    volatility_annual: float
    volume_surprise: float | None       # log(vol / ewma_vol); positive = elevated
    high_52w_ratio: float | None        # price / 52-week high [0,1]
    cross_sectional_rank: float | None  # percentile within universe [0,1]
    recent_events: list[str]            # ["earnings_2024-01-14", "guidance_raised"]
    evidence_summary: dict[str, Any]    # keys from FEATURE_NAMES
    quant_forecast: dict[str, Any] | None  # {expected_return, p_positive, ...}
    universe_size: int = 0

    def to_text(self) -> str:
        """Render a compact text summary for LLM consumption."""
        lines = [
            f"{self.instrument_id} — {self.as_of} — horizon: {self.horizon}",
            "─" * 50,
            f"Price: ${self.price:.2f}",
        ]

        ret_parts = [f"{w}: {v:+.1%}" for w, v in self.returns.items()]
        lines.append("Returns: " + "  ".join(ret_parts))
        lines.append(f"Vol (ann): {self.volatility_annual:.1%}")

        if self.volume_surprise is not None:
            lines.append(f"Volume: {self.volume_surprise:+.2f}x log-surprise vs avg")
        if self.high_52w_ratio is not None:
            lines.append(f"52w-high ratio: {self.high_52w_ratio:.2f}")
        if self.cross_sectional_rank is not None:
            lines.append(
                f"Cross-sectional rank: {self.cross_sectional_rank:.0%} "
                f"(of {self.universe_size} instruments)"
            )

        if self.quant_forecast:
            qf = self.quant_forecast
            lines.extend([
                "─" * 50,
                f"Quant forecast ({self.horizon}):",
                f"  Expected excess return: {qf.get('expected_return', 0):+.2%}",
                f"  P(positive): {qf.get('p_positive', 0.5):.1%}",
                f"  Forecast vol: {qf.get('volatility', 0):.1%}",
            ])
            if qf.get("rank") is not None:
                lines.append(f"  Rank: {qf['rank']:.0%}")

        if self.recent_events:
            lines.extend(["─" * 50, "Recent events:"])
            for ev in self.recent_events[:8]:
                lines.append(f"  • {ev}")

        ev = self.evidence_summary
        if ev and any(v != 0 for v in ev.values() if isinstance(v, (int, float))):
            lines.extend(["─" * 50, "LLM evidence signals:"])
            if ev.get("guidance_change_score", 0) != 0:
                label = "raised" if ev["guidance_change_score"] > 0 else "lowered"
                lines.append(f"  • Guidance: {label}")
            if ev.get("surprise_score", 0) != 0:
                label = "positive" if ev["surprise_score"] > 0 else "negative"
                lines.append(f"  • Earnings surprise: {label}")
            if ev.get("management_tone_score", 0) != 0:
                label = "positive" if ev["management_tone_score"] > 0 else "cautious/negative"
                lines.append(f"  • Management tone: {label}")
            if ev.get("n_risk_flags", 0) > 0:
                lines.append(f"  • Risk flags: {int(ev['n_risk_flags'])}")

        return "\n".join(lines)


def build_fact_sheet(
    instrument_id: str,
    as_of: datetime,
    horizon: str,
    features: dict[str, Any],
    forecast: "Forecast | None" = None,
    evidence: dict[str, Any] | None = None,
    universe_size: int = 0,
    recent_events: list[str] | None = None,
) -> FactSheet:
    """
    Build a FactSheet from pre-computed features and optional forecast.

    Args:
        features:  dict from PriceVolumeFeatureSet (keyed by feature name)
        forecast:  Forecast object from the quant engine
        evidence:  dict from evidence_to_features()
    """
    returns = {
        w: float(features.get(f"ret_{w}d", 0.0) or 0.0)
        for w in [1, 5, 20, 60]
        if f"ret_{w}d" in features
    }

    qf = None
    if forecast is not None:
        qf = {
            "expected_return": float(forecast.expected_return),
            "p_positive": float(forecast.p_positive),
            "volatility": float(forecast.volatility),
            "rank": float(forecast.rank) if forecast.rank is not None else None,
        }

    return FactSheet(
        instrument_id=instrument_id,
        as_of=as_of.strftime("%Y-%m-%d"),
        horizon=horizon,
        price=float(features.get("close", features.get("adj_close", 0.0)) or 0.0),
        returns=returns,
        volatility_annual=float(features.get("ewma_vol_20d", 0.0) or 0.0),
        volume_surprise=features.get("vol_surprise_20d"),
        high_52w_ratio=features.get("high_52w_ratio"),
        cross_sectional_rank=float(forecast.rank) if forecast and forecast.rank else None,
        recent_events=recent_events or [],
        evidence_summary=evidence or {},
        quant_forecast=qf,
        universe_size=universe_size,
    )
