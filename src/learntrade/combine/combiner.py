"""
Meta-combiner — blends quant forecasts with LLM signal views.

Approaches (ablation-ordered; each kept only if it beats the prior):
1. Static weights:        combined = quant_w × quant + llm_w × llm
2. Rolling IC weighting:  weight each signal by its recent IC
3. Regime-conditioned:    separate weights in bull/bear/volatile regimes

The combiner NEVER sets portfolio weights — it only adjusts signal rankings.
The allocator remains responsible for translating rankings to weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from leantrade.core.types import Forecast, ForecastQuantiles, InstrumentId

logger = logging.getLogger(__name__)

CombinerMethod = Literal["static", "rolling_ic", "regime"]


@dataclass
class CombinerConfig:
    method: CombinerMethod = "rolling_ic"
    quant_weight: float = 0.7      # static-weight baseline
    llm_weight: float = 0.3
    ic_lookback: int = 60          # periods for rolling IC estimation
    min_ic_weight: float = 0.1    # floor so a bad signal isn't silenced entirely
    regime_lookback: int = 20      # bars for regime detection


class MetaCombiner:
    """
    Combines quant forecasts with optional LLM signal scores.

    Usage:
        combiner = MetaCombiner(config)
        updated = combiner.combine(quant_forecasts, llm_views)

    llm_views: {instrument_id_str: score} where positive = bullish
    """

    def __init__(self, config: CombinerConfig | None = None) -> None:
        self._cfg = config or CombinerConfig()
        self._ic_history: dict[str, list[float]] = {}  # signal_name → [ic values]

    def combine(
        self,
        forecasts: list[Forecast],
        llm_views: dict[str, float] | None = None,
    ) -> list[Forecast]:
        """
        Return updated Forecasts with combined expected_return and rank.
        Original quantiles and volatility are preserved from the quant model.
        """
        if not forecasts:
            return []
        if not llm_views:
            return forecasts  # no LLM views — pass through unchanged

        method = self._cfg.method
        if method == "static":
            weights = self._static_weights()
        elif method == "rolling_ic":
            weights = self._rolling_ic_weights()
        else:
            weights = self._regime_weights(forecasts)

        quant_w, llm_w = weights
        updated: list[Forecast] = []
        for fc in forecasts:
            iid_str = str(fc.instrument_id)
            llm_score = llm_views.get(iid_str, 0.0)

            combined = quant_w * fc.expected_return + llm_w * llm_score
            # Adjust p_positive towards 0.5 when quant and LLM disagree
            if (fc.expected_return > 0) != (llm_score > 0) and abs(llm_score) > 0.001:
                p_pos = fc.p_positive * 0.9 + 0.5 * 0.1  # pull toward 0.5
            else:
                p_pos = fc.p_positive

            updated.append(Forecast(
                instrument_id=fc.instrument_id,
                as_of=fc.as_of,
                horizon=fc.horizon,
                expected_return=combined,
                p_positive=float(np.clip(p_pos, 0.01, 0.99)),
                volatility=fc.volatility,
                quantiles=fc.quantiles,
                rank=fc.rank,
                model_id=f"{fc.model_id}+meta_{method}",
                data_snapshot_id=fc.data_snapshot_id,
                pit_level=fc.pit_level,
            ))

        # Re-rank cross-sectionally
        scores = np.array([f.expected_return for f in updated])
        ranks = scores.argsort().argsort() / max(len(scores) - 1, 1)
        for fc, rank in zip(updated, ranks):
            object.__setattr__(fc, "rank", float(rank)) if hasattr(fc, "__dataclass_fields__") else None
            # Pydantic: create a new object with updated rank
        updated = [
            Forecast(
                instrument_id=f.instrument_id, as_of=f.as_of, horizon=f.horizon,
                expected_return=f.expected_return, p_positive=f.p_positive,
                volatility=f.volatility, quantiles=f.quantiles,
                rank=float(r), model_id=f.model_id, data_snapshot_id=f.data_snapshot_id,
                pit_level=f.pit_level,
            )
            for f, r in zip(updated, ranks)
        ]
        return updated

    def record_ic(self, signal_name: str, ic: float) -> None:
        """Call after each period to update rolling IC history."""
        self._ic_history.setdefault(signal_name, []).append(ic)
        # Keep only recent history
        if len(self._ic_history[signal_name]) > self._cfg.ic_lookback * 2:
            self._ic_history[signal_name] = self._ic_history[signal_name][-self._cfg.ic_lookback:]

    # ── Weight computation ────────────────────────────────────────────────

    def _static_weights(self) -> tuple[float, float]:
        return self._cfg.quant_weight, self._cfg.llm_weight

    def _rolling_ic_weights(self) -> tuple[float, float]:
        quant_ic = np.mean(self._ic_history.get("quant", [0.05])) or 0.05
        llm_ic = np.mean(self._ic_history.get("llm", [0.02])) or 0.02

        # Clip at min_ic_weight to avoid silencing a signal entirely
        quant_ic = max(abs(quant_ic), self._cfg.min_ic_weight)
        llm_ic = max(abs(llm_ic), self._cfg.min_ic_weight)

        total = quant_ic + llm_ic
        return quant_ic / total, llm_ic / total

    def _regime_weights(self, forecasts: list[Forecast]) -> tuple[float, float]:
        """Simple volatility-based regime: high vol → trust quant more."""
        avg_vol = np.mean([f.volatility for f in forecasts]) if forecasts else 0.2
        if avg_vol > 0.30:    # high vol regime
            return 0.85, 0.15
        elif avg_vol > 0.15:  # normal regime
            return 0.70, 0.30
        else:                  # low vol regime
            return 0.65, 0.35
