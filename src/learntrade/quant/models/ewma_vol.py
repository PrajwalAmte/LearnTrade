"""
EWMA volatility model — position sizing and forecast uncertainty estimation.
No fitting required; parameters are fixed.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from learntrade.core.types import InstrumentId


class EWMAVolModel:
    id: str = "ewma_vol_v1"

    def __init__(
        self,
        horizon: str = "vol",
        feature_set_version: str = "price_volume_v1",
        span: int = 20,
    ) -> None:
        self.horizon = horizon
        self.feature_set_version = feature_set_version
        self.span = span
        self._decay = 1.0 - 2.0 / (span + 1)

    def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> None:
        pass  # EWMA is parameter-free

    def predict_vol(self, returns: np.ndarray, annualise: bool = True) -> float:
        """EWMA variance from recent returns, sqrt for volatility."""
        r = np.asarray(returns, dtype=float)
        r = r[~np.isnan(r)]
        if len(r) < 2:
            return float("nan")
        n = len(r)
        weights = self._decay ** np.arange(n - 1, -1, -1)
        weights /= weights.sum()
        ewma_var = float(np.dot(weights, r ** 2))
        vol = np.sqrt(ewma_var)
        return float(vol * np.sqrt(252) if annualise else vol)

    def predict_vols_cross_section(
        self,
        bars: pl.DataFrame,
        instruments: list[InstrumentId],
        as_of: datetime,
    ) -> dict[str, float]:
        """Return EWMA annualised vol for each instrument."""
        result: dict[str, float] = {}
        bars_sorted = bars.sort(["instrument_id", "event_time"])
        close_col = "adj_close" if "adj_close" in bars_sorted.columns else "close"
        for iid in instruments:
            iid_bars = bars_sorted.filter(pl.col("instrument_id") == str(iid))
            closes = iid_bars[close_col].cast(pl.Float64, strict=False).drop_nulls()
            if len(closes) < 5:
                continue
            returns = np.diff(np.log(closes.to_numpy()))
            result[str(iid)] = self.predict_vol(returns[-self.span * 5:])
        return result

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"span": self.span, "horizon": self.horizon,
                         "feature_set_version": self.feature_set_version}, f)

    @classmethod
    def load(cls, path: str) -> "EWMAVolModel":
        with open(path, "rb") as f:
            s = pickle.load(f)
        return cls(s["horizon"], s["feature_set_version"], s["span"])
