"""
LightGBM LambdaMART cross-sectional ranking model.
Groups instruments by date; ranks within each date.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from leantrade.core.types import Forecast, ForecastQuantiles, InstrumentId


class LGBMRankModel:
    id: str = "lgbm_rank_v1"

    def __init__(
        self,
        horizon: str,
        feature_set_version: str,
        n_estimators: int = 300,
        random_state: int = 42,
        assumed_return_spread: float = 0.05,
    ) -> None:
        self.horizon = horizon
        self.feature_set_version = feature_set_version
        self._n_estimators = n_estimators
        self._random_state = random_state
        # A ranker's raw score is an unbounded, unitless relevance margin —
        # it is NOT a return and must never be reported as `expected_return`
        # directly (it previously was, producing nonsense like "+15% / -69%"
        # expected returns for a ranker that only knows relative ORDER).
        # `assumed_return_spread` is the believed top-to-bottom return
        # dispersion within the universe at this horizon, used only to turn
        # the cross-sectional rank into a plausible-looking pseudo-return
        # for consumers (allocators, display) that expect one. The model's
        # actual output is `rank` (a percentile in [0, 1]); treat
        # `expected_return` here as illustrative, not a calibrated forecast.
        self._assumed_return_spread = assumed_return_spread
        self._model = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        dates: np.ndarray | pl.Series | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Fit the LambdaMART ranker with one LightGBM "query group" per date.

        ``dates``: per-row date/timestamp array (same length and row order as
        X/y) used to build the groups. Without it there is no way to know
        which rows belong to the same cross-section, so this raises rather
        than silently training one giant group — LightGBM's NDCG evaluation
        caps a single query group at 10,000 rows and previously failed with
        "Number of rows ... exceeds upper limit of 10000 for a query" on any
        dataset bigger than that, while also ranking instruments globally
        across dates instead of within each date.
        """
        import lightgbm as lgb

        if dates is None:
            raise ValueError(
                "LGBMRankModel.fit() requires `dates` (one timestamp per row) "
                "to build per-date LambdaMART groups."
            )

        self._feature_names = list(X.columns)
        X_np = X.to_numpy(allow_copy=True).astype(np.float64)
        y_np = y.to_numpy(allow_copy=True).astype(np.float64)
        dates_arr = np.asarray(
            dates.to_numpy() if isinstance(dates, pl.Series) else dates
        )
        if len(dates_arr) != len(y_np):
            raise ValueError(
                f"dates has {len(dates_arr)} rows but X/y have {len(y_np)}"
            )

        # Treat +/-inf as missing, then impute by column median (LightGBM
        # handles NaN natively, but not inf).
        X_np[np.isinf(X_np)] = np.nan
        label_mask = ~np.isnan(y_np)
        X_np, y_np, dates_arr = X_np[label_mask], y_np[label_mask], dates_arr[label_mask]
        if len(X_np) < 30:
            raise ValueError(f"Too few training samples: {len(X_np)}")

        self._medians = np.where(
            np.isnan(np.nanmedian(X_np, axis=0)), 0.0, np.nanmedian(X_np, axis=0)
        )
        for j in range(X_np.shape[1]):
            X_np[np.isnan(X_np[:, j]), j] = self._medians[j]
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

        # Sort by date so each date's rows are contiguous — LightGBM requires
        # `group` sizes to describe contiguous row blocks in input order.
        order = np.argsort(dates_arr, kind="stable")
        X_np, y_np, dates_sorted = X_np[order], y_np[order], dates_arr[order]

        _, group_sizes = np.unique(dates_sorted, return_counts=True)

        # Relevance label: rank WITHIN each date's group (0 = worst that day),
        # scaled to [0, 30] since LightGBM's default NDCG label_gain table
        # only covers 32 relevance levels. Ranking globally across dates (the
        # previous behaviour) mixed returns from different market regimes
        # into one ordering, which is not what a per-date ranker should learn.
        y_rank = np.zeros(len(y_np), dtype=np.int32)
        start = 0
        for size in group_sizes:
            segment = y_np[start:start + size]
            local_rank = np.argsort(np.argsort(segment))
            if size > 1:
                scaled = np.round(local_rank / (size - 1) * 30).astype(np.int32)
            else:
                scaled = np.zeros(size, dtype=np.int32)
            y_rank[start:start + size] = scaled
            start += size

        self._model = lgb.LGBMRanker(
            n_estimators=self._n_estimators,
            random_state=self._random_state,
            verbosity=-1,
            n_jobs=-1,
        )
        self._model.fit(
            X_np, y_rank, group=group_sizes, feature_name=list(self._feature_names)
        )

    def predict_distribution(
        self,
        X: pl.DataFrame,
        instruments: list[InstrumentId],
        as_of: datetime,
    ) -> list[Forecast]:
        if self._model is None:
            raise RuntimeError("Model not fitted")

        X_np = X.to_numpy(allow_copy=True).astype(np.float64)
        X_np[np.isinf(X_np)] = np.nan
        if hasattr(self, "_medians") and self._medians is not None:
            for j in range(X_np.shape[1]):
                X_np[np.isnan(X_np[:, j]), j] = self._medians[j]
        raw = self._model.predict(X_np)
        lo, hi = raw.min(), raw.max()
        norm = (raw - lo) / (hi - lo + 1e-9)   # cross-sectional percentile, [0, 1]
        # Pseudo-return: centred at 0, spanning +/- half the assumed spread
        # at the extremes of this cross-section. This is NOT the raw ranker
        # score (see __init__) — it exists so downstream code that expects
        # a return-shaped number gets something in a sane range.
        half_spread = self._assumed_return_spread / 2.0
        pseudo_return = (norm - 0.5) * 2.0 * half_spread

        return [
            Forecast(
                instrument_id=instruments[i],
                as_of=as_of,
                horizon=self.horizon,
                expected_return=float(pseudo_return[i]),
                p_positive=float(np.clip(norm[i], 0.01, 0.99)),
                volatility=0.2,
                quantiles=ForecastQuantiles(
                    q05=float(pseudo_return[i] - 1.645 * half_spread),
                    q25=float(pseudo_return[i] - 0.674 * half_spread),
                    q50=float(pseudo_return[i]),
                    q75=float(pseudo_return[i] + 0.674 * half_spread),
                    q95=float(pseudo_return[i] + 1.645 * half_spread),
                ),
                rank=float(norm[i]),
                model_id=self.id,
                data_snapshot_id="",
            )
            for i in range(min(len(raw), len(instruments)))
        ]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model, "feature_names": self._feature_names,
                "horizon": self.horizon, "feature_set_version": self.feature_set_version,
                "medians": getattr(self, "_medians", None),
                "assumed_return_spread": self._assumed_return_spread,
            }, f)

    @classmethod
    def load(cls, path: str) -> "LGBMRankModel":
        with open(path, "rb") as f:
            s = pickle.load(f)
        obj = cls(s["horizon"], s["feature_set_version"])
        obj._model = s["model"]
        obj._feature_names = s["feature_names"]
        obj._medians = s.get("medians", None)
        obj._assumed_return_spread = s.get("assumed_return_spread", 0.05)
        return obj
