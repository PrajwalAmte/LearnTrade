"""LightGBM gradient-boosted return model."""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from leantrade.core.types import Forecast, ForecastQuantiles, InstrumentId
from leantrade.quant.models._vol_scale import vol_scales


class LGBMReturnModel:
    id: str = "lgbm_v1"

    def __init__(
        self,
        horizon: str,
        feature_set_version: str,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
    ) -> None:
        self.horizon = horizon
        self.feature_set_version = feature_set_version
        self._params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        self._model = None
        self._feature_names: list[str] = []
        self._pred_std: float = 0.02

    def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> None:
        import lightgbm as lgb

        self._feature_names = list(X.columns)
        X_np = X.to_numpy(allow_copy=True).astype(np.float64)
        y_np = y.to_numpy(allow_copy=True).astype(np.float64)

        # Drop rows with no usable label first (can't impute a label).
        label_mask = ~np.isnan(y_np)
        X_np, y_np = X_np[label_mask], y_np[label_mask]
        if len(X_np) < 30:
            raise ValueError(f"Too few training samples: {len(X_np)}")

        # Treat +/-inf as missing, then impute (not drop) missing features —
        # dropping any row with a single missing feature previously discarded
        # most of the training set and made the "median imputation" below a
        # no-op, since every remaining row was already NaN-free.
        X_np[np.isinf(X_np)] = np.nan
        self._medians = np.where(
            np.isnan(np.nanmedian(X_np, axis=0)), 0.0, np.nanmedian(X_np, axis=0)
        )
        for j in range(X_np.shape[1]):
            X_np[np.isnan(X_np[:, j]), j] = self._medians[j]
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

        self._model = lgb.LGBMRegressor(**self._params)
        self._model.fit(X_np, y_np, feature_name=self._feature_names)
        preds = self._model.predict(X_np)
        self._pred_std = float(np.std(preds - y_np)) or 0.02

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
        sc = vol_scales(X_np, self._feature_names, getattr(self, "_medians", None))
        preds = self._model.predict(X_np)
        s0 = self._pred_std
        return [
            Forecast(
                instrument_id=instruments[i],
                as_of=as_of,
                horizon=self.horizon,
                expected_return=float(p),
                p_positive=float(np.clip(0.5 + p / (2 * s0 + 1e-9) * 0.3, 0.01, 0.99)),
                volatility=float(s0 * sc[i] * np.sqrt(252 / 20)),
                quantiles=ForecastQuantiles(
                    q05=float(p - 1.645 * s0 * sc[i]), q25=float(p - 0.674 * s0 * sc[i]),
                    q50=float(p),
                    q75=float(p + 0.674 * s0 * sc[i]), q95=float(p + 1.645 * s0 * sc[i]),
                ),
                model_id=self.id,
                data_snapshot_id="",
            )
            for i, p in enumerate(preds)
            if not np.isnan(p) and i < len(instruments)
        ]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model, "feature_names": self._feature_names,
                "horizon": self.horizon, "feature_set_version": self.feature_set_version,
                "params": self._params, "pred_std": self._pred_std,
                "medians": getattr(self, "_medians", None),
            }, f)

    @classmethod
    def load(cls, path: str) -> "LGBMReturnModel":
        with open(path, "rb") as f:
            s = pickle.load(f)
        obj = cls(s["horizon"], s["feature_set_version"])
        obj._model = s["model"]
        obj._feature_names = s["feature_names"]
        obj._params = s["params"]
        obj._pred_std = s.get("pred_std", 0.02)
        obj._medians = s.get("medians", None)
        return obj
