"""
Ridge regression cross-sectional return model.
Uses RidgeCV (leave-one-out) to select regularisation automatically.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from learntrade.core.types import Forecast, ForecastQuantiles, InstrumentId
from learntrade.quant.models._vol_scale import vol_scales


class RidgeReturnModel:
    id: str = "ridge_v1"

    def __init__(
        self,
        horizon: str,
        feature_set_version: str,
        alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
    ) -> None:
        self.horizon = horizon
        self.feature_set_version = feature_set_version
        self._alphas = alphas
        self._scaler = None
        self._model = None
        self._feature_names: list[str] = []
        self._residual_std: float = 0.2

    def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> None:
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler

        self._feature_names = list(X.columns)
        X_np = X.to_numpy(allow_copy=True).astype(np.float64)
        y_np = y.to_numpy(allow_copy=True).astype(np.float64)

        # Treat +/-inf (e.g. from a zero-volume bar slipping through a
        # feature set) the same as missing, so a single bad row can't poison
        # the median or blow up StandardScaler with a near-infinite column.
        X_np[np.isinf(X_np)] = np.nan

        # Impute NaN features: use column median, fall back to 0 for all-NaN cols
        self._medians = np.where(
            np.isnan(np.nanmedian(X_np, axis=0)), 0.0, np.nanmedian(X_np, axis=0)
        )
        for j in range(X_np.shape[1]):
            X_np[np.isnan(X_np[:, j]), j] = self._medians[j]
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)  # safety net

        mask = ~np.isnan(y_np)
        X_np, y_np = X_np[mask], y_np[mask]
        if len(X_np) < 30:
            raise ValueError(f"Too few training samples: {len(X_np)}")

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_np)
        self._model = RidgeCV(alphas=self._alphas)
        self._model.fit(X_scaled, y_np)

        resid = y_np - self._model.predict(X_scaled)
        self._residual_std = float(np.std(resid)) or 0.02

    def predict_distribution(
        self,
        X: pl.DataFrame,
        instruments: list[InstrumentId],
        as_of: datetime,
    ) -> list[Forecast]:
        if self._model is None or self._scaler is None:
            raise RuntimeError("Model not fitted")

        X_np = X.to_numpy(allow_copy=True).astype(np.float64)
        X_np[np.isinf(X_np)] = np.nan
        # Apply saved medians for inference-time imputation
        if hasattr(self, "_medians") and self._medians is not None:
            for j in range(X_np.shape[1]):
                X_np[np.isnan(X_np[:, j]), j] = self._medians[j]
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
        sc = vol_scales(X_np, self._feature_names, getattr(self, "_medians", None))
        X_scaled = self._scaler.transform(X_np)
        preds = self._model.predict(X_scaled)
        s0 = self._residual_std

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
                "scaler": self._scaler, "model": self._model,
                "feature_names": self._feature_names,
                "horizon": self.horizon,
                "feature_set_version": self.feature_set_version,
                "residual_std": self._residual_std,
                "medians": getattr(self, "_medians", None),
            }, f)

    @classmethod
    def load(cls, path: str) -> "RidgeReturnModel":
        with open(path, "rb") as f:
            s = pickle.load(f)
        obj = cls(s["horizon"], s["feature_set_version"])
        obj._scaler = s["scaler"]
        obj._model = s["model"]
        obj._feature_names = s["feature_names"]
        obj._residual_std = s.get("residual_std", 0.02)
        obj._medians = s.get("medians", None)
        return obj
