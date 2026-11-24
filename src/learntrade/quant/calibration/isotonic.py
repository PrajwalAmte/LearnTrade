"""
Isotonic regression probability calibrator.
Fit on out-of-fold predictions to avoid overfitting.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class IsotonicCalibrator:
    """Maps raw model scores → calibrated P(positive return)."""

    def __init__(self) -> None:
        self._ir = None

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> None:
        from sklearn.isotonic import IsotonicRegression

        binary = (labels > 0).astype(float)
        self._ir = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self._ir.fit(raw_scores, binary)

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        if self._ir is None:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        return np.clip(self._ir.transform(raw_scores), 0.0, 1.0)

    def save(self, path: str) -> None:
        from learntrade.quant.models import _compat
        _compat.dump(path, {"isotonic": self._ir})

    @classmethod
    def load(cls, path: str) -> "IsotonicCalibrator":
        from learntrade.quant.models import _compat
        cal = cls()
        s = _compat.load(path)
        cal._ir = s["isotonic"] if isinstance(s, dict) and "isotonic" in s else s
        return cal
