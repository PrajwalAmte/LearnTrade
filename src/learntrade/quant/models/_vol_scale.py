"""Per-instrument volatility scaling shared by the return models.

The models estimate ONE residual standard deviation for the whole training set.
Used as-is, every stock would get the same volatility and the same forecast
range (a calm utility and a speculative small-cap alike). We scale that
residual by each stock's own recent volatility relative to the typical stock
seen in training, so riskier stocks get wider ranges.
"""

from __future__ import annotations

import numpy as np

VOL_FEATURE = "ewma_vol_20d"
_MIN_SCALE, _MAX_SCALE = 0.3, 3.0


def vol_column(X: np.ndarray, feature_names: list[str]) -> np.ndarray | None:
    if VOL_FEATURE not in feature_names:
        return None
    return X[:, feature_names.index(VOL_FEATURE)]


def vol_scales(
    X: np.ndarray, feature_names: list[str], medians: np.ndarray | None
) -> np.ndarray:
    """Multiplier per row: own volatility / training-median volatility, clipped."""
    v = vol_column(X, feature_names)
    if v is None or medians is None:
        return np.ones(len(X))
    ref = float(medians[feature_names.index(VOL_FEATURE)])
    if not np.isfinite(ref) or ref <= 0:
        return np.ones(len(X))
    ratio = np.where(np.isfinite(v) & (v > 0), v / ref, 1.0)
    return np.clip(ratio, _MIN_SCALE, _MAX_SCALE)
