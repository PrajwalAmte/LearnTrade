"""learntrade.quant.targets"""

from learntrade.quant.targets.horizon import BUILTIN_HORIZONS, HorizonSpec, get_horizon
from learntrade.quant.targets.labels import compute_forward_returns, merge_features_labels

__all__ = [
    "BUILTIN_HORIZONS",
    "HorizonSpec",
    "compute_forward_returns",
    "get_horizon",
    "merge_features_labels",
]
