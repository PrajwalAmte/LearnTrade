"""
Calibration quality metrics: Brier score, ROC-AUC, calibration buckets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationReport:
    brier_score: float
    roc_auc: float
    directional_accuracy: float
    balanced_accuracy: float
    n_samples: int
    calibration_buckets: list[tuple[float, float, int]]


def compute_calibration_report(
    p_positive: np.ndarray,
    labels: np.ndarray,
    n_buckets: int = 10,
) -> CalibrationReport:
    """
    Compute calibration metrics.
    p_positive: calibrated probability [0, 1]
    labels:     continuous return labels (positive = good)
    """
    from sklearn.metrics import brier_score_loss, roc_auc_score, balanced_accuracy_score

    binary = (labels > 0).astype(int)
    brier = float(brier_score_loss(binary, p_positive))

    try:
        auc = float(roc_auc_score(binary, p_positive))
    except Exception:
        auc = float("nan")

    preds_binary = (p_positive > 0.5).astype(int)
    dir_acc = float(np.mean(preds_binary == binary))

    try:
        bal_acc = float(balanced_accuracy_score(binary, preds_binary))
    except Exception:
        bal_acc = float("nan")

    buckets: list[tuple[float, float, int]] = []
    edges = np.linspace(0, 1, n_buckets + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p_positive >= lo) & (p_positive < hi)
        if mask.sum() > 0:
            buckets.append((
                float(p_positive[mask].mean()),
                float(binary[mask].mean()),
                int(mask.sum()),
            ))

    return CalibrationReport(
        brier_score=brier,
        roc_auc=auc,
        directional_accuracy=dir_acc,
        balanced_accuracy=bal_acc,
        n_samples=len(labels),
        calibration_buckets=buckets,
    )
