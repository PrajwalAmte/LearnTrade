"""
Calibration tests — isotonic calibrator, calibration metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from learntrade.quant.calibration.isotonic import IsotonicCalibrator
from learntrade.quant.calibration.metrics import compute_calibration_report


def _make_data(n: int = 500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores = rng.normal(0, 1, n)
    # True label is noisy but positively correlated with score
    labels = scores * 0.3 + rng.normal(0, 0.5, n)
    return scores, labels


def test_isotonic_calibrator_fit_transform():
    scores, labels = _make_data()
    cal = IsotonicCalibrator()
    cal.fit(scores, labels)
    p = cal.transform(scores)
    assert p.shape == scores.shape
    assert (p >= 0).all()
    assert (p <= 1).all()


def test_isotonic_calibrator_monotone():
    """Calibrated probabilities must be monotonically non-decreasing with score."""
    scores, labels = _make_data()
    cal = IsotonicCalibrator()
    cal.fit(scores, labels)
    sorted_scores = np.sort(scores)
    sorted_p = cal.transform(sorted_scores)
    # Non-decreasing
    assert (np.diff(sorted_p) >= -1e-9).all()


def test_isotonic_calibrator_raises_before_fit():
    cal = IsotonicCalibrator()
    with pytest.raises(RuntimeError):
        cal.transform(np.array([0.5]))


def test_isotonic_calibrator_save_load(tmp_path):
    scores, labels = _make_data()
    cal = IsotonicCalibrator()
    cal.fit(scores, labels)
    path = str(tmp_path / "cal.pkl")
    cal.save(path)
    loaded = IsotonicCalibrator.load(path)
    orig = cal.transform(scores)
    loaded_p = loaded.transform(scores)
    np.testing.assert_array_almost_equal(orig, loaded_p)


def test_calibration_report_brier_better_than_naive():
    """Calibrated model should have lower Brier score than predicting 0.5 always."""
    scores, labels = _make_data(n=1000)
    cal = IsotonicCalibrator()
    cal.fit(scores, labels)
    p = cal.transform(scores)
    report = compute_calibration_report(p, labels)
    # Naive predictor (always 0.5): Brier = 0.25
    assert report.brier_score < 0.26
    assert 0 < report.roc_auc <= 1.0
    assert 0 < report.directional_accuracy <= 1.0


def test_calibration_report_perfect_model():
    """Perfect binary classifier should have Brier=0 and AUC=1."""
    n = 100
    labels = np.concatenate([np.ones(n // 2), -np.ones(n // 2)])
    p = np.concatenate([np.ones(n // 2) * 0.99, np.ones(n // 2) * 0.01])
    report = compute_calibration_report(p, labels)
    assert report.brier_score < 0.01
    assert report.roc_auc > 0.99
