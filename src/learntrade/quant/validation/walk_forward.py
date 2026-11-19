"""
Purged walk-forward cross-validation.

Based on López de Prado (2018), Chapter 7.
Purging removes training observations whose label period overlaps with the test set.
Embargo drops bars near the train/test boundary to prevent feature leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import polars as pl


@dataclass
class WalkForwardFold:
    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: str
    train_end: str
    test_start: str
    test_end: str


class PurgedWalkForwardCV:
    """
    Walk-forward CV with purging and embargo.

    Args:
        n_splits:        number of test folds
        embargo_bars:    bars to drop between train end and test start
        min_train_bars:  minimum observations required in the training set
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_bars: int = 20,
        min_train_bars: int = 252,
    ) -> None:
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars
        self.min_train_bars = min_train_bars

    def split(
        self,
        dates: pl.Series,
        horizon_bars: int = 20,
    ) -> Iterator[WalkForwardFold]:
        """
        Yield WalkForwardFold for each test period.

        dates:        event_time column from the merged features+labels DataFrame
        horizon_bars: forward bar horizon used for purging (labels overlap detection)
        """
        all_dates = dates.to_list()
        unique_sorted = sorted(set(str(d) for d in all_dates))
        n_dates = len(unique_sorted)

        test_size = max(n_dates // (self.n_splits + 1), 21)

        # Map date string → list of row indices in the original DataFrame
        date_to_rows: dict[str, list[int]] = {}
        for row_i, d in enumerate(all_dates):
            key = str(d)
            date_to_rows.setdefault(key, []).append(row_i)

        for fold_id in range(self.n_splits):
            test_start_date_idx = n_dates - (self.n_splits - fold_id) * test_size
            test_end_date_idx = test_start_date_idx + test_size

            # Purge: exclude training bars whose forward label window overlaps
            # the test set. Embargo: additionally drop bars in the
            # `embargo_bars` window right before the test set, independent of
            # the label horizon, as an extra margin against short-horizon
            # feature leakage (e.g. rolling features computed right at the
            # boundary). Take whichever cutoff is stricter (smaller) — for
            # short horizons the embargo is the binding constraint, and
            # previously it was computed but never applied.
            purge_cutoff = test_start_date_idx - horizon_bars - 1
            embargo_cutoff = test_start_date_idx - self.embargo_bars
            train_end_idx = min(purge_cutoff, embargo_cutoff)

            if train_end_idx < self.min_train_bars:
                continue
            if test_start_date_idx >= n_dates:
                continue

            train_dates = unique_sorted[:train_end_idx]
            test_dates = set(unique_sorted[test_start_date_idx:test_end_date_idx])

            if len(train_dates) < self.min_train_bars or not test_dates:
                continue

            train_indices = np.array([
                i for d in train_dates for i in date_to_rows.get(d, [])
            ])
            test_indices = np.array([
                i for d in test_dates for i in date_to_rows.get(d, [])
            ])

            if len(train_indices) == 0 or len(test_indices) == 0:
                continue

            yield WalkForwardFold(
                fold_id=fold_id,
                train_indices=np.sort(train_indices),
                test_indices=np.sort(test_indices),
                train_start=unique_sorted[0],
                train_end=unique_sorted[train_end_idx - 1],
                test_start=unique_sorted[test_start_date_idx],
                test_end=unique_sorted[min(test_end_date_idx - 1, n_dates - 1)],
            )
