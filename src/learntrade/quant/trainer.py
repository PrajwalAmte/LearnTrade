"""
Walk-forward training pipeline.

Fetches bars → computes features → computes labels → purged walk-forward CV →
calibrates OOF probabilities → retrains on full data → saves artifacts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from leantrade.core.types import InstrumentId
from leantrade.data.store.pit import PITStore
from leantrade.features.price_volume import PriceVolumeFeatureSet
from leantrade.markets.loader import get_pack
from leantrade.quant.calibration.isotonic import IsotonicCalibrator
from leantrade.quant.calibration.metrics import CalibrationReport, compute_calibration_report
from leantrade.quant.models.lgbm_rank import LGBMRankModel
from leantrade.quant.models.ridge import RidgeReturnModel
from leantrade.quant.targets.horizon import get_horizon
from leantrade.quant.targets.labels import (
    compute_benchmark_returns,
    compute_forward_returns,
    merge_features_labels,
)
from leantrade.quant.validation.metrics import ICResult, block_bootstrap_ci, ic_by_date, ic_tstat
from leantrade.quant.validation.walk_forward import PurgedWalkForwardCV

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    ic: float
    n_test: int


@dataclass
class TrainingResult:
    horizon: str
    market: str
    model_id: str
    feature_set_version: str
    fold_results: list[FoldResult]
    calibration: CalibrationReport | None
    ic_result: ICResult
    model_path: str | None = None
    calibrator_path: str | None = None
    benchmark_id: str | None = None   # None means labels are RAW returns, not excess

    def summary(self) -> str:
        ic = self.ic_result
        bm = f" vs {self.benchmark_id}" if self.benchmark_id else " (raw return, no benchmark)"
        return (
            f"{self.market} {self.horizon}{bm} | model={self.model_id} | "
            f"IC={ic.ic_mean:.4f} RankIC={ic.rank_ic_mean:.4f} "
            f"t={ic.ic_tstat:.2f} hit={ic.hit_rate:.0%} n={ic.n_periods}"
        )


class QuantTrainer:
    """
    Walk-forward training for a single horizon and model type.
    """

    def __init__(
        self,
        store: PITStore,
        horizon_id: str,
        model_class: type = RidgeReturnModel,
        n_splits: int = 5,
        embargo_bars: int = 20,
        min_train_bars: int = 252,
        artifacts_dir: Path | None = None,
    ) -> None:
        self._store = store
        self._horizon = get_horizon(horizon_id)
        self._model_class = model_class
        self._cv = PurgedWalkForwardCV(n_splits=n_splits, embargo_bars=embargo_bars,
                                       min_train_bars=min_train_bars)
        self._artifacts_dir = artifacts_dir or Path("data/models")
        self._features = PriceVolumeFeatureSet()

    def train(
        self,
        market: str,
        since: datetime,
        until: datetime,
    ) -> TrainingResult:
        logger.info(
            "Training %s for %s [%s → %s]",
            self._horizon.id, market, since.date(), until.date(),
        )

        # Universe: check at until first; fall back to now() (supports historical training)
        from leantrade.core.clock import now as _now
        instruments = self._store.get_universe(market, until)
        if not instruments:
            instruments = self._store.get_universe(market, _now())
        if not instruments:
            raise ValueError(f"No universe for {market}. Import instruments first.")
        logger.info("Universe: %d instruments", len(instruments))

        feature_df = self._features.compute_history(instruments, since, until, self._store)
        if feature_df.is_empty():
            raise ValueError("No feature data. Run data pull first.")

        bars = self._store.query_bars(instruments, as_of=until, since=since)
        if bars.is_empty():
            raise ValueError("No bars in PIT store. Run data pull first.")

        # Label the target as excess return over the market's benchmark when
        # one is configured, instead of always using the instrument's raw
        # return. A model that is "right" only because the whole market
        # moved that way isn't adding anything — excess return is what the
        # quant engine is actually supposed to predict (see README "Quant
        # engine" / HorizonSpec.benchmark).
        benchmark_id = self._horizon.benchmark
        if benchmark_id is None:
            try:
                benchmark_id = get_pack(market).benchmark
            except Exception:
                benchmark_id = None

        benchmark_col = None
        if benchmark_id:
            bench_bars = self._store.query_bars(
                [InstrumentId(benchmark_id)], as_of=until, since=since
            )
            bench_fwd = compute_benchmark_returns(
                bench_bars, benchmark_id, self._horizon.forward_days
            )
            if not bench_fwd.is_empty():
                bars = bars.join(
                    bench_fwd.rename({"benchmark_return": "benchmark_fwd_return"}),
                    on="event_time", how="left",
                )
                benchmark_col = "benchmark_fwd_return"
            else:
                logger.warning(
                    "Benchmark %s has no bars in the PIT store — training on "
                    "RAW returns, not excess return. Run `leantrade data pull` "
                    "with the benchmark in the universe to fix this.",
                    benchmark_id,
                )
                benchmark_id = None

        labels_df = compute_forward_returns(
            bars, self._horizon.forward_days, benchmark_col=benchmark_col
        )
        merged = merge_features_labels(feature_df, labels_df)
        if merged.is_empty():
            raise ValueError("No rows after merging features and labels.")

        logger.info("Dataset: %d rows, features=%s", len(merged), self._features.feature_names[:3])

        feat_cols = [c for c in self._features.feature_names if c in merged.columns]
        X = merged.select(feat_cols)
        y = merged["label"]
        dates = merged["event_time"]

        fold_results: list[FoldResult] = []
        all_oof_scores: list[float] = []
        all_oof_labels: list[float] = []
        all_daily_ic: list[float] = []
        all_daily_rank_ic: list[float] = []
        is_ranker = issubclass(self._model_class, LGBMRankModel)

        for fold in self._cv.split(dates, self._horizon.forward_days):
            X_tr, y_tr = X[fold.train_indices], y[fold.train_indices]
            X_te, y_te = X[fold.test_indices], y[fold.test_indices]
            dates_tr = dates[fold.train_indices]
            dates_te_arr = dates[fold.test_indices].to_numpy()

            model = self._model_class(
                horizon=self._horizon.id,
                feature_set_version=self._features.id,
            )
            try:
                if is_ranker:
                    model.fit(X_tr, y_tr, dates=dates_tr)
                else:
                    model.fit(X_tr, y_tr)
                iids_te = [
                    InstrumentId(r) if ":" in str(r) else InstrumentId(f"{market}:{r}")
                    for r in merged["instrument_id"][fold.test_indices].to_list()
                ]
                forecasts = model.predict_distribution(X_te, iids_te, until)
                oof_s = np.array([f.expected_return for f in forecasts])
                oof_l = y_te.to_numpy()[:len(oof_s)]
                oof_dates = dates_te_arr[:len(oof_s)]

                # Per-date IC/Rank-IC, not one correlation pooled across every
                # (date, instrument) row in the fold — see ic_by_date().
                per_date = ic_by_date(oof_dates, oof_s, oof_l)
                fold_ic = float(np.mean(per_date["ic"])) if len(per_date["ic"]) else float("nan")
                fold_rank_ic = (
                    float(np.mean(per_date["rank_ic"])) if len(per_date["rank_ic"]) else float("nan")
                )
                fold_results.append(FoldResult(
                    fold_id=fold.fold_id,
                    train_start=fold.train_start, train_end=fold.train_end,
                    test_start=fold.test_start, test_end=fold.test_end,
                    ic=fold_rank_ic if not np.isnan(fold_rank_ic) else 0.0,
                    n_test=len(fold.test_indices),
                ))
                all_oof_scores.extend(oof_s.tolist())
                all_oof_labels.extend(oof_l.tolist())
                all_daily_ic.extend(per_date["ic"].tolist())
                all_daily_rank_ic.extend(per_date["rank_ic"].tolist())
                logger.info(
                    "  Fold %d: IC=%.4f RankIC=%.4f n_test=%d n_days=%d",
                    fold.fold_id, fold_ic, fold_rank_ic, len(fold.test_indices),
                    len(per_date["rank_ic"]),
                )
            except Exception as exc:
                logger.warning("  Fold %d failed: %s", fold.fold_id, exc)

        if not fold_results:
            raise ValueError("All CV folds failed.")
        if not all_daily_rank_ic:
            raise ValueError(
                "No fold produced a usable per-date IC (every test date had "
                "fewer than 5 instruments or all-NaN forecasts)."
            )

        ic_daily_arr = np.array(all_daily_ic)
        rank_ic_daily_arr = np.array(all_daily_rank_ic)
        ic_result = ICResult(
            ic_mean=float(np.mean(ic_daily_arr)) if len(ic_daily_arr) else float("nan"),
            ic_std=float(np.std(rank_ic_daily_arr, ddof=1)) if len(rank_ic_daily_arr) > 1 else 0.0,
            ic_tstat=ic_tstat(rank_ic_daily_arr) if len(rank_ic_daily_arr) > 1 else 0.0,
            rank_ic_mean=float(np.mean(rank_ic_daily_arr)),
            hit_rate=float(np.mean(rank_ic_daily_arr > 0)),
            n_periods=len(rank_ic_daily_arr),
        )
        ic_ci = (
            block_bootstrap_ci(rank_ic_daily_arr) if len(rank_ic_daily_arr) > 2
            else (float("nan"), float("nan"))
        )
        logger.info("IC summary: %s (95%% CI on Rank IC: %s)", ic_result, ic_ci)

        calibration = None
        calibrator = None
        if all_oof_scores:
            oof_s_arr = np.array(all_oof_scores)
            oof_l_arr = np.array(all_oof_labels)
            calibrator = IsotonicCalibrator()
            calibrator.fit(oof_s_arr, oof_l_arr)
            cal_p = calibrator.transform(oof_s_arr)
            calibration = compute_calibration_report(cal_p, oof_l_arr)

        # Retrain on full data
        final_model = self._model_class(
            horizon=self._horizon.id, feature_set_version=self._features.id
        )
        if is_ranker:
            final_model.fit(X, y, dates=dates)
        else:
            final_model.fit(X, y)
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(
            self._artifacts_dir / f"{market}_{self._horizon.id}_{final_model.id}.pkl"
        )
        final_model.save(model_path)

        cal_path = None
        if calibrator:
            cal_path = str(self._artifacts_dir / f"{market}_{self._horizon.id}_calibrator.pkl")
            calibrator.save(cal_path)

        return TrainingResult(
            horizon=self._horizon.id,
            market=market,
            model_id=final_model.id,
            feature_set_version=self._features.id,
            fold_results=fold_results,
            calibration=calibration,
            ic_result=ic_result,
            model_path=model_path,
            calibrator_path=cal_path,
            benchmark_id=benchmark_id,
        )
