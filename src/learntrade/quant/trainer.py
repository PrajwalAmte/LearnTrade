"""
Walk-forward training pipeline.

Fetches bars → computes features → computes labels → purged walk-forward CV →
calibrates OOF probabilities → retrains on full data → saves artifacts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from learntrade.core.types import InstrumentId
from learntrade.data.store.pit import PITStore
from learntrade.features.price_volume import PriceVolumeFeatureSet, default_feature_set
from learntrade.markets.loader import get_pack
from learntrade.quant.calibration.isotonic import IsotonicCalibrator
from learntrade.quant.calibration.metrics import CalibrationReport, compute_calibration_report
from learntrade.quant.models.lgbm_rank import LGBMRankModel
from learntrade.quant.models.ridge import RidgeReturnModel
from learntrade.quant.targets.horizon import get_horizon
from learntrade.quant.targets.labels import (
    compute_benchmark_returns,
    compute_forward_returns,
    merge_features_labels,
)
from learntrade.quant.validation.metrics import (
    ICResult, block_bootstrap_ci, ic_by_date, ic_tstat, ic_tstat_overlap,
)
from learntrade.quant.validation.walk_forward import PurgedWalkForwardCV

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
    report: dict[str, Any] = field(default_factory=dict)
    report_path: str | None = None

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
        holdout_days: int = 0,
        features: str = "auto",
        only: set[str] | None = None,
    ) -> None:
        """``features``: "auto" (financials when stored), "price" or "price+fund".
        ``only``: restrict training to these symbols (e.g. one size group of the list)."""
        self._store = store
        self._horizon = get_horizon(horizon_id)
        self._model_class = model_class
        self._cv = PurgedWalkForwardCV(n_splits=n_splits, embargo_bars=embargo_bars,
                                       min_train_bars=min_train_bars)
        self._artifacts_dir = artifacts_dir or Path("data/models")
        self._holdout_days = holdout_days
        self._membership: dict[str, Any] = {}
        self._fundamentals_info: dict[str, Any] = {"used": False}
        self._feature_choice = features
        self._only = {s.split(":")[-1].upper() for s in only} if only else None
        self._features = default_feature_set(store)

    def _build_dataset(
        self, market: str, since: datetime, until: datetime
    ) -> tuple[pl.DataFrame, list[str], str | None, int]:
        """Features + labels for the whole period. Returns (merged, feat_cols, benchmark_id, n_instruments)."""
        # The feature set depends on what this market has stored (e.g. company financials).
        if self._feature_choice == "auto":
            self._features = default_feature_set(self._store, market)
        else:
            self._features = PriceVolumeFeatureSet(fundamentals=self._feature_choice == "price+fund")
        logger.info(
            "Training %s for %s [%s → %s]",
            self._horizon.id, market, since.date(), until.date(),
        )

        # Universe: check at until first; fall back to now() (supports historical training)
        from learntrade.core.clock import now as _now
        from learntrade.data.universe import membership_filter
        history = self._store.get_universe_history(market, until)
        if not history:
            history = self._store.get_universe_history(market, _now())
        instruments = self._store.get_universe_union(market, until) or \
            self._store.get_universe_union(market, _now())
        if self._only is not None:
            instruments = [i for i in instruments if str(i).split(":")[-1].upper() in self._only]
            history = [(at, [m for m in mem if m.split(":")[-1].upper() in self._only])
                       for at, mem in history]
        if not instruments:
            raise ValueError(f"No universe for {market}. Import instruments first.")
        logger.info("Universe: %d instruments across %d snapshot(s)", len(instruments), len(history))

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
                    "RAW returns, not excess return. Run `learntrade data pull` "
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

        # Survivorship: when membership history exists, keep each row only if the stock was in
        # the universe on that date (instead of treating today's list as the list for all time).
        merged, mem = membership_filter(merged, history, market)
        have_bars = {str(i).split(":")[-1].upper() for i in bars["instrument_id"].unique().to_list()}
        listed = {str(i).split(":")[-1].upper() for i in instruments}
        mem["coverage"] = len(listed & have_bars) / max(len(listed), 1)
        mem["missing_symbols"] = sorted(listed - have_bars)[:25]
        self._membership = mem
        self._fundamentals_info = {"used": False}
        if self._features.fundamentals and "earnings_yield" in merged.columns and len(merged):
            cov = 1.0 - merged["earnings_yield"].null_count() / len(merged)
            self._fundamentals_info = {"used": True, "coverage": round(float(cov), 3)}
        if merged.is_empty():
            raise ValueError("No rows left after applying membership dates.")

        logger.info("Dataset: %d rows, features=%s", len(merged), self._features.feature_names[:3])

        feat_cols = [c for c in self._features.feature_names if c in merged.columns]
        return merged, feat_cols, benchmark_id, len(instruments)

    def _split_holdout(self, full: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        """
        Reserve the most recent ``holdout_days`` trading dates as a final test set.

        Rows whose forward-return window reaches into the holdout are purged from the
        development set (a gap of ``forward_days`` trading dates), so nothing the model
        trains on or is tuned on overlaps the holdout outcomes.
        """
        fwd = self._horizon.forward_days
        uniq = full["event_time"].unique().sort().to_list()
        need = self._holdout_days + fwd + self._cv.min_train_bars + 60
        if len(uniq) < need:
            logger.warning(
                "Holdout skipped: need ~%d trading dates of history, have %d.", need, len(uniq)
            )
            return full, None
        hold_start = uniq[-self._holdout_days]
        dev_end = uniq[len(uniq) - self._holdout_days - 1 - fwd]
        dev = full.filter(pl.col("event_time") <= dev_end)
        hold = full.filter(pl.col("event_time") >= hold_start)
        logger.info(
            "Holdout: %d rows from %s (development ends %s, %d-day purge)",
            hold.height, str(hold_start)[:10], str(dev_end)[:10], fwd,
        )
        return dev, hold

    def build_dataset(self, market: str, since: datetime, until: datetime) -> dict[str, Any]:
        """Build features + labels once, to share across several models (see run_evaluation)."""
        merged, feat_cols, bm, n_inst = self._build_dataset(market, since, until)
        return {"merged": merged, "feat_cols": feat_cols, "benchmark_id": bm, "n_instruments": n_inst,
                "features": self._features, "membership": self._membership,
                "fundamentals_info": self._fundamentals_info}

    def holdout_signals(
        self, market: str, since: datetime, until: datetime, signal: str = "model",
        dataset: dict[str, Any] | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame, int]:
        """
        Out-of-sample signals for the reserved holdout dates.

        ``signal="model"``: fit on the development period ONLY, then score the holdout.
        ``signal="momentum"``: the 20-day-return baseline (no fitting at all).
        Returns (signals[date, instrument_id, signal], holdout_rows_with_labels, n_instruments).
        """
        if self._holdout_days <= 0:
            raise ValueError("holdout_days must be > 0 for an out-of-sample evaluation")
        if dataset is None:
            dataset = self.build_dataset(market, since, until)
        else:
            self._features, self._membership = dataset["features"], dataset["membership"]
        merged, feat_cols, n_inst = dataset["merged"], dataset["feat_cols"], dataset["n_instruments"]
        dev, hold = self._split_holdout(merged)
        if hold is None or hold.is_empty():
            raise ValueError("Not enough history to reserve a holdout. Pull more data first.")
        if signal == "momentum":
            col = "ret_20d" if "ret_20d" in hold.columns else feat_cols[0]
            scores = hold[col].fill_null(0.0).to_numpy()
        else:
            is_ranker = issubclass(self._model_class, LGBMRankModel)
            m = self._model_class(horizon=self._horizon.id, feature_set_version=self._features.id)
            if is_ranker:
                m.fit(dev.select(feat_cols), dev["label"], dates=dev["event_time"])
            else:
                m.fit(dev.select(feat_cols), dev["label"])
            iids = [
                InstrumentId(r) if ":" in str(r) else InstrumentId(f"{market}:{r}")
                for r in hold["instrument_id"].to_list()
            ]
            fcs = m.predict_distribution(hold.select(feat_cols), iids, until)
            if len(fcs) != hold.height:
                raise ValueError("Model dropped rows while scoring the holdout")
            scores = np.array([f.expected_return for f in fcs])
        sigs = pl.DataFrame({
            "date": hold["event_time"], "instrument_id": hold["instrument_id"], "signal": scores,
        }).drop_nulls()
        return sigs, hold, n_inst

    def train(
        self,
        market: str,
        since: datetime,
        until: datetime,
        dataset: dict[str, Any] | None = None,
    ) -> TrainingResult:
        if dataset is None:
            merged, feat_cols, benchmark_id, n_instruments = self._build_dataset(market, since, until)
        else:      # shared with other models (see backtest.sweep)
            self._features, self._membership = dataset["features"], dataset["membership"]
            self._fundamentals_info = dataset.get("fundamentals_info", {"used": False})
            merged, feat_cols = dataset["merged"], dataset["feat_cols"]
            benchmark_id, n_instruments = dataset["benchmark_id"], dataset["n_instruments"]
        full = merged
        holdout_df: pl.DataFrame | None = None
        if self._holdout_days > 0:
            merged, holdout_df = self._split_holdout(full)


        X = merged.select(feat_cols)
        y = merged["label"]
        dates = merged["event_time"]

        fold_results: list[FoldResult] = []
        all_oof_scores: list[float] = []
        all_oof_labels: list[float] = []
        all_daily_ic: list[float] = []
        all_daily_rank_ic: list[float] = []
        is_ranker = issubclass(self._model_class, LGBMRankModel)

        last_fold_error: Exception | None = None
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
                if len(forecasts) != len(fold.test_indices):
                    # Truncating would pair every later score with the wrong outcome.
                    raise ValueError(
                        f"model returned {len(forecasts)} forecasts for "
                        f"{len(fold.test_indices)} test rows")
                oof_s = np.array([f.expected_return for f in forecasts])
                oof_l = y_te.to_numpy()
                oof_dates = dates_te_arr

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
                last_fold_error = exc
                logger.warning("  Fold %d failed: %s", fold.fold_id, exc)

        if fold_results and len(fold_results) < self._cv.n_splits:
            logger.warning(
                "Only %d of %d cross-validation folds had enough history; the test statistics "
                "rest on fewer periods than configured.", len(fold_results), self._cv.n_splits)
        if not fold_results:
            from learntrade.core.errors import friendly_error
            why = f" {friendly_error(last_fold_error)}" if last_fold_error else ""
            raise ValueError(f"All CV folds failed.{why}")
        if not all_daily_rank_ic:
            raise ValueError(
                "No fold produced a usable per-date IC (every test date had "
                "fewer than 5 instruments or all-NaN forecasts)."
            )

        ic_daily_arr = np.array(all_daily_ic)
        rank_ic_daily_arr = np.array(all_daily_rank_ic)
        fwd_days = self._horizon.forward_days
        t_naive = ic_tstat(rank_ic_daily_arr) if len(rank_ic_daily_arr) > 1 else float("nan")
        t_adj = ic_tstat_overlap(rank_ic_daily_arr, fwd_days)
        ic_result = ICResult(
            ic_mean=float(np.mean(ic_daily_arr)) if len(ic_daily_arr) else float("nan"),
            ic_std=float(np.std(rank_ic_daily_arr, ddof=1)) if len(rank_ic_daily_arr) > 1 else 0.0,
            # Overlap-adjusted: consecutive daily ICs share most of their outcome window.
            ic_tstat=0.0 if np.isnan(t_adj) else t_adj,
            rank_ic_mean=float(np.mean(rank_ic_daily_arr)),
            hit_rate=float(np.mean(rank_ic_daily_arr > 0)),
            n_periods=len(rank_ic_daily_arr),
        )
        ic_ci = (
            block_bootstrap_ci(rank_ic_daily_arr, min_block=fwd_days) if len(rank_ic_daily_arr) > 2
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

        # Final untouched test: a model fit on the development period only, scored once
        # on the reserved most-recent dates. Never used to pick anything.
        holdout_metrics: dict[str, Any] | None = None
        if holdout_df is not None and holdout_df.height:
            holdout_metrics = self._score_holdout(
                X, y, dates, holdout_df, feat_cols, calibrator, market, until, is_ranker, fwd_days
            )

        # Retrain on ALL data (development + holdout) for live use
        final_model = self._model_class(
            horizon=self._horizon.id, feature_set_version=self._features.id
        )
        X_all = full.select(feat_cols)
        if is_ranker:
            final_model.fit(X_all, full["label"], dates=full["event_time"])
        else:
            final_model.fit(X_all, full["label"])
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(
            self._artifacts_dir / f"{market}_{self._horizon.id}_{final_model.id}.pkl"
        )
        final_model.save(model_path)

        cal_path = None
        if calibrator:
            cal_path = str(self._artifacts_dir / f"{market}_{self._horizon.id}_{final_model.id}_calibrator.pkl")
            calibrator.save(cal_path)

        report: dict[str, Any] = {
            "trained_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "market": market, "horizon": self._horizon.id,
            "model_id": final_model.id, "feature_set_version": self._features.id,
            "benchmark_id": benchmark_id,
            "label_type": "excess" if benchmark_id else "raw",
            "n_rows": int(full.height), "n_instruments": int(full["instrument_id"].n_unique()),
            "n_days": int(full["event_time"].n_unique()),
            "data_start": str(full["event_time"].min())[:10],
            "data_end": str(full["event_time"].max())[:10],
            "membership": self._membership,
            "fundamentals": self._fundamentals_info,
            "survivorship_note": None,   # derived from `membership` by reliability.assess_reliability
            "cv": {
                "rank_ic_mean": ic_result.rank_ic_mean,
                "rank_ic_ci": [float(ic_ci[0]), float(ic_ci[1])],
                "tstat_naive": None if np.isnan(t_naive) else float(t_naive),
                "tstat_overlap_adjusted": None if np.isnan(t_adj) else float(t_adj),
                "hit_rate": ic_result.hit_rate,
                "n_days": ic_result.n_periods, "n_folds": len(fold_results),
                "n_folds_requested": self._cv.n_splits,
            },
            "calibration": None if calibration is None else {
                "brier_score": calibration.brier_score, "roc_auc": calibration.roc_auc,
                "directional_accuracy": calibration.directional_accuracy,
                "n_samples": calibration.n_samples,
            },
            "holdout": holdout_metrics,
        }
        report_path = str(
            self._artifacts_dir / f"{market}_{self._horizon.id}_{final_model.id}_report.json"
        )
        Path(report_path).write_text(json.dumps(report, indent=2, default=str))

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
            report=report,
            report_path=report_path,
        )

    def _score_holdout(
        self, X: pl.DataFrame, y: pl.Series, dates: pl.Series, holdout_df: pl.DataFrame,
        feat_cols: list[str], calibrator: Any, market: str, until: datetime,
        is_ranker: bool, fwd_days: int,
    ) -> dict[str, Any] | None:
        try:
            m = self._model_class(
                horizon=self._horizon.id, feature_set_version=self._features.id
            )
            if is_ranker:
                m.fit(X, y, dates=dates)
            else:
                m.fit(X, y)
            iids = [
                InstrumentId(r) if ":" in str(r) else InstrumentId(f"{market}:{r}")
                for r in holdout_df["instrument_id"].to_list()
            ]
            fcs = m.predict_distribution(holdout_df.select(feat_cols), iids, until)
            if len(fcs) != holdout_df.height:
                logger.warning("Holdout: model returned %d of %d forecasts; skipped",
                               len(fcs), holdout_df.height)
                return None
            scores = np.array([f.expected_return for f in fcs])
            labels = holdout_df["label"].to_numpy()
            per_date = ic_by_date(holdout_df["event_time"].to_numpy(), scores, labels)
            r = per_date["rank_ic"]
            if len(r) < 3:
                return None
            lo, hi = block_bootstrap_ci(r, min_block=fwd_days)
            out: dict[str, Any] = {
                "rank_ic_mean": float(np.mean(r)), "rank_ic_ci": [float(lo), float(hi)],
                "hit_rate": float(np.mean(r > 0)), "n_days": int(len(r)),
                "tstat_overlap_adjusted": (
                    None if np.isnan(ic_tstat_overlap(r, fwd_days))
                    else float(ic_tstat_overlap(r, fwd_days))),
                "start": str(holdout_df["event_time"].min())[:10],
                "end": str(holdout_df["event_time"].max())[:10],
            }
            if calibrator is not None:
                cal = compute_calibration_report(calibrator.transform(scores), labels)
                out["brier_score"], out["roc_auc"] = cal.brier_score, cal.roc_auc
            logger.info("Holdout: rank IC %.4f over %d days (%s → %s)",
                        out["rank_ic_mean"], out["n_days"], out["start"], out["end"])
            return out
        except Exception as exc:
            logger.warning("Holdout scoring failed: %s", exc)
            return None
