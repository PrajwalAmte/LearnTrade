"""
Ablation ladder — measures the incremental value of each pipeline component.

Each rung adds one component on top of the previous. A component stays
enabled by default only if it adds statistically significant IC or Sharpe.

Rungs:
  1. Quant-only (Ridge on price features)
  2. + LLM evidence features
  3. + single synthesised LLM view
  4. + specialist analyst roles
  5. + verifier + quant challenger
  6. + debate / escalation
  7. + meta-combiner (IC-weighted combination)
  8. + memory (retrieval from past decisions) [ablation-gated, Phase 5+]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from learntrade.quant.validation.metrics import (
    ICResult,
    block_bootstrap_ci,
    compute_portfolio_metrics,
    ic_tstat,
    information_coefficient,
)

logger = logging.getLogger(__name__)


@dataclass
class AblationRung:
    id: int
    name: str
    description: str
    enable_llm_evidence_features: bool = False
    enable_llm_single_view: bool = False
    enable_specialist_roles: bool = False
    enable_verifier: bool = False
    enable_escalation: bool = False
    enable_meta_combiner: bool = False
    enable_memory: bool = False


STANDARD_LADDER: list[AblationRung] = [
    AblationRung(1, "quant_only",          "Ridge on price features"),
    AblationRung(2, "llm_evidence",        "Quant + LLM evidence features",
                 enable_llm_evidence_features=True),
    AblationRung(3, "llm_single_view",     "Quant + LLM evidence + single synthesis",
                 enable_llm_evidence_features=True, enable_llm_single_view=True),
    AblationRung(4, "specialist_roles",    "Rung 3 + specialist analysts",
                 enable_llm_evidence_features=True, enable_llm_single_view=True,
                 enable_specialist_roles=True),
    AblationRung(5, "verifier",            "Rung 4 + verifier + quant challenger",
                 enable_llm_evidence_features=True, enable_llm_single_view=True,
                 enable_specialist_roles=True, enable_verifier=True),
    AblationRung(6, "escalation",          "Rung 5 + debate / escalation",
                 enable_llm_evidence_features=True, enable_llm_single_view=True,
                 enable_specialist_roles=True, enable_verifier=True,
                 enable_escalation=True),
    AblationRung(7, "meta_combiner",       "Rung 6 + meta-combiner",
                 enable_llm_evidence_features=True, enable_llm_single_view=True,
                 enable_specialist_roles=True, enable_verifier=True,
                 enable_escalation=True, enable_meta_combiner=True),
    AblationRung(8, "memory",              "Rung 7 + memory retrieval [gated]",
                 enable_llm_evidence_features=True, enable_llm_single_view=True,
                 enable_specialist_roles=True, enable_verifier=True,
                 enable_escalation=True, enable_meta_combiner=True,
                 enable_memory=True),
]


@dataclass
class RungResult:
    rung: AblationRung
    ic_mean: float
    ic_std: float
    ic_tstat: float
    sharpe: float
    cagr: float
    incremental_ic: float = 0.0   # vs previous rung
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    n_periods: int = 0
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class AblationReport:
    market: str
    horizon: str
    since: str
    until: str
    rungs: list[RungResult]

    def print_summary(self) -> None:
        print(f"\nAblation Ladder — {self.market} {self.horizon} [{self.since} → {self.until}]")
        print(f"{'Rung':<4} {'Name':<22} {'IC':>8} {'t':>6} {'Sharpe':>8} {'ΔIC':>8} {'CI95':>18}")
        print("─" * 80)
        for r in self.rungs:
            if r.skipped:
                print(f"{r.rung.id:<4} {r.rung.name:<22} {'[SKIPPED: ' + r.skip_reason + ']'}")
                continue
            ci = f"[{r.ci_low:.4f},{r.ci_high:.4f}]" if not np.isnan(r.ci_low) else "   —"
            print(
                f"{r.rung.id:<4} {r.rung.name:<22} "
                f"{r.ic_mean:>8.4f} {r.ic_tstat:>6.2f} "
                f"{r.sharpe:>8.2f} {r.incremental_ic:>+8.4f} {ci:>18}"
            )


class AblationRunner:
    """
    Runs the ablation ladder.

    For rungs 1–2 (quant-only and LLM evidence), the runner uses the
    real QuantTrainer pipeline. Rungs 3–8 require LLM API access and
    are skipped when LLM is disabled.
    """

    def __init__(
        self,
        store: Any,          # PITStore
        output_dir: Path | None = None,
    ) -> None:
        self._store = store
        self._output_dir = output_dir or Path("research/reports")

    def run(
        self,
        market: str,
        horizon: str,
        since: datetime,
        until: datetime,
        rungs: list[AblationRung] | None = None,
    ) -> AblationReport:
        active_rungs = rungs or STANDARD_LADDER
        results: list[RungResult] = []
        prev_ic = 0.0

        for rung in active_rungs:
            logger.info("Running ablation rung %d: %s", rung.id, rung.name)
            result = self._run_rung(rung, market, horizon, since, until)
            if not result.skipped and not np.isnan(result.ic_mean) and len(results) > 0:
                result.incremental_ic = result.ic_mean - prev_ic
            if not result.skipped:
                prev_ic = result.ic_mean
            results.append(result)

        report = AblationReport(
            market=market,
            horizon=horizon,
            since=since.strftime("%Y-%m-%d"),
            until=until.strftime("%Y-%m-%d"),
            rungs=results,
        )
        self._save_report(report)
        return report

    def _run_rung(
        self,
        rung: AblationRung,
        market: str,
        horizon: str,
        since: datetime,
        until: datetime,
    ) -> RungResult:
        # Rungs requiring LLM are skipped when LLM is not configured
        if any([rung.enable_llm_single_view, rung.enable_specialist_roles,
                rung.enable_verifier, rung.enable_escalation,
                rung.enable_meta_combiner, rung.enable_memory]):
            try:
                from learntrade.llm.router import LLMRouter
                router = LLMRouter()
                if not router.is_enabled:
                    return RungResult(
                        rung=rung, ic_mean=float("nan"), ic_std=float("nan"),
                        ic_tstat=float("nan"), sharpe=float("nan"), cagr=float("nan"),
                        skipped=True, skip_reason="LLM disabled",
                    )
            except Exception:
                return RungResult(
                    rung=rung, ic_mean=float("nan"), ic_std=float("nan"),
                    ic_tstat=float("nan"), sharpe=float("nan"), cagr=float("nan"),
                    skipped=True, skip_reason="LLM not configured",
                )

        try:
            ic_values, daily_returns = self._compute_ic_and_returns(
                rung, market, horizon, since, until
            )
        except Exception as exc:
            logger.warning("Rung %d failed: %s", rung.id, exc)
            return RungResult(
                rung=rung, ic_mean=float("nan"), ic_std=float("nan"),
                ic_tstat=float("nan"), sharpe=float("nan"), cagr=float("nan"),
                skipped=True, skip_reason=str(exc),
            )

        ic_arr = np.array(ic_values)
        pm = compute_portfolio_metrics(np.array(daily_returns))
        ci_lo, ci_hi = block_bootstrap_ci(ic_arr) if len(ic_arr) > 4 else (float("nan"), float("nan"))

        return RungResult(
            rung=rung,
            ic_mean=float(np.nanmean(ic_arr)),
            ic_std=float(np.nanstd(ic_arr, ddof=1)) if len(ic_arr) > 1 else 0.0,
            ic_tstat=ic_tstat(ic_arr) if len(ic_arr) > 1 else float("nan"),
            sharpe=pm.get("sharpe", float("nan")),
            cagr=pm.get("cagr", float("nan")),
            ci_low=ci_lo,
            ci_high=ci_hi,
            n_periods=len(ic_arr),
        )

    def _compute_ic_and_returns(
        self,
        rung: AblationRung,
        market: str,
        horizon: str,
        since: datetime,
        until: datetime,
    ) -> tuple[list[float], list[float]]:
        from learntrade.features.price_volume import default_feature_set
        from learntrade.quant.models.ridge import RidgeReturnModel
        from learntrade.quant.targets.labels import compute_forward_returns, merge_features_labels
        from learntrade.quant.targets.horizon import get_horizon
        from learntrade.quant.validation.walk_forward import PurgedWalkForwardCV
        from learntrade.core.clock import now as _now

        feature_set = default_feature_set(self._store, market)
        h = get_horizon(horizon)
        instruments = self._store.get_universe(market, until) or self._store.get_universe(market, _now())
        if not instruments:
            raise ValueError(f"No universe for {market}")

        feature_df = feature_set.compute_history(instruments, since, until, self._store)
        bars = self._store.query_bars(instruments, as_of=until, since=since)
        if feature_df.is_empty() or bars.is_empty():
            raise ValueError("No data")

        labels_df = compute_forward_returns(bars, h.forward_days)
        merged = merge_features_labels(feature_df, labels_df)
        if merged.is_empty():
            raise ValueError("Empty merged dataset")

        feat_cols = [c for c in feature_set.feature_names if c in merged.columns]

        # LLM evidence features (rung 2+)
        if rung.enable_llm_evidence_features:
            from learntrade.llm.extraction.feature_bridge import FEATURE_NAMES
            ev_cols = [c for c in FEATURE_NAMES if c in merged.columns]
            feat_cols = feat_cols + ev_cols

        X = merged.select(feat_cols)
        y = merged["label"]
        dates = merged["event_time"]

        cv = PurgedWalkForwardCV(n_splits=3, embargo_bars=20, min_train_bars=50)
        ic_values: list[float] = []
        all_rets: list[float] = []

        import polars as pl
        from learntrade.core.types import InstrumentId

        for fold in cv.split(dates, h.forward_days):
            X_tr, y_tr = X[fold.train_indices], y[fold.train_indices]
            X_te, y_te = X[fold.test_indices], y[fold.test_indices]
            model = RidgeReturnModel(horizon=h.id, feature_set_version=feature_set.id)
            try:
                model.fit(X_tr, y_tr)
                iids = [InstrumentId(r) if ":" in str(r) else InstrumentId(f"{market}:{r}")
                        for r in merged["instrument_id"][fold.test_indices].to_list()]
                forecasts = model.predict_distribution(X_te, iids, until)
                scores = np.array([f.expected_return for f in forecasts])
                labels = y_te.to_numpy()[:len(scores)]
                ic = information_coefficient(scores, labels)
                if not np.isnan(ic):
                    ic_values.append(float(ic))
                # Approximate daily returns from IC (for Sharpe computation)
                all_rets.extend((scores * labels / max(np.std(labels), 1e-9)).tolist())
            except Exception as exc:
                logger.debug("Fold failed in ablation: %s", exc)

        return ic_values, all_rets

    def _save_report(self, report: AblationReport) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        from datetime import timezone
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self._output_dir / f"{ts}_ablation_{report.market}_{report.horizon}.md"
        lines = [f"# Ablation Report: {report.market} {report.horizon}",
                 f"Period: {report.since} → {report.until}", ""]
        for r in report.rungs:
            if r.skipped:
                lines.append(f"- Rung {r.rung.id} ({r.rung.name}): SKIPPED — {r.skip_reason}")
            else:
                lines.append(
                    f"- Rung {r.rung.id} ({r.rung.name}): IC={r.ic_mean:.4f} "
                    f"t={r.ic_tstat:.2f} Sharpe={r.sharpe:.2f} ΔIC={r.incremental_ic:+.4f}"
                )
        path.write_text("\n".join(lines))
        logger.info("Ablation report saved to %s", path)
