"""
Evaluation sweep: does ANY reasonable configuration show a real, repeatable edge?

Every combination of stock group x horizon x feature set x model is trained and tested the same
way (purged walk-forward cross-validation plus an untouched final period), and judged by ONE
rule fixed in advance:

    PASS  =  cross-validation rank IC >= 0.02
         AND overlap-adjusted t-statistic >= 2.0
         AND rank IC on the untouched final period > 0

Testing many configurations makes some pass by luck. With a t >= 2 bar, about 2.5% of
configurations with no real edge pass anyway, so the report states how many passes to expect
by chance. A pass is a lead to confirm (more history, another period), not proof.

Model files are written to a temporary folder: a sweep never replaces the models in use.
"""

from __future__ import annotations

import itertools
import logging
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MIN_RANK_IC, MIN_T = 0.02, 2.0
CHANCE_PASS_RATE = 0.025          # one-sided t >= 2 with no real edge


@dataclass
class SweepRow:
    group: str
    horizon: str
    features: str
    model: str
    n_stocks: int = 0
    cv_rank_ic: float | None = None
    cv_ci: tuple[float, float] | None = None
    t_adj: float | None = None
    n_folds: int | None = None
    holdout_rank_ic: float | None = None
    holdout_ci: tuple[float, float] | None = None
    fundamentals_coverage: float | None = None
    passed: bool = False
    error: str | None = None
    seconds: float = 0.0


@dataclass
class SweepResult:
    market: str
    since: str
    rule: str = (f"CV rank IC >= {MIN_RANK_IC} and overlap-adjusted t >= {MIN_T} and "
                 "untouched-period rank IC > 0")
    rows: list[SweepRow] = field(default_factory=list)

    @property
    def tested(self) -> int:
        return sum(1 for r in self.rows if r.error is None)

    @property
    def passed(self) -> list[SweepRow]:
        return [r for r in self.rows if r.passed]

    def to_dict(self) -> dict[str, Any]:
        return {"market": self.market, "since": self.since, "rule": self.rule,
                "tested": self.tested, "passed": len(self.passed),
                "expected_by_chance": round(self.tested * CHANCE_PASS_RATE, 2),
                "rows": [asdict(r) for r in self.rows]}


def judge(cv_ic: float | None, t: float | None, hold_ic: float | None) -> bool:
    return (cv_ic is not None and t is not None and hold_ic is not None
            and cv_ic >= MIN_RANK_IC and t >= MIN_T and hold_ic > 0)


def run_sweep(
    store: Any, market: str, since: datetime, until: datetime,
    groups: dict[str, set[str] | None], horizons: list[str], feature_sets: list[str],
    models: dict[str, type], holdout_days: int = 120,
) -> SweepResult:
    from learntrade.quant.trainer import QuantTrainer

    res = SweepResult(market=market, since=since.date().isoformat())
    with tempfile.TemporaryDirectory() as tmp:
        for group, horizon, feats in itertools.product(groups, horizons, feature_sets):
            dataset = None
            for model_name, cls in models.items():
                row = SweepRow(group=group, horizon=horizon, features=feats, model=model_name)
                t0 = time.monotonic()
                try:
                    tr = QuantTrainer(store, horizon, model_class=cls, holdout_days=holdout_days,
                                      artifacts_dir=Path(tmp), features=feats, only=groups[group])
                    if dataset is None:
                        dataset = tr.build_dataset(market, since, until)
                    rep = tr.train(market, since, until, dataset=dataset).report
                    cv, hold = rep.get("cv") or {}, rep.get("holdout") or {}
                    row.n_stocks = int(rep.get("n_instruments") or 0)
                    row.cv_rank_ic = cv.get("rank_ic_mean")
                    row.cv_ci = tuple(cv["rank_ic_ci"]) if cv.get("rank_ic_ci") else None
                    row.t_adj = cv.get("tstat_overlap_adjusted")
                    row.n_folds = cv.get("n_folds")
                    row.holdout_rank_ic = hold.get("rank_ic_mean")
                    row.holdout_ci = tuple(hold["rank_ic_ci"]) if hold.get("rank_ic_ci") else None
                    row.fundamentals_coverage = (rep.get("fundamentals") or {}).get("coverage")
                    row.passed = judge(row.cv_rank_ic, row.t_adj, row.holdout_rank_ic)
                except Exception as exc:
                    from learntrade.core.errors import friendly_error
                    row.error = friendly_error(exc)
                    logger.warning("%s/%s/%s/%s failed: %s", group, horizon, feats, model_name, exc)
                    if dataset is None:
                        row.seconds = time.monotonic() - t0
                        res.rows.append(row)
                        break        # the data for this combination could not be built
                row.seconds = time.monotonic() - t0
                res.rows.append(row)
                logger.info("%s", row)
    return res


def _f(v: float | None, signed: bool = True) -> str:
    return "n/a" if v is None else (f"{v:+.3f}" if signed else f"{v:.2f}")


def to_markdown(res: SweepResult) -> str:
    d = res.to_dict()
    lines = [
        f"# Evaluation sweep: {res.market}",
        "",
        f"Data from {res.since}. Pass rule (fixed in advance): **{res.rule}**.",
        "",
        f"**{d['passed']} of {d['tested']} configurations passed.** With no real edge anywhere, about "
        f"**{d['expected_by_chance']}** would pass by luck alone, so a pass is a lead to confirm, "
        "not proof.",
        "",
        "| Group | Horizon | Features | Model | Stocks | CV rank IC (95% range) | t (adj.) | Folds | "
        "Untouched period | Pass |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res.rows:
        if r.error:
            lines.append(f"| {r.group} | {r.horizon} | {r.features} | {r.model} | | "
                         f"could not run: {r.error[:80]} | | | | |")
            continue
        ci = f" ({r.cv_ci[0]:+.3f} to {r.cv_ci[1]:+.3f})" if r.cv_ci else ""
        lines.append(
            f"| {r.group} | {r.horizon} | {r.features} | {r.model} | {r.n_stocks} | "
            f"{_f(r.cv_rank_ic)}{ci} | {_f(r.t_adj, False)} | {r.n_folds or ''} | "
            f"{_f(r.holdout_rank_ic)} | {'yes' if r.passed else 'no'} |")
    lines += ["", "Rank IC: how well the order of the list matched the order of later returns "
              "(0 = no better than chance, 1 = perfect). t (adj.) accounts for overlapping return "
              "windows."]
    return "\n".join(lines)
