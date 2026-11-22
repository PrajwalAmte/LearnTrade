"""
Plain-language read-out of how well a trained model has done in testing.

Tone: say what the evidence shows and what would improve it, without alarm. The numbers
are the same honest ones as before (overlap-corrected significance, untouched final test);
only the framing is constructive.
"""

from __future__ import annotations

from datetime import date
from typing import Any

MIN_UNIVERSE = 20
MIN_MEANINGFUL_IC = 0.01   # below this the average ranking skill is indistinguishable from none
MIN_DAYS = 500


def assess_reliability(report: dict[str, Any], last_bar: str | None = None,
                       today: date | None = None) -> dict[str, Any]:
    """Return {level, headline, reasons[], facts{}} for the UI/CLI.

    level: "no_evidence" | "unproven" | "modest"
    """
    today = today or date.today()
    cv = report.get("cv") or {}
    hold = report.get("holdout")
    reasons: list[str] = []

    mean = cv.get("rank_ic_mean")
    lo, hi = (cv.get("rank_ic_ci") or [None, None])
    # Both checks must agree: the range clears zero AND the overlap-adjusted t-stat is
    # conventionally significant. Either alone is too easy to hit by luck.
    t_adj = cv.get("tstat_overlap_adjusted")
    clears_zero = lo is not None and lo > 0 and (t_adj is None or t_adj >= 2.0)

    if mean is None:
        level, headline = "no_evidence", "Not measured yet. Update data to test the model."
    elif mean <= MIN_MEANINGFUL_IC:
        level, headline = "no_evidence", (
            "For this group of stocks, the ranking did not beat chance in testing. "
            "A longer history or more stocks often helps.")
        reasons.append("Large, heavily followed stocks are the hardest to rank from price data "
                       "alone; less-followed stocks often show more.")
    elif not clears_zero:
        level, headline = "unproven", (
            "The ranking leans the right way in testing. More history will show how solid it is.")
    else:
        level, headline = "modest", (
            "The ranking has beaten chance in testing, and the edge is larger than the noise.")

    if hold:
        h = hold.get("rank_ic_mean")
        if h is not None and h <= 0:
            reasons.append(
                f"On the most recent untouched months the ranking did not hold up "
                f"(rank IC {h:+.3f}), so treat the older results with care.")
            if level == "modest":
                level = "unproven"
        elif h is not None:
            reasons.append(f"It also held up on the most recent untouched months "
                           f"(rank IC {h:+.3f}).")
    else:
        reasons.append("No untouched final test period was kept aside. Re-train to add one.")

    n_inst = report.get("n_instruments", 0)
    if n_inst < MIN_UNIVERSE:
        reasons.append(f"Only {n_inst} stocks in the list. 30 or more gives steadier results.")
    n_days = report.get("n_days", 0)
    if n_days < MIN_DAYS:
        reasons.append(f"{n_days / 252:.1f} years of history so far; 3+ years gives a firmer read.")
    if report.get("label_type") != "excess":
        reasons.append("Measured against raw returns (no market index loaded); "
                       "Update data to load the index.")

    mem = report.get("membership") or {}
    if mem.get("mode") == "point_in_time":
        cov = mem.get("coverage")
        text = "Uses who was in the index on each date, so removed stocks are not ignored."
        if cov is not None and cov < 0.95:
            text += (f" Price data was unavailable for {100 * (1 - cov):.0f}% of past members "
                     f"(usually delisted).")
        reasons.append(text)
    else:
        reasons.append("Uses today's stock list for all of history, so past results can look a "
                       "bit better than reality. Add index history to correct this.")

    stale_days = None
    if last_bar:
        try:
            stale_days = (today - date.fromisoformat(last_bar[:10])).days
        except ValueError:
            pass
        if stale_days is not None and stale_days > 7:
            reasons.append(f"Prices are {stale_days} days old. Tap Update data.")

    facts = {
        "model_id": report.get("model_id"),
        "trained_at": report.get("trained_at"),
        "data_through": report.get("data_end"),
        "n_instruments": n_inst, "n_days": n_days,
        "rank_ic": mean, "rank_ic_ci": [lo, hi] if lo is not None else None,
        "t_stat_adjusted": t_adj,
        "hit_rate": cv.get("hit_rate"),
        "brier": (report.get("calibration") or {}).get("brier_score"),
        "auc": (report.get("calibration") or {}).get("roc_auc"),
        "holdout_rank_ic": hold.get("rank_ic_mean") if hold else None,
        "holdout_days": hold.get("n_days") if hold else None,
        "benchmark": report.get("benchmark_id"),
        "label_type": report.get("label_type"),
        "membership_mode": mem.get("mode", "current_list"),
        "fundamentals_used": bool((report.get("fundamentals") or {}).get("used")),
        "fundamentals_coverage": (report.get("fundamentals") or {}).get("coverage"),
        "price_data_age_days": stale_days,
    }
    return {"level": level, "headline": headline, "reasons": reasons, "facts": facts}
