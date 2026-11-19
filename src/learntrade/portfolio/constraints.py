"""
Portfolio constraints — post-processing of raw allocator weights.

Constraints are applied in order:
1. Position limit  (clip each weight to max_position)
2. Long-only guard (zero out short weights if long_only=True)
3. Minimum weight  (drop holdings below min_weight, re-normalise)
4. Re-normalise to sum = 1.0

Constraints do NOT solve a QP — they are fast O(n) post-processing.
For binding constraints (e.g. sector limits) the allocator should be
configured to include them. QP-based constraint enforcement is Phase 5+.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from learntrade.core.types import Forecast, Holding, InstrumentId


@dataclass
class PortfolioConstraints:
    max_position: float = 0.05          # max weight per instrument
    max_sector: float = 0.25            # max total weight per sector
    max_country: float = 1.0            # max total weight per country
    max_currency_exposure: dict[str, float] = field(default_factory=dict)
    max_liquidity_pct: float = 0.10     # max % of average daily volume
    max_turnover: float = 2.0           # max fraction of portfolio traded per step
    min_weight: float = 0.001           # positions below this are closed
    long_only: bool = True


def apply_constraints(
    weights: dict[InstrumentId, float],
    constraints: PortfolioConstraints,
    forecasts: list[Forecast] | None = None,
    current_weights: dict[InstrumentId, float] | None = None,
) -> dict[InstrumentId, float]:
    """
    Apply portfolio constraints to raw allocator weights.

    Returns weights that respect ``max_position`` and sum to at most 1.0.
    Never modifies the input dict.
    """
    result = dict(weights)

    # Long-only guard
    if constraints.long_only:
        result = {k: max(v, 0.0) for k, v in result.items()}

    # Drop sub-minimum positions up front (before capping/redistribution).
    result = {k: v for k, v in result.items() if v >= constraints.min_weight}
    if not result:
        return result

    # Scale the raw weights to sum to 1.0, then enforce the position cap via
    # water-filling: redistribute whatever a capped name gives up across the
    # *uncapped* names only, and leave any residual that genuinely can't be
    # placed within the cap as uninvested cash. The previous version capped
    # each weight and then renormalised the WHOLE set back to sum=1, which
    # scales already-capped weights back above the cap again — e.g. 15 names
    # equal-weighted with a 5% cap renormalises to 6.67% each, silently
    # breaking the limit it just applied.
    total = sum(result.values())
    if total > 1e-9:
        result = {k: v / total for k, v in result.items()}
    result = _cap_with_waterfill(result, constraints.max_position)

    # Turnover limit: blend toward the (already cap-respecting) target weights
    # without moving the portfolio by more than max_turnover this step.
    # The blended result is a transitional position — it's a weighted
    # average of the current holding (which may predate today's constraints,
    # e.g. a legacy position already above the cap) and the capped target,
    # so it is intentionally not re-clipped to the cap here: doing so would
    # make the turnover limit itself ineffective whenever the cap is small
    # relative to the position being wound down.
    if current_weights and constraints.max_turnover < 2.0:
        result = _apply_turnover_limit(result, current_weights, constraints.max_turnover)

    # Drop sub-minimum positions created by capping/turnover blending. This
    # intentionally does NOT renormalise afterward — a dropped position's
    # weight is left uninvested (cash) rather than silently reappearing on
    # the remaining names and breaking their cap again.
    result = {k: v for k, v in result.items() if v >= constraints.min_weight}

    return result


def _cap_with_waterfill(
    weights: dict[InstrumentId, float],
    cap: float,
) -> dict[InstrumentId, float]:
    """
    Cap every weight at ``cap`` while keeping the total as close to 1.0 as
    the cap allows, by repeatedly redistributing the amount given up by
    capped names across the names that are still under the cap.

    If ``cap * n_names < 1.0`` the result sums to less than 1.0 (the rest is
    legitimately uninvested cash) rather than violating the cap. If the cap
    is infeasible in the other direction (too many names forced to exactly
    ``cap`` to be scaled down to fit in 1.0 total), capped weights are
    scaled down together as a last resort.
    """
    if cap <= 0 or not weights:
        return dict(weights)

    w = dict(weights)
    capped: set[InstrumentId] = set()

    for _ in range(len(w) + 1):
        free = [k for k in w if k not in capped]
        if not free:
            break
        capped_total = sum(w[k] for k in capped)
        remaining = 1.0 - capped_total
        free_total = sum(w[k] for k in free)
        if remaining <= 1e-12 or free_total <= 1e-12:
            for k in free:
                w[k] = 0.0
            break
        scale = remaining / free_total
        newly_capped = []
        for k in free:
            w[k] = w[k] * scale
            if w[k] >= cap - 1e-12:
                w[k] = cap
                newly_capped.append(k)
        if not newly_capped:
            break
        capped.update(newly_capped)

    # Infeasible cap (e.g. cap=0.03 but only 10 names can ever be selected,
    # so 10 x 0.03 = 0.30 < 1.0 is fine and left as cash — but if instead
    # *every* name is capped and their sum still exceeds 1.0, which can only
    # happen if the cap itself is set above 1.0/n in a way that still blows
    # the budget due to floating point, scale everything down together).
    total = sum(w.values())
    if total > 1.0 + 1e-9:
        w = {k: v / total for k, v in w.items()}

    return w


def compute_turnover(
    new_weights: dict[InstrumentId, float],
    old_weights: dict[InstrumentId, float],
) -> float:
    """Return the total absolute weight change (round-trip turnover)."""
    all_ids = set(new_weights) | set(old_weights)
    return sum(abs(new_weights.get(iid, 0.0) - old_weights.get(iid, 0.0))
               for iid in all_ids)


def _apply_turnover_limit(
    target: dict[InstrumentId, float],
    current: dict[InstrumentId, float],
    max_turnover: float,
) -> dict[InstrumentId, float]:
    """Blend current and target weights to stay within turnover budget."""
    to = compute_turnover(target, current)
    if to <= max_turnover:
        return target
    # Scale the trade toward target proportionally to the budget
    blend = max_turnover / to
    result = {}
    for iid in set(target) | set(current):
        cur_w = current.get(iid, 0.0)
        tgt_w = target.get(iid, 0.0)
        result[iid] = cur_w + blend * (tgt_w - cur_w)
    return result
