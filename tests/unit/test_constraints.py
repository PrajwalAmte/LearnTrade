"""
Portfolio constraints tests.
"""

from __future__ import annotations

import pytest

from leantrade.core.types import InstrumentId
from leantrade.portfolio.constraints import (
    PortfolioConstraints,
    apply_constraints,
    compute_turnover,
)


def _weights(*pairs: tuple[str, float]) -> dict[InstrumentId, float]:
    return {InstrumentId(iid): w for iid, w in pairs}


def test_position_limit_clips():
    # Three instruments: two large, one small. With a 30% cap, the most this
    # portfolio can ever hold is 3 x 0.30 = 0.90 — the remaining 10% is
    # legitimately uninvested cash, not something to redistribute back onto
    # the capped names (which would silently break the cap again).
    w = _weights(("XNYS:A", 0.50), ("XNYS:B", 0.40), ("XNYS:C", 0.10))
    c = PortfolioConstraints(max_position=0.30)
    result = apply_constraints(w, c)
    for v in result.values():
        assert v <= c.max_position + 1e-9
    assert sum(result.values()) <= 3 * c.max_position + 1e-9
    assert sum(result.values()) > 0.5  # not all cash either


def test_weights_renormalise_after_clip():
    # 2 names x 30% cap = 60% max investable; the cap must hold even though
    # that leaves 40% as cash rather than being scaled back above the cap.
    w = _weights(("XNYS:A", 0.60), ("XNYS:B", 0.40))
    c = PortfolioConstraints(max_position=0.30)
    result = apply_constraints(w, c)
    for v in result.values():
        assert v <= c.max_position + 1e-9
    assert abs(sum(result.values()) - 2 * c.max_position) < 1e-9


def test_cap_not_violated_when_enough_names_to_fill_budget():
    # With enough names under the cap, the portfolio should still end up
    # fully invested (sum to 1.0) — the cap only holds back investment when
    # there genuinely aren't enough uncapped names to reach 100%.
    w = _weights(*[(f"XNYS:S{i}", 1.0) for i in range(20)])  # equal raw weights
    c = PortfolioConstraints(max_position=0.10)
    result = apply_constraints(w, c)
    for v in result.values():
        assert v <= c.max_position + 1e-9
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_min_weight_drops_tiny_positions():
    w = _weights(("XNYS:A", 0.50), ("XNYS:B", 0.00001), ("XNYS:C", 0.49999))
    c = PortfolioConstraints(min_weight=0.01)
    result = apply_constraints(w, c)
    assert InstrumentId("XNYS:B") not in result


def test_long_only_zeros_negatives():
    w = _weights(("XNYS:A", 0.60), ("XNYS:B", -0.20), ("XNYS:C", 0.40))
    c = PortfolioConstraints(long_only=True)
    result = apply_constraints(w, c)
    for v in result.values():
        assert v >= 0


def test_compute_turnover_no_change():
    w = _weights(("XNYS:A", 0.50), ("XNYS:B", 0.50))
    assert compute_turnover(w, w) == pytest.approx(0.0)


def test_compute_turnover_full_replacement():
    old = _weights(("XNYS:A", 1.0))
    new = _weights(("XNYS:B", 1.0))
    assert compute_turnover(new, old) == pytest.approx(2.0)


def test_turnover_limit_blends_weights():
    old = _weights(("XNYS:A", 1.0))
    new = _weights(("XNYS:B", 1.0))
    c = PortfolioConstraints(max_turnover=0.5)
    result = apply_constraints(new, c, current_weights=old)
    # Turnover of result should be ≤ max_turnover + small tolerance
    to = compute_turnover(result, old)
    assert to <= 0.5 + 0.01
