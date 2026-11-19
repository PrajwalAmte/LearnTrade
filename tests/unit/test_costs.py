"""
Cost model tests — full Market Pack pipeline.
"""

from __future__ import annotations

import pytest

from learntrade.execution.costs import TradeCost, compute_commission, compute_trade_cost
from learntrade.markets.loader import get_pack
from learntrade.markets.schema import CommissionConfig, MarketPack


def test_zero_commission():
    cfg = CommissionConfig(type="zero")
    assert compute_commission(cfg, 10_000.0) == 0.0


def test_flat_commission():
    cfg = CommissionConfig(type="flat", flat=5.0)
    assert compute_commission(cfg, 10_000.0) == pytest.approx(5.0)


def test_bps_commission():
    cfg = CommissionConfig(type="bps", bps=10.0)
    assert compute_commission(cfg, 10_000.0) == pytest.approx(10.0)  # 10 bps × 10000 = 10


def test_flat_or_bps_min():
    cfg = CommissionConfig(type="flat_or_bps", flat=5.0, bps=10.0, take="min")
    # bps cost = 1_000_000 × 10/10000 = 1000, flat = 5 → min = 5
    assert compute_commission(cfg, 1_000_000.0) == pytest.approx(5.0)
    # bps cost = 10 × 10/10000 = 0.01, flat = 5 → min = 0.01
    assert compute_commission(cfg, 10.0) == pytest.approx(0.01)


def test_full_cost_xnys_buy():
    pack = get_pack("XNYS")
    cost = compute_trade_cost(
        side="buy",
        quantity=100.0,
        price=50.0,          # $5000 notional
        adv=1_000_000.0,
        volatility_annual=0.20,
        pack=pack,
    )
    assert isinstance(cost, TradeCost)
    assert cost.notional == pytest.approx(5000.0)
    assert cost.total >= 0
    assert cost.total_bps >= 0
    # XNYS has zero commission at retail brokers
    assert cost.commission == pytest.approx(0.0)
    # SEC fee applies only to sells, not buys
    assert cost.taxes == pytest.approx(0.0)


def test_full_cost_xnys_sell():
    pack = get_pack("XNYS")
    cost = compute_trade_cost(
        side="sell",
        quantity=100.0,
        price=50.0,
        adv=1_000_000.0,
        volatility_annual=0.20,
        pack=pack,
    )
    # SEC fee and FINRA TAF apply on sells
    assert cost.taxes > 0


def test_impact_increases_with_participation():
    pack = get_pack("XNYS")
    cost_small = compute_trade_cost("buy", 100.0, 50.0, 1_000_000.0, 0.20, pack)
    cost_large = compute_trade_cost("buy", 100_000.0, 50.0, 1_000_000.0, 0.20, pack)
    # Larger participation → larger impact
    assert cost_large.impact_cost > cost_small.impact_cost


def test_zero_quantity_returns_zero_cost():
    pack = get_pack("XNYS")
    cost = compute_trade_cost("buy", 0.0, 50.0, 1_000_000.0, 0.20, pack)
    assert cost.total == 0.0
    assert cost.total_bps == 0.0


def test_vat_applied_xnse():
    pack = get_pack("XNSE")
    cost = compute_trade_cost(
        side="buy",
        quantity=10.0,
        price=1000.0,
        adv=100_000.0,
        volatility_annual=0.25,
        pack=pack,
    )
    # XNSE has 18% VAT on commission + exchange fees
    assert cost.vat_on_fees > 0


def test_total_bps_roundtrip():
    pack = get_pack("XNYS")
    cost = compute_trade_cost("sell", 500.0, 100.0, 500_000.0, 0.15, pack)
    expected_bps = cost.total / cost.notional * 10_000
    assert cost.total_bps == pytest.approx(expected_bps, rel=1e-6)
