"""
Full Market Pack cost pipeline — commission, taxes, fees, VAT, spread, impact.

The only public entry point is ``compute_trade_cost()``.
All costs are returned as absolute amounts in the instrument's currency,
plus a summary in basis-points of notional.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from leantrade.markets.schema import CommissionConfig, CostsConfig, MarketPack


@dataclass(frozen=True)
class TradeCost:
    commission: float
    taxes: float
    exchange_fees: float
    regulatory_fees: float
    vat_on_fees: float
    spread_cost: float    # half-spread paid by the aggressive side
    impact_cost: float    # square-root market impact
    total: float
    total_bps: float      # total / notional * 10_000
    notional: float


def compute_commission(cfg: CommissionConfig, notional: float) -> float:
    bps_cost = notional * cfg.bps / 10_000
    if cfg.type == "zero":
        return 0.0
    if cfg.type == "flat":
        return cfg.flat
    if cfg.type == "bps":
        return bps_cost
    if cfg.type == "flat_plus_bps":
        return cfg.flat + bps_cost
    if cfg.type == "flat_or_bps":
        if cfg.take == "min":
            return min(cfg.flat, bps_cost)
        if cfg.take == "max":
            return max(cfg.flat, bps_cost)
    return bps_cost


def compute_trade_cost(
    side: Literal["buy", "sell"],
    quantity: float,
    price: float,
    adv: float,
    volatility_annual: float,
    pack: MarketPack,
    broker_commission_override: CommissionConfig | None = None,
    trade_type: Literal["delivery", "intraday"] = "delivery",
) -> TradeCost:
    """
    Compute the full cost of one trade using the Market Pack cost model.

    Args:
        side:                    "buy" or "sell"
        quantity:                number of shares / units
        price:                   execution price (in instrument currency)
        adv:                     average daily volume (shares, for impact model)
        volatility_annual:       annualised return volatility (0.20 = 20%)
        pack:                    Market Pack for the instrument's exchange
        broker_commission_override: if set, replaces pack commission config
        trade_type:              "delivery" (position held overnight — the
                                  only kind this simulator produces today,
                                  since it fills on the next bar's open) or
                                  "intraday" (opened and closed same session;
                                  not yet modelled by the event-driven
                                  simulator). Selects which `applies_to`-tagged
                                  taxes fire — e.g. NSE's stt_delivery and
                                  stt_intraday are mutually exclusive and must
                                  never both be charged on the same trade.
    """
    if quantity <= 0 or price <= 0:
        return TradeCost(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    notional = quantity * price
    costs = pack.costs
    commission_cfg = broker_commission_override or costs.commission

    commission = compute_commission(commission_cfg, notional)

    # Side- and trade-type-specific taxes. `applies_to == "all"` always
    # fires; anything else (e.g. "delivery"/"intraday") only fires for a
    # matching trade_type.
    taxes = sum(
        notional * t.bps / 10_000
        for t in costs.taxes
        if (t.side == "both" or t.side == side)
        and (t.applies_to == "all" or t.applies_to == trade_type)
    )

    # Exchange and regulatory fees (always applied)
    exchange_fee = notional * costs.exchange_fees.get("bps", 0.0) / 10_000
    regulatory_fee = notional * costs.regulatory_fees.get("bps", 0.0) / 10_000

    # VAT/GST on specified fee types
    vat_pct = pack.costs.vat_on_fees.get("percent", 0.0) / 100.0
    vat_applies_to: list[str] = pack.costs.vat_on_fees.get("applies_to", [])
    fees_for_vat = 0.0
    if "commission" in vat_applies_to:
        fees_for_vat += commission
    if "exchange_fees" in vat_applies_to:
        fees_for_vat += exchange_fee
    vat = fees_for_vat * vat_pct

    # Spread cost (half-spread for a market order)
    spread_cost = 0.0
    if costs.spread_model == "from_quotes_or_volatility" and volatility_annual > 0:
        # Half-spread empirical proxy: ~0.5% of daily vol for liquid large-caps
        daily_vol = volatility_annual / math.sqrt(252)
        spread_cost = notional * daily_vol * 0.005

    # Square-root market impact
    impact_cost = 0.0
    impact_cfg = costs.impact_model
    if impact_cfg.get("type") == "square_root" and adv > 0 and volatility_annual > 0:
        daily_vol = volatility_annual / math.sqrt(252)
        participation = quantity / adv
        coeff = impact_cfg.get("coefficient", 0.1)
        impact_pct = coeff * daily_vol * math.sqrt(participation)
        impact_cost = notional * impact_pct

    total = commission + taxes + exchange_fee + regulatory_fee + vat + spread_cost + impact_cost
    total_bps = (total / notional * 10_000) if notional > 0 else 0.0

    return TradeCost(
        commission=commission,
        taxes=taxes,
        exchange_fees=exchange_fee,
        regulatory_fees=regulatory_fee,
        vat_on_fees=vat,
        spread_cost=spread_cost,
        impact_cost=impact_cost,
        total=total,
        total_bps=total_bps,
        notional=notional,
    )
