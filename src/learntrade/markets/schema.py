"""
Pydantic schema for Market Pack YAML files.

Numbers in the example packs are illustrative; always verify with
your exchange or broker before using in production.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SessionConfig(BaseModel):
    open: str    # "HH:MM" in the market's local time zone
    close: str


class TickBand(BaseModel):
    below: float    # upper bound of this band (use float('inf') for the last band)
    tick: float


class PriceLimitsConfig(BaseModel):
    type: Literal["none", "percent_band", "absolute"] = "none"
    bands_source: Literal["static", "provider"] = "static"
    daily_limit_pct: float | None = None


class ShortSellingConfig(BaseModel):
    intraday: bool = False
    overnight: bool = False
    borrow_cost_bps_annual: float = 0.0


class CommissionConfig(BaseModel):
    type: Literal["flat", "bps", "flat_or_bps", "flat_plus_bps", "zero"] = "bps"
    flat: float = 0.0
    bps: float = 10.0
    take: Literal["min", "max", "both"] = "max"


class TaxConfig(BaseModel):
    name: str
    side: Literal["buy", "sell", "both"]
    bps: float = 0.0
    # "all" always applies; "delivery"/"intraday" (or any custom tag) only
    # applies when compute_trade_cost() is called with a matching
    # trade_type — e.g. NSE's stt_delivery and stt_intraday are mutually
    # exclusive and must never both be charged on the same trade.
    applies_to: str = "all"


class CostsConfig(BaseModel):
    commission: CommissionConfig = Field(default_factory=CommissionConfig)
    taxes: list[TaxConfig] = Field(default_factory=list)
    exchange_fees: dict[str, float] = Field(default_factory=lambda: {"bps": 0.3})
    regulatory_fees: dict[str, float] = Field(default_factory=lambda: {"bps": 0.01})
    vat_on_fees: dict[str, Any] = Field(default_factory=dict)
    spread_model: Literal["from_quotes_or_volatility", "fixed_bps", "none"] = (
        "from_quotes_or_volatility"
    )
    impact_model: dict[str, Any] = Field(
        default_factory=lambda: {"type": "square_root", "coefficient": 0.1}
    )


class FxConfig(BaseModel):
    quote_currency: str = "USD"


class DataDefaultsConfig(BaseModel):
    prices: list[str] = Field(default_factory=lambda: ["yahoo"])
    filings: list[str] = Field(default_factory=list)
    fundamentals: list[str] = Field(default_factory=list)
    news: list[str] = Field(default_factory=list)


class MarketPack(BaseModel):
    """
    Fully parsed representation of one Market Pack YAML file.
    """

    mic: str
    name: str = ""
    country: str = ""
    currency: str = "USD"
    timezone: str = "UTC"
    calendar: str = "GENERIC"           # exchange_calendars key or custom holiday file
    sessions: dict[str, SessionConfig] = Field(
        default_factory=lambda: {"regular": SessionConfig(open="09:30", close="16:00")}
    )
    settlement: str = "T+2"
    benchmark: str | None = None
    lot_size: int = 1
    tick_size: list[TickBand] = Field(
        default_factory=lambda: [TickBand(below=float("inf"), tick=0.01)]
    )
    price_limits: PriceLimitsConfig = Field(default_factory=PriceLimitsConfig)
    short_selling: ShortSellingConfig = Field(default_factory=ShortSellingConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    fx: FxConfig = Field(default_factory=FxConfig)
    data_defaults: DataDefaultsConfig = Field(default_factory=DataDefaultsConfig)
