"""
Core domain types for LearnTrade.

All modules share these Pydantic models as their typed contract.
Timestamps are always UTC; local display is handled at the interface layer.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, GetCoreSchemaHandler, model_validator
from pydantic_core import CoreSchema, core_schema


# ---------------------------------------------------------------------------
# Instrument identity
# ---------------------------------------------------------------------------


class InstrumentId(str):
    """
    '<MIC>:<SYMBOL>'  e.g. 'XNYS:IBM', 'XNSE:RELIANCE', 'XTKS:7203'.

    Optionally linked to ISIN / FIGI for cross-listing and symbol-change resolution.
    The string representation is always upper-cased and normalised.
    """

    __slots__ = ()

    def __new__(cls, value: str) -> "InstrumentId":
        value = value.upper().strip()
        if ":" not in value:
            raise ValueError(f"InstrumentId must be '<MIC>:<SYMBOL>', got {value!r}")
        return str.__new__(cls, value)

    @property
    def mic(self) -> str:
        return self.split(":")[0]

    @property
    def symbol(self) -> str:
        return self.split(":")[1]

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )

    @classmethod
    def _validate(cls, value: Any) -> "InstrumentId":
        if isinstance(value, cls):
            return value
        return cls(str(value))


# ---------------------------------------------------------------------------
# Instrument
# ---------------------------------------------------------------------------

AssetClass = Literal["equity", "etf", "index", "fx", "future", "crypto"]


class Instrument(BaseModel):
    id: InstrumentId
    mic: str
    symbol: str
    isin: str | None = None
    figi: str | None = None
    currency: str                           # ISO 4217
    asset_class: AssetClass = "equity"
    listed_from: date | None = None
    delisted_on: date | None = None         # required for survivorship-free universes
    sector: str | None = None               # taxonomy configurable (GICS-like, ICB, custom)
    name: str | None = None

    @model_validator(mode="after")
    def _validate_mic_symbol_match(self) -> "Instrument":
        if self.id.mic != self.mic or self.id.symbol != self.symbol:
            raise ValueError("id.mic / id.symbol must match mic / symbol fields")
        return self


# ---------------------------------------------------------------------------
# Observation (point-in-time record)
# ---------------------------------------------------------------------------

PitLevel = Literal["full", "partial", "none"]


class Observation(BaseModel):
    """
    A single observation from a data provider.

    ``available_at`` is the moment the data became publicly knowable (UTC).
    ``event_time``   is the moment the event occurred or the period it refers to.
    All queries must filter ``available_at <= as_of`` — never ``event_time <= as_of``.
    """

    instrument_id: InstrumentId | None = None   # None for macro / market-wide data
    source_id: str
    kind: Literal["bar", "fundamental", "filing", "news", "macro", "corporate_action",
                  "universe", "quote", "other"]
    event_time: datetime
    available_at: datetime                      # UTC; enforced by the PIT store
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    payload: dict[str, Any]
    source_version: str = "1"
    pit_level: PitLevel = "partial"


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------


class ForecastQuantiles(BaseModel):
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float


class Forecast(BaseModel):
    """
    Output of the quant engine for one instrument × horizon.

    ``expected_return`` is excess over the market benchmark, not raw return.
    All probabilities are calibrated.
    """

    instrument_id: InstrumentId
    as_of: datetime                 # UTC; the decision time
    horizon: str                    # e.g. "swing_5d", "positional_60d"
    expected_return: float          # excess over benchmark
    p_positive: float               # calibrated P(excess return > 0)
    volatility: float               # forecast annualised vol
    quantiles: ForecastQuantiles
    rank: float | None = None       # cross-sectional percentile [0, 1] within universe
    model_id: str
    data_snapshot_id: str
    pit_level: PitLevel = "partial"


# ---------------------------------------------------------------------------
# Holdings and orders
# ---------------------------------------------------------------------------


class Holding(BaseModel):
    instrument_id: InstrumentId
    quantity: float
    average_cost: float | None = None
    currency: str


class Order(BaseModel):
    instrument_id: InstrumentId
    side: Literal["buy", "sell", "sell_short", "cover"]
    order_type: Literal["market", "limit", "stop", "stop_limit"] = "market"
    quantity: float
    limit_price: float | None = None
    stop_price: float | None = None
    horizon: str | None = None          # which sleeve this order belongs to
    strategy_id: str | None = None
    notes: str | None = None


class OrderAck(BaseModel):
    order_id: str
    status: Literal["submitted", "filled", "partial", "rejected", "cancelled"]
    filled_quantity: float = 0.0
    filled_avg_price: float | None = None
    message: str | None = None


# ---------------------------------------------------------------------------
# Decision attribution
# ---------------------------------------------------------------------------


class EvidenceClaim(BaseModel):
    source_id: str
    available_at: datetime
    claim: str
    verified: bool = False


class ForecastAttribution(BaseModel):
    models: dict[str, float]            # model_id → weight contribution
    llm_cost_usd: float = 0.0


class DecisionRecord(BaseModel):
    """
    Full auditable record for one instrument × horizon decision.
    Written by the attribution store after each run; outcomes are appended later.
    """

    decision_id: str                    # "{as_of}/{instrument_id}/{horizon}"
    instrument_id: InstrumentId
    horizon: str
    as_of: datetime
    forecast: Forecast
    thesis: str | None = None
    counter_thesis: str | None = None
    invalidation_conditions: list[str] = Field(default_factory=list)
    evidence: list[EvidenceClaim] = Field(default_factory=list)
    agreement_with_quant: Literal["agree", "disagree", "neutral"] | None = None
    target_weight: float | None = None
    current_weight: float | None = None
    trade: Literal["BUY", "SELL", "HOLD"] | None = None
    est_cost_bps: float | None = None
    attribution: ForecastAttribution | None = None
    data_quality: dict[str, Any] = Field(default_factory=dict)
    realised_return: float | None = None    # filled in after the horizon elapses


# ---------------------------------------------------------------------------
# Data quality report
# ---------------------------------------------------------------------------


class DataQualityReport(BaseModel):
    snapshot_id: str
    as_of: datetime
    issues: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    passed: bool = True
