"""
Contamination guard — blocks LLM signals where the model's training
cutoff overlaps the evaluation period.

LLMs memorise historical data. A model with a training cutoff of 2024-04-01
asked about an event from 2023 may already "know" the outcome, invalidating
any evaluation of its analytical quality.

Three defences:
1. is_contaminated(model_id, as_of): returns True if as_of ≤ cutoff + margin.
   Call this before adding any LLM signal to a training row.
2. mask_document(text): anonymises ticker symbols, instrument names, and dates,
   and rescales prices. Use in masking-mode backtests.
3. Backtest reports flag contaminated rows and exclude them from "validated" metrics.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone


# Registry: model_id → training cutoff (UTC).
_CUTOFFS: dict[str, datetime] = {
    "gpt-4o":              datetime(2023, 10, 1, tzinfo=timezone.utc),
    "gpt-4o-mini":         datetime(2023, 10, 1, tzinfo=timezone.utc),
    "claude-opus-4-5":     datetime(2024, 4, 1, tzinfo=timezone.utc),
    "claude-sonnet-4-5":   datetime(2024, 4, 1, tzinfo=timezone.utc),
    "claude-haiku-4-5":    datetime(2024, 4, 1, tzinfo=timezone.utc),
    "qwen2.5:14b-instruct":datetime(2024, 9, 1, tzinfo=timezone.utc),
}


def is_contaminated(model_id: str, as_of: datetime, safety_margin_days: int = 30) -> bool:
    """
    Return True if ``as_of`` falls within the model's training window.

    Unknown models return True (assume contaminated — safe default).
    When True, any LLM-derived signal for this date must be excluded from
    validated out-of-sample metrics.
    """
    cutoff = _CUTOFFS.get(model_id)
    if cutoff is None:
        return True  # unknown model → contaminated
    return as_of <= cutoff + timedelta(days=safety_margin_days)


def register_cutoff(model_id: str, cutoff: datetime) -> None:
    """Add or update a model's training cutoff in the registry."""
    _CUTOFFS[model_id] = cutoff


def list_cutoffs() -> dict[str, str]:
    return {k: v.isoformat() for k, v in _CUTOFFS.items()}


# ── Masking mode ──────────────────────────────────────────────────────────────

_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")
_DATE_RE = re.compile(
    r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]20\d{2}|"
    r"Q[1-4]\s+20\d{2}|FY\s*20\d{2})\b"
)
_PRICE_RE = re.compile(r"\$\s*(\d[\d,]*\.?\d*)")


def mask_document(
    text: str,
    price_scale: float = 1.0,
    replace_tickers: bool = True,
    replace_dates: bool = True,
    replace_prices: bool = True,
) -> str:
    """
    Anonymise a document for masking-mode backtests.

    Replaces ticker symbols, dates, and prices to test whether LLM analytical
    value comes from reasoning or from memorised historical outcomes.
    """
    if replace_tickers:
        # Replace common ticker-like tokens (AAPL → ZZZZ, etc.)
        # We only replace tokens that look like US equity tickers
        text = _TICKER_RE.sub(lambda m: "X" * len(m.group(1)), text)
    if replace_dates:
        text = _DATE_RE.sub("[DATE REDACTED]", text)
    if replace_prices and price_scale != 1.0:
        def _scale(m: re.Match) -> str:
            try:
                val = float(m.group(1).replace(",", "")) * price_scale
                return f"${val:.2f}"
            except ValueError:
                return m.group(0)
        text = _PRICE_RE.sub(_scale, text)
    return text
