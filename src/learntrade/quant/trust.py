"""
What LearnTrade is allowed to SHOW, given how much evidence there is behind it.

The rule: a forecast number (outlook, chance of beating the market, "would rank N of M")
is only shown when testing found a real edge, and is pulled toward "no view" in proportion
to how weak that edge is. Otherwise the user sees the order of their list and facts that are
measured, not predicted (how much a stock moves, its financials, its price history).

Also here:
  * asset type: ETFs, funds and indices are not companies; they get price facts, never a score
  * familiarity: a stock very unlike the ones the model learned from gets no score
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

# How much of a forecast survives, by reliability level (see quant.reliability).
EVIDENCE_WEIGHT: dict[str, float] = {"modest": 1.0, "unproven": 0.5}

NO_EDGE_NOTE = (
    "No proven edge: in testing, this ranking did not beat chance for these stocks, so no "
    "outlook or probability is shown. The order is the model's score, useful for sorting your "
    "list, not a forecast."
)
QUICK_NOTE = (
    "No trained model yet: the order is recent price momentum only, so no outlook or "
    "probability is shown. Tap Update data to train and test a model."
)


def evidence_weight(level: str | None) -> float:
    return EVIDENCE_WEIGHT.get(level or "", 0.0)


def gate_forecast(expected_return: float, p_up: float, low: float, high: float,
                  level: str | None) -> dict[str, Any]:
    """
    Numbers fit to show for one forecast.

    With no evidence the outlook and probability are withheld (None) and the range is
    re-centred on zero, so it describes how much the stock moves, not where it is headed.
    With partial evidence both are shrunk toward 'no view'.
    """
    w = evidence_weight(level)
    if w == 0.0:
        return {"gated": True, "expected_return": None, "p_up": None,
                "low": low - expected_return, "high": high - expected_return}
    er = expected_return * w
    return {"gated": False, "expected_return": er, "p_up": 0.5 + (p_up - 0.5) * w,
            "low": low - expected_return + er, "high": high - expected_return + er}


# ── Asset type ────────────────────────────────────────────────────────────────

_FUND_SUFFIXES = ("BEES", "ETF", "IETF")
_FUND_WORDS = (" ETF", "BEES", " FUND", "INDEX FUND", "EXCHANGE TRADED", " TRUST")
_NON_COMPANY = {"ETF", "MUTUALFUND", "INDEX", "CRYPTOCURRENCY", "CURRENCY", "FUTURE"}


def is_company(asset_type: str | None, symbol: str, name: str | None = None) -> bool:
    """
    False for ETFs, funds and indices.

    The price source's type is not enough on its own: Yahoo reports NSE ETFs such as GOLDBEES
    and SILVERBEES as EQUITY. So the symbol (NSE ETFs end in BEES/ETF) and the fund's full name
    ("... ETF", "... BeES", "... Fund") are checked as well.
    """
    if asset_type and asset_type.upper() in _NON_COMPANY:
        return False
    s = symbol.upper().split(":")[-1]
    if s.startswith("^") or s.endswith(_FUND_SUFFIXES):
        return False
    n = f" {(name or '').upper()}"
    return not any(w in n for w in _FUND_WORDS)


def non_company_note(asset_type: str | None) -> str:
    kind = {"MUTUALFUND": "a mutual fund", "INDEX": "a market index"}.get(
        (asset_type or "").upper(), "a fund (such as an ETF) or an index, not a company")
    return (f"This is {kind}. LearnTrade's model scores companies, so there is no outlook or "
            "ranking for it, and no company financials. Shown instead: how its price has behaved.")


def price_summary(closes: list[float]) -> dict[str, Any]:
    """Measured facts from a daily price series (oldest first): returns, volatility, drawdown."""
    px = np.array([c for c in closes if c and c > 0], dtype=float)
    if len(px) < 20:
        return {}
    rets = np.diff(np.log(px))
    peak = np.maximum.accumulate(px)
    out = {
        "return_3m": float(px[-1] / px[-min(63, len(px))] - 1),
        "volatility": float(np.std(rets, ddof=1) * math.sqrt(252)),
        "max_drawdown": float((px / peak - 1).min()),
        "days": int(len(px)),
    }
    if len(px) >= 250:
        out["return_1y"] = float(px[-1] / px[-250] - 1)
    return out


# ── Familiarity: is this stock like the ones the model learned from? ─────────

def _profiles(store: Any, ids: list[str], now: datetime) -> dict[str, dict[str, float]]:
    from learntrade.core.types import InstrumentId
    bars = store.query_bars([InstrumentId(i) for i in ids], as_of=now, since=now - timedelta(days=180))
    out: dict[str, dict[str, float]] = {}
    if bars.is_empty():
        return out
    bars = bars.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in ("close", "volume")])
    for iid, g in bars.sort("event_time").group_by("instrument_id"):
        iid = iid[0] if isinstance(iid, tuple) else iid
        c = g["close"].drop_nulls().to_numpy()
        v = g["volume"].fill_null(0).to_numpy()
        if len(c) < 40:
            continue
        r = np.diff(np.log(c[c > 0]))
        out[str(iid)] = {
            "volatility": float(np.std(r, ddof=1) * math.sqrt(252)),
            "traded_value": float(np.median((g["close"].to_numpy() * v)[-60:])),
            "close": float(c[-1]),
        }
    try:   # market value from the latest statement's share count, where financials exist
        for period in ("annual", "quarterly"):
            st = store.query_fundamentals(now, [InstrumentId(i) for i in ids], period=period)
            if st.is_empty():
                continue
            last = st.drop_nulls("shares").sort("event_time").group_by("instrument_id").last()
            for row in last.iter_rows(named=True):
                p = out.get(str(row["instrument_id"]))
                if p is not None and row["shares"] and "market_value" not in p:
                    p["market_value"] = float(row["shares"]) * p["close"]
    except Exception:
        pass
    return out


def familiarity(store: Any, iid: str, members: list[str], now: datetime) -> dict[str, Any]:
    """
    Compare a stock outside the list with the list the model was trained on.

    Unfamiliar = clearly smaller (market value, or trading activity when financials are
    missing) than the smallest stock in the list, or much more volatile than the most volatile
    one. A model has no evidence about stocks outside what it learned from, so those get no score.
    """
    others = [m for m in members if m != iid]
    prof = _profiles(store, others + [iid], now)
    me = prof.get(iid)
    ref = [prof[m] for m in others if m in prof]
    if me is None or len(ref) < 5:
        return {"familiar": True, "reasons": []}
    reasons: list[str] = []
    ref_mv = [p["market_value"] for p in ref if "market_value" in p]
    if "market_value" in me and len(ref_mv) >= 5:
        smallest = min(ref_mv)
        if me["market_value"] < 0.67 * smallest:
            reasons.append(f"It is a much smaller company: about {smallest / me['market_value']:.0f}x "
                           "smaller by market value than the smallest stock the model learned from.")
    else:
        least = min(p["traded_value"] for p in ref)
        if me["traded_value"] < 0.33 * least:
            reasons.append(f"It trades far less: about {least / max(me['traded_value'], 1):.0f}x less "
                           "money changes hands daily than in the least-traded stock the model learned from.")
    top_vol = max(p["volatility"] for p in ref)
    if me["volatility"] > 1.25 * top_vol:
        reasons.append(f"It moves much more: {me['volatility']:.0%} a year against at most "
                       f"{top_vol:.0%} for the stocks the model learned from.")
    return {"familiar": not reasons, "reasons": reasons}
