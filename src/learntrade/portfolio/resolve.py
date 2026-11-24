"""
Turn imported rows into instruments LearnTrade can price, and say what each one is.

  * Indian stocks and ETFs -> an NSE (or BSE) symbol, found by ISIN, symbol or company name
    (Yahoo Finance search), priced from daily exchange prices.
  * Mutual funds -> an AMFI scheme code (from AMFI's free daily NAV file, by ISIN or name),
    priced from NAV history (mfapi.in, which republishes AMFI data).

Each holding also gets an asset class (Indian stocks, equity fund, gold, silver, debt, ...) so
the portfolio can be broken down by what it actually holds.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from learntrade.portfolio.importer import RawHolding

logger = logging.getLogger(__name__)

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
_SEARCH = "https://query1.finance.yahoo.com/v1/finance/search"
_AMFI = "https://portal.amfiindia.com/spages/NAVAll.txt"


@dataclass
class Holding:
    name: str
    quantity: float
    avg_price: float | None
    id: str | None = None             # "XNSE:TCS" or "AMFI:122639"; None if not found
    kind: str = "unknown"             # stock | etf | fund | unknown
    asset_class: str = "Other"        # Indian stocks | Equity funds | Gold | Silver | Debt | ...
    sector: str | None = None
    category: str | None = None       # fund category, e.g. "Equity Scheme - Small Cap Fund"
    isin: str | None = None
    file_price: float | None = None   # price/NAV in the broker file (fallback if no live price)
    note: str | None = None           # why it could not be resolved, if so

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Holding":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


# ── Classification ───────────────────────────────────────────────────────────

def _has(text: str, *words: str) -> bool:
    t = f" {text.upper()} "
    return any(w in t for w in words)


def etf_class(name: str) -> str:
    if _has(name, "GOLD"):
        return "Gold"
    if _has(name, "SILVER"):
        return "Silver"
    if _has(name, "LIQUID", "GILT", "BOND", "G-SEC", "GSEC", "SDL", "DEBT", "MONEY MARKET", "OVERNIGHT"):
        return "Debt"
    if _has(name, "NASDAQ", "S&P 500", "S&P500", "HANG SENG", " US ", "GLOBAL", "INTERNATIONAL",
            "FANG", "MSCI", "NYSE", "MAFANG", "WORLD"):
        return "International"
    return "Equity funds"


def fund_class(category: str | None, name: str) -> str:
    c = (category or "").upper()
    if _has(name, "GOLD") and not _has(name, "GOLDMAN"):
        return "Gold"
    if _has(name, "SILVER"):
        return "Silver"
    if "DEBT SCHEME" in c or "LIQUID" in c or "GILT" in c or "OVERNIGHT" in c or "MONEY MARKET" in c:
        return "Debt"
    if "HYBRID" in c or "SOLUTION ORIENTED" in c or "ARBITRAGE" in c:
        return "Hybrid"
    if _has(name, "NASDAQ", "S&P 500", "US ", "GLOBAL", "INTERNATIONAL", "WORLD", "FANG", "OVERSEAS"):
        return "International"
    if "EQUITY SCHEME" in c or "INDEX" in c or "ELSS" in c or c == "":
        return "Equity funds"
    return "Other"


def size_hint(category: str | None, name: str) -> str | None:
    """large / mid / small / mixed from a fund's category or name, if it says."""
    t = f"{category or ''} {name}".upper()
    if "LARGE & MID" in t or "LARGE AND MID" in t:
        return "mixed"
    for key, val in (("SMALL CAP", "small"), ("SMALLCAP", "small"), ("MID CAP", "mid"),
                     ("MIDCAP", "mid"), ("LARGE CAP", "large"), ("LARGECAP", "large"),
                     ("NIFTY 50", "large"), ("NIFTY50", "large"), ("SENSEX", "large"),
                     ("NIFTYBEES", "large")):
        if key in t:
            return val
    if any(k in t for k in ("FLEXI", "MULTI CAP", "MULTICAP", "FOCUSED", "VALUE", "CONTRA", "ELSS")):
        return "mixed"
    return None


# ── Lookups (cached in the data folder) ──────────────────────────────────────

class Resolver:
    def __init__(self, cache_dir: Path, offline: bool = False) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._offline = offline
        self._search_cache_path = self._dir / "symbol_lookup.json"
        try:
            self._search_cache: dict[str, Any] = json.loads(self._search_cache_path.read_text())
        except (FileNotFoundError, ValueError):
            self._search_cache = {}
        self._amfi: list[dict[str, str]] | None = None
        self._last = 0.0

    # Yahoo search ------------------------------------------------------------
    def _search(self, q: str) -> list[dict[str, Any]]:
        key = q.upper()
        if key in self._search_cache:
            return self._search_cache[key]
        if self._offline:
            return []
        wait = 0.3 - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        try:
            r = httpx.get(_SEARCH, params={"q": q, "quotesCount": 8, "newsCount": 0},
                          headers=_UA, timeout=20)
            r.raise_for_status()
            quotes = [
                {k: x.get(k) for k in ("symbol", "exchange", "quoteType", "sector", "longname",
                                       "shortname")}
                for x in r.json().get("quotes", [])
            ]
        except Exception as exc:
            logger.warning("Symbol search failed for %r: %s", q, exc)
            return []
        self._search_cache[key] = quotes
        self._search_cache_path.write_text(json.dumps(self._search_cache))
        return quotes

    def find_listed(self, q: str, exact_symbol: str | None = None) -> dict[str, Any] | None:
        """Best NSE (else BSE) listing for an ISIN, symbol or company name."""
        quotes = [x for x in self._search(q) if x.get("exchange") in ("NSI", "BSE")
                  and x.get("quoteType") in ("EQUITY", "ETF")]
        if exact_symbol:
            want = {f"{exact_symbol}.NS", f"{exact_symbol}.BO"}
            quotes = [x for x in quotes if x.get("symbol") in want] or []
        quotes.sort(key=lambda x: 0 if x.get("exchange") == "NSI" else 1)
        return quotes[0] if quotes else None

    # AMFI scheme list ----------------------------------------------------------
    def _amfi_rows(self) -> list[dict[str, str]]:
        if self._amfi is not None:
            return self._amfi
        path = self._dir / "amfi_navall.txt"
        fresh = path.exists() and (time.time() - path.stat().st_mtime) < 86400
        if not fresh and not self._offline:
            try:
                r = httpx.get(_AMFI, headers=_UA, timeout=60, follow_redirects=True)
                r.raise_for_status()
                path.write_text(r.text)
            except Exception as exc:
                logger.warning("Could not download AMFI scheme list: %s", exc)
        self._amfi = parse_amfi(path.read_text()) if path.exists() else []
        return self._amfi

    def find_fund(self, isin: str | None, name: str | None) -> dict[str, str] | None:
        rows = self._amfi_rows()
        if isin:
            for r in rows:
                if isin in (r["isin_growth"], r["isin_reinvest"]):
                    return r
        if name:
            want = _tokens(name)
            best, score = None, 0.0
            for r in rows:
                have = _tokens(r["name"])
                if not have:
                    continue
                s = len(want & have) / len(want | have)
                if s > score:
                    best, score = r, s
            if best and score >= 0.6:
                return best
        return None


def _tokens(name: str) -> set[str]:
    stop = {"FUND", "THE", "OF", "PLAN", "OPTION", "-", "&", "AND", "SCHEME"}
    return {t for t in re.split(r"[^A-Z0-9]+", name.upper()) if t and t not in stop}


def parse_amfi(text: str) -> list[dict[str, str]]:
    """AMFI NAVAll.txt -> rows with scheme code, ISINs, name and the category heading above it."""
    rows: list[dict[str, str]] = []
    category = ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ";" not in line:
            m = re.match(r".*Schemes?\s*\((.*)\)\s*$", line)
            if m:
                category = m.group(1).strip()
            continue
        parts = line.split(";")
        if not parts[0].isdigit() or len(parts) < 6:
            continue
        name = parts[3]
        if len(parts) >= 8:          # newer layout: name;plan;option;nav;date
            name = f"{parts[3]} - {parts[4]} - {parts[5]}"
        rows.append({"code": parts[0], "isin_growth": parts[1], "isin_reinvest": parts[2],
                     "name": name, "category": category, "nav": parts[-2], "date": parts[-1]})
    return rows


def resolve(raw: list[RawHolding], resolver: Resolver) -> list[Holding]:
    """Find the instrument behind each imported row and classify it."""
    from learntrade.quant.trust import is_company

    out: list[Holding] = []
    for r in raw:
        name = r.name or r.symbol or r.isin or "?"
        h = Holding(name=name, quantity=float(r.quantity or 0), avg_price=r.avg_price,
                    isin=r.isin, sector=r.sector, category=r.category,
                    file_price=r.price or (r.value / r.quantity if r.value and r.quantity else None))
        sym = r.symbol if r.symbol and " " not in r.symbol else None
        listed = None
        fund = None
        if r.isin and r.isin.startswith("INF") or (r.name and _has(r.name, "FUND", "PLAN", "SCHEME")
                                                  and not sym):
            fund = resolver.find_fund(r.isin, r.name or r.symbol)
        if fund is None or "ETF" in (fund.get("category", "") + fund.get("name", "")).upper():
            if r.isin and r.isin.startswith("IN"):
                listed = resolver.find_listed(r.isin)
            if listed is None and sym:
                listed = resolver.find_listed(sym, exact_symbol=sym)
            if listed is None and r.name and not (r.isin or "").startswith("INF"):
                listed = resolver.find_listed(r.name)
        if listed is not None:
            ysym = listed["symbol"]
            mic = "XNSE" if ysym.endswith(".NS") else "XBOM"
            h.id = f"{mic}:{ysym.rsplit('.', 1)[0]}"
            full = listed.get("longname") or listed.get("shortname") or name
            h.name = full if not r.name else r.name
            if listed.get("quoteType") == "ETF" or not is_company(None, h.id, full):
                h.kind, h.asset_class = "etf", etf_class(full)
            else:
                h.kind, h.asset_class = "stock", "Indian stocks"
                h.sector = h.sector or listed.get("sector")
        elif fund is not None:
            h.id, h.kind = f"AMFI:{fund['code']}", "fund"
            h.category = fund.get("category") or h.category
            h.name = r.name or fund["name"]
            h.asset_class = fund_class(h.category, fund["name"])
            if "ETF" in (h.category or "").upper():
                h.kind = "etf"
        else:
            h.note = "Could not find this on NSE/BSE or in the AMFI fund list."
        out.append(h)
    return out
