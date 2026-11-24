"""
The user's saved holdings and the prices behind them.

Holdings are kept in ``<data>/portfolio/holdings.json`` (one portfolio; a new import replaces
it). Prices go into the point-in-time store like any other data: exchange prices for stocks and
ETFs from Yahoo Finance, daily NAVs for mutual funds from mfapi.in.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from learntrade.core.types import InstrumentId, Observation
from learntrade.portfolio.resolve import Holding

logger = logging.getLogger(__name__)
UTC = timezone.utc
_MFAPI = "https://api.mfapi.in/mf/{code}"


def portfolio_dir(data_dir: Path) -> Path:
    d = Path(data_dir) / "portfolio"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_holdings(data_dir: Path, holdings: list[Holding], source: str) -> Path:
    p = portfolio_dir(data_dir) / "holdings.json"
    p.write_text(json.dumps({
        "imported_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "source": source,
        "holdings": [h.to_dict() for h in holdings],
    }, indent=2))
    return p


def load_holdings(data_dir: Path) -> tuple[list[Holding], dict[str, Any]]:
    p = Path(data_dir) / "portfolio" / "holdings.json"
    if not p.exists():
        return [], {}
    d = json.loads(p.read_text())
    return [Holding.from_dict(h) for h in d.get("holdings", [])], {
        "imported_at": d.get("imported_at"), "source": d.get("source")}


def parse_mfapi(code: str, data: dict) -> list[Observation]:
    """mfapi.in scheme history -> one 'bar' per day with the NAV as the close."""
    out: list[Observation] = []
    iid = InstrumentId(f"AMFI:{code}")
    now = datetime.now(tz=UTC)
    for row in (data or {}).get("data") or []:
        try:
            day = datetime.strptime(row["date"], "%d-%m-%Y").replace(tzinfo=UTC)
            nav = float(row["nav"])
        except (KeyError, ValueError, TypeError):
            continue
        if nav <= 0:
            continue
        out.append(Observation(instrument_id=iid, source_id="mfapi", kind="bar", event_time=day,
                               available_at=now, pit_level="none",
                               payload={"close": nav, "adj_close": nav, "open": nav, "high": nav,
                                        "low": nav, "volume": 0}))
    return out


def fetch_prices(store: Any, holdings: list[Holding], days: int = 420) -> dict[str, str]:
    """Download recent prices/NAVs for every resolved holding. Returns {id: problem} for failures."""
    from learntrade.data.providers.base import FetchRequest
    from learntrade.data.providers.yahoo import YahooProvider

    problems: dict[str, str] = {}
    now = datetime.now(tz=UTC)
    yp = YahooProvider()
    listed = sorted({h.id for h in holdings if h.id and not h.id.startswith("AMFI:")})
    for iid in listed:
        try:
            obs = list(yp.fetch(FetchRequest("bars_daily", now - timedelta(days=days), now,
                                             instruments={InstrumentId(iid)})))
            if not any(o.kind == "bar" for o in obs):
                problems[iid] = "no prices found"
            store.ingest(obs, replace=True)
            kind = next((h.kind for h in holdings if h.id == iid), "")
            # Splits and bonuses: a broker file made before one has the old quantity and cost.
            store.ingest(list(yp._fetch_splits(InstrumentId(iid))))
            if kind == "stock":      # share count -> market value -> size bucket
                store.ingest(list(yp._fetch_fundamentals(InstrumentId(iid))))
        except Exception as exc:
            problems[iid] = str(exc)
    for code in sorted({h.id.split(":", 1)[1] for h in holdings if h.id and h.id.startswith("AMFI:")}):
        try:
            r = httpx.get(_MFAPI.format(code=code), timeout=30)
            r.raise_for_status()
            obs = [o for o in parse_mfapi(code, r.json()) if o.event_time >= now - timedelta(days=days)]
            if not obs:
                problems[f"AMFI:{code}"] = "no NAV history found"
            store.ingest(obs, replace=True)
        except Exception as exc:
            problems[f"AMFI:{code}"] = str(exc)
    return problems


def import_portfolio(store: Any, data_dir: Path, files: list[tuple[bytes, str]],
                     offline: bool = False) -> dict[str, Any]:
    """
    Parse one or more broker files, find each holding, save them as THE portfolio, and download
    prices. Several files (e.g. a Zerodha stocks export and a Groww funds export) are combined.
    """
    from learntrade.portfolio.importer import parse_holdings
    from learntrade.portfolio.resolve import Resolver, resolve

    raw, columns, skipped = [], {}, []
    for data, name in files:
        res = parse_holdings(data, name)
        raw += res.holdings
        columns.update({f"{name} / {k}": v for k, v in res.columns.items()})
        skipped += [f"{name}: {s}" for s in res.skipped]
    if not raw:
        raise ValueError("No holdings found. The file needs a table with a quantity column and a "
                         "name, symbol or ISIN column. " + " ".join(skipped))
    holdings = resolve(raw, Resolver(portfolio_dir(data_dir) / "cache", offline=offline))
    save_holdings(data_dir, holdings, ", ".join(n for _, n in files))
    problems = {} if offline else fetch_prices(store, holdings)
    return {"holdings": [h.to_dict() for h in holdings], "columns": columns,
            "skipped": skipped, "price_problems": problems}


def refresh_prices(store: Any, data_dir: Path) -> dict[str, str]:
    holdings, _ = load_holdings(data_dir)
    return fetch_prices(store, holdings) if holdings else {}
