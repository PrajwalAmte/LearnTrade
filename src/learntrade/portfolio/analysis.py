"""
Portfolio analysis that needs no forecast: what you own, how it is spread, how it has behaved.

Everything here is measured from prices, NAVs and the holdings file:
  * value, cost and profit/loss per holding and in total
  * allocation by asset class, by sector and by company size (directly held stocks)
  * concentration: biggest holding, top five, and the "effective number of holdings"
  * risk of the CURRENT mix over the last year: volatility, worst fall, holdings that move
    together
  * plain "worth knowing" facts (a holding over 15%, a sector over 30%, ...). They describe
    the portfolio; they are not advice on what to buy or sell.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import polars as pl

from learntrade.core.types import InstrumentId
from learntrade.portfolio.resolve import Holding, size_hint

UTC = timezone.utc

# Company size by market value (INR). Approximates SEBI's ranking (top 100 large, next 150 mid);
# AMFI publishes the exact list twice a year and the cut-offs move with the market.
LARGE_CAP_INR = 1.0e12     # about ₹1 lakh crore
MID_CAP_INR = 3.3e11       # about ₹33,000 crore

HOLDING_LIMIT, SECTOR_LIMIT, SMALL_MID_LIMIT, CORR_LIMIT = 0.15, 0.30, 0.50, 0.80


def _size_bucket(market_value: float | None) -> str | None:
    if not market_value:
        return None
    return "large" if market_value >= LARGE_CAP_INR else "mid" if market_value >= MID_CAP_INR else "small"


def _price_frame(store: Any, ids: list[str], now: datetime) -> pl.DataFrame:
    if not ids:
        return pl.DataFrame()
    bars = store.query_bars([InstrumentId(i) for i in ids], as_of=now, since=now - timedelta(days=400))
    if bars.is_empty():
        return bars
    return (bars.with_columns(pl.col("close").cast(pl.Float64, strict=False),
                              pl.col("event_time").dt.date().alias("day"))
                .drop_nulls("close").filter(pl.col("close") > 0)
                .group_by(["instrument_id", "day"]).agg(pl.col("close").last())
                .sort(["instrument_id", "day"]))


def _market_values(store: Any, stock_ids: list[str], last_price: dict[str, float], now: datetime) -> dict[str, float]:
    out: dict[str, float] = {}
    if not stock_ids:
        return out
    try:
        for period in ("quarterly", "annual"):
            st = store.query_fundamentals(now, [InstrumentId(i) for i in stock_ids], period=period)
            if st.is_empty():
                continue
            last = st.drop_nulls("shares").sort("event_time").group_by("instrument_id").last()
            for row in last.iter_rows(named=True):
                iid = str(row["instrument_id"])
                if iid not in out and row["shares"] and iid in last_price:
                    out[iid] = float(row["shares"]) * last_price[iid]
    except Exception:
        pass
    return out


def _recent_splits(store: Any, ids: list[str], now: datetime) -> dict[str, list[tuple[datetime, float]]]:
    out: dict[str, list[tuple[datetime, float]]] = {}
    try:
        sp = store.query_splits()
    except Exception:
        return out
    if sp.is_empty():
        return out
    cutoff = now - timedelta(days=3 * 365)
    for r in sp.iter_rows(named=True):
        t = r["event_time"] if r["event_time"].tzinfo else r["event_time"].replace(tzinfo=UTC)
        if str(r["instrument_id"]) in ids and t >= cutoff and r["ratio"] and r["ratio"] > 1:
            out.setdefault(str(r["instrument_id"]), []).append((t, float(r["ratio"])))
    return {k: sorted(v) for k, v in out.items()}


def _split_since_file(splits: list[tuple[datetime, float]], gap: float) -> tuple[float | None, str | None]:
    """
    If the file's price is about N times today's price and N matches the splits/bonuses of the
    last 3 years (the latest one, or the latest few combined), the file predates them.
    Ordinary price moves never match, so they are left alone.
    """
    factor = 1.0
    for when, ratio in reversed(splits):
        factor *= ratio
        if abs(gap / factor - 1) < 0.20:
            return factor, when.date().isoformat()
    return None, None


def _share(groups: dict[str, float], total: float) -> list[dict[str, Any]]:
    return [{"label": k, "value": v, "weight": v / total if total else 0.0}
            for k, v in sorted(groups.items(), key=lambda kv: -kv[1])]


def analyse(store: Any, holdings: list[Holding], now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(tz=UTC)
    ids = sorted({h.id for h in holdings if h.id})
    px = _price_frame(store, ids, now)
    series: dict[str, pl.DataFrame] = (
        {str(k[0] if isinstance(k, tuple) else k): g for k, g in px.group_by("instrument_id")}
        if not px.is_empty() else {})
    last_price = {i: float(g["close"][-1]) for i, g in series.items()}
    last_day = {i: g["day"][-1].isoformat() for i, g in series.items()}
    mvals = _market_values(store, [h.id for h in holdings if h.kind == "stock" and h.id], last_price, now)
    splits = _recent_splits(store, ids, now)

    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for h in holdings:
        price = last_price.get(h.id) if h.id else None
        price_src = "market" if price is not None else None
        qty, avg = h.quantity, h.avg_price
        if price is None and h.file_price:
            price, price_src = h.file_price, "broker file"
        elif price is not None and h.file_price and h.id:
            f, when = _split_since_file(splits.get(h.id, []), h.file_price / price)
            if f:
                # The file was made before this split/bonus: put quantity and cost on today's basis.
                qty, avg = qty * f, (avg / f if avg else avg)
                notes.append(f"{h.name}: adjusted for the {f:g}-for-1 split or bonus on {when}. "
                             "Your broker file was made before it, so its quantity and cost were on "
                             "the old basis.")
        value = qty * price if price is not None else None
        invested = qty * avg if avg else None
        g = series.get(h.id) if h.id else None
        closes = g["close"].to_numpy() if g is not None else np.array([])
        r1y = vol = dd_high = None
        if len(closes) >= 60:
            rets = np.diff(np.log(closes))
            vol = float(np.std(rets, ddof=1) * math.sqrt(252))
            dd_high = float(closes[-1] / closes.max() - 1)
            if len(closes) >= 230:
                r1y = float(closes[-1] / closes[-min(len(closes), 250)] - 1)
        size = _size_bucket(mvals.get(h.id)) if h.kind == "stock" else size_hint(h.category, h.name)
        rows.append({
            "name": h.name, "id": h.id, "kind": h.kind, "asset_class": h.asset_class,
            "sector": h.sector if h.kind == "stock" else None, "size": size,
            "category": h.category, "quantity": qty, "avg_price": avg,
            "price": price, "price_source": price_src, "price_date": last_day.get(h.id) if h.id else None,
            "value": value, "invested": invested,
            "pnl": (value - invested) if value is not None and invested else None,
            "pnl_pct": (value / invested - 1) if value is not None and invested else None,
            "return_1y": r1y, "volatility": vol, "from_1y_high": dd_high,
            "market_value": mvals.get(h.id), "note": h.note,
        })
        if h.id is None:
            notes.append(f"{h.name}: not found on NSE/BSE or in the AMFI fund list, so it is valued "
                         "from the broker file only" + ("" if value is not None else " (no value in the file either)") + ".")
        elif price_src == "broker file":
            notes.append(f"{h.name}: no recent market price could be downloaded; the broker file's price is used.")


    total = sum(r["value"] for r in rows if r["value"] is not None)
    invested_total = sum(r["invested"] for r in rows if r["invested"] is not None and r["value"] is not None)
    for r in rows:
        r["weight"] = (r["value"] / total) if total and r["value"] is not None else None
    rows.sort(key=lambda r: -(r["value"] or 0))

    # Allocation ------------------------------------------------------------------------------
    by_class: dict[str, float] = defaultdict(float)
    by_sector: dict[str, float] = defaultdict(float)
    by_size: dict[str, float] = defaultdict(float)
    stock_total = 0.0
    for r in rows:
        if r["value"] is None:
            continue
        by_class[r["asset_class"]] += r["value"]
        if r["kind"] == "stock":
            stock_total += r["value"]
            by_sector[r["sector"] or "Unknown"] += r["value"]
            by_size[r["size"] or "unknown"] += r["value"]
    size_labels = {"large": "Large companies", "mid": "Mid-size companies",
                   "small": "Small companies", "unknown": "Size not known"}
    sector_rows = _share(by_sector, stock_total)
    for s in sector_rows:
        s["weight_of_total"] = s["value"] / total if total else 0.0

    # Concentration -----------------------------------------------------------------------------
    weights = np.array([r["weight"] for r in rows if r["weight"]])
    hhi = float(np.sum(weights ** 2)) if len(weights) else 0.0
    concentration = {
        "largest": {"name": rows[0]["name"], "weight": rows[0]["weight"]} if rows and rows[0]["weight"] else None,
        "top5_weight": float(np.sort(weights)[::-1][:5].sum()) if len(weights) else 0.0,
        "effective_holdings": (1.0 / hhi) if hhi else 0.0,
        "holdings": len(rows),
    }

    # Risk of the current mix ---------------------------------------------------------------------
    risk: dict[str, Any] = {}
    priced = [r for r in rows if r["id"] in series and r["weight"]]
    if len(priced) >= 1:
        wide = None
        for r in priced:
            g = series[r["id"]].select(["day", pl.col("close").alias(r["id"])])
            wide = g if wide is None else wide.join(g, on="day", how="full", coalesce=True)
        wide = wide.sort("day").filter(pl.col("day") >= (now - timedelta(days=366)).date())
        wide = wide.fill_null(strategy="forward").drop_nulls()
        if wide.height >= 60:
            cols = [r["id"] for r in priced]
            P = wide.select(cols).to_numpy()
            R = P[1:] / P[:-1] - 1
            w = np.array([r["weight"] for r in priced])
            w = w / w.sum()
            port = R @ w
            nav = np.cumprod(1 + port)
            risk = {
                "days": int(len(port)),
                "covered_weight": float(sum(r["weight"] for r in priced)),
                "volatility": float(np.std(port, ddof=1) * math.sqrt(252)),
                "return_period": float(nav[-1] - 1),
                "max_drawdown": float((nav / np.maximum.accumulate(nav) - 1).min()),
                "start": str(wide["day"][0]), "end": str(wide["day"][-1]),
            }
            pairs = []
            if len(cols) >= 2:
                C = np.corrcoef(R.T)
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        if C[i, j] >= CORR_LIMIT:
                            pairs.append({"a": priced[i]["name"], "b": priced[j]["name"],
                                          "correlation": float(C[i, j])})
            risk["moving_together"] = sorted(pairs, key=lambda p: -p["correlation"])[:10]

    # Worth knowing (facts, not advice) ------------------------------------------------------------
    facts: list[str] = []
    for r in rows:
        if r["weight"] and r["weight"] > HOLDING_LIMIT:
            facts.append(f"{r['name']} is {r['weight']:.0%} of the portfolio; a 20% fall in it alone "
                         f"would take about {0.2 * r['weight']:.0%} off the total.")
    for s in sector_rows:
        if s["label"] != "Unknown" and s["weight"] > SECTOR_LIMIT and stock_total:
            facts.append(f"{s['label']} is {s['weight']:.0%} of your directly held stocks "
                         f"({s['weight_of_total']:.0%} of everything).")
    if stock_total:
        sm = (by_size.get("small", 0) + by_size.get("mid", 0)) / stock_total
        if sm > SMALL_MID_LIMIT:
            facts.append(f"Small and mid-size companies are {sm:.0%} of your directly held stocks. "
                         "They usually swing more than large ones, in both directions.")
    metals = by_class.get("Gold", 0) + by_class.get("Silver", 0)
    if total and metals / total > 0.05:
        facts.append(f"Gold and silver are {metals / total:.0%} of the portfolio.")
    for r in rows:
        if r["pnl_pct"] is not None and r["pnl_pct"] <= -0.20:
            facts.append(f"{r['name']} is {-r['pnl_pct']:.0%} below what you paid.")
    for p in risk.get("moving_together", [])[:3]:
        facts.append(f"{p['a']} and {p['b']} have moved almost in step over the last year "
                     f"(correlation {p['correlation']:.2f}), so they add less spread than two holdings usually do.")
    if any(r["kind"] == "fund" for r in rows):
        notes.append("Mutual funds hold many stocks of their own. LearnTrade cannot see inside them yet, "
                     "so the sector and company-size figures cover directly held stocks only.")
    if stock_total and by_size.get("unknown"):
        notes.append("Company size is unknown for some stocks (no share count available).")
    notes.append("Company size uses market value: about ₹1 lakh crore and above = large, ₹33,000 crore "
                 "and above = mid, below that = small (approximately SEBI's top-100 / next-150 split).")

    return {
        "as_of": now.isoformat(timespec="seconds"), "currency": "INR",
        "totals": {
            "value": total, "invested": invested_total or None,
            "pnl": (total - invested_total) if invested_total else None,
            "pnl_pct": (total / invested_total - 1) if invested_total else None,
            "holdings": len(rows), "priced_from_market": sum(1 for r in rows if r["price_source"] == "market"),
        },
        "holdings": rows,
        "allocation": {
            "asset_class": _share(by_class, total),
            "sector": sector_rows,
            "size": [{**x, "label": size_labels.get(x["label"], x["label"])} for x in _share(by_size, stock_total)],
        },
        "concentration": concentration,
        "risk": risk,
        "worth_knowing": facts,
        "notes": notes,
    }
