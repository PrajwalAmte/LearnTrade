"""
Fundamental features: valuation, quality and growth, computed point-in-time.

For every price date, the statement used is the latest one that was already public on that
date (``available_at <= date``), so a number never reaches the model before the market could
have known it. Valuation ratios use that day's price, so they move daily while the
statement stays fixed until the next one is published.

Raw features (before the cross-sectional normalisation applied in price_volume):

  earnings_yield   EPS / price                   (inverse P/E; negative when loss-making)
  book_to_price    book value per share / price  (inverse P/B)
  sales_to_price   sales per share / price       (inverse P/S)
  roe              net income / equity           (quality: profitability)
  gross_margin     gross profit / revenue        (quality)
  cfo_to_assets    operating cash flow / assets  (quality: cash earnings)
  debt_to_assets   debt / assets                 (balance-sheet risk)
  rev_growth       revenue growth vs the prior annual statement
  earn_growth      net-income growth vs the prior annual statement
  log_market_value log(shares x price): company size, so small and large companies are not
                   treated alike (small caps behave differently)
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

FUNDAMENTAL_FEATURES = [
    "earnings_yield", "book_to_price", "sales_to_price",
    "roe", "gross_margin", "cfo_to_assets", "debt_to_assets",
    "rev_growth", "earn_growth", "log_market_value",
]

_ANNUAL_GAP_DAYS = (300, 430)


def _naive(col: str) -> pl.Expr:
    """Drop the timezone and fix microsecond precision so asof joins line up."""
    return pl.col(col).dt.replace_time_zone(None).cast(pl.Datetime("us"))


def _safe_ratio(num: str, den: str) -> pl.Expr:
    return pl.when(pl.col(den).abs() > 0).then(pl.col(num) / pl.col(den)).otherwise(None)


def _growth(col: str) -> pl.Expr:
    prev = pl.col(col).shift(1).over("instrument_id")
    gap = (pl.col("event_time") - pl.col("event_time").shift(1).over("instrument_id")).dt.total_days()
    ok = gap.is_between(*_ANNUAL_GAP_DAYS) & (prev.abs() > 0)
    return pl.when(ok).then((pl.col(col) - prev) / prev.abs()).otherwise(None).clip(-1.0, 3.0)


def _split_factors(stmts: pl.DataFrame, splits: pl.DataFrame | None) -> list[float]:
    """
    Stock splits that happened AFTER each statement was published (product of ratios).

    Prices are adjusted for every later split but a statement's per-share figures are as they
    were when filed, so EPS must be divided (and share counts multiplied) by this factor to
    be in the same units as the price. Only SEC statements need it (Yahoo's are already
    adjusted). This is a change of units, not information from the future.
    """
    if splits is None or splits.is_empty():
        return [1.0] * len(stmts)
    by_inst: dict[str, list[tuple]] = {}
    for r in splits.iter_rows(named=True):
        by_inst.setdefault(str(r["instrument_id"]), []).append(
            (r["event_time"].replace(tzinfo=None), float(r["ratio"])))
    out: list[float] = []
    for r in stmts.iter_rows(named=True):
        f = 1.0
        if r.get("source_id") == "sec_edgar":
            avail = r["available_at"].replace(tzinfo=None)
            for when, ratio in by_inst.get(str(r["instrument_id"]), []):
                if when > avail and ratio > 0:
                    f *= ratio
        out.append(f)
    return out


def statement_ratios(stmts: pl.DataFrame, splits: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per-statement ratios that do not need a price. Input from PITStore.query_fundamentals."""
    if stmts.is_empty():
        return stmts
    stmts = stmts.sort(["instrument_id", "event_time"])
    stmts = stmts.with_columns(pl.Series("_split", _split_factors(stmts, splits), dtype=pl.Float64))
    s = stmts.with_columns([
        (pl.col("shares") * pl.col("_split")).alias("_sh"),
        (pl.col("eps") / pl.col("_split")).alias("eps"),
    ]).with_columns(
        [_safe_ratio("equity", "_sh").alias("bvps"), _safe_ratio("revenue", "_sh").alias("sps")]
    )
    s = s.with_columns([
        _safe_ratio("net_income", "equity").clip(-1.0, 1.0).alias("roe"),
        _safe_ratio("gross_profit", "revenue").clip(-1.0, 1.0).alias("gross_margin"),
        _safe_ratio("cfo", "assets").clip(-1.0, 1.0).alias("cfo_to_assets"),
        _safe_ratio("debt", "assets").clip(0.0, 2.0).alias("debt_to_assets"),
        _growth("revenue").alias("rev_growth"),
        _growth("net_income").alias("earn_growth"),
    ])
    return s


def attach_fundamentals(
    rows: pl.DataFrame, stmts: pl.DataFrame, date_col: str, price_col: str = "close",
    splits: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Add FUNDAMENTAL_FEATURES to ``rows`` (needs instrument_id, ``date_col`` and ``price_col``).

    Rows with no public statement yet get null features. The caller normalises them across
    stocks and treats null as neutral.
    """
    empty = {f: pl.Series(f, [None] * len(rows), dtype=pl.Float64) for f in FUNDAMENTAL_FEATURES}
    if rows.is_empty() or stmts.is_empty():
        return rows.with_columns(list(empty.values()))

    s = statement_ratios(stmts, splits).select(
        ["instrument_id", "available_at", "eps", "bvps", "sps", "_sh",
         "roe", "gross_margin", "cfo_to_assets", "debt_to_assets", "rev_growth", "earn_growth"]
    ).with_columns(_naive("available_at").alias("_avail")).drop("available_at").sort("_avail")

    left = rows.with_columns(_naive(date_col).alias("_d")).sort("_d")
    joined = left.join_asof(s, left_on="_d", right_on="_avail", by="instrument_id", strategy="backward",
                             check_sortedness=False)
    px = pl.col(price_col).cast(pl.Float64, strict=False)
    px_ok = pl.when(px > 0)
    joined = joined.with_columns([
        px_ok.then(pl.col("eps") / px).otherwise(None).clip(-1.0, 1.0).alias("earnings_yield"),
        px_ok.then(pl.col("bvps") / px).otherwise(None).clip(-5.0, 5.0).alias("book_to_price"),
        px_ok.then(pl.col("sps") / px).otherwise(None).clip(0.0, 20.0).alias("sales_to_price"),
        pl.when((px > 0) & (pl.col("_sh") > 0)).then((pl.col("_sh") * px).log())
        .otherwise(None).alias("log_market_value"),
    ])
    return joined.drop(["_d", "_avail", "eps", "bvps", "sps", "_sh"])


def snapshot(store, instrument_id: str, price: float | None, as_of: datetime) -> dict:
    """Readable valuation/quality numbers for one stock (for the detail page). Empty if unknown."""
    st = store.query_fundamentals(as_of, None)
    if st.is_empty():
        return {}
    st = st.filter(pl.col("instrument_id") == str(instrument_id))
    if st.is_empty():
        return {}
    r = statement_ratios(st, store.query_splits()).sort("event_time").tail(1).row(0, named=True)
    out: dict = {"as_of_period": r["event_time"].date().isoformat() if r.get("event_time") else None}
    if price and price > 0:
        if r.get("eps") and r["eps"] > 0:
            out["pe"] = round(price / r["eps"], 1)
        if r.get("bvps") and r["bvps"] > 0:
            out["pb"] = round(price / r["bvps"], 2)
    for k in ("roe", "gross_margin", "rev_growth", "earn_growth", "debt_to_assets"):
        v = r.get(k)
        if v is not None:
            out[k] = round(float(v), 3)
    return out
