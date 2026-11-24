"""
Price / volume feature computation.

All features are computed from adjusted close prices and volume.
The function respects PIT because bars are fetched with available_at <= as_of
before being passed here — no data beyond as_of is ever read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from learntrade.data.store.pit import PITStore

FEATURE_SET_VERSION = "price_volume_v2_cs"
FEATURE_SET_VERSION_FUND = "price_fund_v4_cs"   # v4: + company size

_MIN_FUNDAMENTAL_STOCKS = 5   # need a cross-section for the normalisation to mean anything


def compute_price_volume_features(
    bars: pl.DataFrame,
    return_windows: list[int] | None = None,
    mom_windows: list[int] | None = None,
    vol_window: int = 20,
) -> pl.DataFrame:
    """
    Compute price/volume features from OHLCV bars in-place.

    Input columns required: instrument_id, event_time, close, volume
    Optional: adj_close (falls back to close if absent)
    Returns the same DataFrame with feature columns appended.
    """
    return_windows = return_windows or [1, 5, 20, 60]
    mom_windows = mom_windows or [5, 20]

    bars = bars.sort(["instrument_id", "event_time"])

    # Use adjusted close if available; fall back to close
    has_adj = "adj_close" in bars.columns
    close_expr = (
        pl.when(pl.col("adj_close").is_not_null()).then(pl.col("adj_close")).otherwise(pl.col("close"))
        if has_adj
        else pl.col("close")
    )

    exprs: list[pl.Expr] = []

    # Log returns over multiple windows
    for w in return_windows:
        exprs.append(
            (close_expr / close_expr.shift(w).over("instrument_id"))
            .log()
            .alias(f"ret_{w}d")
        )

    # Momentum: return from t-w-1 to t-1 (excludes most recent bar — avoids short-term reversal)
    for w in mom_windows:
        exprs.append(
            (
                close_expr.shift(1).over("instrument_id")
                / close_expr.shift(w + 1).over("instrument_id")
            )
            .log()
            .alias(f"mom_{w}d")
        )

    # EWMA volatility (annualised from daily log returns)
    log_ret = (close_expr / close_expr.shift(1).over("instrument_id")).log()
    exprs.append(
        log_ret.ewm_std(span=vol_window, adjust=True)
        .over("instrument_id")
        .mul(math.sqrt(252))
        .alias(f"ewma_vol_{vol_window}d")
    )

    # Volume surprise: log(vol / ewma_vol). Clip to >= 1 before logging so a
    # zero-volume bar (illiquid stock, holiday half-day) produces 0 instead of
    # -inf — an infinite feature value crashes every downstream model (ridge,
    # LightGBM) that isn't expecting it.
    vol_f = pl.col("volume").cast(pl.Float64, strict=False)
    vol_f_safe = vol_f.clip(lower_bound=1.0)
    exprs.append(
        (vol_f_safe.log() - vol_f_safe.ewm_mean(span=vol_window).over("instrument_id").log())
        .alias(f"vol_surprise_{vol_window}d")
    )

    # Amihud illiquidity proxy: |1-day return| / volume (rolling mean over vol_window)
    amihud_raw = log_ret.abs() / vol_f.clip(lower_bound=1.0)
    exprs.append(
        amihud_raw.rolling_mean(window_size=vol_window).over("instrument_id")
        .alias(f"amihud_{vol_window}d")
    )

    # 52-week high ratio (momentum at the slowest horizon)
    exprs.append(
        (close_expr / close_expr.rolling_max(window_size=252).over("instrument_id"))
        .alias("high_52w_ratio")
    )

    return bars.with_columns(exprs)


def feature_names(
    return_windows: list[int] | None = None,
    mom_windows: list[int] | None = None,
    vol_window: int = 20,
) -> list[str]:
    """Return the feature column names produced by compute_price_volume_features."""
    return_windows = return_windows or [1, 5, 20, 60]
    mom_windows = mom_windows or [5, 20]
    names = [f"ret_{w}d" for w in return_windows]
    names += [f"mom_{w}d" for w in mom_windows]
    names += [f"ewma_vol_{vol_window}d", f"vol_surprise_{vol_window}d",
              f"amihud_{vol_window}d", "high_52w_ratio"]
    return names


_MIN_NAMES_FOR_ZSCORE = 5


def add_cross_sectional(df: pl.DataFrame, cols: list[str], date_col: str | None) -> pl.DataFrame:
    """
    Append ``{col}_cs``: each feature z-scored across stocks on the same date (clipped to +/-5).

    The model then learns "stronger than the other stocks TODAY" instead of raw values that
    drift with the market regime. With fewer than 5 stocks on a date there is no meaningful
    cross-section, so the value is 0 (neutral) rather than noise.
    """
    exprs = []
    for c in cols:
        mean, std, cnt = pl.col(c).mean(), pl.col(c).std(), pl.col(c).count()
        if date_col:
            mean, std, cnt = mean.over(date_col), std.over(date_col), cnt.over(date_col)
        z = ((pl.col(c) - mean) / (std + 1e-8)).clip(-5.0, 5.0)
        exprs.append(
            pl.when(cnt >= _MIN_NAMES_FOR_ZSCORE).then(z).otherwise(0.0)
            .fill_nan(0.0).fill_null(0.0).alias(f"{c}_cs")   # missing value = neutral
        )
    return df.with_columns(exprs)


@dataclass
class PriceVolumeFeatureSet:
    """
    Versioned feature set for price/volume features.
    Implements the FeatureSet protocol.
    """

    id: str = FEATURE_SET_VERSION
    version: str = "1"
    requires: list[str] = field(default_factory=lambda: ["bars_daily"])
    return_windows: list[int] = field(default_factory=lambda: [1, 5, 20, 60])
    mom_windows: list[int] = field(default_factory=lambda: [5, 20])
    vol_window: int = 20
    cross_sectional: bool = True
    fundamentals: bool = False   # add valuation / quality / growth (see features.fundamentals)

    def __post_init__(self) -> None:
        if self.fundamentals and self.id == FEATURE_SET_VERSION:
            self.id = FEATURE_SET_VERSION_FUND
            self.requires = self.requires + ["fundamentals"]

    @property
    def fundamental_names(self) -> list[str]:
        from learntrade.features.fundamentals import FUNDAMENTAL_FEATURES
        return list(FUNDAMENTAL_FEATURES) if self.fundamentals else []

    @property
    def raw_feature_names(self) -> list[str]:
        """The plain indicators (ret_20d, ...), kept in the output for display and fallbacks."""
        return feature_names(self.return_windows, self.mom_windows, self.vol_window)

    @property
    def feature_names(self) -> list[str]:
        """What the models are trained on: cross-sectionally normalised indicators, plus the
        raw volatility level (used to scale each stock's forecast range)."""
        raw = self.raw_feature_names
        if not self.cross_sectional:
            return raw + self.fundamental_names
        return ([f"{c}_cs" for c in raw] + [f"ewma_vol_{self.vol_window}d"]
                + [f"{c}_cs" for c in self.fundamental_names])

    def _out_cols(self) -> list[str]:
        raw = self.raw_feature_names + self.fundamental_names
        return raw + ([f"{c}_cs" for c in raw] if self.cross_sectional else [])

    def _with_fundamentals(
        self, df: pl.DataFrame, store: "PITStore", as_of: datetime, date_col: str | None,
    ) -> pl.DataFrame:
        """Join valuation/quality/growth features onto rows that carry ``close``."""
        from learntrade.features.fundamentals import FUNDAMENTAL_FEATURES, attach_fundamentals
        stmts = store.query_fundamentals(as_of, None)
        splits = store.query_splits()
        if date_col is None:     # inference: price date is the latest bar
            df = df.with_columns(pl.lit(as_of).alias("_asof_date"))
            df = attach_fundamentals(df, stmts, "_asof_date", splits=splits).drop("_asof_date")
        else:
            df = attach_fundamentals(df, stmts, date_col, splits=splits)
        return add_cross_sectional(df, FUNDAMENTAL_FEATURES, date_col) if self.cross_sectional else df

    def compute(
        self,
        instruments: list,
        as_of: datetime,
        store: "PITStore",
        lookback_days: int = 400,
    ) -> pl.DataFrame:
        """Return the most recent feature row per instrument (inference)."""
        since = as_of - timedelta(days=lookback_days)
        bars = store.query_bars(instruments, as_of=as_of, since=since)
        if bars.is_empty():
            return pl.DataFrame()
        bars = _cast_ohlcv(bars)
        featured = compute_price_volume_features(
            bars, self.return_windows, self.mom_windows, self.vol_window
        )
        latest = (
            featured.sort(["instrument_id", "event_time"])
            .group_by("instrument_id")
            .last()
            .select(["instrument_id", "close"] + self.raw_feature_names)
        )
        if self.fundamentals:
            latest = self._with_fundamentals(latest, store, as_of, None)
        if self.cross_sectional:
            latest = add_cross_sectional(latest, self.raw_feature_names, None)
        return latest.select(["instrument_id"] + self._out_cols())

    def compute_history(
        self,
        instruments: list,
        since: datetime,
        until: datetime,
        store: "PITStore",
        lookback_days: int = 400,
    ) -> pl.DataFrame:
        """Return feature rows for all dates in [since, until] (training)."""
        fetch_since = since - timedelta(days=lookback_days)
        bars = store.query_bars(instruments, as_of=until, since=fetch_since)
        if bars.is_empty():
            return pl.DataFrame()
        bars = _cast_ohlcv(bars)
        featured = compute_price_volume_features(
            bars, self.return_windows, self.mom_windows, self.vol_window
        )
        required = self.raw_feature_names[:3]  # need at least ret_1d, ret_5d, ret_20d
        hist = (
            featured
            .filter(
                (pl.col("event_time") >= since) & (pl.col("event_time") <= until)
            )
            .select(["instrument_id", "event_time", "close"] + self.raw_feature_names)
            .drop_nulls(subset=[c for c in required if c in featured.columns])
        )
        if self.fundamentals:
            hist = self._with_fundamentals(hist, store, until, "event_time")
        if self.cross_sectional:
            hist = add_cross_sectional(hist, self.raw_feature_names, "event_time")
        return hist.select(["instrument_id", "event_time"] + self._out_cols())


def _cast_ohlcv(bars: pl.DataFrame) -> pl.DataFrame:
    casts = {c: pl.Float64 for c in ["open", "high", "low", "close", "volume", "adj_close"]
             if c in bars.columns}
    return bars.with_columns([pl.col(c).cast(t, strict=False) for c, t in casts.items()])


def has_fundamentals(store: "PITStore", market: str | None = None) -> bool:
    """True once enough stocks OF THIS MARKET have statements for the features to mean anything.

    Per market: financials stored for one market must not change another market's features
    (that would invalidate its trained model and add all-zero columns)."""
    try:
        return store.count_fundamental_stocks(market) >= _MIN_FUNDAMENTAL_STOCKS
    except Exception:
        return False


def default_feature_set(store: "PITStore", market: str | None = None) -> PriceVolumeFeatureSet:
    """The feature set for a market: fundamentals are included when that market has them."""
    return PriceVolumeFeatureSet(fundamentals=has_fundamentals(store, market))
