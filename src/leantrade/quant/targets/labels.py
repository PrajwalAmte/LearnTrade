"""
Forward return label generation.

Labels are only valid when the full forward window has elapsed.
label_available_at marks when the return becomes observable —
this enforces PIT correctness when training.
"""

from __future__ import annotations

import polars as pl


def compute_forward_returns(
    bars: pl.DataFrame,
    forward_days: int,
    benchmark_col: str | None = None,
) -> pl.DataFrame:
    """
    Compute forward (excess) log returns.

    Args:
        bars:          DataFrame with [instrument_id, event_time, close, (adj_close)]
        forward_days:  number of trading bars to look forward
        benchmark_col: if set, subtract this per-date column to compute excess returns

    Returns:
        DataFrame [instrument_id, event_time, label, label_available_at]
        label_available_at = event_time of the bar at t + forward_days
    """
    if forward_days <= 0:
        raise ValueError(f"forward_days must be positive, got {forward_days}")

    bars = bars.sort(["instrument_id", "event_time"])
    close_col = "adj_close" if "adj_close" in bars.columns else "close"

    bars = bars.with_columns(pl.col(close_col).cast(pl.Float64, strict=False))

    bars = bars.with_columns([
        pl.col(close_col).shift(-forward_days).over("instrument_id").alias("_future_close"),
        pl.col("event_time").shift(-forward_days).over("instrument_id").alias("label_available_at"),
    ])

    bars = bars.with_columns(
        (pl.col("_future_close") / pl.col(close_col)).log().alias("_raw_return")
    )

    if benchmark_col and benchmark_col in bars.columns:
        bars = bars.with_columns(
            (pl.col("_raw_return") - pl.col(benchmark_col)).alias("label")
        )
    else:
        bars = bars.with_columns(pl.col("_raw_return").alias("label"))

    return (
        bars.select(["instrument_id", "event_time", "label", "label_available_at"])
        .drop_nulls(["label", "label_available_at"])
    )


def compute_benchmark_returns(
    bars: pl.DataFrame,
    benchmark_id: str,
    forward_days: int,
) -> pl.DataFrame:
    """
    Extract forward returns for a single benchmark instrument,
    to be joined with universe bars for excess-return computation.
    """
    bench = bars.filter(pl.col("instrument_id") == benchmark_id)
    if bench.is_empty():
        return pl.DataFrame()
    close_col = "adj_close" if "adj_close" in bench.columns else "close"
    bench = bench.sort("event_time")
    bench = bench.with_columns(pl.col(close_col).cast(pl.Float64, strict=False))
    bench = bench.with_columns(
        (pl.col(close_col).shift(-forward_days) / pl.col(close_col)).log()
        .alias("benchmark_return")
    )
    return bench.select(["event_time", "benchmark_return"]).drop_nulls()


def merge_features_labels(
    features: pl.DataFrame,
    labels: pl.DataFrame,
    date_col: str = "event_time",
) -> pl.DataFrame:
    """
    Inner-join features and labels on (instrument_id, date).
    Only returns rows where both are available and the label is not null.
    """
    return (
        features.join(labels, on=["instrument_id", date_col], how="inner")
        .drop_nulls(subset=["label"])
    )
