"""
Cross-sectional transforms — rank and z-score within each date.
"""

from __future__ import annotations

import polars as pl


def cross_sectional_rank(
    df: pl.DataFrame,
    feature_cols: list[str],
    date_col: str = "event_time",
) -> pl.DataFrame:
    """
    Append percentile-rank versions of feature_cols, computed within each date.
    Output column name: {col}_rank in [0, 1].
    """
    exprs = [
        (pl.col(c).rank().over(date_col) / pl.col(c).count().over(date_col))
        .alias(f"{c}_rank")
        for c in feature_cols
        if c in df.columns
    ]
    return df.with_columns(exprs) if exprs else df


def cross_sectional_zscore(
    df: pl.DataFrame,
    feature_cols: list[str],
    date_col: str = "event_time",
) -> pl.DataFrame:
    """
    Append cross-sectionally z-scored versions of feature_cols.
    Output column name: {col}_cs (clipped to [-5, 5] to limit outlier influence).
    """
    exprs = [
        (
            (pl.col(c) - pl.col(c).mean().over(date_col))
            / (pl.col(c).std().over(date_col) + 1e-8)
        )
        .clip(-5.0, 5.0)
        .alias(f"{c}_cs")
        for c in feature_cols
        if c in df.columns
    ]
    return df.with_columns(exprs) if exprs else df
