"""
Data quality check tests.
"""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from leantrade.data.quality.checks import check_duplicates, check_leakage, run_all_checks


def _make_df(available_at_dates: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument_id": ["XNYS:IBM"] * len(available_at_dates),
            "source_id": ["yahoo"] * len(available_at_dates),
            "event_time": available_at_dates,
            "available_at": available_at_dates,
        }
    )


def test_no_leakage_passes():
    as_of = datetime(2026, 1, 10, tzinfo=timezone.utc)
    df = _make_df([datetime(2026, 1, 1, tzinfo=timezone.utc)])
    assert check_leakage(df, as_of) == []


def test_leakage_detected():
    as_of = datetime(2026, 1, 1, tzinfo=timezone.utc)
    df = _make_df([datetime(2026, 1, 5, tzinfo=timezone.utc)])  # future
    issues = check_leakage(df, as_of)
    assert len(issues) == 1
    assert issues[0]["check"] == "leakage"


def test_run_all_checks_no_issues():
    as_of = datetime(2026, 1, 10, tzinfo=timezone.utc)
    df = _make_df([datetime(2026, 1, 1, tzinfo=timezone.utc)])
    report = run_all_checks(df, as_of, snapshot_id="test")
    assert report.passed
