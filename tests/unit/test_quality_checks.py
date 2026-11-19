"""
Comprehensive quality check tests — all 7 checks.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from learntrade.data.quality.checks import (
    check_corporate_actions,
    check_duplicates,
    check_leakage,
    check_missing_bars,
    check_outliers,
    check_stale_prices,
    check_survivorship,
    check_timezone_consistency,
    run_all_checks,
)

UTC = timezone.utc


def _bars_df(dates_closes: list[tuple[datetime, float]], available_at_offset_days: int = 0) -> pl.DataFrame:
    return pl.DataFrame({
        "instrument_id": ["XNYS:IBM"] * len(dates_closes),
        "source_id": ["test"] * len(dates_closes),
        "event_time": [d for d, _ in dates_closes],
        "available_at": [d + timedelta(days=available_at_offset_days) for d, _ in dates_closes],
        "close": [c for _, c in dates_closes],
        "open": [c for _, c in dates_closes],
        "high": [c for _, c in dates_closes],
        "low": [c for _, c in dates_closes],
        "volume": [1000] * len(dates_closes),
    })


# ── Check 1: Leakage ─────────────────────────────────────────────────────────

def test_leakage_none() -> None:
    as_of = datetime(2024, 6, 1, tzinfo=UTC)
    df = _bars_df([(datetime(2024, 1, d + 1, tzinfo=UTC), 100.0) for d in range(5)])
    assert check_leakage(df, as_of) == []


def test_leakage_detected() -> None:
    as_of = datetime(2024, 1, 1, tzinfo=UTC)
    # available_at is 1 day after event_time, putting it after as_of
    df = _bars_df([(datetime(2024, 1, 2, tzinfo=UTC), 100.0)], available_at_offset_days=1)
    issues = check_leakage(df, as_of)
    assert len(issues) == 1
    assert issues[0]["severity"] == "error"
    assert issues[0]["count"] == 1


def test_leakage_partial() -> None:
    as_of = datetime(2024, 1, 3, tzinfo=UTC)
    df = pl.DataFrame({
        "instrument_id": ["XNYS:IBM", "XNYS:IBM"],
        "source_id": ["t", "t"],
        "event_time": [datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)],
        "available_at": [datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 5, tzinfo=UTC)],
        "close": [100.0, 101.0],
    })
    issues = check_leakage(df, as_of)
    assert issues[0]["count"] == 1


# ── Check 2: Missing bars ─────────────────────────────────────────────────────

def test_missing_bars_no_calendar_graceful() -> None:
    # With an invalid MIC, check should return a warning, not raise
    df = _bars_df([(datetime(2024, 1, 2, tzinfo=UTC), 100.0)])
    issues = check_missing_bars(df, "XXXX", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 5, tzinfo=UTC))
    # Should return a warning, not crash
    assert all(i["severity"] in {"error", "warning"} for i in issues)


# ── Check 3: Duplicates ───────────────────────────────────────────────────────

def test_duplicates_none() -> None:
    df = _bars_df([(datetime(2024, 1, d + 1, tzinfo=UTC), 100.0) for d in range(5)])
    assert check_duplicates(df) == []


def test_duplicates_detected() -> None:
    dt = datetime(2024, 1, 2, tzinfo=UTC)
    df = pl.DataFrame({
        "instrument_id": ["XNYS:IBM", "XNYS:IBM"],
        "event_time": [dt, dt],
        "source_id": ["test", "test"],
        "close": [100.0, 100.0],
    })
    issues = check_duplicates(df)
    assert len(issues) == 1
    assert issues[0]["severity"] == "warning"


# ── Check 4: Timezone consistency ─────────────────────────────────────────────

def test_timezone_weekdays_pass() -> None:
    # 2024-01-02=Tue, 01-03=Wed, 01-04=Thu, 01-05=Fri — all weekdays
    df = _bars_df([(datetime(2024, 1, d + 2, tzinfo=UTC), 100.0) for d in range(4)])
    issues = check_timezone_consistency(df, "XNYS")
    assert issues == []


def test_timezone_weekend_flagged() -> None:
    # 2024-01-06 is Saturday (weekday 5)
    df = _bars_df([(datetime(2024, 1, 6, tzinfo=UTC), 100.0)])
    issues = check_timezone_consistency(df, "XNYS")
    assert len(issues) == 1


# ── Check 5: Corporate actions ────────────────────────────────────────────────

def test_corp_actions_clean() -> None:
    dates = [datetime(2024, 1, d + 2, tzinfo=UTC) for d in range(5)]
    df = _bars_df(list(zip(dates, [100.0, 101.0, 100.5, 102.0, 101.5])))
    assert check_corporate_actions(df, max_ratio=5.0) == []


def test_corp_actions_flagged() -> None:
    dates = [datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC)]
    # 2000% jump — clearly unadjusted split
    df = _bars_df(list(zip(dates, [100.0, 2100.0])))
    issues = check_corporate_actions(df, max_ratio=10.0)
    assert len(issues) == 1


# ── Check 6: Survivorship ─────────────────────────────────────────────────────

def test_survivorship_no_delisted_warns() -> None:
    issues = check_survivorship(
        universe_instruments=["XNYS:IBM", "XNYS:AAPL"],
        all_known_instruments=["XNYS:IBM", "XNYS:AAPL"],
    )
    assert len(issues) == 1
    assert issues[0]["severity"] == "warning"


def test_survivorship_with_delisted_passes() -> None:
    issues = check_survivorship(
        universe_instruments=["XNYS:IBM", "XNYS:AAPL"],
        all_known_instruments=["XNYS:IBM", "XNYS:AAPL", "XNYS:OLDCO"],  # OLDCO is delisted
    )
    assert issues == []


# ── Check 7a: Outliers ────────────────────────────────────────────────────────

def test_outliers_clean() -> None:
    dates = [datetime(2024, 1, d + 2, tzinfo=UTC) for d in range(20)]
    prices = [100.0 + i * 0.5 for i in range(20)]
    df = _bars_df(list(zip(dates, prices)))
    assert check_outliers(df) == []


def test_outliers_flagged() -> None:
    # 60 normal days then one 1000x spike — z-score well above 5
    dates = [datetime(2024, 1, 2, tzinfo=UTC) + __import__('datetime').timedelta(days=d) for d in range(61)]
    prices = [100.0] * 60 + [100000.0]
    df = _bars_df(list(zip(dates, prices)))
    issues = check_outliers(df, z_threshold=5.0)
    assert len(issues) == 1


# ── Check 7b: Stale prices ────────────────────────────────────────────────────

def test_stale_prices_clean() -> None:
    dates = [datetime(2024, 1, d + 2, tzinfo=UTC) for d in range(10)]
    prices = [100.0 + i for i in range(10)]
    df = _bars_df(list(zip(dates, prices)))
    assert check_stale_prices(df, max_stale_days=5) == []


def test_stale_prices_flagged() -> None:
    dates = [datetime(2024, 1, d + 2, tzinfo=UTC) for d in range(10)]
    prices = [100.0] * 10  # all identical
    df = _bars_df(list(zip(dates, prices)))
    issues = check_stale_prices(df, max_stale_days=5)
    assert len(issues) == 1


# ── Composite run_all_checks ──────────────────────────────────────────────────

def test_run_all_checks_passes_clean_data() -> None:
    as_of = datetime(2024, 6, 1, tzinfo=UTC)
    since = datetime(2024, 1, 1, tzinfo=UTC)
    dates = [datetime(2024, 1, d + 2, tzinfo=UTC) for d in range(10)]
    prices = [100.0 + i for i in range(10)]
    df = _bars_df(list(zip(dates, prices)))
    report = run_all_checks(df, as_of=as_of, snapshot_id="test", since=since, until=as_of)
    assert report.passed


def test_run_all_checks_fails_on_leakage() -> None:
    as_of = datetime(2024, 1, 1, tzinfo=UTC)
    # available_at is after as_of
    df = _bars_df([(datetime(2024, 1, 2, tzinfo=UTC), 100.0)], available_at_offset_days=5)
    report = run_all_checks(df, as_of=as_of, snapshot_id="test")
    assert not report.passed
    assert any(i["check"] == "leakage" for i in report.issues)
