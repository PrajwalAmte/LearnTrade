"""
End-to-end leakage pipeline test.

Simulates a full data pull → PIT store ingest → quality check cycle
to verify the leakage test suite works on a realistic dataset.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from leantrade.core.types import InstrumentId, Observation
from leantrade.data.quality.checks import run_all_checks
from leantrade.data.store.pit import PITStore

UTC = timezone.utc


def _simulate_pull(store: PITStore, n_instruments: int = 5, n_days: int = 20) -> None:
    """Simulate a provider pull: ingest clean bars with available_at = event_time."""
    obs = []
    for i in range(n_instruments):
        symbol = f"SYM{i:02d}"
        iid = InstrumentId(f"XNYS:{symbol}")
        base_price = 100.0 + i * 10
        for d in range(n_days):
            dt = datetime(2024, 1, d + 2, tzinfo=UTC)
            obs.append(Observation(
                instrument_id=iid,
                source_id="test_provider",
                kind="bar",
                event_time=dt,
                available_at=dt,  # clean: available same day
                payload={
                    "open": base_price,
                    "high": base_price + 1,
                    "low": base_price - 1,
                    "close": base_price + (d % 3) * 0.5,
                    "volume": 100_000,
                    "adj_close": base_price,
                },
                pit_level="none",
            ))
    store.ingest(obs)


@pytest.fixture()
def store(tmp_path: Path) -> PITStore:
    s = PITStore(tmp_path)
    yield s
    s.close()


def test_clean_pipeline_passes_all_checks(store: PITStore) -> None:
    _simulate_pull(store, n_instruments=3, n_days=10)
    as_of = datetime(2024, 12, 31, tzinfo=UTC)
    since = datetime(2024, 1, 1, tzinfo=UTC)

    df = store.query_observations(as_of=as_of, since=since, kind="bar")
    report = run_all_checks(df, as_of=as_of, snapshot_id="e2e_test", since=since, until=as_of)

    assert report.passed, f"Unexpected errors: {report.issues}"


def test_pipeline_catches_leakage(store: PITStore) -> None:
    """Inject future-available observations and verify the check catches them."""
    future_available = datetime(2025, 6, 1, tzinfo=UTC)
    obs = [
        Observation(
            instrument_id=InstrumentId("XNYS:LEAK"),
            source_id="leaky_provider",
            kind="bar",
            event_time=datetime(2024, 3, 1, tzinfo=UTC),
            available_at=future_available,  # leak: available_at is in the future
            payload={"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
                     "volume": 1000, "adj_close": 100.5},
            pit_level="none",
        )
    ]
    store.ingest(obs)

    as_of = datetime(2024, 12, 31, tzinfo=UTC)
    since = datetime(2024, 1, 1, tzinfo=UTC)

    # The leaky row IS visible in a raw query but should NOT be in PIT query
    pit_df = store.query_bars([InstrumentId("XNYS:LEAK")], as_of=as_of, since=since)
    assert len(pit_df) == 0, "Leaky row leaked through PIT query!"

    # Running checks on all observations (including raw) should detect the issue
    all_df = store._raw_query(
        "SELECT * FROM observations WHERE source_id = 'leaky_provider'"
    )
    report = run_all_checks(all_df, as_of=as_of, snapshot_id="leak_test")
    assert not report.passed
    assert any(i["check"] == "leakage" for i in report.issues)


def test_duplicate_observations_flagged(store: PITStore) -> None:
    dt = datetime(2024, 1, 2, tzinfo=UTC)
    obs = Observation(
        instrument_id=InstrumentId("XNYS:DUPE"),
        source_id="dup_provider",
        kind="bar",
        event_time=dt,
        available_at=dt,
        payload={"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                 "volume": 1000, "adj_close": 100.0},
    )
    store.ingest([obs])  # ingest twice — should deduplicate
    store.ingest([obs])

    df = store.query_observations(
        as_of=datetime(2024, 12, 31, tzinfo=UTC),
        since=datetime(2024, 1, 1, tzinfo=UTC),
        kind="bar",
    )
    assert len(df) == 1  # deduplication works via row_key hash
