"""
PITStore tests — schema, ingest, PIT golden rule enforcement, universe, snapshots.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from learntrade.core.types import InstrumentId, Instrument, Observation
from learntrade.data.store.pit import PITStore

UTC = timezone.utc


@pytest.fixture()
def store(tmp_path: Path) -> PITStore:
    s = PITStore(tmp_path)
    yield s
    s.close()


def _make_obs(
    instrument_id: str,
    event_time: datetime,
    available_at: datetime,
    close: float = 100.0,
    source_id: str = "test",
) -> Observation:
    return Observation(
        instrument_id=InstrumentId(instrument_id),
        source_id=source_id,
        kind="bar",
        event_time=event_time,
        available_at=available_at,
        payload={"open": close, "high": close, "low": close, "close": close, "volume": 1000, "adj_close": close},
    )


def test_ingest_and_count(store: PITStore) -> None:
    obs = [_make_obs("XNYS:IBM", datetime(2024, 1, d, tzinfo=UTC), datetime(2024, 1, d, tzinfo=UTC))
           for d in range(2, 6)]
    n = store.ingest(obs)
    assert n == 4
    stats = store.stats()
    assert stats["total_observations"] == 4


def test_ingest_idempotent(store: PITStore) -> None:
    obs = [_make_obs("XNYS:IBM", datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC))]
    store.ingest(obs)
    store.ingest(obs)  # duplicate — must not double-count
    assert store.stats()["total_observations"] == 1


def test_pit_golden_rule(store: PITStore) -> None:
    """query_bars must NEVER return rows where available_at > as_of."""
    past = datetime(2024, 1, 2, tzinfo=UTC)
    future = datetime(2025, 6, 1, tzinfo=UTC)
    store.ingest([
        _make_obs("XNYS:IBM", past, past, 100.0),          # visible
        _make_obs("XNYS:IBM", past, future, 110.0),        # future available_at — must be hidden
    ])
    as_of = datetime(2024, 12, 31, tzinfo=UTC)
    since = datetime(2023, 1, 1, tzinfo=UTC)
    df = store.query_bars([InstrumentId("XNYS:IBM")], as_of, since)
    assert len(df) == 1
    assert float(df["close"][0]) == 100.0


def test_query_bars_respects_since(store: PITStore) -> None:
    for d in range(1, 10):
        dt = datetime(2024, 1, d + 1, tzinfo=UTC)
        store.ingest([_make_obs("XNYS:IBM", dt, dt)])
    as_of = datetime(2024, 12, 31, tzinfo=UTC)
    since = datetime(2024, 1, 5, tzinfo=UTC)
    df = store.query_bars([InstrumentId("XNYS:IBM")], as_of, since)
    assert all(
        datetime.fromisoformat(str(t)).replace(tzinfo=UTC) >= since
        for t in df["event_time"].to_list()
    )


def test_register_instrument(store: PITStore) -> None:
    inst = Instrument(
        id=InstrumentId("XNYS:IBM"),
        mic="XNYS",
        symbol="IBM",
        currency="USD",
    )
    store.register_instrument(inst)
    instruments = store.get_instruments("XNYS")
    assert len(instruments) == 1
    assert instruments[0]["symbol"] == "IBM"


def test_universe_roundtrip(store: PITStore) -> None:
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    store.ingest([
        Observation(
            instrument_id=None,
            source_id="test",
            kind="universe",
            event_time=ts,
            available_at=ts,
            payload={"mic": "XNYS", "members": ["IBM", "AAPL", "MSFT"]},
        )
    ])
    universe = store.get_universe("XNYS", datetime(2024, 6, 1, tzinfo=UTC))
    symbols = {iid.symbol for iid in universe}
    assert symbols == {"IBM", "AAPL", "MSFT"}


def test_universe_pit_enforcement(store: PITStore) -> None:
    """Universe registered in the future must not be visible at an earlier as_of."""
    future_ts = datetime(2025, 1, 1, tzinfo=UTC)
    store.ingest([
        Observation(
            instrument_id=None,
            source_id="test",
            kind="universe",
            event_time=future_ts,
            available_at=future_ts,
            payload={"mic": "XNYS", "members": ["IBM"]},
        )
    ])
    past_as_of = datetime(2024, 1, 1, tzinfo=UTC)
    universe = store.get_universe("XNYS", past_as_of)
    assert universe == []


def test_create_snapshot(store: PITStore) -> None:
    snapshot_id = store.create_snapshot("test snapshot", datetime(2024, 1, 1, tzinfo=UTC))
    assert len(snapshot_id) == 36  # UUID format


def test_stats_by_kind(store: PITStore) -> None:
    store.ingest([_make_obs("XNYS:IBM", datetime(2024, 1, d + 1, tzinfo=UTC), datetime(2024, 1, d + 1, tzinfo=UTC))
                  for d in range(3)])
    stats = store.stats()
    assert stats["by_kind"]["bar"] == 3
