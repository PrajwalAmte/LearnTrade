"""
Universe manager tests.
"""

from __future__ import annotations

import csv
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from leantrade.core.types import InstrumentId
from leantrade.data.store.pit import PITStore
from leantrade.data.universe import UniverseManager

UTC = timezone.utc


@pytest.fixture()
def store(tmp_path: Path) -> PITStore:
    s = PITStore(tmp_path)
    yield s
    s.close()


def test_import_explicit_list(store: PITStore) -> None:
    mgr = UniverseManager(store)
    mgr.import_explicit_list("XNYS", ["IBM", "AAPL", "MSFT"])
    universe = mgr.get_universe("XNYS", datetime.now(tz=UTC))
    symbols = {iid.symbol for iid in universe}
    assert symbols == {"IBM", "AAPL", "MSFT"}


def test_import_csv(store: PITStore, tmp_path: Path) -> None:
    csv_path = tmp_path / "universe.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["symbol"])
        writer.writeheader()
        for sym in ["GOOG", "AMZN", "META"]:
            writer.writerow({"symbol": sym})

    mgr = UniverseManager(store)
    n = mgr.import_csv("XNAS", csv_path)
    assert n == 1  # one snapshot with all symbols
    universe = mgr.get_universe("XNAS", datetime.now(tz=UTC))
    symbols = {iid.symbol for iid in universe}
    assert symbols == {"GOOG", "AMZN", "META"}


def test_pit_universe_at_past_date(store: PITStore) -> None:
    """Universe registered later must not be visible at an earlier as_of."""
    mgr = UniverseManager(store)
    past = datetime(2020, 1, 1, tzinfo=UTC)
    present = datetime(2024, 1, 1, tzinfo=UTC)
    mgr.import_explicit_list("XNYS", ["IBM"], available_at=present)
    # Query before the universe was registered
    before = mgr.get_universe("XNYS", past)
    assert before == []


def test_different_mics_isolated(store: PITStore) -> None:
    mgr = UniverseManager(store)
    mgr.import_explicit_list("XNYS", ["IBM", "AAPL"])
    mgr.import_explicit_list("XNSE", ["RELIANCE", "INFY"])
    nys = {iid.symbol for iid in mgr.get_universe("XNYS", datetime.now(tz=UTC))}
    nse = {iid.symbol for iid in mgr.get_universe("XNSE", datetime.now(tz=UTC))}
    assert nys == {"IBM", "AAPL"}
    assert nse == {"RELIANCE", "INFY"}
