"""Point-in-time universe: stocks are only used on dates when they were actually members."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl

from learntrade.data.index_history import parse_sp500_table
from learntrade.data.store.pit import PITStore
from learntrade.data.universe import (
    UniverseManager, membership_filter, snapshots_from_added_dates,
)

UTC = timezone.utc
D = lambda y, m, d: datetime(y, m, d, tzinfo=UTC)  # noqa: E731


def test_join_dates_become_cumulative_snapshots_with_collapse():
    added = {"OLD": D(1999, 1, 1), "OLD2": D(2005, 5, 5), "NEW": D(2024, 6, 1), "NEWER": D(2025, 2, 1)}
    snaps = snapshots_from_added_dates(added, collapse_before=D(2023, 1, 1))
    assert [d for d, _ in snaps] == [D(2023, 1, 1), D(2024, 6, 1), D(2025, 2, 1)]
    assert snaps[0][1] == ["OLD", "OLD2"]
    assert snaps[1][1] == ["NEW", "OLD", "OLD2"]
    assert snaps[2][1] == ["NEW", "NEWER", "OLD", "OLD2"]


def _rows():
    days = [D(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    return pl.DataFrame({
        "event_time": days * 2,
        "instrument_id": ["XNYS:AAA"] * 10 + ["XNYS:BBB"] * 10,
        "x": [1.0] * 20,
    })


def test_membership_filter_excludes_dates_before_joining_and_after_leaving():
    history = [(D(2024, 1, 1), ["AAA", "BBB"]), (D(2024, 1, 5), ["AAA"]), ]  # BBB leaves on the 5th
    out, info = membership_filter(_rows(), history, "XNYS")
    assert info["mode"] == "point_in_time"
    bbb = out.filter(pl.col("instrument_id") == "XNYS:BBB")
    assert bbb["event_time"].max() < D(2024, 1, 5)
    assert out.filter(pl.col("instrument_id") == "XNYS:AAA").height == 10


def test_membership_filter_new_member_excluded_before_join():
    history = [(D(2024, 1, 1), ["AAA"]), (D(2024, 1, 6), ["AAA", "BBB"])]
    out, _ = membership_filter(_rows(), history, "XNYS")
    bbb = out.filter(pl.col("instrument_id") == "XNYS:BBB")
    assert bbb["event_time"].min() >= D(2024, 1, 6)


def test_single_snapshot_is_a_plain_list_and_untouched():
    out, info = membership_filter(_rows(), [(D(2024, 1, 1), ["AAA"])], "XNYS")
    assert info["mode"] == "current_list" and out.height == 20


def test_import_join_dates_and_union_loads_past_members(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text("symbol,date_added\nAAA,1990-01-01\nBBB,2024-06-01\n")
    with PITStore(tmp_path / "d") as store:
        n = UniverseManager(store).import_csv(
            "XNYS", csv, added_column="date_added", collapse_before=D(2023, 1, 1))
        assert n == 2
        hist = store.get_universe_history("XNYS", D(2025, 1, 1))
        assert len(hist) == 2
        assert {str(i) for i in store.get_universe_union("XNYS", D(2025, 1, 1))} == {
            "XNYS:AAA", "XNYS:BBB"}
        # Before BBB joined it is not in the universe visible at that time.
        assert [str(i) for i in store.get_universe("XNYS", D(2024, 1, 1))] == ["XNYS:AAA"]


def test_parse_sp500_table():
    page = '''<table class="wikitable" id="constituents"><tbody>
    <tr><th>Symbol</th><th>Security</th><th>Date added</th></tr>
    <tr><td><a>MMM</a></td><td>3M</td><td>1957-03-04</td></tr>
    <tr><td>BRK.B</td><td>Berkshire</td><td>2010-02-16 [3]</td></tr>
    <tr><td>XYZ</td><td>NoDate</td><td></td></tr>
    </tbody></table><table id="other"></table>'''
    rows = parse_sp500_table(page)
    assert rows[0] == ("MMM", D(1957, 3, 4))
    assert rows[1] == ("BRK.B", D(2010, 2, 16))
    assert rows[2] == ("XYZ", None)


def test_trainer_dataset_respects_membership(tmp_path, monkeypatch):
    from learntrade.quant.trainer import QuantTrainer
    from learntrade.setup.demo import generate_demo_data

    with PITStore(tmp_path) as store:
        generate_demo_data(store, mic="XNYS", n_days=420, seed=3)
        join = datetime.now(tz=UTC) - timedelta(days=200)
        everyone = [f"SYM{i:02d}" for i in range(50)]
        history = [(datetime.now(tz=UTC) - timedelta(days=900), everyone[1:]),
                   (join, everyone)]
        monkeypatch.setattr(store, "get_universe_history", lambda m, a: history)
        t = QuantTrainer(store, "swing_20d")
        merged, _cols, _bm, _n = t._build_dataset(
            "XNYS", datetime.now(tz=UTC) - timedelta(days=700), datetime.now(tz=UTC))
        sym0 = merged.filter(pl.col("instrument_id").str.ends_with("SYM00"))
        assert sym0.height > 0
        assert sym0["event_time"].min() >= join - timedelta(days=1)
        assert t._membership["mode"] == "point_in_time"
        # a regular member has rows before the join date
        sym1 = merged.filter(pl.col("instrument_id").str.ends_with("SYM01"))
        assert sym1["event_time"].min() < join - timedelta(days=100)


def test_old_leantrade_database_file_is_migrated(tmp_path):
    """Workspaces created before the LearnTrade rename must keep their data."""
    from learntrade.core.types import InstrumentId, Observation

    with PITStore(tmp_path) as store:
        store.ingest([Observation(
            instrument_id=InstrumentId("XNYS:AAA"), source_id="t", kind="bar",
            event_time=D(2024, 1, 2), available_at=D(2024, 1, 2),
            payload={"close": 1.0})])
    (tmp_path / "learntrade.duckdb").rename(tmp_path / "leantrade.duckdb")
    with PITStore(tmp_path) as store:
        assert store.stats()["total_observations"] == 1
    assert (tmp_path / "learntrade.duckdb").exists() and not (tmp_path / "leantrade.duckdb").exists()


def test_old_leantrade_env_names_still_work(monkeypatch, tmp_path):
    from learntrade.core.config import AppSecrets, _migrate_legacy_names

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LEARNTRADE_PORT", raising=False)
    monkeypatch.setenv("LEANTRADE_PORT", "9123")
    _migrate_legacy_names()
    assert AppSecrets().port == 9123
    monkeypatch.setenv("LEARNTRADE_PORT", "9456")        # explicit new name wins
    _migrate_legacy_names()
    assert AppSecrets().port == 9456


def test_migration_replaces_an_empty_new_file_but_never_a_populated_one(tmp_path):
    from learntrade.core.types import InstrumentId, Observation

    def one_obs() -> Observation:
        return Observation(instrument_id=InstrumentId("XNYS:AAA"), source_id="t", kind="bar",
                           event_time=D(2024, 1, 2), available_at=D(2024, 1, 2), payload={"close": 1.0})

    with PITStore(tmp_path) as store:
        store.ingest([one_obs()])
    (tmp_path / "learntrade.duckdb").rename(tmp_path / "leantrade.duckdb")
    with PITStore(tmp_path / "other") as empty:      # make an empty new-name file in tmp_path
        pass
    (tmp_path / "other" / "learntrade.duckdb").rename(tmp_path / "learntrade.duckdb")
    with PITStore(tmp_path) as store:                # empty new file -> legacy wins
        assert store.stats()["total_observations"] == 1
    assert (tmp_path / "learntrade.duckdb.empty.bak").exists()
