"""Regression tests for the 10 issues found in the code review (2026-11-22)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import pytest

from learntrade.core.config import AppConfig, DataConfig, DataProvidersConfig
from learntrade.core.types import InstrumentId, Observation
from learntrade.data.engine import DataEngine
from learntrade.data.providers.base import FetchRequest
from learntrade.data.store.pit import PITStore
from learntrade.data.universe import membership_filter
from learntrade.setup.demo import generate_demo_data

UTC = timezone.utc


# 1 ── prices re-adjusted after a split are reloaded, not left on the old basis ──────────

class SplitProvider:
    """Serves 40 days of a stock that did a 4-for-1 split on day 30 (split-adjusted history)."""
    id = "splitty"; capabilities = {"bars_daily"}; markets = "*"
    pit_level = {"bars_daily": "none"}; licence = "test"

    def __init__(self) -> None:
        self.split_done = False
        self.requests: list[FetchRequest] = []

    def fetch(self, request: FetchRequest) -> Iterable[Observation]:
        self.requests.append(request)
        for iid in request.instruments:
            for d in range(40):
                day = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=d)
                if day < request.since or day > request.until:
                    continue
                px = 400.0 + d                      # unadjusted price path
                if self.split_done:
                    px = px / 4.0                   # Yahoo rewrites ALL history after the split
                yield Observation(instrument_id=iid, source_id=self.id, kind="bar", event_time=day,
                                  available_at=datetime.now(tz=UTC),
                                  payload={"close": px, "adj_close": px, "open": px, "high": px,
                                           "low": px, "volume": 1000})


def test_refresh_after_split_reloads_whole_history(tmp_path):
    cfg = AppConfig()
    cfg.markets_enabled = ["XNYS"]
    cfg.data = DataConfig(providers=DataProvidersConfig(prices=["splitty"]))
    cfg.universe = {"XNYS": type("U", (), {"source": "explicit", "instruments": ["ACME"],
                                           "index": None, "n": None, "min_adv": None,
                                           "point_in_time": True})()}
    prov = SplitProvider()
    with PITStore(tmp_path) as store:
        engine = DataEngine(cfg, store)
        engine._registry._cache["splitty"] = prov
        engine.pull(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 30, tzinfo=UTC),
                    capabilities={"bars_daily"})
        prov.split_done = True
        # A short refresh (last ~10 days), like the web page's Update data.
        engine.pull(datetime(2024, 1, 25, tzinfo=UTC), datetime(2024, 2, 9, tzinfo=UTC),
                    capabilities={"bars_daily"})
        bars = store.query_bars([InstrumentId("XNYS:ACME")], as_of=datetime.now(tz=UTC),
                                since=datetime(2023, 12, 1, tzinfo=UTC))
    closes = bars.sort("event_time")["close"].cast(pl.Float64).to_numpy()
    rets = closes[1:] / closes[:-1] - 1
    assert len(closes) == 40
    assert np.abs(rets).max() < 0.05, "no fake split jump may remain in the stored series"
    assert closes[0] == pytest.approx(100.0)        # the oldest day was rewritten on the new basis
    assert any(r.since == datetime(2024, 1, 1, tzinfo=UTC) for r in prov.requests[2:])


def test_refresh_without_adjustment_change_does_not_reload(tmp_path):
    cfg = AppConfig()
    cfg.markets_enabled = ["XNYS"]
    cfg.data = DataConfig(providers=DataProvidersConfig(prices=["splitty"]))
    cfg.universe = {"XNYS": type("U", (), {"source": "explicit", "instruments": ["ACME"],
                                           "index": None, "n": None, "min_adv": None,
                                           "point_in_time": True})()}
    prov = SplitProvider()
    with PITStore(tmp_path) as store:
        engine = DataEngine(cfg, store)
        engine._registry._cache["splitty"] = prov
        engine.pull(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 30, tzinfo=UTC),
                    capabilities={"bars_daily"})
        n = len(prov.requests)
        engine.pull(datetime(2024, 1, 25, tzinfo=UTC), datetime(2024, 2, 9, tzinfo=UTC),
                    capabilities={"bars_daily"})
    assert len(prov.requests) == n + 1             # just the refresh itself


# 2 ── a new stock list replaces the old one; it is not index history ─────────────────

def _universe(mic, members, at, source="csv:ui_universe.csv"):
    return Observation(instrument_id=None, source_id=source, kind="universe", event_time=at,
                       available_at=at, payload={"mic": mic, "members": members})


def test_reimported_list_replaces_previous_list(tmp_path):
    now = datetime.now(tz=UTC)
    with PITStore(tmp_path) as store:
        store.ingest([_universe("XNSE", ["A", "B", "C"], now - timedelta(days=10))])
        store.ingest([_universe("XNSE", ["C", "D", "E"], now - timedelta(days=1))])   # same file name
        hist = store.get_universe_history("XNSE", now)
        assert len(hist) == 1 and hist[0][1] == ["C", "D", "E"]
        assert [str(i) for i in store.get_universe("XNSE", now)] == ["XNSE:C", "XNSE:D", "XNSE:E"]
        assert {str(i) for i in store.get_universe_union("XNSE", now)} == {"XNSE:C", "XNSE:D", "XNSE:E"}
    df = pl.DataFrame({"event_time": [now - timedelta(days=300)] * 3,
                       "instrument_id": ["XNSE:C", "XNSE:D", "XNSE:E"]})
    out, info = membership_filter(df, hist, "XNSE")
    assert info["mode"] == "current_list" and out.height == 3          # nothing dropped


def test_index_history_is_still_one_import_with_many_snapshots(tmp_path):
    now = datetime.now(tz=UTC)
    with PITStore(tmp_path) as store:
        store.ingest([_universe("XNYS", ["A", "B"], now - timedelta(days=400), "index:SP500"),
                      _universe("XNYS", ["A", "B", "C"], now - timedelta(days=100), "index:SP500")])
        assert len(store.get_universe_history("XNYS", now)) == 2
        # A plain list typed in later wins over the index import.
        store.ingest([_universe("XNYS", ["X", "Y"], now, "csv:mine.csv")])
        assert [h[1] for h in store.get_universe_history("XNYS", now)] == [["X", "Y"]]


# 3 ── calibrators are stored per model ──────────────────────────────────────────────

def test_each_model_has_its_own_calibrator(tmp_path):
    from learntrade.cli.main import _model_artifact_paths
    cfg = AppConfig()
    cfg.workspace.data_dir = tmp_path
    _, cal_ridge = _model_artifact_paths(cfg, "XNYS", "swing_20d", "ridge")
    _, cal_lgbm = _model_artifact_paths(cfg, "XNYS", "swing_20d", "lgbm")
    assert cal_ridge != cal_lgbm and "ridge" in cal_ridge.name


# 4 ── financials of one market do not change another market's features ───────────────

def test_fundamentals_switch_is_per_market(tmp_path):
    from learntrade.features.price_volume import default_feature_set
    now = datetime.now(tz=UTC)
    obs = [Observation(instrument_id=InstrumentId(f"XNYS:S{i}"), source_id="yahoo", kind="fundamental",
                       event_time=now - timedelta(days=400), available_at=now - timedelta(days=300),
                       source_version="annual", payload={"period": "annual", "eps": 1.0})
           for i in range(6)]
    with PITStore(tmp_path) as store:
        store.ingest(obs)
        assert default_feature_set(store, "XNYS").fundamentals
        assert not default_feature_set(store, "XNSE").fundamentals


# 6 + 7 ── every offered horizon is trained, and MCP names real horizons ───────────────

def test_setup_trains_every_horizon_the_page_offers():
    from learntrade.api.routes.workspace import HORIZON_LABELS, _train_steps
    steps = _train_steps("XNSE", "2023-01-01")
    trained = {s.args[s.args.index("--horizon") + 1] for s in steps}
    assert trained == set(HORIZON_LABELS)
    assert [s.optional for s in steps if "swing_20d" in s.args] == [False]


def test_mcp_horizons_exist_and_bad_ones_are_explained():
    from learntrade.mcp_server import ToolError, _HORIZONS, _check_horizon
    from learntrade.quant.targets.horizon import get_horizon
    for h in _HORIZONS:
        assert get_horizon(h).id == h
    with pytest.raises(ToolError, match="positional_60d"):
        _check_horizon("swing_60d")


# 8 ── cross-validation refuses misaligned forecasts ─────────────────────────────────

def test_cv_fold_fails_instead_of_misaligning(tmp_path):
    from learntrade.quant.models.ridge import RidgeReturnModel
    from learntrade.quant.trainer import QuantTrainer

    class Dropper(RidgeReturnModel):
        def predict_distribution(self, X, ids, as_of):
            return super().predict_distribution(X, ids, as_of)[:-1]   # silently drops a row

    with PITStore(tmp_path) as store:
        generate_demo_data(store, mic="XNYS", n_days=700, seed=1)
        t = QuantTrainer(store, "swing_20d", model_class=Dropper, artifacts_dir=tmp_path / "m",
                         holdout_days=0)
        with pytest.raises(ValueError, match="All CV folds failed"):
            t.train("XNYS", datetime.now(tz=UTC) - timedelta(days=1000), datetime.now(tz=UTC))


# 9 ── processes coordinate on the single-writer data file ───────────────────────────

def test_store_waits_for_another_process_to_release_the_file(tmp_path):
    PITStore(tmp_path).close()
    holder = subprocess.Popen([sys.executable, "-c", (
        "import duckdb,time,sys; c=duckdb.connect(sys.argv[1]); print('ok',flush=True); time.sleep(2)"),
        str(tmp_path / "learntrade.duckdb")], stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "ok"
    t0 = time.monotonic()
    with PITStore(tmp_path, lock_wait=20):
        waited = time.monotonic() - t0
    holder.wait()
    assert waited > 0.5


def test_job_marker_is_seen_by_other_processes_and_cleared_when_stale(tmp_path):
    from learntrade.core.activity import clear_job, mark_job, running_job
    mark_job(tmp_path, "Updating your data")
    assert running_job(tmp_path)["title"] == "Updating your data"
    clear_job(tmp_path)
    assert running_job(tmp_path) is None
    (tmp_path / ".learntrade_job.json").write_text(json.dumps({"pid": 999999, "title": "x"}))
    assert running_job(tmp_path) is None and not (tmp_path / ".learntrade_job.json").exists()


# 10 ── evaluation builds the dataset once ───────────────────────────────────────────

def test_evaluation_builds_dataset_once(tmp_path, monkeypatch):
    from learntrade.backtest.evaluation import run_evaluation
    from learntrade.quant.models.ridge import RidgeReturnModel
    from learntrade.quant.trainer import QuantTrainer
    calls = {"n": 0}
    orig = QuantTrainer._build_dataset

    def counting(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(QuantTrainer, "_build_dataset", counting)
    with PITStore(tmp_path) as store:
        generate_demo_data(store, mic="XNYS", n_days=700, seed=4)
        res = run_evaluation(store, "XNYS", "swing_20d", datetime.now(tz=UTC) - timedelta(days=1100),
                             60, {"ridge": RidgeReturnModel, "ridge2": RidgeReturnModel})
    assert len(res.candidates) == 3 and calls["n"] == 1
