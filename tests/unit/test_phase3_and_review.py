"""Phase 3 (Nifty lists, size feature, sweep) and the fixes from the outside review."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from learntrade.data.store.pit import PITStore
from learntrade.setup.demo import generate_demo_data

UTC = timezone.utc


def test_nse_index_csv_parses_todays_members():
    from learntrade.data.index_history import SUPPORTED, parse_nse_index_csv
    text = "﻿Company Name,Industry,Symbol,Series,ISIN Code\nABB India Ltd.,Capital Goods,ABB,EQ,INE117A01022\n"
    assert parse_nse_index_csv(text) == [("ABB", None)]
    assert SUPPORTED["NIFTY500"][0] == "XNSE"
    with pytest.raises(ValueError):
        parse_nse_index_csv("a,b\n1,2\n")


def test_size_feature_is_log_market_value():
    import polars as pl
    from learntrade.features.fundamentals import FUNDAMENTAL_FEATURES, attach_fundamentals
    stmts = pl.DataFrame({
        "instrument_id": ["XNSE:A"], "source_id": ["yahoo"],
        "event_time": [datetime(2024, 3, 31, tzinfo=UTC)], "available_at": [datetime(2024, 6, 29, tzinfo=UTC)],
        "revenue": [1000.0], "gross_profit": [400.0], "net_income": [100.0], "eps": [10.0],
        "equity": [500.0], "shares": [1e6], "assets": [1500.0], "debt": [300.0], "cfo": [150.0]})
    rows = pl.DataFrame({"instrument_id": ["XNSE:A"], "event_time": [datetime(2024, 9, 1, tzinfo=UTC)],
                         "close": [250.0]})
    out = attach_fundamentals(rows, stmts, "event_time")
    assert "log_market_value" in FUNDAMENTAL_FEATURES
    assert out["log_market_value"][0] == pytest.approx(np.log(1e6 * 250.0))


def test_sweep_judges_with_the_fixed_rule_and_never_touches_saved_models(tmp_path):
    from learntrade.backtest.sweep import judge, run_sweep, to_markdown
    from learntrade.quant.models.ridge import RidgeReturnModel
    assert judge(0.03, 2.5, 0.01) and not judge(0.03, 1.9, 0.01) and not judge(0.03, 2.5, -0.01)
    assert not judge(None, 3.0, 0.1)
    with PITStore(tmp_path) as store:
        generate_demo_data(store, mic="XNYS", n_days=800, seed=3)
        now = datetime.now(tz=UTC)
        syms = {str(i).split(":")[1] for i in store.get_universe("XNYS", now)}
        half = set(sorted(syms)[:25])
        res = run_sweep(store, "XNYS", now - timedelta(days=1300), now,
                        {"Whole list": None, "Half": half}, ["swing_20d"], ["price"],
                        {"ridge": RidgeReturnModel}, holdout_days=60)
    assert [r.group for r in res.rows] == ["Whole list", "Half"]
    assert all(r.error is None for r in res.rows), [r.error for r in res.rows]
    assert res.rows[1].n_stocks == 25 and res.rows[0].n_stocks > 25
    md = to_markdown(res)
    assert "would pass by luck" in md and "Whole list" in md
    assert not (tmp_path / "models").exists()                   # sweep models go to a temp dir


def test_model_files_record_library_versions(tmp_path):
    from learntrade.quant.models import _compat
    from learntrade.quant.models.ridge import RidgeReturnModel
    import polars as pl
    m = RidgeReturnModel("swing_20d", "v")
    X = pl.DataFrame({"a": np.random.default_rng(0).normal(size=50)})
    m.fit(X, pl.Series(np.random.default_rng(1).normal(size=50)))
    p = str(tmp_path / "m.pkl")
    m.save(p)
    assert RidgeReturnModel.load(p).feature_set_version == "v"
    with open(p, "rb") as f:
        header = pickle.load(f)
        payload = pickle.load(f)
    header["versions"]["scikit-learn"] = "0.1"                  # pretend an old library made it
    with open(p, "wb") as f:
        pickle.dump(header, f)
        pickle.dump(payload, f)
    with pytest.raises(_compat.ModelCompatibilityError, match="scikit-learn 0.1"):
        RidgeReturnModel.load(p)
    legacy = str(tmp_path / "old.pkl")                         # files from before headers still load
    with open(legacy, "wb") as f:
        pickle.dump(payload, f)
    assert RidgeReturnModel.load(legacy).feature_set_version == "v"


def test_llm_daily_budget_survives_a_restart(tmp_path):
    from learntrade.llm.router import BudgetExceededError, BudgetTracker
    ledger = tmp_path / "spend.json"
    a = BudgetTracker(per_run_limit=10, per_day_limit=1.0, ledger_path=ledger)
    a.record(0.8)
    b = BudgetTracker(per_run_limit=10, per_day_limit=1.0, ledger_path=ledger)   # new process
    assert b.day_spent == pytest.approx(0.8)
    with pytest.raises(BudgetExceededError):
        b.check(0.3)
    assert json.loads(ledger.read_text())[b._today()] == pytest.approx(0.8)


def test_fewer_folds_than_requested_is_reported():
    from learntrade.quant.reliability import assess_reliability
    rep = {"cv": {"rank_ic_mean": 0.0, "rank_ic_ci": [-0.05, 0.05], "n_days": 300,
                  "n_folds": 2, "n_folds_requested": 5}, "n_instruments": 40, "n_days": 800,
           "label_type": "excess"}
    reasons = assess_reliability(rep)["reasons"]
    assert any("Only 2 of 5 test periods" in r for r in reasons)


def test_docker_publishes_only_on_this_computer():
    compose = (Path(__file__).parents[2] / "docker-compose.yml").read_text()
    assert '"127.0.0.1:8787:8787"' in compose


def test_store_public_summaries(tmp_path):
    with PITStore(tmp_path) as store:
        generate_demo_data(store, mic="XNYS", n_days=100, seed=1)
        rows = store.bar_summary_by_market()
        assert rows[0]["mic"] == "XNYS" and rows[0]["demo"]
        assert store.last_bar_time("XNYS") is not None and store.last_bar_time("XNSE") is None
