"""Fundamentals: parsing, point-in-time visibility, features, and the feature-set switch."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from learntrade.core.types import InstrumentId, Observation
from learntrade.data.providers.yahoo import FUNDAMENTAL_LAG_DAYS, parse_fundamentals
from learntrade.data.store.pit import PITStore
from learntrade.features.fundamentals import FUNDAMENTAL_FEATURES, attach_fundamentals
from learntrade.features.price_volume import (
    FEATURE_SET_VERSION, FEATURE_SET_VERSION_FUND, PriceVolumeFeatureSet, default_feature_set,
)
from learntrade.setup.demo import generate_demo_data

UTC = timezone.utc


def _series(name, vals):
    return {"meta": {"type": [name]}, name: [
        {"asOfDate": d, "reportedValue": {"raw": v}} for d, v in vals]}


def _yahoo_payload():
    return {"timeseries": {"result": [
        _series("annualNetIncome", [("2023-03-31", 100.0), ("2024-03-31", 130.0)]),
        _series("annualTotalRevenue", [("2023-03-31", 1000.0), ("2024-03-31", 1200.0)]),
        _series("quarterlyNetIncome", [("2024-03-31", 40.0)]),
        _series("annualSomethingElse", [("2024-03-31", 1.0)]),
    ]}}


def test_parse_marks_statements_public_only_after_the_reporting_lag():
    obs = parse_fundamentals(InstrumentId("XNSE:TCS"), _yahoo_payload())
    annual = [o for o in obs if o.source_version == "annual"]
    quarterly = [o for o in obs if o.source_version == "quarterly"]
    assert len(annual) == 2 and len(quarterly) == 1
    a = annual[-1]
    assert a.kind == "fundamental" and a.payload["net_income"] == 130.0 and a.payload["revenue"] == 1200.0
    assert a.available_at - a.event_time == timedelta(days=FUNDAMENTAL_LAG_DAYS["annual"])
    assert quarterly[0].available_at - quarterly[0].event_time == timedelta(days=FUNDAMENTAL_LAG_DAYS["quarterly"])


def test_store_hides_statements_until_available(tmp_path):
    obs = parse_fundamentals(InstrumentId("XNSE:TCS"), _yahoo_payload())
    with PITStore(tmp_path) as store:
        assert store.ingest(obs) == len(obs)
        assert store.ingest(obs) == 0                       # no duplicates on re-pull
        before = store.query_fundamentals(datetime(2024, 5, 1, tzinfo=UTC))      # FY24 not yet public
        after = store.query_fundamentals(datetime(2024, 7, 1, tzinfo=UTC))
        assert len(before) == 1 and len(after) == 2
        assert store.count_fundamental_stocks() == 1


def _stmts():
    return pl.DataFrame({
        "instrument_id": ["XNSE:A", "XNSE:A"],
        "event_time": [datetime(2023, 3, 31, tzinfo=UTC), datetime(2024, 3, 31, tzinfo=UTC)],
        "available_at": [datetime(2023, 6, 29, tzinfo=UTC), datetime(2024, 6, 29, tzinfo=UTC)],
        "revenue": [1000.0, 1200.0], "gross_profit": [400.0, 500.0], "net_income": [100.0, 130.0],
        "eps": [10.0, 13.0], "equity": [500.0, 600.0], "shares": [10.0, 10.0],
        "assets": [1500.0, 1600.0], "debt": [300.0, 320.0], "cfo": [150.0, 170.0],
    })


def test_features_use_only_the_latest_public_statement_and_todays_price():
    rows = pl.DataFrame({
        "instrument_id": ["XNSE:A"] * 3,
        "event_time": [datetime(2023, 5, 1, tzinfo=UTC), datetime(2023, 9, 1, tzinfo=UTC),
                       datetime(2024, 9, 1, tzinfo=UTC)],
        "close": [100.0, 100.0, 130.0],
    })
    out = attach_fundamentals(rows, _stmts(), "event_time").sort("event_time")
    assert out["earnings_yield"][0] is None                       # nothing public yet
    assert abs(out["earnings_yield"][1] - 10.0 / 100.0) < 1e-9     # FY23 EPS / price
    assert abs(out["earnings_yield"][2] - 13.0 / 130.0) < 1e-9     # FY24 EPS, today's price
    assert abs(out["book_to_price"][1] - (500 / 10) / 100.0) < 1e-9
    assert out["rev_growth"][1] is None and abs(out["rev_growth"][2] - 0.2) < 1e-9
    assert abs(out["roe"][2] - 130 / 600) < 1e-9
    assert set(FUNDAMENTAL_FEATURES) <= set(out.columns)


def _add_fundamentals(store, ids):
    now = datetime.now(tz=UTC)
    rng = np.random.default_rng(3)
    obs = []
    for iid in ids:
        for yrs_ago in (2, 1):
            end = now - timedelta(days=365 * yrs_ago + 120)
            obs.append(Observation(
                instrument_id=iid, source_id="yahoo", kind="fundamental", event_time=end,
                available_at=end + timedelta(days=90), source_version="annual", pit_level="partial",
                payload={"period": "annual", "revenue": 1000.0 + rng.normal(0, 50),
                         "gross_profit": 400.0, "net_income": 100.0 + rng.normal(0, 20),
                         "eps": 5.0 + rng.normal(0, 1), "equity": 500.0, "shares": 20.0,
                         "assets": 1500.0, "debt": 300.0, "cfo": 120.0}))
    store.ingest(obs)


def test_feature_set_includes_fundamentals_only_when_stored(tmp_path):
    with PITStore(tmp_path) as store:
        generate_demo_data(store, mic="XNYS", n_days=300, seed=5)
        ids = store.get_universe("XNYS", datetime.now(tz=UTC))
        plain = default_feature_set(store)
        assert plain.id == FEATURE_SET_VERSION and not plain.fundamentals
        _add_fundamentals(store, ids)
        fs = default_feature_set(store)
        assert fs.fundamentals and fs.id == FEATURE_SET_VERSION_FUND and "fundamentals" in fs.requires
        assert "earnings_yield_cs" in fs.feature_names
        now = datetime.now(tz=UTC)
        hist = fs.compute_history(ids, now - timedelta(days=150), now, store)
        latest = fs.compute(ids, now, store)
    assert set(fs.feature_names) <= set(hist.columns) and set(fs.feature_names) <= set(latest.columns)
    assert hist["earnings_yield_cs"].null_count() == 0           # missing is neutral, never null
    assert latest["earnings_yield"].null_count() == 0
    assert abs(latest["earnings_yield_cs"].mean()) < 1e-6


def test_old_price_only_feature_set_is_unchanged(tmp_path):
    with PITStore(tmp_path) as store:
        generate_demo_data(store, mic="XNYS", n_days=300, seed=5)
        fs = PriceVolumeFeatureSet()
    assert fs.id == FEATURE_SET_VERSION and not any(n.startswith("earnings") for n in fs.feature_names)


# ── SEC EDGAR company facts and split handling ───────────────────────────────

from learntrade.data.providers.sec_edgar import parse_company_facts
from learntrade.data.providers.yahoo import parse_splits
from learntrade.features.fundamentals import statement_ratios


def _fact(end, val, filed, start=None, form="10-K"):
    r = {"end": end, "val": val, "filed": filed, "form": form, "accn": "x"}
    if start:
        r["start"] = start
    return r


def _facts():
    def tag(rows, unit="USD"):
        return {"units": {unit: rows}}
    return {"facts": {"us-gaap": {
        "Revenues": tag([_fact("2023-12-31", 900.0, "2024-02-20", "2023-01-01")]),
        # Tag change in 2024: the second tag fills the later year.
        "RevenueFromContractWithCustomerExcludingAssessedTax": tag([
            _fact("2024-12-31", 1100.0, "2025-02-18", "2024-01-01"),
            _fact("2023-12-31", 900.0, "2025-02-18", "2023-01-01")]),   # comparison, filed later
        "NetIncomeLoss": tag([
            _fact("2023-12-31", 90.0, "2024-02-20", "2023-01-01"),
            _fact("2024-12-31", 120.0, "2025-02-18", "2024-01-01"),
            _fact("2024-03-31", 25.0, "2024-05-02", "2024-01-01", form="10-Q")]),
        "EarningsPerShareDiluted": tag([_fact("2023-12-31", 4.5, "2024-02-20", "2023-01-01")], "USD/shares"),
        "StockholdersEquity": tag([_fact("2023-12-31", 400.0, "2024-02-20"),
                                   _fact("2024-12-31", 450.0, "2025-02-18")]),
        "Assets": tag([_fact("2023-12-31", 1000.0, "2024-02-20"), _fact("2024-12-31", 1200.0, "2025-02-18")]),
        "CommonStockSharesOutstanding": tag([_fact("2023-12-31", 20.0, "2024-02-20")], "shares"),
    }}}


def test_sec_facts_use_true_filing_dates_and_first_reported_values():
    obs = parse_company_facts(InstrumentId("XNYS:ACME"), _facts(), datetime(2020, 1, 1, tzinfo=UTC))
    annual = {o.payload["period_end"]: o for o in obs if o.source_version == "annual"}
    assert set(annual) == {"2023-12-31", "2024-12-31"}
    fy23 = annual["2023-12-31"]
    assert fy23.available_at == datetime(2024, 2, 21, tzinfo=UTC)         # filed 2024-02-20 + 1 day
    assert fy23.payload["revenue"] == 900.0 and fy23.payload["eps"] == 4.5 and fy23.payload["shares"] == 20.0
    assert fy23.pit_level == "full" and fy23.source_id == "sec_edgar"
    assert annual["2024-12-31"].payload["revenue"] == 1100.0                # picked up via the newer tag
    assert annual["2024-12-31"].available_at == datetime(2025, 2, 19, tzinfo=UTC)
    assert not [o for o in obs if o.source_version == "quarterly"]          # too few numbers


def test_sec_statements_replace_yahoo_ones_and_are_split_adjusted(tmp_path):
    inst = InstrumentId("XNYS:ACME")
    sec = parse_company_facts(inst, _facts(), datetime(2020, 1, 1, tzinfo=UTC))
    yahoo = parse_fundamentals(inst, _yahoo_payload())
    split = parse_splits(inst, {"chart": {"result": [{"events": {"splits": {
        "1": {"date": int(datetime(2024, 8, 1, tzinfo=UTC).timestamp()), "numerator": 4.0, "denominator": 1.0}}}}]}})
    assert len(split) == 1 and split[0].payload["ratio"] == 4.0
    with PITStore(tmp_path) as store:
        store.ingest(sec + yahoo + split)
        st = store.query_fundamentals(datetime(2026, 1, 1, tzinfo=UTC))
        assert set(st["source_id"].to_list()) == {"sec_edgar"}              # Yahoo's rows dropped for this stock
        sp = store.query_splits()
        r = statement_ratios(st, sp).sort("event_time")
    # FY2023 was filed Feb 2024, BEFORE the Aug 2024 4-for-1 split: EPS 4.5 -> 1.125, shares 20 -> 80
    assert abs(r["eps"][0] - 1.125) < 1e-9 and abs(r["bvps"][0] - 400.0 / 80.0) < 1e-9
    assert r["_split"].to_list() == [4.0, 1.0]       # FY2024 was filed after the split: unchanged
