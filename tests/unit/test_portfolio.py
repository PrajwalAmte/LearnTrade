"""Portfolio import and analysis: broker layouts, fund lookup, allocation, concentration, risk."""

from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from learntrade.core.types import InstrumentId, Observation
from learntrade.data.store.pit import PITStore
from learntrade.portfolio.analysis import analyse
from learntrade.portfolio.book import load_holdings, parse_mfapi, save_holdings
from learntrade.portfolio.importer import parse_holdings
from learntrade.portfolio.resolve import (
    Holding, Resolver, etf_class, fund_class, parse_amfi, resolve, size_hint,
)

UTC = timezone.utc

KITE = b"""Instrument,Qty.,Avg. cost,LTP,Invested,Cur. val,P&L,Net chg.,Day chg.
HDFCBANK,40,805.25,950,32210,38000,5790,17.98,0.5
GOLDBEES-EQ,300,62,75,18600,22500,3900,20.9,0.0
Total,,,,50810,60500,,,
"""

GROWW_MF = b"""Holdings statement as on 20-11-2026
Name,Test User
,
Scheme Name,AMC,Category,Sub-category,Folio No.,Source,Units,Invested Value,Current Value,Returns,XIRR
Parag Parikh Flexi Cap Fund Direct Growth,PPFAS Mutual Fund,Equity,Flexi Cap,123,Groww,520.5,40000,49700,9700,14%
"""


def test_kite_csv_columns_and_series_suffix():
    res = parse_holdings(KITE, "holdings.csv")
    assert res.columns["csv"]["symbol"] == "Instrument" and res.columns["csv"]["quantity"] == "Qty."
    assert [h.symbol for h in res.holdings] == ["HDFCBANK", "GOLDBEES"]        # total row ignored
    assert res.holdings[0].avg_price == pytest.approx(805.25) and res.holdings[0].price == 950


def test_groww_funds_with_preamble_rows():
    res = parse_holdings(GROWW_MF, "mf.csv")
    h = res.holdings[0]
    assert h.name.startswith("Parag Parikh") and h.quantity == pytest.approx(520.5)
    assert h.avg_price == pytest.approx(40000 / 520.5)                        # from invested value
    assert h.value == 49700


def test_console_style_excel_with_two_sheets():
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Equity"
    ws.append(["Client ID", "AB1234"])
    ws.append([])
    ws.append(["Symbol", "ISIN", "Sector", "Quantity Available", "Quantity Discrepant",
               "Average Price", "Previous Closing Price", "Unrealized P&L"])
    ws.append(["SBIN", "INE062A01020", "BANKS", 50, 0, 720, 800, 4000])
    mf = wb.create_sheet("Mutual Funds")
    mf.append(["Symbol", "ISIN", "Instrument Type", "Quantity Available", "Average Price",
               "Previous Closing Price"])
    mf.append(["PARAG PARIKH FLEXI CAP FUND - DIRECT PLAN", "INF879O01027", "Equity", 100.5, 60, 95])
    buf = io.BytesIO()
    wb.save(buf)
    res = parse_holdings(buf.getvalue(), "holdings.xlsx")
    assert {h.sheet for h in res.holdings} == {"Equity", "Mutual Funds"}
    sbin = next(h for h in res.holdings if h.sheet == "Equity")
    assert (sbin.symbol, sbin.isin, sbin.sector, sbin.quantity) == ("SBIN", "INE062A01020", "BANKS", 50)
    fund = next(h for h in res.holdings if h.sheet == "Mutual Funds")
    assert fund.isin == "INF879O01027" and fund.quantity == pytest.approx(100.5)


def test_no_table_is_reported_not_crashed():
    res = parse_holdings(b"hello,world\n1,2\n", "x.csv")
    assert res.holdings == [] and "no holdings table" in res.skipped[0]


AMFI_TEXT = """Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Plan;Option;Net Asset Value;Date

Open Ended Schemes(Equity Scheme - Flexi Cap Fund)

PPFAS Mutual Fund

122639;INF879O01027;-;Parag Parikh Flexi Cap Fund;Direct Plan;Growth;95.4585;20-Nov-2026

Open Ended Schemes(Other Scheme - Gold ETF)

140088;INF204KB17I5;-;Nippon India ETF Gold BeES;-;-;75.10;20-Nov-2026
"""


def test_amfi_parse_and_fund_classes():
    rows = parse_amfi(AMFI_TEXT)
    assert rows[0]["code"] == "122639" and rows[0]["category"] == "Equity Scheme - Flexi Cap Fund"
    assert rows[1]["category"] == "Other Scheme - Gold ETF"
    assert fund_class("Equity Scheme - Small Cap Fund", "X Small Cap Fund") == "Equity funds"
    assert fund_class("Debt Scheme - Liquid Fund", "X Liquid") == "Debt"
    assert fund_class("Other Scheme - FoF Domestic", "X Gold Savings Fund") == "Gold"
    assert etf_class("Nippon India Silver ETF") == "Silver" and etf_class("Nippon India ETF Nifty 50 BeES") == "Equity funds"
    assert size_hint("Equity Scheme - Small Cap Fund", "") == "small"
    assert size_hint("Equity Scheme - Flexi Cap Fund", "") == "mixed"


def test_resolve_offline_from_cached_lookups(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "amfi_navall.txt").write_text(AMFI_TEXT)
    (cache / "symbol_lookup.json").write_text(json.dumps({
        "HDFCBANK": [{"symbol": "HDFCBANK.NS", "exchange": "NSI", "quoteType": "EQUITY",
                      "sector": "Financial Services", "longname": "HDFC Bank Limited"}],
        "GOLDBEES": [{"symbol": "GOLDBEES.NS", "exchange": "NSI", "quoteType": "EQUITY",
                      "longname": "Nippon India ETF Gold BeES"}],
    }))
    raw = parse_holdings(KITE, "k.csv").holdings + parse_holdings(GROWW_MF, "m.csv").holdings
    hs = resolve(raw, Resolver(cache, offline=True))
    by = {h.id: h for h in hs}
    assert by["XNSE:HDFCBANK"].kind == "stock" and by["XNSE:HDFCBANK"].sector == "Financial Services"
    assert by["XNSE:GOLDBEES"].kind == "etf" and by["XNSE:GOLDBEES"].asset_class == "Gold"
    fund = by["AMFI:122639"]
    assert fund.kind == "fund" and fund.asset_class == "Equity funds" and "Flexi Cap" in fund.category


def test_mfapi_nav_history_becomes_daily_prices():
    obs = parse_mfapi("122639", {"data": [{"date": "20-11-2026", "nav": "95.4585"},
                                          {"date": "bad", "nav": "1"}]})
    assert len(obs) == 1 and obs[0].payload["close"] == pytest.approx(95.4585)
    assert str(obs[0].instrument_id) == "AMFI:122639"


def _series(iid, start_px, drift, vol, seed, n=300):
    rng = np.random.default_rng(seed)
    now = datetime.now(tz=UTC)
    px, out = start_px, []
    for d in range(n):
        px *= float(np.exp(drift / 252 + rng.normal(0, vol / np.sqrt(252))))
        t = (now - timedelta(days=n - d)).replace(hour=10)
        out.append(Observation(instrument_id=InstrumentId(iid), source_id="t", kind="bar",
                               event_time=t, available_at=t, payload={"close": px, "volume": 1}))
    return out


def test_analysis_allocation_concentration_and_facts(tmp_path):
    hs = [
        Holding(name="Bank A", quantity=100, avg_price=100, id="XNSE:BANKA", kind="stock",
                asset_class="Indian stocks", sector="Financial Services"),
        Holding(name="Bank B", quantity=100, avg_price=100, id="XNSE:BANKB", kind="stock",
                asset_class="Indian stocks", sector="Financial Services"),
        Holding(name="Tech C", quantity=10, avg_price=500, id="XNSE:TECHC", kind="stock",
                asset_class="Indian stocks", sector="Technology"),
        Holding(name="Gold ETF", quantity=50, avg_price=60, id="XNSE:GOLDBEES", kind="etf",
                asset_class="Gold"),
        Holding(name="Mystery Ltd", quantity=10, avg_price=10, file_price=12, note="not found"),
    ]
    obs = (_series("XNSE:BANKA", 100, 0.1, 0.2, 1) + _series("XNSE:BANKB", 100, 0.1, 0.2, 2)
           + _series("XNSE:TECHC", 500, -0.6, 0.3, 3) + _series("XNSE:GOLDBEES", 60, 0.0, 0.1, 4))
    with PITStore(tmp_path) as store:
        store.ingest(obs)
        save_holdings(tmp_path, hs, "test.csv")
        loaded, meta = load_holdings(tmp_path)
        a = analyse(store, loaded)
    assert meta["source"] == "test.csv" and len(loaded) == 5
    t = a["totals"]
    assert t["holdings"] == 5 and t["priced_from_market"] == 4
    assert sum(h["weight"] for h in a["holdings"]) == pytest.approx(1.0)
    classes = {x["label"]: x["weight"] for x in a["allocation"]["asset_class"]}
    assert set(classes) == {"Indian stocks", "Gold", "Other"}
    sectors = {x["label"]: x["weight"] for x in a["allocation"]["sector"]}
    assert sectors["Financial Services"] > 0.30 and sum(sectors.values()) == pytest.approx(1.0)
    assert any("Financial Services is" in f for f in a["worth_knowing"])
    assert any("Tech C is" in f and "below what you paid" in f for f in a["worth_knowing"])
    assert any("Mystery Ltd" in n for n in a["notes"])
    assert 1 < a["concentration"]["effective_holdings"] <= 5
    assert a["risk"]["days"] > 100 and a["risk"]["max_drawdown"] < 0


def test_file_made_before_a_bonus_is_put_on_todays_basis(tmp_path):
    hs = [Holding(name="Bonus Co", quantity=40, avg_price=1610, id="XNSE:BONUSCO", kind="stock",
                  asset_class="Indian stocks", file_price=1900)]
    when = datetime.now(tz=UTC) - timedelta(days=100)
    bonus = Observation(instrument_id=InstrumentId("XNSE:BONUSCO"), source_id="yahoo",
                        kind="corporate_action", event_time=when, available_at=when,
                        source_version="split", payload={"type": "split", "ratio": 2.0})
    with PITStore(tmp_path) as store:
        store.ingest(_series("XNSE:BONUSCO", 950, 0.0, 0.01, 5) + [bonus])   # market: new basis
        a = analyse(store, hs)
    h = a["holdings"][0]
    assert h["quantity"] == 80 and h["avg_price"] == pytest.approx(805)
    assert h["price_source"] == "market"
    assert h["pnl_pct"] == pytest.approx(h["price"] / 805 - 1)          # not a fake -50%
    assert any("2-for-1 split or bonus" in n for n in a["notes"])


def test_ordinary_price_move_is_not_mistaken_for_a_split(tmp_path):
    hs = [Holding(name="Mover", quantity=10, avg_price=700, id="XNSE:MOVER", kind="stock",
                  asset_class="Indian stocks", file_price=800)]
    with PITStore(tmp_path) as store:
        store.ingest(_series("XNSE:MOVER", 1130, 0.0, 0.01, 6))          # up ~40% since the file
        a = analyse(store, hs)
    h = a["holdings"][0]
    assert h["quantity"] == 10 and h["price_source"] == "market" and h["price"] > 1000
    assert not any("split or bonus" in n for n in a["notes"])


def test_web_import_starts_a_job_and_saves_the_file(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from learntrade.api import jobs
    from learntrade.api.app import create_app
    from learntrade.core import config as cfgmod

    monkeypatch.setenv("LEARNTRADE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cfgmod, "_config", None)
    started = {}
    monkeypatch.setattr(jobs, "start", lambda title, steps: started.update(title=title, steps=steps)
                        or type("J", (), {"public": lambda self: {"id": "x"}})())
    c = TestClient(create_app())
    r = c.post("/api/app/portfolio/import", files=[("files", ("my holdings.csv", KITE, "text/csv"))])
    assert r.status_code == 200, r.text
    args = started["steps"][0].args
    assert args[:2] == ["portfolio", "import"] and args[2].endswith("my_holdings.csv")
    assert (tmp_path / "portfolio" / "uploads").exists()
    bad = c.post("/api/app/portfolio/import", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))])
    assert bad.status_code == 400
    assert c.get("/api/app/portfolio").json() == {"ready": False}
    monkeypatch.setattr(cfgmod, "_config", None)
