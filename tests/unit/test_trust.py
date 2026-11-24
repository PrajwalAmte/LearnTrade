"""Trust rules: forecasts are shown only with evidence; funds and unfamiliar stocks get no score."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from learntrade.core.types import InstrumentId, Observation
from learntrade.data.providers.yahoo import YahooProvider
from learntrade.data.store.pit import PITStore
from learntrade.quant.trust import (
    familiarity, gate_forecast, is_company, non_company_note, price_summary,
)

UTC = timezone.utc


@pytest.mark.parametrize("level", ["no_evidence", "none", "unknown", None])
def test_no_evidence_withholds_outlook_and_probability(level):
    g = gate_forecast(0.03, 0.66, -0.07, 0.13, level)
    assert g["gated"] and g["expected_return"] is None and g["p_up"] is None
    assert g["low"] == pytest.approx(-0.10) and g["high"] == pytest.approx(0.10)   # re-centred: a swing


def test_partial_evidence_is_shrunk_and_full_evidence_kept():
    half = gate_forecast(0.04, 0.70, -0.06, 0.14, "unproven")
    assert not half["gated"]
    assert half["expected_return"] == pytest.approx(0.02) and half["p_up"] == pytest.approx(0.60)
    full = gate_forecast(0.04, 0.70, -0.06, 0.14, "modest")
    assert full["expected_return"] == pytest.approx(0.04) and full["p_up"] == pytest.approx(0.70)


def test_funds_and_indices_are_not_companies():
    assert not is_company("ETF", "XNSE:GOLDBEES")
    assert not is_company(None, "XNSE:SILVERBEES")          # name fallback when type unknown
    assert not is_company(None, "XNSE:NIFTYIETF")
    assert not is_company("INDEX", "XNSE:NIFTY50")
    assert is_company("EQUITY", "XNSE:TCS") and is_company(None, "XNSE:TCS")
    assert "ETF" in non_company_note("ETF")
    # Yahoo reports NSE ETFs as EQUITY; the symbol or the fund's name must still catch them.
    assert not is_company("EQUITY", "XNSE:GOLDBEES")
    assert not is_company("EQUITY", "XNSE:GOLD1", "Nippon India ETF Gold BeES")
    assert not is_company("EQUITY", "XNYS:GLD", "SPDR Gold Trust")
    assert is_company("EQUITY", "XNSE:KAYNES", "Kaynes Technology India Limited")


def test_yahoo_reports_asset_type_and_store_returns_it(tmp_path, monkeypatch):
    payload = {"chart": {"result": [{
        "meta": {"instrumentType": "ETF", "longName": "Nippon India ETF Gold BeES"},
        "timestamp": [1735000000], "indicators": {"quote": [{"close": [60.0], "open": [60.0],
        "high": [60.0], "low": [60.0], "volume": [1]}], "adjclose": [{"adjclose": [60.0]}]}}]}}
    p = YahooProvider()
    monkeypatch.setattr(p, "_get_chart", lambda *a, **k: payload)
    obs = list(p._fetch_bars(InstrumentId("XNSE:GOLDBEES"), datetime(2024, 1, 1, tzinfo=UTC),
                             datetime(2030, 1, 1, tzinfo=UTC)))
    assert sum(o.kind == "bar" for o in obs) == 1
    with PITStore(tmp_path) as store:
        store.ingest(obs)
        assert store.instrument_types(["XNSE:GOLDBEES", "XNSE:TCS"]) == {"XNSE:GOLDBEES": "ETF"}


def test_price_summary_measures_drawdown_and_returns():
    px = list(np.linspace(100, 200, 200)) + list(np.linspace(200, 150, 60))
    s = price_summary(px)
    assert s["max_drawdown"] == pytest.approx(-0.25, abs=1e-6)
    assert s["return_1y"] == pytest.approx(150 / px[-250] - 1)
    assert price_summary([1.0] * 5) == {}


def _bars(iid, px, vol, n=150, seed=0):
    rng = np.random.default_rng(seed)
    now = datetime.now(tz=UTC)
    out, p = [], px
    for d in range(n):
        p *= float(np.exp(rng.normal(0, vol / np.sqrt(252))))
        t = now - timedelta(days=n - d)
        out.append(Observation(instrument_id=InstrumentId(iid), source_id="t", kind="bar",
                               event_time=t, available_at=t,
                               payload={"close": p, "adj_close": p, "volume": 1_000_000}))
    return out


def _annual(iid, shares):
    t = datetime.now(tz=UTC) - timedelta(days=200)
    return Observation(instrument_id=InstrumentId(iid), source_id="yahoo", kind="fundamental",
                       event_time=t, available_at=t, source_version="annual",
                       payload={"period": "annual", "shares": shares, "eps": 1.0})


def test_much_smaller_or_wilder_stock_is_unfamiliar(tmp_path):
    members = [f"XNSE:BIG{i}" for i in range(8)]
    obs = []
    for i, m in enumerate(members):
        obs += _bars(m, 1000, 0.20, seed=i) + [_annual(m, 1e9)]          # ~1e12 market value
    obs += _bars("XNSE:TINY", 300, 0.60, seed=99) + [_annual("XNSE:TINY", 1e8)]   # ~3e10, wild
    obs += _bars("XNSE:PEER", 900, 0.22, seed=50) + [_annual("XNSE:PEER", 1e9)]
    now = datetime.now(tz=UTC)
    with PITStore(tmp_path) as store:
        store.ingest(obs)
        tiny = familiarity(store, "XNSE:TINY", members, now)
        peer = familiarity(store, "XNSE:PEER", members, now)
    assert not tiny["familiar"] and len(tiny["reasons"]) == 2
    assert "smaller" in tiny["reasons"][0] and "moves much more" in tiny["reasons"][1]
    assert peer["familiar"] and peer["reasons"] == []
