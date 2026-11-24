"""
Routes behind the simple web UI.

Everything here is read-only analysis or a background data/training job. Nothing
in this module can place an order: the UI is paper/analysis only by design.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

import polars as pl
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from learntrade.api import jobs

router = APIRouter(tags=["workspace"])

HORIZON_LABELS = {
    "swing_5d": "about 1 week",
    "swing_20d": "about 1 month",
    "positional_60d": "about 3 months",
}
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.&\-]{0,19}$")


def _cfg() -> Any:
    from learntrade.core.config import get_config
    return get_config()


def _open_store() -> Any:
    from learntrade.data.store.pit import PITStore
    try:
        return PITStore(_cfg().workspace.data_dir, lock_wait=3)
    except Exception as exc:  # DuckDB: another process holds the file
        raise HTTPException(
            status_code=503,
            detail="LearnTrade is busy updating data. Try again in a minute.",
        ) from exc


@router.get("/state")
def state() -> dict[str, Any]:
    from pathlib import Path

    job = jobs.active_job()
    out: dict[str, Any] = {
        "busy": jobs.is_busy(),
        "job": job.public() if job else None,
        "markets": [], "models": [], "llm_enabled": False, "news_articles": 0,
    }
    if out["busy"]:
        return out

    cfg = _cfg()
    out["llm_enabled"] = cfg.llm.enabled
    store = _open_store()
    try:
        for r in store.bar_summary_by_market():
            # Count the stock LIST, not every symbol ever priced (benchmark, single-stock checks).
            # A market without a list (e.g. prices fetched only for portfolio holdings, or fund
            # NAVs) is not offered on the stock screen.
            listed = len(store.get_universe(r["mic"], datetime.now(tz=timezone.utc)))
            if not listed and not r["demo"]:
                continue
            out["markets"].append({
                "mic": r["mic"], "instruments": listed or r["n"],
                "last_bar": r["last_bar"].date().isoformat() if r["last_bar"] else None,
                "demo": bool(r["demo"]),
            })
    finally:
        store.close()

    out["portfolio"] = (Path(cfg.workspace.data_dir) / "portfolio" / "holdings.json").exists()
    models_dir = Path(cfg.workspace.data_dir) / "models"
    if models_dir.exists():
        for p in sorted(models_dir.glob("*.pkl")):
            if "calibrator" in p.name:
                continue
            mic, _, rest = p.stem.partition("_")
            horizon = "_".join(rest.split("_")[:2])
            out["models"].append({"market": mic, "horizon": horizon, "file": p.name})

    from learntrade.cli.main import _corpus_path
    cp = _corpus_path(cfg)
    if cp.exists():
        from learntrade.data.corpus.store import CorpusStore
        c = CorpusStore(cp)
        out["news_articles"] = c.document_count()
        c.close()
    return out


@router.get("/markets")
def markets() -> dict[str, Any]:
    from learntrade.markets.loader import list_packs, get_pack

    rows = []
    for mic in list_packs():
        try:
            p = get_pack(mic)
            rows.append({"mic": mic, "name": p.name, "country": p.country, "currency": p.currency})
        except Exception:
            rows.append({"mic": mic, "name": mic, "country": "", "currency": ""})
    return {"markets": rows}


class StartRequest(BaseModel):
    mode: str                    # "demo" | "real"
    mic: str | None = None
    symbols: str | None = None   # free text: comma / space / newline separated
    index: str | None = None     # e.g. "SP500": use a built-in index with its join dates


def _train_steps(mic: str, since: str, prefix: str = "Teaching the model") -> list[jobs.Step]:
    """One training step per horizon offered on the page (1 week, 1 month, 3 months).

    The 1-month model is required; the others are optional so a short history (too little for
    the 3-month horizon) does not fail the whole setup."""
    return [
        jobs.Step(f"{prefix} ({HORIZON_LABELS[h]})",
                  ["train", "--market", mic, "--since", since, "--horizon", h],
                  optional=(h != "swing_20d"))
        for h in ("swing_20d", "swing_5d", "positional_60d")
    ]


def _parse_symbols(text: str) -> list[str]:
    seen: list[str] = []
    for tok in re.split(r"[\s,;]+", (text or "").upper()):
        if tok and tok not in seen:
            seen.append(tok)
    bad = [t for t in seen if not _SYMBOL_RE.match(t)]
    if bad:
        raise HTTPException(400, f"These don't look like stock symbols: {', '.join(bad[:5])}")
    return seen


@router.post("/start")
def start(req: StartRequest) -> dict[str, Any]:
    if jobs.is_busy():
        raise HTTPException(409, "Another task is already running. Please wait.")
    since = (datetime.now(tz=timezone.utc) - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    if req.mode == "demo":
        steps = [
            jobs.Step("Creating practice data (fake stocks)", ["demo"]),
            *_train_steps("XNYS", "2015-01-01", prefix="Teaching the model on that data"),
        ]
        return jobs.start("Setting up the demo", steps).public()

    if req.mode != "real":
        raise HTTPException(400, "mode must be 'demo' or 'real'")

    if req.index:
        from learntrade.data.index_history import SUPPORTED
        key = req.index.upper()
        if key not in SUPPORTED:
            raise HTTPException(400, f"Built-in indexes: {', '.join(SUPPORTED)}.")
        mic = SUPPORTED[key][0]
        steps = [
            jobs.Step(f"Loading the {SUPPORTED[key][1]} members and when each joined",
                      ["universe", "fetch-index", key, "--since", since]),
            jobs.Step("Downloading 3 years of prices (a few minutes for ~500 stocks)", [
                "data", "pull", "--market", mic, "--since", since, "--capabilities", "bars_daily"]),
            jobs.Step("Loading company financials (earnings, valuation)", [
                "data", "pull", "--market", mic, "--capabilities", "fundamentals,corporate_actions"], optional=True),
            *_train_steps(mic, since),
        ]
        return jobs.start(f"Setting up the {SUPPORTED[key][1]}", steps).public()

    from learntrade.markets.loader import list_packs
    mic = (req.mic or "").upper()
    if mic not in list_packs():
        raise HTTPException(400, f"Unknown market {mic!r}.")
    symbols = _parse_symbols(req.symbols or "")
    if len(symbols) < 8:
        raise HTTPException(
            400, "Please enter at least 8 stocks — the tool compares stocks against each "
                 "other, so it needs a handful to work with.")
    if len(symbols) > 100:
        raise HTTPException(400, "Please start with 100 stocks or fewer.")

    data_dir = _cfg().workspace.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"ui_universe_{mic}.csv"
    csv_path.write_text("symbol\n" + "\n".join(symbols) + "\n")

    steps = [
        jobs.Step("Saving your stock list", ["universe", "import", str(csv_path), "--mic", mic]),
        jobs.Step("Downloading 3 years of prices", [
            "data", "pull", "--market", mic, "--since", since, "--capabilities", "bars_daily"]),
        jobs.Step("Loading company financials (earnings, valuation)", [
            "data", "pull", "--market", mic, "--capabilities", "fundamentals,corporate_actions"], optional=True),
        *_train_steps(mic, since),
        jobs.Step("Fetching recent news headlines", ["news", "pull", "--market", mic],
                  optional=True),
    ]
    return jobs.start(f"Setting up {mic}", steps).public()


@router.post("/refresh")
def refresh() -> dict[str, Any]:
    if jobs.is_busy():
        raise HTTPException(409, "Another task is already running. Please wait.")
    real = [m for m in state()["markets"] if not m["demo"]]
    if not real:
        raise HTTPException(400, "There is no real market data to refresh. The demo uses fake data.")
    since = (datetime.now(tz=timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    train_since = (datetime.now(tz=timezone.utc) - timedelta(days=3 * 365)).strftime("%Y-%m-%d")
    steps: list[jobs.Step] = []
    for m in real:
        mic = m["mic"]
        steps += [
            jobs.Step(f"Getting the latest {mic} prices", [
                "data", "pull", "--market", mic, "--since", since, "--capabilities", "bars_daily"]),
            jobs.Step(f"Fetching {mic} news", ["news", "pull", "--market", mic], optional=True),
            jobs.Step(f"Updating {mic} company financials", [
                "data", "pull", "--market", mic, "--capabilities", "fundamentals,corporate_actions"], optional=True),
            *_train_steps(mic, train_since, prefix=f"Re-teaching the {mic} model"),
        ]
    return jobs.start("Updating your data", steps).public()


@router.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown task")
    return job.public()


def _label(rank_pct: float) -> str:
    if rank_pct >= 0.8:
        return "Among the strongest"
    if rank_pct >= 0.6:
        return "Above average"
    if rank_pct > 0.4:
        return "Middle of the pack"
    if rank_pct > 0.2:
        return "Below average"
    return "Among the weakest"


def _forecasts(market: str, horizon: str, model: str) -> tuple[list[Any], str]:
    from learntrade.cli.main import _generate_current_forecasts
    store = _open_store()
    try:
        return _generate_current_forecasts(_cfg(), store, market, horizon, model)
    finally:
        store.close()


def _reliability(market: str, horizon: str, model: str, rough: bool) -> dict[str, Any]:
    """Plain-language reliability of the model behind a forecast (from its training report)."""
    import json

    from learntrade.cli.main import _model_artifact_paths
    from learntrade.quant.reliability import assess_reliability

    if rough:
        return {"level": "none", "headline": "No trained model yet: these are rough guesses from recent price momentum only.",
                "reasons": [], "facts": {}}
    model_path, _ = _model_artifact_paths(_cfg(), market, horizon, model)
    report_path = model_path.with_name(model_path.stem + "_report.json")
    if not report_path.exists():
        return {"level": "unknown",
                "headline": "This model has no test report yet, so how reliable it is can't be shown.",
                "reasons": ["Tap Update data to re-train it and measure it."], "facts": {}}
    report = json.loads(report_path.read_text())
    last_bar = None
    store = _open_store()
    try:
        t = store.last_bar_time(market)
        if t is not None:
            last_bar = t.date().isoformat()
    finally:
        store.close()
    return assess_reliability(report, last_bar=last_bar)


def _asset_meta(ids: list[str]) -> dict[str, dict]:
    store = _open_store()
    try:
        return store.instrument_meta(ids)
    finally:
        store.close()


def _company(meta: dict[str, dict], iid: str) -> bool:
    from learntrade.quant.trust import is_company
    m = meta.get(iid) or {}
    return is_company(m.get("type"), iid, m.get("name"))


@router.get("/rank")
def rank(market: str, horizon: str = "swing_20d", model: str = "ridge") -> dict[str, Any]:
    from learntrade.quant.trust import NO_EDGE_NOTE, QUICK_NOTE, gate_forecast, is_company

    if jobs.is_busy():
        raise HTTPException(503, "LearnTrade is busy updating data. Try again in a minute.")
    forecasts, source = _forecasts(market, horizon, model)
    if source in ("no_universe", "no_features") or not forecasts:
        return {"ready": False, "reason": source}
    rough = source.startswith("raw momentum")
    reliability = _reliability(market, horizon, model, rough)
    level = reliability.get("level")

    # Funds and indices are not companies: list them apart, never ranked.
    meta = _asset_meta([str(f.instrument_id) for f in forecasts])
    not_ranked = [
        {"instrument": str(f.instrument_id), "symbol": f.instrument_id.symbol,
         "name": (meta.get(str(f.instrument_id)) or {}).get("name")}
        for f in forecasts if not _company(meta, str(f.instrument_id))
    ]
    skip = {r["instrument"] for r in not_ranked}
    ordered = sorted((f for f in forecasts if str(f.instrument_id) not in skip),
                     key=lambda f: f.expected_return, reverse=True)
    n = len(ordered)
    rows = []
    gated = False
    for i, f in enumerate(ordered):
        pct = 1 - i / max(n - 1, 1)
        g = gate_forecast(float(f.expected_return), float(f.p_positive),
                          float(f.quantiles.q05), float(f.quantiles.q95), level)
        gated = g["gated"]
        rows.append({
            "instrument": str(f.instrument_id),
            "symbol": f.instrument_id.symbol,
            "expected_return": g["expected_return"],
            "p_up": g["p_up"],
            "risk": float(f.volatility),
            # A strength label is a forecast too, so it is only given with evidence.
            "label": None if gated else _label(pct),
            "rank": i + 1,
        })
    return {
        "ready": True, "market": market, "horizon": horizon,
        "horizon_label": HORIZON_LABELS.get(horizon, horizon),
        "source": source, "rough": rough, "count": n, "rows": rows,
        "gated": gated, "gate_note": (QUICK_NOTE if rough else NO_EDGE_NOTE) if gated else None,
        "not_ranked": not_ranked,
        "reliability": reliability,
    }


@router.get("/stock/{instrument_id}")
def stock(instrument_id: str, horizon: str = "swing_20d", model: str = "ridge") -> dict[str, Any]:
    from learntrade.cli.main import _news_context
    from learntrade.core.types import InstrumentId

    if jobs.is_busy():
        raise HTTPException(503, "LearnTrade is busy updating data. Try again in a minute.")
    try:
        iid = InstrumentId(instrument_id)
    except Exception:
        raise HTTPException(400, "Expected something like XNYS:SYM01") from None

    forecasts, source = _forecasts(iid.mic, horizon, model)
    fc = next((f for f in forecasts if str(f.instrument_id) == str(iid)), None)
    outside: dict[str, Any] | None = None
    if fc is None:
        from learntrade.cli.main import _forecast_outside_universe
        store = _open_store()
        try:
            fc, ctx, msg = _forecast_outside_universe(_cfg(), store, iid, horizon, model, forecasts)
        finally:
            store.close()
        if fc is None:
            raise HTTPException(404, msg)
        outside, source = ctx, "trained model, single-stock check"

    now = datetime.now(tz=timezone.utc)
    store = _open_store()
    try:
        bars_1y = store.query_bars([iid], as_of=now, since=now - timedelta(days=380))
    finally:
        store.close()
    bars_1y = bars_1y.with_columns(pl.col("close").cast(pl.Float64, strict=False))
    closes = [
        {"d": r["event_time"].date().isoformat(), "c": float(r["close"])}
        for r in bars_1y.iter_rows(named=True) if r["close"] is not None
    ][-120:]

    events, _ = _news_context(_cfg(), str(iid), now)
    from learntrade.quant.trust import (
        NO_EDGE_NOTE, QUICK_NOTE, familiarity, gate_forecast, is_company, non_company_note,
        price_summary,
    )
    rough = source.startswith("raw momentum")
    reliability = _reliability(iid.mic, horizon, model, rough)
    meta = _asset_meta([str(iid)])
    asset_type = (meta.get(str(iid)) or {}).get("type")
    company = _company(meta, str(iid))
    g = gate_forecast(float(fc.expected_return), float(fc.p_positive),
                      float(fc.quantiles.q05), float(fc.quantiles.q95), reliability.get("level"))
    note = (QUICK_NOTE if rough else NO_EDGE_NOTE) if g["gated"] else None
    unfamiliar: list[str] = []
    if outside is not None and company:
        store = _open_store()
        try:
            members = [str(m) for m in store.get_universe(iid.mic, now)]
            fam = familiarity(store, str(iid), members, now)
        finally:
            store.close()
        if not fam["familiar"]:
            unfamiliar = fam["reasons"]
            g = {"gated": True, "expected_return": None, "p_up": None,
                 "low": g["low"] - (g["expected_return"] or 0.0),
                 "high": g["high"] - (g["expected_return"] or 0.0)}
            note = ("Not scored: this stock is very different from the stocks the model learned "
                    "from, so its score would not mean anything. " + " ".join(unfamiliar))
    if g["gated"] and outside is not None:
        outside = {"of": outside.get("of")}          # no "would rank N of M" without evidence
    if not company:
        g = {"gated": True, "expected_return": None, "p_up": None, "low": None, "high": None}
        note, outside = non_company_note(asset_type), ({"of": outside.get("of")} if outside else None)
    fundamentals: dict = {}
    try:
        from learntrade.features.fundamentals import snapshot
        store = _open_store()
        try:
            fundamentals = snapshot(store, str(iid), closes[-1]["c"] if closes else None, now)
        finally:
            store.close()
    except Exception:
        fundamentals = {}
    return {
        "instrument": str(iid), "symbol": iid.symbol, "horizon": horizon,
        "horizon_label": HORIZON_LABELS.get(horizon, horizon),
        "source": source, "rough": rough,
        "is_company": company, "asset_type": asset_type if company else (
            asset_type if asset_type and asset_type.upper() != "EQUITY" else "FUND_OR_INDEX"),
        "name": (meta.get(str(iid)) or {}).get("name"),
        "gated": g["gated"], "gate_note": note, "unfamiliar": unfamiliar,
        "expected_return": g["expected_return"], "p_up": g["p_up"],
        "risk": float(fc.volatility),
        "low": g["low"], "high": g["high"],
        "price_summary": price_summary(
            [float(r["close"]) for r in bars_1y.iter_rows(named=True) if r["close"] is not None]),
        "closes": closes, "news": events,
        "fundamentals": fundamentals if company else {}, "outside": outside,
        "reliability": reliability,
    }


# ── My portfolio ──────────────────────────────────────────────────────────────

from fastapi import File, UploadFile  # noqa: E402

_MAX_UPLOAD = 5 * 1024 * 1024


@router.get("/portfolio")
def portfolio() -> dict[str, Any]:
    """Analysis of the saved holdings (allocation, concentration, P&L, risk)."""
    from learntrade.portfolio.analysis import analyse
    from learntrade.portfolio.book import load_holdings

    if jobs.is_busy():
        raise HTTPException(503, "LearnTrade is busy updating data. Try again in a minute.")
    holdings, meta = load_holdings(_cfg().workspace.data_dir)
    if not holdings:
        return {"ready": False}
    store = _open_store()
    try:
        out = analyse(store, holdings)
    finally:
        store.close()
    return {"ready": True, **meta, **out}


@router.post("/portfolio/import")
async def portfolio_import(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Save the uploaded broker exports and start a job that imports them and fetches prices."""
    if jobs.is_busy():
        raise HTTPException(409, "Another task is already running. Please wait.")
    from learntrade.portfolio.book import portfolio_dir
    up = portfolio_dir(_cfg().workspace.data_dir) / "uploads"
    up.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    paths: list[str] = []
    for i, f in enumerate(files):
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", _file_name(f.filename or f"file{i}.csv"))[-80:]
        if not name.lower().endswith((".csv", ".xlsx", ".xlsm", ".txt")):
            raise HTTPException(400, f"{f.filename}: please upload a CSV or Excel (.xlsx) file.")
        data = await f.read()
        if len(data) > _MAX_UPLOAD:
            raise HTTPException(400, f"{f.filename} is larger than 5 MB; that is not a holdings file.")
        p = up / f"{stamp}_{i}_{name}"
        p.write_bytes(data)
        paths.append(str(p))
    if not paths:
        raise HTTPException(400, "No file received.")
    steps = [jobs.Step("Reading your holdings and finding each one", ["portfolio", "import", *paths])]
    return jobs.start("Importing your portfolio", steps).public()


@router.post("/portfolio/refresh")
def portfolio_refresh() -> dict[str, Any]:
    if jobs.is_busy():
        raise HTTPException(409, "Another task is already running. Please wait.")
    return jobs.start("Updating portfolio prices",
                      [jobs.Step("Getting the latest prices and NAVs", ["portfolio", "refresh"])]).public()


def _file_name(filename: str) -> str:
    """Just the file name from whatever path a browser sends."""
    return filename.replace("\\", "/").rsplit("/", 1)[-1]
