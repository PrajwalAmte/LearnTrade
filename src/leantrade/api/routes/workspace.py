"""
Routes behind the simple web UI.

Everything here is read-only analysis or a background data/training job. Nothing
in this module can place an order: the UI is paper/analysis only by design.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from leantrade.api import jobs

router = APIRouter(tags=["workspace"])

HORIZON_LABELS = {
    "swing_5d": "about 1 week",
    "swing_20d": "about 1 month",
    "positional_60d": "about 3 months",
}
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.&\-]{0,19}$")


def _cfg() -> Any:
    from leantrade.core.config import get_config
    return get_config()


def _open_store() -> Any:
    from leantrade.data.store.pit import PITStore
    try:
        return PITStore(_cfg().workspace.data_dir)
    except Exception as exc:  # DuckDB: another process holds the file
        raise HTTPException(
            status_code=503,
            detail="LeanTrade is busy updating data. Try again in a minute.",
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
        df = store._fetch_pl(
            "SELECT split_part(instrument_id, ':', 1) AS mic, "
            "COUNT(DISTINCT instrument_id) AS n, MAX(event_time) AS last_bar, "
            "BOOL_OR(source_id = 'demo') AS demo "
            "FROM observations WHERE kind = 'bar' GROUP BY 1 ORDER BY 1"
        )
        for r in df.iter_rows(named=True):
            out["markets"].append({
                "mic": r["mic"], "instruments": r["n"],
                "last_bar": r["last_bar"].date().isoformat() if r["last_bar"] else None,
                "demo": bool(r["demo"]),
            })
    finally:
        store.close()

    models_dir = Path(cfg.workspace.data_dir) / "models"
    if models_dir.exists():
        for p in sorted(models_dir.glob("*.pkl")):
            if "calibrator" in p.name:
                continue
            mic, _, rest = p.stem.partition("_")
            horizon = "_".join(rest.split("_")[:2])
            out["models"].append({"market": mic, "horizon": horizon, "file": p.name})

    from leantrade.cli.main import _corpus_path
    cp = _corpus_path(cfg)
    if cp.exists():
        from leantrade.data.corpus.store import CorpusStore
        c = CorpusStore(cp)
        out["news_articles"] = c.document_count()
        c.close()
    return out


@router.get("/markets")
def markets() -> dict[str, Any]:
    from leantrade.markets.loader import list_packs, get_pack

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
            jobs.Step("Teaching the model on that data", ["train", "--market", "XNYS"]),
        ]
        return jobs.start("Setting up the demo", steps).public()

    if req.mode != "real":
        raise HTTPException(400, "mode must be 'demo' or 'real'")

    from leantrade.markets.loader import list_packs
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
        jobs.Step("Teaching the model on that history", [
            "train", "--market", mic, "--since", since]),
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
            jobs.Step(f"Re-teaching the {mic} model", ["train", "--market", mic, "--since", train_since]),
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
    from leantrade.cli.main import _generate_current_forecasts
    store = _open_store()
    try:
        return _generate_current_forecasts(_cfg(), store, market, horizon, model)
    finally:
        store.close()


@router.get("/rank")
def rank(market: str, horizon: str = "swing_20d", model: str = "ridge") -> dict[str, Any]:
    if jobs.is_busy():
        raise HTTPException(503, "LeanTrade is busy updating data. Try again in a minute.")
    forecasts, source = _forecasts(market, horizon, model)
    if source in ("no_universe", "no_features") or not forecasts:
        return {"ready": False, "reason": source}
    ordered = sorted(forecasts, key=lambda f: f.expected_return, reverse=True)
    n = len(ordered)
    rows = []
    for i, f in enumerate(ordered):
        pct = 1 - i / max(n - 1, 1)
        rows.append({
            "instrument": str(f.instrument_id),
            "symbol": f.instrument_id.symbol,
            "expected_return": float(f.expected_return),
            "p_up": float(f.p_positive),
            "risk": float(f.volatility),
            "label": _label(pct),
            "rank": i + 1,
        })
    rough = source.startswith("raw momentum")
    return {
        "ready": True, "market": market, "horizon": horizon,
        "horizon_label": HORIZON_LABELS.get(horizon, horizon),
        "source": source, "rough": rough, "count": n, "rows": rows,
    }


@router.get("/stock/{instrument_id}")
def stock(instrument_id: str, horizon: str = "swing_20d", model: str = "ridge") -> dict[str, Any]:
    from leantrade.cli.main import _news_context
    from leantrade.core.types import InstrumentId

    if jobs.is_busy():
        raise HTTPException(503, "LeanTrade is busy updating data. Try again in a minute.")
    try:
        iid = InstrumentId(instrument_id)
    except Exception:
        raise HTTPException(400, "Expected something like XNYS:SYM01") from None

    forecasts, source = _forecasts(iid.mic, horizon, model)
    fc = next((f for f in forecasts if str(f.instrument_id) == str(iid)), None)
    if fc is None:
        raise HTTPException(404, "No data for that stock yet.")

    now = datetime.now(tz=timezone.utc)
    store = _open_store()
    try:
        bars = store.query_bars([iid], as_of=now, since=now - timedelta(days=200))
    finally:
        store.close()
    closes = [
        {"d": r["event_time"].date().isoformat(), "c": float(r["close"])}
        for r in bars.iter_rows(named=True) if r["close"] is not None
    ][-120:]

    events, _ = _news_context(_cfg(), str(iid), now)
    return {
        "instrument": str(iid), "symbol": iid.symbol, "horizon": horizon,
        "horizon_label": HORIZON_LABELS.get(horizon, horizon),
        "source": source, "rough": source.startswith("raw momentum"),
        "expected_return": float(fc.expected_return), "p_up": float(fc.p_positive),
        "risk": float(fc.volatility),
        "low": float(fc.quantiles.q05), "high": float(fc.quantiles.q95),
        "closes": closes, "news": events,
    }
