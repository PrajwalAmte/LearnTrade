"""
LearnTrade as an MCP server (Model Context Protocol), for Claude Code, Codex and any other
MCP client.

Read-only: these tools look at the data and models already on this computer. They never
place orders (LearnTrade cannot) and never change your stock lists or retrain models. The one
exception is opt-in per call: ``stock(..., download=True)`` fetches prices for a symbol that
is not in your list and caches them locally, so it can be scored.

Transport is stdio, so nothing may be printed to stdout except protocol messages; all logging
goes to stderr.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from learntrade import __version__ as _version

INSTRUCTIONS = """\
LearnTrade ranks stocks in the user's own list by expected performance versus the market and
reports how well that ranking has worked in testing. Start with `status` to see which markets
have data, then `rank` for a market, and `stock` for one company. Always pass along the
`reliability` verdict with any ranking: it says how much weight the ranking deserves. This is
information, not financial advice, and LearnTrade never trades.
"""


from mcp.server.mcpserver.exceptions import ToolError   # its message is shown to the model as-is


def _not_busy() -> None:
    """Fail fast with a clear message while the web page is updating the data."""
    from learntrade.core.activity import running_job
    from learntrade.core.config import get_config
    job = running_job(get_config().workspace.data_dir)
    if job:
        raise ToolError(f"LearnTrade is busy: '{job.get('title')}' has been running since "
                        f"{job.get('started')}. Try again when it finishes (usually a few minutes).")


def _call(fn, *args, **kwargs) -> Any:
    """Run a web-route function and turn its HTTP errors into plain tool errors."""
    _not_busy()
    try:
        return fn(*args, **kwargs)
    except HTTPException as exc:
        raise ToolError(str(exc.detail)) from None


_HORIZONS = {"swing_5d": "about 1 week", "swing_20d": "about 1 month", "positional_60d": "about 3 months"}


def _check_horizon(horizon: str) -> str:
    if horizon not in _HORIZONS:
        raise ToolError(f"Unknown horizon {horizon!r}. Use one of: "
                        + ", ".join(f"{k} ({v})" for k, v in _HORIZONS.items()) + ".")
    return horizon


def _market_of(symbol: str, market: str | None) -> str:
    if ":" in symbol:
        return symbol.split(":")[0].upper()
    if market:
        return market.upper()
    raise ToolError("Give the market, e.g. market='XNSE', or write the symbol as 'XNSE:TCS'.")


# ── Tool implementations (plain functions, easy to test) ─────────────────────

def tool_status() -> dict[str, Any]:
    from learntrade.api.routes import workspace as w
    st = _call(w.state)
    return {
        "markets": st["markets"],
        "trained_models": st["models"],
        "news_articles": st["news_articles"],
        "ai_research_enabled": st["llm_enabled"],
        "note": "last_bar is the date of the newest stored price. 'demo' markets use made-up data.",
    }


def tool_rank(market: str, horizon: str = "swing_20d", model: str = "ridge", top: int = 15) -> dict[str, Any]:
    from learntrade.api.routes import workspace as w
    res = _call(w.rank, market.upper(), _check_horizon(horizon), model)
    if not res.get("ready"):
        raise ToolError(f"No ranking for {market}: {res.get('reason')}. "
                        "Set up the market first (the web page or `learntrade data pull`).")
    rows = res["rows"]
    top = max(1, min(int(top), len(rows)))
    keep = lambda r: {k: r[k] for k in ("rank", "instrument", "expected_return", "p_up", "risk", "label")}
    return {
        "market": res["market"], "horizon": res["horizon_label"], "model_source": res["source"],
        "stocks": res["count"],
        "strongest": [keep(r) for r in rows[:top]],
        "weakest": [keep(r) for r in rows[-top:]] if len(rows) > top else [],
        "reliability": res["reliability"],
        "units": "expected_return is expected excess return over the benchmark for the horizon "
                 "(0.02 = +2%); p_up is the chance it beats the benchmark over the horizon (not the chance its price rises); risk is the volatility estimate.",
    }


def tool_stock(symbol: str, market: str | None = None, horizon: str = "swing_20d",
               model: str = "ridge", download: bool = False) -> dict[str, Any]:
    from learntrade.api.routes import workspace as w
    mic = _market_of(symbol, market)
    _check_horizon(horizon)
    iid = symbol.upper() if ":" in symbol else f"{mic}:{symbol.upper()}"
    if not download:
        # Only answer for stocks already in the list; a download changes local data.
        forecasts, _ = _call(w._forecasts, mic, horizon, model)
        if not any(str(f.instrument_id) == iid for f in forecasts):
            raise ToolError(f"{iid} is not in your {mic} list. Call again with download=True to "
                            "fetch its prices and score it against your list.")
    d = _call(w.stock, iid, horizon, model)
    d.pop("closes", None)
    d["news"] = d.get("news", [])[:8]
    return d


def tool_reliability(market: str, horizon: str = "swing_20d", model: str = "ridge") -> dict[str, Any]:
    from learntrade.api.routes import workspace as w
    market = market.upper()
    forecasts, source = _call(w._forecasts, market, _check_horizon(horizon), model)
    if not forecasts:
        raise ToolError(f"Nothing to assess for {market} ({source}).")
    return w._reliability(market, horizon, model, source.startswith("raw momentum"))


def tool_evaluate(market: str, horizon: str = "swing_20d", holdout_days: int = 120,
                  since: str = "2015-01-01", top_n: int = 10) -> dict[str, Any]:
    from learntrade.backtest.evaluation import run_evaluation, to_markdown
    from learntrade.cli.main import _require_model_class
    from learntrade.core.config import get_config
    from learntrade.data.store.pit import PITStore

    _not_busy()
    _check_horizon(horizon)
    cfg = get_config()
    since_dt = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    models = {n: _require_model_class(n) for n in ("ridge", "lgbm", "lgbm_rank")}
    try:
        store = PITStore(cfg.workspace.data_dir)
    except Exception as exc:
        raise ToolError("The LearnTrade data is in use by another process (for example the web "
                        "page while it updates). Try again in a minute.") from exc
    try:
        res = run_evaluation(store, market.upper(), horizon, since_dt, holdout_days, models, top_n=top_n)
    finally:
        store.close()
    if not res.candidates:
        raise ToolError("Nothing could be evaluated: " + "; ".join(res.notes))
    return {"report_markdown": to_markdown(res),
            "note": "Models are fit on the earlier data and scored on the untouched final days."}


# ── Server ────────────────────────────────────────────────────────────────────

def build_server():
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    @asynccontextmanager
    async def lifespan(_server):
        # The transport already holds the real stdout. From here on, a stray print() inside
        # any library goes to stderr instead of corrupting the protocol stream.
        sys.stdout = sys.stderr
        yield {}

    server = MCPServer("learntrade", instructions=INSTRUCTIONS, lifespan=lifespan, version=_version)
    ro = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    server.tool(name="status", description="Which markets have data (with the newest price date), "
                "which models are trained, and whether AI research is enabled.", annotations=ro)(tool_status)
    server.tool(name="rank", description="Ranked list of the stocks in one market, strongest first: "
                "expected excess return, chance of beating the market, risk, plus the reliability verdict that "
                "says how well this ranking has worked in testing. market is a MIC like XNSE or XNYS; "
                "horizon is swing_5d (about 1 week), swing_20d (about 1 month) or positional_60d (about 3 months).", annotations=ro)(tool_rank)
    server.tool(name="stock", description="One company: forecast, rank context, key financials "
                "(P/E, ROE, growth), recent headlines and the reliability verdict. symbol may be "
                "'XNSE:TCS' or 'TCS' with market. Stocks outside the list need download=True, which "
                "fetches and caches that stock's prices locally.", annotations=ToolAnnotations(
                    readOnlyHint=False, destructiveHint=False, openWorldHint=True))(tool_stock)
    server.tool(name="reliability", description="How well the model behind a market's ranking did in "
                "testing: verdict, reasons and the numbers (rank IC, untouched final test, stock-list "
                "basis).", annotations=ro)(tool_reliability)
    server.tool(name="evaluate", description="Fair out-of-sample comparison of a momentum baseline, ridge, "
                "LightGBM and the ranker on the same stocks, dates and costs. Can take from seconds "
                "to several minutes depending on the number of stocks.", annotations=ro)(tool_evaluate)
    return server


def serve() -> None:
    """Run the MCP server on stdio."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="%(message)s")
    try:   # keep the CLI's pretty console silent: stdout belongs to the protocol
        from learntrade.cli import main as cli
        cli.console.quiet = True
    except Exception:
        pass
    build_server().run("stdio")
