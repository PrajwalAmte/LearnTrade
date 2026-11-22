"""
LearnTrade CLI.

Usage:
    learntrade <command> [options]

All commands that need a workspace first run `learntrade init` (or the browser
wizard at `learntrade serve --open`) to create config/learntrade.toml.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

app = typer.Typer(
    name="learntrade",
    help="Local-first trading intelligence engine.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

console = Console()

# Model name -> "module:ClassName", shared by `train` and every command that
# needs to load a previously trained model back off disk (rank, rebalance,
# backtest, analyze). Keeping one registry means the artifact path a loader
# looks for always matches the path `train` actually wrote.
_MODEL_REGISTRY: dict[str, str] = {
    "ridge": "learntrade.quant.models.ridge:RidgeReturnModel",
    "lgbm": "learntrade.quant.models.lgbm_return:LGBMReturnModel",
    "lgbm_rank": "learntrade.quant.models.lgbm_rank:LGBMRankModel",
}


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)],
    )


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command()
def mcp(
    data_dir: Optional[str] = typer.Option(
        None, help="Folder with your LearnTrade data (default: ./data or LEARNTRADE_DATA_DIR)"),
) -> None:
    """Run LearnTrade as an MCP server (stdio) for Claude Code, Codex and other MCP clients."""
    import os
    if data_dir:
        os.environ["LEARNTRADE_DATA_DIR"] = str(Path(data_dir).expanduser().resolve())
    from learntrade.mcp_server import serve as _serve
    _serve()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address"),
    port: int = typer.Option(8787, help="Port"),
    open_browser: bool = typer.Option(False, "--open", help="Open the browser after start"),
    log_level: str = typer.Option("info", help="Uvicorn log level"),
) -> None:
    """Start the local web UI and REST API."""
    import uvicorn

    _setup_logging(log_level.upper())
    console.print(f"[bold green]LearnTrade[/] starting at http://{host}:{port}")

    if open_browser:
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()

    uvicorn.run(
        "learntrade.api.app:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False,
    )


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


@app.command()
def demo(
    days: int = typer.Option(600, help="Trading days of fake history to create"),
) -> None:
    """Create clearly-labelled FAKE market data (50 made-up stocks) to try the tool offline."""
    from learntrade.core.config import get_config
    from learntrade.data.store.pit import PITStore
    from learntrade.setup.demo import generate_demo_data

    _setup_logging()
    cfg = get_config()
    with PITStore(cfg.workspace.data_dir) as store:
        n = generate_demo_data(store, "XNYS", n_days=days)
    console.print(f"[green]✓[/] Created {n} fake price points for 50 made-up stocks (market XNYS).")


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@app.command()
def init(
    config_dir: Path = typer.Option(Path("config"), help="Directory for learntrade.toml"),
) -> None:
    """Interactive workspace initialisation wizard (terminal version)."""
    from learntrade.setup.wizard import run_terminal_wizard
    run_terminal_wizard(config_dir)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


data_app = typer.Typer(help="Data management commands.")
app.add_typer(data_app, name="data")


@data_app.command("pull")
def data_pull(
    since: str = typer.Option("2015-01-01", help="Start date YYYY-MM-DD"),
    until: Optional[str] = typer.Option(None, help="End date YYYY-MM-DD (default: today)"),
    market: Optional[str] = typer.Option(None, help="Limit to one MIC"),
    capabilities: Optional[str] = typer.Option(
        None, help="Comma-separated capabilities (bars_daily,macro,filings)"
    ),
) -> None:
    """Pull prices, fundamentals, macro and filings from configured providers."""
    import json
    from datetime import date
    from learntrade.core.config import get_config
    from learntrade.data.engine import DataEngine
    from learntrade.data.store.pit import PITStore

    _setup_logging()
    cfg = get_config()
    data_dir = cfg.workspace.data_dir

    since_dt = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    until_dt = (
        datetime.strptime(until, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if until
        else datetime.now(tz=timezone.utc)
    )
    markets_list = [market] if market else None
    caps_set = set(capabilities.split(",")) if capabilities else None

    with PITStore(data_dir) as store:
        engine = DataEngine.from_config(store, cfg)
        with console.status("Pulling data…"):
            summary = engine.pull(since_dt, until_dt, markets_list, caps_set)

    console.print(f"\n[bold green]Pull complete.[/] {summary.total_added} observations added.")
    for r in summary.results:
        status = "[green]✓[/]" if r.success else "[red]✗[/]"
        console.print(
            f"  {status} {r.provider_id:12} {r.capability:20} {r.market:6} "
            f"{r.observations_added:>8} obs  {r.duration_seconds:.1f}s"
        )
    if summary.errors:
        console.print(f"\n[yellow]{len(summary.errors)} provider(s) failed.[/]")


@data_app.command("check")
def data_check(
    market: Optional[str] = typer.Option(None, help="Limit to one MIC"),
    since: str = typer.Option("2015-01-01", help="Start date for bar check YYYY-MM-DD"),
) -> None:
    """Run point-in-time leakage, gap and quality checks."""
    from learntrade.core.config import get_config
    from learntrade.data.quality.checks import run_all_checks
    from learntrade.data.store.pit import PITStore

    _setup_logging()
    cfg = get_config()
    markets_list = [market] if market else cfg.markets_enabled

    since_dt = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(tz=timezone.utc)

    any_error = False
    with PITStore(cfg.workspace.data_dir) as store:
        stats = store.stats()
        console.print(f"Store: {stats['total_observations']} total observations")
        for mic in markets_list:
            console.print(f"\n[bold]Checking {mic}…[/]")
            ids = store.get_universe(mic, now)
            df = store.query_bars(ids, as_of=now, since=since_dt) if ids else (
                store.query_observations(as_of=now, since=since_dt, kind="bar")
            )
            report = run_all_checks(
                df, as_of=now, snapshot_id="cli", mic=mic, since=since_dt, until=now
            )
            for issue in report.issues:
                console.print(f"  [red][ERROR][/] {issue['check']}: {issue['message']}")
                any_error = True
            for warn in report.warnings:
                console.print(f"  [yellow][WARN][/]  {warn['check']}: {warn['message']}")
            if report.passed and not report.warnings:
                console.print(f"  [green]All checks passed for {mic}[/]")

    if any_error:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@app.command()
def train(
    horizon: str = typer.Option("swing_20d", help="Horizon to train (swing_5d, swing_20d, positional_60d)"),
    market: Optional[str] = typer.Option(None, help="Limit to one MIC (default: all enabled markets)"),
    since: str = typer.Option("2015-01-01", help="Training data start date YYYY-MM-DD"),
    model: str = typer.Option("ridge", help="Model type: ridge | lgbm | lgbm_rank"),
    n_splits: int = typer.Option(5, help="Walk-forward CV folds"),
    holdout_days: int = typer.Option(
        120, help="Most recent trading days reserved as a final untouched test (0 = none)"),
) -> None:
    """Walk-forward train and calibrate a quant model."""
    from learntrade.core.config import get_config
    from learntrade.data.store.pit import PITStore
    from learntrade.quant.trainer import QuantTrainer

    _setup_logging()
    cfg = get_config()
    markets_list = [market] if market else cfg.markets_enabled
    since_dt = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(tz=timezone.utc)

    if model not in _MODEL_REGISTRY:
        console.print(f"[red]Unknown model {model!r}. Choose: {list(_MODEL_REGISTRY)}")
        raise typer.Exit(1)

    mod_path, cls_name = _MODEL_REGISTRY[model].split(":")
    import importlib
    model_cls = getattr(importlib.import_module(mod_path), cls_name)

    with PITStore(cfg.workspace.data_dir) as store:
        for mic in markets_list:
            console.print(f"[bold]Training {model} for {mic} {horizon}…[/]")
            try:
                trainer = QuantTrainer(
                    store=store,
                    horizon_id=horizon,
                    model_class=model_cls,
                    n_splits=n_splits,
                    artifacts_dir=cfg.workspace.data_dir / "models",
                    holdout_days=holdout_days,
                )
                result = trainer.train(mic, since_dt, now)
                console.print(f"[green]✓[/] {result.summary()}")
                if result.calibration:
                    console.print(
                        f"  Brier={result.calibration.brier_score:.4f} "
                        f"AUC={result.calibration.roc_auc:.4f} "
                        f"dirAcc={result.calibration.directional_accuracy:.2%}"
                    )
                from learntrade.quant.reliability import assess_reliability
                rel = assess_reliability(result.report)
                h = result.report.get("holdout")
                if h:
                    console.print(
                        f"  Final untouched test ({h['start']} → {h['end']}, {h['n_days']} days): "
                        f"rank IC {h['rank_ic_mean']:+.3f}, 95% range "
                        f"[{h['rank_ic_ci'][0]:+.3f}, {h['rank_ic_ci'][1]:+.3f}]"
                    )
                console.print(f"  [bold]Verdict:[/] {rel['headline']}")
                for why in rel["reasons"]:
                    console.print(f"   - {why}")
            except Exception as exc:
                from learntrade.core.errors import friendly_error
                console.print(f"[red]✗ {mic}: {friendly_error(exc)}[/]")


# ---------------------------------------------------------------------------
# Shared: load a trained model + calibrator back off disk
# ---------------------------------------------------------------------------


def _model_artifact_paths(cfg: Any, market: str, horizon: str, model_name: str) -> tuple[Path, Path]:
    model_cls = _require_model_class(model_name)
    models_dir = cfg.workspace.data_dir / "models"
    model_path = models_dir / f"{market}_{horizon}_{model_cls.id}.pkl"
    # One calibrator per model: score scales differ between models.
    cal_path = models_dir / f"{market}_{horizon}_{model_cls.id}_calibrator.pkl"
    return model_path, cal_path


def _require_model_class(model_name: str) -> Any:
    import importlib

    if model_name not in _MODEL_REGISTRY:
        console.print(f"[red]Unknown model {model_name!r}. Choose: {list(_MODEL_REGISTRY)}[/]")
        raise typer.Exit(1)
    mod_path, cls_name = _MODEL_REGISTRY[model_name].split(":")
    return getattr(importlib.import_module(mod_path), cls_name)


def _load_trained_model(
    cfg: Any, market: str, horizon: str, model_name: str, store: Any = None
) -> tuple[Any, Any]:
    """
    Load a previously trained model and its (optional) calibrator.
    Returns (None, None) if `learntrade train` hasn't been run for this
    market/horizon/model combination yet — callers decide how to degrade.
    """
    model_cls = _require_model_class(model_name)
    model_path, cal_path = _model_artifact_paths(cfg, market, horizon, model_name)
    if not model_path.exists():
        return None, None
    model = model_cls.load(str(model_path))
    from learntrade.features.price_volume import default_feature_set
    # The feature set depends on the data: company financials are included once stored.
    expected = default_feature_set(store, market).id if store is not None else None
    trained_on = getattr(model, "feature_set_version", expected)
    if expected is not None and trained_on != expected:
        # Trained on an older feature definition; its inputs would not line up. Treat as
        # "not trained" so callers fall back and the UI asks for an update.
        logging.getLogger(__name__).warning(
            "Model %s was trained on features %r, current is %r - re-train with `learntrade train`.",
            model_path.name, trained_on, expected)
        return None, None
    calibrator = None
    if cal_path.exists():
        from learntrade.quant.calibration.isotonic import IsotonicCalibrator
        calibrator = IsotonicCalibrator.load(str(cal_path))
    return model, calibrator


def _apply_calibration(forecasts: list[Any], calibrator: Any) -> list[Any]:
    """Overwrite each forecast's p_positive with the fitted calibrator's
    output instead of the model's own uncalibrated guess."""
    if calibrator is None or not forecasts:
        return forecasts
    import numpy as np
    raw = np.array([f.expected_return for f in forecasts])
    calibrated = calibrator.transform(raw)
    for fc, p in zip(forecasts, calibrated, strict=True):
        fc.p_positive = float(p)
    return forecasts


def _generate_current_forecasts(
    cfg: Any, store: Any, market: str, horizon: str, model_name: str
) -> tuple[list[Any], str]:
    """
    Build today's cross-section of forecasts for a market: one row per
    instrument, using the trained model + calibrator when available.

    Returns (forecasts, source_description). Shared by `rank`, `rebalance`
    and `analyze` so all three commands agree on what a "forecast" is,
    instead of each hand-rolling its own (previously: a constant
    p_positive=0.5 and made-up quantiles, regardless of whether a model had
    ever been trained).
    """
    from learntrade.core.types import Forecast, ForecastQuantiles, InstrumentId
    from learntrade.features.price_volume import default_feature_set

    now = datetime.now(tz=timezone.utc)
    instruments = store.get_universe(market, now)
    if not instruments:
        return [], "no_universe"

    feature_set = default_feature_set(store, market)
    features = feature_set.compute(instruments, now, store)
    if features.is_empty():
        return [], "no_features"

    trained_model, calibrator = _load_trained_model(cfg, market, horizon, model_name, store)
    if trained_model is not None:
        feat_cols = [c for c in feature_set.feature_names if c in features.columns]
        raw_iids = features["instrument_id"].to_list()
        iids_list = [
            InstrumentId(r) if ":" in str(r) else InstrumentId(f"{market}:{r}")
            for r in raw_iids
        ]
        forecasts = trained_model.predict_distribution(features.select(feat_cols), iids_list, now)
        forecasts = _apply_calibration(forecasts, calibrator)
        if forecasts:
            return forecasts, f"trained model ({trained_model.id})"

    # Fallback: an uncalibrated momentum proxy, clearly labelled as such.
    # p_positive=0.5 here means "unknown", not "50% confident" — there is no
    # calibrated probability without a trained model.
    signal_col = "ret_20d" if "ret_20d" in features.columns else features.columns[1]
    fallback: list[Any] = []
    for row in features.drop_nulls(subset=[signal_col]).iter_rows(named=True):
        raw_iid = row["instrument_id"]
        iid = InstrumentId(raw_iid) if ":" in str(raw_iid) else InstrumentId(f"{market}:{raw_iid}")
        p = float(row[signal_col])
        vol = float(row.get("ewma_vol_20d") or 0.2)
        fallback.append(Forecast(
            instrument_id=iid, as_of=now, horizon=horizon,
            expected_return=p, p_positive=0.5, volatility=vol,
            quantiles=ForecastQuantiles(
                q05=p - 1.645 * vol, q25=p - 0.674 * vol,
                q50=p, q75=p + 0.674 * vol, q95=p + 1.645 * vol,
            ),
            model_id="momentum_fallback", data_snapshot_id="",
        ))
    return fallback, "raw momentum fallback (no trained model)"


def _forecast_outside_universe(
    cfg: Any, store: Any, iid: Any, horizon: str, model_name: str, universe_forecasts: list[Any]
) -> tuple[Any | None, dict[str, Any], str]:
    """
    Score ONE stock that is not in the stored universe.

    Downloads ~2 years of its prices (so features can be computed), scores it with
    the already-trained model, and reports where it would rank among the universe's
    current forecasts. Returns (forecast_or_None, context, message).

    Only absolute-score models can do this: the ranker min-max-normalises within the
    cross-section, so a lone stock has no meaningful score. For those we say so.
    """
    from datetime import timedelta
    from learntrade.core.config import get_secrets
    from learntrade.data.providers.base import FetchRequest
    from learntrade.data.providers.registry import ProviderRegistry
    import polars as pl
    from learntrade.features.price_volume import default_feature_set

    market = iid.mic
    model, calibrator = _load_trained_model(cfg, market, horizon, model_name, store)
    if model is None:
        return None, {}, (f"No trained {model_name} model for {market}. Set up and train on a list "
                          f"of stocks first.")
    if model_name == "lgbm_rank":
        return None, {}, ("The ranking model only compares stocks inside a list. "
                          "Use the ridge or lgbm model to check a single stock.")

    now = datetime.now(tz=timezone.utc)
    pid = (cfg.data.providers.prices or ["yahoo"])[0]
    provider = ProviderRegistry(cfg, get_secrets()).get(pid)
    if provider is None:
        return None, {}, f"Price source {pid!r} is not available."
    req = FetchRequest("bars_daily", now - timedelta(days=730), now,
                       instruments={iid}, market=market)
    try:
        obs = list(provider.fetch(req))
    except Exception as exc:
        return None, {}, f"Could not download prices for {iid}: {exc}"
    if len(obs) < 60:
        return None, {}, (f"Not enough price history for {iid} ({len(obs)} days found). "
                          f"Check the symbol.")
    store.ingest(obs)
    try:   # financials for the same stock, so it is scored on the same features as the list
        from learntrade.data.providers.yahoo import YahooProvider
        yp = YahooProvider()
        store.ingest(list(yp._fetch_fundamentals(iid)) + list(yp._fetch_splits(iid)))
    except Exception:
        pass

    # Re-take 'now': the new bars carry their download time as available_at, so a
    # timestamp from before the fetch would (correctly) hide them as 'future' data.
    now = datetime.now(tz=timezone.utc)
    fs = default_feature_set(store, market)
    # Features are normalised across stocks, so score the new stock alongside the universe
    # (its row is then read off) instead of in isolation.
    members = store.get_universe(market, now)
    feats = fs.compute(list(members) + [iid], now, store)
    feats = feats.filter(pl.col("instrument_id") == str(iid)) if not feats.is_empty() else feats
    if feats.is_empty():
        return None, {}, f"Could not compute indicators for {iid} (too little history)."
    cols = [c for c in fs.feature_names if c in feats.columns]
    fcs = model.predict_distribution(feats.select(cols), [iid], now)
    fcs = _apply_calibration(fcs, calibrator)
    if not fcs:
        return None, {}, f"The model could not score {iid}."
    fc = fcs[0]
    higher = sum(1 for f in universe_forecasts if f.expected_return > fc.expected_return)
    ctx = {"would_rank": higher + 1, "of": len(universe_forecasts) + 1}
    return fc, ctx, "ok"


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------


@app.command()
def backtest(
    strategy: str = typer.Option("default", help="Strategy config name"),
    from_date: str = typer.Option("2018-01-01", "--from", help="Start date YYYY-MM-DD"),
    horizon: str = typer.Option("swing_20d", help="Horizon"),
    market: str = typer.Option("XNYS", help="MIC code"),
    model: str = typer.Option("ridge", help="Model: ridge | lgbm | lgbm_rank"),
    engine: str = typer.Option("event_driven", help="Engine: event_driven | vectorised"),
) -> None:
    """Run a backtest and write a report to research/reports/."""
    from learntrade.backtest.engine import BacktestConfig, EventDrivenBacktest, VectorisedBacktest
    from learntrade.backtest.report import generate_report
    from learntrade.core.config import get_config
    from learntrade.data.store.pit import PITStore
    from learntrade.features.price_volume import default_feature_set
    from learntrade.markets import get_pack
    from learntrade.quant.targets.labels import compute_forward_returns, merge_features_labels

    _setup_logging()
    cfg = get_config()
    since_dt = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(tz=timezone.utc)

    with PITStore(cfg.workspace.data_dir) as store:
        instruments = store.get_universe(market, now)
        if not instruments:
            console.print(f"[red]No universe for {market}. Run `learntrade universe import`.[/]")
            raise typer.Exit(1)

        feature_set = default_feature_set(store, market)
        bars = store.query_bars(instruments, as_of=now, since=since_dt)
        if bars.is_empty():
            console.print("[red]No bars found. Run `learntrade data pull` first.[/]")
            raise typer.Exit(1)

        from learntrade.quant.targets.horizon import get_horizon
        import polars as pl
        h = get_horizon(horizon)
        features = feature_set.compute_history(instruments, since_dt, now, store)
        if features.is_empty():
            console.print("[red]No features. Run `learntrade data pull` first.[/]")
            raise typer.Exit(1)

        trained_model, calibrator = _load_trained_model(cfg, market, horizon, model, store)
        signal_source: str
        if trained_model is not None:
            # Use the TRAINED model's forecast as the signal, not a raw
            # feature column — this is what the model/calibration/backtest
            # pipeline is actually supposed to be testing.
            from learntrade.core.types import InstrumentId
            feat_cols = [c for c in feature_set.feature_names if c in features.columns]
            dates_list = features["event_time"].to_list()
            raw_iids = features["instrument_id"].to_list()
            iids_list = [
                InstrumentId(r) if ":" in str(r) else InstrumentId(f"{market}:{r}")
                for r in raw_iids
            ]
            # `predict_distribution` is called once per (date, instrument) row
            # here, not per cross-section, so its output must line up
            # POSITIONALLY with the input rows — a model that silently drops
            # a row (e.g. on a NaN prediction) would otherwise misalign every
            # later row's date/instrument.
            forecasts = trained_model.predict_distribution(
                features.select(feat_cols), iids_list, now
            )
            if len(forecasts) != len(dates_list):
                console.print(
                    f"[yellow]Model returned {len(forecasts)} forecasts for "
                    f"{len(dates_list)} input rows (some were dropped) — "
                    f"falling back to a raw momentum signal.[/]"
                )
                trained_model = None
            else:
                forecasts = _apply_calibration(forecasts, calibrator)
                signals = pl.DataFrame({
                    "date": dates_list,
                    "instrument_id": raw_iids,
                    "signal": [fc.expected_return for fc in forecasts],
                }).drop_nulls()
                signal_source = f"trained model ({trained_model.id})"
        if trained_model is None:
            console.print(
                f"[yellow]Using a raw momentum signal (ret_20d), not a trained model. "
                f"Run `learntrade train --model {model} --market {market} "
                f"--horizon {horizon}` first for a real signal.[/]"
            )
            labels = compute_forward_returns(bars, h.forward_days)
            if labels.is_empty():
                console.print("[red]No labels could be computed either. Run `learntrade data pull` first.[/]")
                raise typer.Exit(1)
            merged = merge_features_labels(features, labels)
            signal_col = "ret_20d" if "ret_20d" in merged.columns else merged.columns[2]
            signals = merged.select([
                pl.col("event_time").alias("date"),
                pl.col("instrument_id"),
                pl.col(signal_col).alias("signal"),
            ]).drop_nulls()
            signal_source = "raw momentum fallback (no trained model)"

        if signals.is_empty():
            console.print("[red]No signals produced. Run `learntrade train` first.[/]")
            raise typer.Exit(1)

        pack = get_pack(market)
        back_cfg = BacktestConfig(
            pit_level="none",
            survivorship_note="No survivorship check applied",
            engine=engine,
            rebalance_every=h.forward_days,   # trade on the forecast horizon, not daily
            ic_horizon=h.forward_days,        # score IC over the same horizon as the forecast
        )

        with console.status(f"Running {engine} backtest…"):
            if engine == "event_driven":
                runner = EventDrivenBacktest()
            else:
                runner = VectorisedBacktest()
            result = runner.run(signals=signals, bars=bars, market_pack=pack, config=back_cfg)

        report_dir = Path("research/reports")
        report = generate_report(result, horizon, market, [model], report_dir)

        pm = report.portfolio_metrics
        ic = report.ic_metrics
        console.print(f"\n[bold]Backtest: {market} {horizon}[/] [dim]({engine}, signal: {signal_source})[/]")
        console.print(
            f"  CAGR={pm.get('cagr', float('nan')):.2%}  "
            f"Sharpe={pm.get('sharpe', float('nan')):.2f}  "
            f"MaxDD={pm.get('max_drawdown', float('nan')):.2%}  "
            f"CostDrag={report.cost_drag_bps_annual:.1f}bps/yr"
        )
        if ic:
            console.print(f"  IC={ic.get('ic_mean', float('nan')):.4f}  t={ic.get('ic_tstat', float('nan')):.2f}")
        console.print(f"  PIT level: [yellow]{result.pit_level}[/]")
        if trained_model is not None:
            console.print(
                "  [yellow]In-sample:[/] this model was trained on the same period it is being "
                "tested on, so these numbers flatter it. For an honest out-of-sample test run "
                "`learntrade evaluate`."
            )
        console.print(f"  Report: [cyan]research/reports/{report.run_id}_{market}_{horizon}.*[/]")


# ---------------------------------------------------------------------------
# rank
# ---------------------------------------------------------------------------


@app.command()
def rank(
    market: str = typer.Option("XNYS", help="MIC code"),
    top: int = typer.Option(20, help="Number of top-ranked instruments to show"),
    horizon: str = typer.Option("swing_20d", help="Forecast horizon"),
    model: str = typer.Option("ridge", help="Model: ridge | lgbm | lgbm_rank"),
) -> None:
    """Show today's ranked opportunities for a market."""
    from learntrade.core.config import get_config
    from learntrade.data.store.pit import PITStore

    _setup_logging()
    cfg = get_config()

    with PITStore(cfg.workspace.data_dir) as store:
        forecasts, source = _generate_current_forecasts(cfg, store, market, horizon, model)

    if source == "no_universe":
        console.print(f"[red]No universe for {market}. Run `learntrade universe import`.[/]")
        raise typer.Exit(1)
    if source == "no_features":
        console.print("[red]No features. Run `learntrade data pull` first.[/]")
        raise typer.Exit(1)
    if not forecasts:
        console.print("[red]No forecasts produced.[/]")
        raise typer.Exit(1)
    if source.startswith("raw momentum"):
        console.print(
            f"[yellow]{source} — run `learntrade train --model {model} "
            f"--market {market} --horizon {horizon}` for calibrated forecasts.[/]"
        )

    ranked = sorted(forecasts, key=lambda f: f.expected_return, reverse=True)[:top]
    console.print(f"\n[bold]Top {len(ranked)} — {market} {horizon} as of {datetime.now(tz=timezone.utc).date()}[/] [dim]({source})[/]\n")
    for i, fc in enumerate(ranked, 1):
        console.print(
            f"  {i:2d}. {str(fc.instrument_id):20s}  "
            f"ExpRet={fc.expected_return:+.2%}  P(pos)={fc.p_positive:.1%}  "
            f"Vol={fc.volatility:.1%}"
        )


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@app.command()
def evaluate(
    market: str = typer.Option("XNYS", help="MIC code"),
    horizon: str = typer.Option("swing_20d", help="Horizon"),
    since: str = typer.Option("2015-01-01", help="Data start date YYYY-MM-DD"),
    holdout_days: int = typer.Option(120, help="Most recent trading days kept untouched for testing"),
    top_n: int = typer.Option(10, help="Stocks held in the simulated portfolio"),
    engine: str = typer.Option("vectorised", help="Engine: vectorised | event_driven"),
) -> None:
    """Fair, out-of-sample comparison of momentum, ridge, lgbm and lgbm_rank."""
    from learntrade.backtest.evaluation import run_evaluation, to_markdown
    from learntrade.core.config import get_config
    from learntrade.data.store.pit import PITStore

    _setup_logging("WARNING")
    cfg = get_config()
    since_dt = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    models = {name: _require_model_class(name) for name in ("ridge", "lgbm", "lgbm_rank")}
    with PITStore(cfg.workspace.data_dir) as store:
        with console.status("Fitting on the development period and testing on the untouched period…"):
            res = run_evaluation(store, market, horizon, since_dt, holdout_days, models,
                                 top_n=top_n, engine=engine)
    if not res.candidates:
        console.print("[red]Nothing could be evaluated.[/]")
        for n in res.notes:
            console.print(f"  - {n}")
        raise typer.Exit(1)
    md = to_markdown(res)
    out_dir = Path("research/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{stamp}_{market}_{horizon}_evaluation.md"
    path.write_text(md)
    console.print(md)
    console.print(f"Saved: [cyan]{path}[/]")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@app.command()
def analyze(
    instrument: str = typer.Argument(help="Instrument ID, e.g. XNYS:IBM"),
    horizon: str = typer.Option("swing_20d", help="Forecast horizon"),
    model: str = typer.Option("ridge", help="Model: ridge | lgbm | lgbm_rank"),
) -> None:
    """Generate a research report for one instrument."""
    from learntrade.core.config import get_config
    from learntrade.core.types import InstrumentId
    from learntrade.data.store.pit import PITStore

    _setup_logging()
    cfg = get_config()
    try:
        iid = InstrumentId(instrument)
    except Exception:
        console.print(f"[red]Invalid instrument id {instrument!r} — expected 'MIC:SYMBOL', e.g. 'XNYS:IBM'[/]")
        raise typer.Exit(1) from None
    market = iid.mic

    with PITStore(cfg.workspace.data_dir) as store:
        forecasts, source = _generate_current_forecasts(cfg, store, market, horizon, model)
        fc = next((f for f in forecasts if str(f.instrument_id) == str(iid)), None)

        if source == "no_universe":
            console.print(f"[red]No universe for {market}. Run `learntrade universe import`.[/]")
            raise typer.Exit(1)
        outside_ctx: dict[str, Any] = {}
        if fc is None:
            # Not in the stored list: score it on its own against the trained model.
            console.print(f"[dim]{instrument} is not in your list — fetching its prices…[/]")
            fc, outside_ctx, msg = _forecast_outside_universe(
                cfg, store, iid, horizon, model, forecasts)
            if fc is None:
                console.print(f"[red]{msg}[/]")
                raise typer.Exit(1)
            source = f"trained model, single-stock check ({fc.model_id})"
        elif source == "no_features":
            console.print(f"[red]No data for {instrument}. Run `learntrade data pull`.[/]")
            raise typer.Exit(1)

        console.print(f"\n[bold]{instrument}[/] — {horizon} [dim]({source})[/]")
        console.print(
            f"  Expected excess return: [bold]{fc.expected_return:+.2%}[/]  "
            f"P(positive): {fc.p_positive:.1%}  Volatility: {fc.volatility:.1%}"
        )
        console.print(
            f"  Range (5–95%): {fc.quantiles.q05:+.2%} to {fc.quantiles.q95:+.2%}"
        )
        if outside_ctx:
            console.print(
                f"  Would rank [bold]{outside_ctx['would_rank']}[/] of {outside_ctx['of']} "
                f"compared with your list's current forecasts."
            )
        if source.startswith("raw momentum"):
            console.print(
                f"[yellow]This is an uncalibrated momentum proxy, not a trained "
                f"model's forecast. Run `learntrade train` first.[/]"
            )

        headlines, _ = _news_context(cfg, str(iid), datetime.now(tz=timezone.utc))
        if headlines:
            console.print("\n[bold]Recent news[/]")
            for h in headlines:
                console.print(f"  • {h}")
        else:
            console.print(
                "[dim]No news stored for this instrument. "
                "Run `learntrade news pull` to fetch headlines.[/]"
            )

        thesis = None
        counter_thesis = None
        agreement = None
        llm_cost = 0.0
        if cfg.llm.enabled:
            import asyncio
            from learntrade.features.price_volume import default_feature_set
            from learntrade.llm.research.fact_sheet import build_fact_sheet
            from learntrade.llm.research.orchestrator import ResearchOrchestrator
            from learntrade.llm.router import LLMRouter
            try:
                router = LLMRouter()
                feat_row_df = default_feature_set(store, market).compute([iid], datetime.now(tz=timezone.utc), store)
                feat_row = feat_row_df.row(0, named=True) if not feat_row_df.is_empty() else {}
                events, news_evidence = _news_context(cfg, str(iid), datetime.now(tz=timezone.utc))
                fact_sheet = build_fact_sheet(
                    str(iid), datetime.now(tz=timezone.utc), horizon, feat_row,
                    forecast=fc, universe_size=len(forecasts),
                    recent_events=events, evidence=news_evidence,
                )
                orchestrator = ResearchOrchestrator(router)
                result = asyncio.run(orchestrator.analyze(fact_sheet, fc))
                thesis, counter_thesis = result.thesis, result.counter_thesis
                agreement, llm_cost = result.agreement_with_quant, result.llm_cost_usd
                console.print(f"\n[bold]Research thesis:[/] {thesis}")
                console.print(f"[bold]Counter-thesis:[/] {counter_thesis}")
                console.print(f"[dim]Agreement with quant forecast: {agreement} | LLM cost: ${llm_cost:.4f}[/]")
            except Exception as exc:
                console.print(f"[yellow]LLM research failed ({exc}); showing quant-only analysis.[/]")
        else:
            console.print(
                "[dim]LLM research is disabled ([llm] enabled = false) — "
                "showing the quant forecast only.[/]"
            )

        from learntrade.attribution.store import AttributionStore
        from learntrade.core.types import DecisionRecord, ForecastAttribution
        now = datetime.now(tz=timezone.utc)
        record = DecisionRecord(
            decision_id=f"{now.isoformat()}/{iid}/{horizon}/analyze",
            instrument_id=iid, horizon=horizon, as_of=now, forecast=fc,
            thesis=thesis, counter_thesis=counter_thesis,
            agreement_with_quant=agreement,
            attribution=ForecastAttribution(models={fc.model_id: 1.0}, llm_cost_usd=llm_cost),
        )
        db_path = cfg.workspace.data_dir / "attribution.duckdb"
        with AttributionStore(db_path) as attr_store:
            attr_store.write(record)


# ---------------------------------------------------------------------------
# news
# ---------------------------------------------------------------------------

news_app = typer.Typer(help="News: pull headlines, extract evidence, browse.")
app.add_typer(news_app, name="news")


def _corpus_path(cfg: Any) -> Path:
    return cfg.workspace.data_dir / "corpus.duckdb"


def _news_context(cfg: Any, instrument_id: str, as_of: datetime) -> tuple[list[str], dict[str, Any]]:
    """Headlines (+ event tags) and averaged LLM evidence for one instrument, PIT at ``as_of``."""
    from learntrade.data.corpus.news import evidence_summary_for, recent_headlines, tag_events
    from learntrade.data.corpus.store import CorpusStore

    path = _corpus_path(cfg)
    if not path.exists():
        return [], {}
    corpus = CorpusStore(path)
    try:
        heads = recent_headlines(corpus, instrument_id, as_of)
        tags = sorted(tag_events(heads))
        events = [f"[{t}]" for t in tags] + heads
        return events, evidence_summary_for(corpus, instrument_id, as_of)
    finally:
        corpus.close()


@news_app.command("pull")
def news_pull(
    market: Optional[str] = typer.Option(None, help="Limit to one MIC (default: all enabled)"),
    days: int = typer.Option(7, help="How many days back to fetch"),
) -> None:
    """Fetch headlines for the universe and store them (point-in-time)."""
    from datetime import timedelta
    from learntrade.core.config import get_config
    from learntrade.data.corpus.news import ingest_news
    from learntrade.data.corpus.store import CorpusStore
    from learntrade.data.engine import DataEngine
    from learntrade.data.store.pit import PITStore

    _setup_logging()
    cfg = get_config()
    if not cfg.data.providers.news:
        cfg.data.providers.news = ["rss"]
        console.print("[dim]No news provider configured — using the free 'rss' default.[/]")
    now = datetime.now(tz=timezone.utc)
    since = now - timedelta(days=days)
    markets = [market] if market else cfg.markets_enabled

    with PITStore(cfg.workspace.data_dir) as store:
        summary = DataEngine.from_config(store, cfg).pull(
            since, now, markets=markets, capabilities={"news"}
        )
        for r in summary.results:
            status = "[green]✓[/]" if r.success else f"[red]✗ {r.error}[/]"
            console.print(f"  {status} {r.provider_id} {r.market}: {r.observations_added} items")
        df = store.query_observations(as_of=now, since=since, kind="news")
    from learntrade.core.types import InstrumentId, Observation
    rows = []
    for r in df.iter_rows(named=True):
        payload = r["payload"]
        rows.append(Observation(
            instrument_id=InstrumentId(r["instrument_id"]) if r["instrument_id"] else None,
            source_id=r["source_id"], kind="news",
            event_time=r["event_time"], available_at=r["available_at"],
            payload=json.loads(payload) if isinstance(payload, str) else payload,
            pit_level=r["pit_level"],
        ))
    corpus = CorpusStore(_corpus_path(cfg))
    try:
        added = ingest_news(rows, corpus)
        console.print(f"[green]✓[/] {added} new articles in corpus ({corpus.document_count()} total)")
    finally:
        corpus.close()


@news_app.command("extract")
def news_extract(
    limit: int = typer.Option(50, help="Max articles to process this run"),
) -> None:
    """Run the LLM over new articles once each (cached; needs [llm] enabled)."""
    import asyncio
    from learntrade.core.config import get_config
    from learntrade.data.corpus.store import CorpusStore

    _setup_logging()
    cfg = get_config()
    if not cfg.llm.enabled:
        console.print("[yellow]LLM is disabled ([llm] enabled = false). Headlines still work "
                      "without it; enable the LLM only if you want structured evidence.[/]")
        raise typer.Exit(0)
    path = _corpus_path(cfg)
    if not path.exists():
        console.print("[red]No corpus yet — run `learntrade news pull` first.[/]")
        raise typer.Exit(1)

    from learntrade.llm.extraction.extractor import EvidenceExtractor
    from learntrade.llm.router import LLMRouter

    corpus = CorpusStore(path)
    try:
        docs = corpus.unextracted_documents(datetime.now(tz=timezone.utc).replace(tzinfo=None),
                                            limit=limit)
        if not docs:
            console.print("Nothing new to extract.")
            return
        extractor = EvidenceExtractor(LLMRouter())

        async def run() -> int:
            n = 0
            for d in docs:
                try:
                    ev = await extractor.extract(
                        d["doc_id"], f"{d['title']}\n{d['body'] or ''}", d["instrument_ids"],
                        d["available_at"].replace(tzinfo=timezone.utc), verify=False,
                    )
                    corpus.save_evidence(ev)
                    n += 1
                except Exception as exc:
                    console.print(f"[yellow]skip {d['title'][:50]!r}: {exc}[/]")
            return n

        n = asyncio.run(run())
        console.print(f"[green]✓[/] extracted evidence for {n}/{len(docs)} articles")
    finally:
        corpus.close()


@news_app.command("show")
def news_show(
    instrument: str = typer.Argument(help="Instrument ID, e.g. XNSE:RELIANCE"),
    days: int = typer.Option(14, help="Look-back window in days"),
) -> None:
    """Show stored headlines for one instrument."""
    from learntrade.core.config import get_config
    from learntrade.data.corpus.news import recent_headlines
    from learntrade.data.corpus.store import CorpusStore

    cfg = get_config()
    path = _corpus_path(cfg)
    if not path.exists():
        console.print("[red]No corpus yet — run `learntrade news pull` first.[/]")
        raise typer.Exit(1)
    corpus = CorpusStore(path)
    try:
        heads = recent_headlines(corpus, instrument.upper(), datetime.now(tz=timezone.utc),
                                 days=days, limit=30)
    finally:
        corpus.close()
    if not heads:
        console.print(f"No stored news for {instrument} in the last {days} days.")
    for h in heads:
        console.print(f"  • {h}")


# ---------------------------------------------------------------------------
# portfolio
# ---------------------------------------------------------------------------


portfolio_app = typer.Typer(help="Portfolio commands.")
app.add_typer(portfolio_app, name="portfolio")


@portfolio_app.command("import")
def portfolio_import(
    file: Path = typer.Argument(help="Holdings CSV or XLSX file"),
    mic: str = typer.Option("XNYS", help="Exchange MIC for the holdings"),
) -> None:
    """Import holdings from a broker-exported CSV / XLSX file."""
    from learntrade.execution.brokers.csv_import import CsvImportAdapter
    adapter = CsvImportAdapter(file, mic)
    holdings = adapter.holdings()
    console.print(f"Imported [bold]{len(holdings)}[/] holdings from {file}")
    for h in holdings:
        console.print(f"  {h.instrument_id}  qty={h.quantity}")


# ---------------------------------------------------------------------------
# rebalance
# ---------------------------------------------------------------------------


@app.command()
def rebalance(
    dry_run: bool = typer.Option(True, "--dry-run/--execute", help="Propose without submitting"),
    market: str = typer.Option("XNYS", help="MIC code"),
    horizon: str = typer.Option("swing_20d", help="Forecast horizon"),
    allocator: str = typer.Option("hrp", help="Allocator: equal_weight | inverse_vol | hrp | mv | risk_parity"),
    model: str = typer.Option("ridge", help="Model: ridge | lgbm | lgbm_rank"),
) -> None:
    """Propose (or execute) a portfolio rebalance."""
    from learntrade.attribution.store import AttributionStore
    from learntrade.core.config import get_config
    from learntrade.core.types import DecisionRecord, ForecastAttribution
    from learntrade.data.store.pit import PITStore
    from learntrade.portfolio.allocators import (
        EqualWeightAllocator, HRPAllocator, InverseVolAllocator,
        MeanVarianceAllocator, RiskParityAllocator,
    )
    from learntrade.portfolio.constraints import PortfolioConstraints, apply_constraints

    _setup_logging()
    cfg = get_config()
    now = datetime.now(tz=timezone.utc)
    alloc_map = {
        "equal_weight": EqualWeightAllocator(),
        "inverse_vol": InverseVolAllocator(),
        "hrp": HRPAllocator(),
        "mv": MeanVarianceAllocator(),
        "risk_parity": RiskParityAllocator(),
    }
    alloc = alloc_map.get(allocator, HRPAllocator())
    constraints = PortfolioConstraints(
        max_position=cfg.portfolio.max_position,
        long_only=cfg.portfolio.long_only,
    )

    with PITStore(cfg.workspace.data_dir) as store:
        forecasts, source = _generate_current_forecasts(cfg, store, market, horizon, model)
        if source == "no_universe":
            console.print(f"[red]No universe for {market}. Run `learntrade universe import`.[/]")
            raise typer.Exit(1)
        if source == "no_features" or not forecasts:
            console.print("[red]No features/forecasts. Run `learntrade data pull` first.[/]")
            raise typer.Exit(1)
        if source.startswith("raw momentum"):
            console.print(
                f"[yellow]{source} — run `learntrade train --model {model} "
                f"--market {market} --horizon {horizon}` for calibrated forecasts.[/]"
            )

        raw_weights = alloc.allocate(forecasts)
        weights = apply_constraints(raw_weights, constraints)

        # Record every proposal in the attribution store, whether or not it
        # is ever submitted — this is what makes `decision-history` and
        # eventual realised-return feedback possible.
        db_path = cfg.workspace.data_dir / "attribution.duckdb"
        fc_by_iid = {str(f.instrument_id): f for f in forecasts}
        with AttributionStore(db_path) as attr_store:
            for iid_str, w in weights.items():
                fc = fc_by_iid.get(str(iid_str))
                if fc is None:
                    continue
                record = DecisionRecord(
                    decision_id=f"{now.isoformat()}/{iid_str}/{horizon}/rebalance",
                    instrument_id=fc.instrument_id, horizon=horizon, as_of=now,
                    forecast=fc, target_weight=w, trade="BUY",
                    attribution=ForecastAttribution(models={fc.model_id: 1.0}),
                )
                attr_store.write(record)

    console.print(f"\n[bold]Proposed allocation — {market} {horizon} ({allocator})[/] [dim]({source})[/]")
    for iid, w in sorted(weights.items(), key=lambda x: -x[1])[:20]:
        console.print(f"  {str(iid):25s}  {w:.2%}")

    if dry_run:
        console.print("\n[dim](dry-run — use --execute to submit orders)[/]")
        return

    # --execute: route through the same three-gate live-execution safety
    # check used everywhere else, instead of silently behaving exactly like
    # --dry-run. With no broker adapter wired in yet, this will almost
    # always refuse (clearly — not silently) unless the user has
    # deliberately opted into live trading via config + session flag.
    from learntrade.core.types import InstrumentId, Order
    from learntrade.execution.live import LiveExecutionError, check_live_gates
    orders = [
        Order(instrument_id=InstrumentId(str(iid)), side="buy", quantity=1, horizon=horizon)
        for iid in weights
    ]
    try:
        confirmed = check_live_gates(orders)
        console.print(f"[green]{len(confirmed)} order(s) passed live-execution gates.[/]")
    except LiveExecutionError as exc:
        console.print(f"[red]--execute refused: {exc}[/]")
        console.print(
            "[dim]No broker order routing is wired in yet; --execute only "
            "checks the live-execution safety gates.[/]"
        )
        raise typer.Exit(1) from None


# ---------------------------------------------------------------------------
# decision-history
# ---------------------------------------------------------------------------


@app.command("decision-history")
def decision_history(
    market: Optional[str] = typer.Option(None, help="Filter by MIC prefix"),
    horizon: Optional[str] = typer.Option(None, help="Filter by horizon"),
    limit: int = typer.Option(20, help="Number of rows to show"),
) -> None:
    """Show the decision attribution history."""
    from learntrade.attribution.store import AttributionStore
    from learntrade.core.config import get_config

    _setup_logging()
    cfg = get_config()
    db_path = cfg.workspace.data_dir / "attribution.duckdb"
    if not db_path.exists():
        console.print("[yellow]No decision history found.[/]")
        return
    with AttributionStore(db_path) as store:
        df = store.query(market=market, horizon=horizon, limit=limit)
    if df.is_empty():
        console.print("[yellow]No decisions found.[/]")
        return
    console.print(df)


# ---------------------------------------------------------------------------
# market
# ---------------------------------------------------------------------------


market_app = typer.Typer(help="Market Pack commands.")
app.add_typer(market_app, name="market")


@market_app.command("list")
def market_list() -> None:
    """List all available Market Packs."""
    from learntrade.markets import list_packs
    packs = list_packs()
    if not packs:
        console.print("No Market Packs found.")
        return
    for mic in packs:
        console.print(f"  {mic}")


@market_app.command("validate")
def market_validate(
    mic: Optional[str] = typer.Argument(default=None, help="MIC to validate (or omit for --all)"),
    all_packs: bool = typer.Option(False, "--all", help="Validate every pack"),
) -> None:
    """Validate one or all Market Packs."""
    from learntrade.markets.validate import validate_all_packs, validate_pack

    _setup_logging()
    if all_packs or mic is None:
        reports = validate_all_packs()
    else:
        reports = {mic.upper(): validate_pack(mic.upper())}

    any_error = False
    for m, report in reports.items():
        status = "[green]PASS[/]" if report.passed else "[red]FAIL[/]"
        console.print(f"{status} {m}")
        for issue in report.issues:
            sev = "[red]error[/]" if issue.severity == "error" else "[yellow]warn[/]"
            console.print(f"     {sev}  {issue.check}: {issue.message}")
            if issue.severity == "error":
                any_error = True
    if any_error:
        raise typer.Exit(1)


@market_app.command("new")
def market_new(
    mic: str = typer.Option(..., help="ISO 10383 MIC code"),
) -> None:
    """Scaffold a new Market Pack YAML."""
    import shutil
    from learntrade.markets.loader import _BUILTIN_MARKET_PACKS_DIR

    target = Path("markets") / f"{mic.upper()}.yaml"
    if target.exists():
        console.print(f"[yellow]{target} already exists[/]")
        raise typer.Exit(1)
    generic = _BUILTIN_MARKET_PACKS_DIR / "generic.yaml"
    if not generic.exists():
        console.print("[red]generic.yaml not found[/]")
        raise typer.Exit(1)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(generic, target)
    # Patch MIC in the new file
    text = target.read_text()
    text = text.replace("mic: GENERIC", f"mic: {mic.upper()}")
    target.write_text(text)
    console.print(f"[green]Scaffolded {target}[/] — edit costs and calendar, then run `learntrade market validate {mic.upper()}`")


# ---------------------------------------------------------------------------
# llm
# ---------------------------------------------------------------------------


llm_app = typer.Typer(help="LLM management commands.")
app.add_typer(llm_app, name="llm")


@llm_app.command("eval")
def llm_eval(
    task: str = typer.Option("extract", help="Task: extract | research | verify | escalation"),
    models: str = typer.Option("", help="Comma-separated model IDs (uses config default if empty)"),
) -> None:
    """Run the LLM evaluation suite for a task."""
    import asyncio
    from learntrade.llm.eval.suite import EvalSuite, EVAL_CASES
    from learntrade.llm.router import LLMRouter

    _setup_logging()
    router = LLMRouter()
    if not router.is_enabled:
        console.print("[red]LLM is disabled. Set [llm] enabled = true in learntrade.toml.[/]")
        raise typer.Exit(1)

    suite = EvalSuite(router)
    console.print(f"Running LLM eval suite: task=[bold]{task}[/]  model=[bold]{router._get_client(task).model}[/]")
    results = asyncio.run(suite.run(task=task))
    suite.print_summary(results)
    passed = sum(1 for r in results if r.passed)
    console.print(f"\n[{'green' if passed == len(results) else 'yellow'}]{passed}/{len(results)} cases passed[/]")


@llm_app.command("cache-stats")
def llm_cache_stats() -> None:
    """Show LLM response cache statistics."""
    from pathlib import Path
    from learntrade.llm.cache import LLMCache
    from learntrade.core.config import get_config

    _setup_logging()
    cache_path = get_config().workspace.data_dir / "llm_cache.duckdb"
    if not cache_path.exists():
        console.print("[yellow]No cache found.[/]")
        return
    cache = LLMCache(cache_path)
    stats = cache.stats()
    cache.close()
    console.print(f"Cache entries: [bold]{stats['total_entries']}[/]")
    console.print(f"Total hits:    [bold]{stats['total_hits']}[/]")


# ---------------------------------------------------------------------------
# universe
# ---------------------------------------------------------------------------

universe_app = typer.Typer(help="Universe management commands.")
app.add_typer(universe_app, name="universe")


@universe_app.command("import")
def universe_import(
    file: Path = typer.Argument(help="CSV file with instrument symbols"),
    mic: str = typer.Option(..., help="Exchange MIC code"),
    symbol_column: str = typer.Option("symbol", help="Column name for symbols"),
    date_column: Optional[str] = typer.Option(
        None, help="Membership history: each row = 'this symbol was in the index on this date'"),
    added_column: Optional[str] = typer.Option(
        None, help="Join dates: each row = 'this symbol joined the index on this date'"),
    since: Optional[str] = typer.Option(
        None, help="With --added-column: merge join dates before this date (YYYY-MM-DD)"),
) -> None:
    """Import universe membership from a CSV file (a plain list, a history, or join dates)."""
    from learntrade.core.config import get_config
    from learntrade.data.store.pit import PITStore
    from learntrade.data.universe import UniverseManager

    _setup_logging()
    cfg = get_config()
    with PITStore(cfg.workspace.data_dir) as store:
        mgr = UniverseManager(store)
        collapse = (datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    if since else None)
        n = mgr.import_csv(mic.upper(), file, symbol_column=symbol_column,
                           date_column=date_column, added_column=added_column,
                           collapse_before=collapse)
    console.print(f"[green]Imported universe for {mic.upper()}[/] ({n} snapshot(s))")


@universe_app.command("fetch-index")
def universe_fetch_index(
    index: str = typer.Argument(help="Index name. Built in: SP500"),
    since: str = typer.Option("2015-01-01", help="Start of your data window YYYY-MM-DD"),
) -> None:
    """Download an index's members with their join dates (removes 'joined after winning' bias)."""
    from learntrade.core.config import get_config
    from learntrade.data.index_history import fetch_index
    from learntrade.data.store.pit import PITStore
    from learntrade.data.universe import UniverseManager, snapshots_from_added_dates
    from learntrade.core.types import Observation

    _setup_logging()
    cfg = get_config()
    try:
        mic, rows = fetch_index(index)
    except Exception as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from None
    floor = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    snaps = snapshots_from_added_dates({sym: d for sym, d in rows}, floor)
    obs = [Observation(instrument_id=None, source_id=f"index:{index.upper()}", kind="universe",
                       event_time=d, available_at=d,
                       payload={"mic": mic, "members": members}, pit_level="partial")
           for d, members in snaps]
    with PITStore(cfg.workspace.data_dir) as store:
        n = store.ingest(obs)
    joined_late = sum(1 for _s, d in rows if d and d > floor)
    console.print(
        f"[green]Imported {index.upper()} for {mic}[/]: {len(rows)} members, "
        f"{joined_late} joined after {since}, {n} membership snapshot(s)."
    )
    console.print(
        "[dim]Join dates are applied from now on in training and evaluation. Companies that "
        "already left the index are not in this list; for those, import a membership history "
        "CSV with --date-column.[/]"
    )


@universe_app.command("list")
def universe_list(
    mic: str = typer.Option(..., help="Exchange MIC code"),
) -> None:
    """List current universe members for a market."""
    from datetime import timezone
    from learntrade.core.config import get_config
    from learntrade.data.store.pit import PITStore

    _setup_logging()
    cfg = get_config()
    now = datetime.now(tz=timezone.utc)
    with PITStore(cfg.workspace.data_dir) as store:
        ids = store.get_universe(mic.upper(), now)
    if not ids:
        console.print(f"[yellow]No universe configured for {mic.upper()}[/]")
        return
    console.print(f"[bold]{len(ids)} instruments in {mic.upper()} universe[/]")
    for iid in sorted(ids)[:50]:
        console.print(f"  {iid}")
    if len(ids) > 50:
        console.print(f"  … and {len(ids) - 50} more")


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


@app.command()
def doctor() -> None:
    """Diagnose common installation, data and configuration issues."""
    from learntrade.setup.doctor import run_doctor

    _setup_logging()
    issues = run_doctor()
    console.print("\n[bold]LearnTrade Doctor[/]\n")

    if not issues:
        console.print("[green]No issues found.[/]")
        return

    for issue in issues:
        icon = {"error": "[red]✗[/]", "warning": "[yellow]![/]", "info": "[green]✓[/]"}.get(issue.severity, " ")
        console.print(f"  {icon}  [{issue.severity.upper()}] {issue.check}: {issue.message}")
        if issue.fix:
            console.print(f"       → Fix: [dim]{issue.fix}[/]")

    errors = sum(1 for i in issues if i.severity == "error")
    if errors:
        console.print(f"\n[red]{errors} error(s) found.[/] Run the suggested fixes above.")
        raise typer.Exit(1)
    else:
        console.print(f"\n[yellow]{len(issues)} warning(s).[/] System is functional but not optimal.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
