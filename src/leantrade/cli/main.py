"""
LearnTrade CLI.

Usage:
    leantrade <command> [options]

All commands that need a workspace first run `leantrade init` (or the browser
wizard at `leantrade serve --open`) to create config/leantrade.toml.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

app = typer.Typer(
    name="leantrade",
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
    "ridge": "leantrade.quant.models.ridge:RidgeReturnModel",
    "lgbm": "leantrade.quant.models.lgbm_return:LGBMReturnModel",
    "lgbm_rank": "leantrade.quant.models.lgbm_rank:LGBMRankModel",
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
        "leantrade.api.app:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False,
    )


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@app.command()
def init(
    config_dir: Path = typer.Option(Path("config"), help="Directory for leantrade.toml"),
) -> None:
    """Interactive workspace initialisation wizard (terminal version)."""
    from leantrade.setup.wizard import run_terminal_wizard
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
    from leantrade.core.config import get_config
    from leantrade.data.engine import DataEngine
    from leantrade.data.store.pit import PITStore

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
    from leantrade.core.config import get_config
    from leantrade.data.quality.checks import run_all_checks
    from leantrade.data.store.pit import PITStore

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
) -> None:
    """Walk-forward train and calibrate a quant model."""
    from leantrade.core.config import get_config
    from leantrade.data.store.pit import PITStore
    from leantrade.quant.trainer import QuantTrainer

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
                )
                result = trainer.train(mic, since_dt, now)
                console.print(f"[green]✓[/] {result.summary()}")
                if result.calibration:
                    console.print(
                        f"  Brier={result.calibration.brier_score:.4f} "
                        f"AUC={result.calibration.roc_auc:.4f} "
                        f"dirAcc={result.calibration.directional_accuracy:.2%}"
                    )
            except Exception as exc:
                console.print(f"[red]✗ {mic}: {exc}[/]")


# ---------------------------------------------------------------------------
# Shared: load a trained model + calibrator back off disk
# ---------------------------------------------------------------------------


def _model_artifact_paths(cfg: Any, market: str, horizon: str, model_name: str) -> tuple[Path, Path]:
    model_cls = _require_model_class(model_name)
    models_dir = cfg.workspace.data_dir / "models"
    model_path = models_dir / f"{market}_{horizon}_{model_cls.id}.pkl"
    cal_path = models_dir / f"{market}_{horizon}_calibrator.pkl"
    return model_path, cal_path


def _require_model_class(model_name: str) -> Any:
    import importlib

    if model_name not in _MODEL_REGISTRY:
        console.print(f"[red]Unknown model {model_name!r}. Choose: {list(_MODEL_REGISTRY)}[/]")
        raise typer.Exit(1)
    mod_path, cls_name = _MODEL_REGISTRY[model_name].split(":")
    return getattr(importlib.import_module(mod_path), cls_name)


def _load_trained_model(
    cfg: Any, market: str, horizon: str, model_name: str
) -> tuple[Any, Any]:
    """
    Load a previously trained model and its (optional) calibrator.
    Returns (None, None) if `leantrade train` hasn't been run for this
    market/horizon/model combination yet — callers decide how to degrade.
    """
    model_cls = _require_model_class(model_name)
    model_path, cal_path = _model_artifact_paths(cfg, market, horizon, model_name)
    if not model_path.exists():
        return None, None
    model = model_cls.load(str(model_path))
    calibrator = None
    if cal_path.exists():
        from leantrade.quant.calibration.isotonic import IsotonicCalibrator
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
    from leantrade.core.types import Forecast, ForecastQuantiles, InstrumentId
    from leantrade.features.price_volume import PriceVolumeFeatureSet

    now = datetime.now(tz=timezone.utc)
    instruments = store.get_universe(market, now)
    if not instruments:
        return [], "no_universe"

    feature_set = PriceVolumeFeatureSet()
    features = feature_set.compute(instruments, now, store)
    if features.is_empty():
        return [], "no_features"

    trained_model, calibrator = _load_trained_model(cfg, market, horizon, model_name)
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
    from leantrade.backtest.engine import BacktestConfig, EventDrivenBacktest, VectorisedBacktest
    from leantrade.backtest.report import generate_report
    from leantrade.core.config import get_config
    from leantrade.data.store.pit import PITStore
    from leantrade.features.price_volume import PriceVolumeFeatureSet
    from leantrade.markets import get_pack
    from leantrade.quant.targets.labels import compute_forward_returns, merge_features_labels

    _setup_logging()
    cfg = get_config()
    since_dt = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(tz=timezone.utc)

    with PITStore(cfg.workspace.data_dir) as store:
        instruments = store.get_universe(market, now)
        if not instruments:
            console.print(f"[red]No universe for {market}. Run `leantrade universe import`.[/]")
            raise typer.Exit(1)

        feature_set = PriceVolumeFeatureSet()
        bars = store.query_bars(instruments, as_of=now, since=since_dt)
        if bars.is_empty():
            console.print("[red]No bars found. Run `leantrade data pull` first.[/]")
            raise typer.Exit(1)

        from leantrade.quant.targets.horizon import get_horizon
        import polars as pl
        h = get_horizon(horizon)
        features = feature_set.compute_history(instruments, since_dt, now, store)
        if features.is_empty():
            console.print("[red]No features. Run `leantrade data pull` first.[/]")
            raise typer.Exit(1)

        trained_model, calibrator = _load_trained_model(cfg, market, horizon, model)
        signal_source: str
        if trained_model is not None:
            # Use the TRAINED model's forecast as the signal, not a raw
            # feature column — this is what the model/calibration/backtest
            # pipeline is actually supposed to be testing.
            from leantrade.core.types import InstrumentId
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
                f"Run `leantrade train --model {model} --market {market} "
                f"--horizon {horizon}` first for a real signal.[/]"
            )
            labels = compute_forward_returns(bars, h.forward_days)
            if labels.is_empty():
                console.print("[red]No labels could be computed either. Run `leantrade data pull` first.[/]")
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
            console.print("[red]No signals produced. Run `leantrade train` first.[/]")
            raise typer.Exit(1)

        pack = get_pack(market)
        back_cfg = BacktestConfig(
            pit_level="none",
            survivorship_note="No survivorship check applied",
            engine=engine,
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
    from leantrade.core.config import get_config
    from leantrade.data.store.pit import PITStore

    _setup_logging()
    cfg = get_config()

    with PITStore(cfg.workspace.data_dir) as store:
        forecasts, source = _generate_current_forecasts(cfg, store, market, horizon, model)

    if source == "no_universe":
        console.print(f"[red]No universe for {market}. Run `leantrade universe import`.[/]")
        raise typer.Exit(1)
    if source == "no_features":
        console.print("[red]No features. Run `leantrade data pull` first.[/]")
        raise typer.Exit(1)
    if not forecasts:
        console.print("[red]No forecasts produced.[/]")
        raise typer.Exit(1)
    if source.startswith("raw momentum"):
        console.print(
            f"[yellow]{source} — run `leantrade train --model {model} "
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
# analyze
# ---------------------------------------------------------------------------


@app.command()
def analyze(
    instrument: str = typer.Argument(help="Instrument ID, e.g. XNYS:IBM"),
    horizon: str = typer.Option("swing_20d", help="Forecast horizon"),
    model: str = typer.Option("ridge", help="Model: ridge | lgbm | lgbm_rank"),
) -> None:
    """Generate a research report for one instrument."""
    from leantrade.core.config import get_config
    from leantrade.core.types import InstrumentId
    from leantrade.data.store.pit import PITStore

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
            console.print(f"[red]No universe for {market}. Run `leantrade universe import`.[/]")
            raise typer.Exit(1)
        if source == "no_features" or fc is None:
            console.print(
                f"[red]No data for {instrument}. Run `leantrade data pull` "
                f"(and check it's in the {market} universe).[/]"
            )
            raise typer.Exit(1)

        console.print(f"\n[bold]{instrument}[/] — {horizon} [dim]({source})[/]")
        console.print(
            f"  Expected excess return: [bold]{fc.expected_return:+.2%}[/]  "
            f"P(positive): {fc.p_positive:.1%}  Volatility: {fc.volatility:.1%}"
        )
        console.print(
            f"  Range (5–95%): {fc.quantiles.q05:+.2%} to {fc.quantiles.q95:+.2%}"
        )
        if source.startswith("raw momentum"):
            console.print(
                f"[yellow]This is an uncalibrated momentum proxy, not a trained "
                f"model's forecast. Run `leantrade train` first.[/]"
            )

        thesis = None
        counter_thesis = None
        agreement = None
        llm_cost = 0.0
        if cfg.llm.enabled:
            import asyncio
            from leantrade.features.price_volume import PriceVolumeFeatureSet
            from leantrade.llm.research.fact_sheet import build_fact_sheet
            from leantrade.llm.research.orchestrator import ResearchOrchestrator
            from leantrade.llm.router import LLMRouter
            try:
                router = LLMRouter()
                feat_row_df = PriceVolumeFeatureSet().compute([iid], datetime.now(tz=timezone.utc), store)
                feat_row = feat_row_df.row(0, named=True) if not feat_row_df.is_empty() else {}
                fact_sheet = build_fact_sheet(
                    str(iid), datetime.now(tz=timezone.utc), horizon, feat_row,
                    forecast=fc, universe_size=len(forecasts),
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

        from leantrade.attribution.store import AttributionStore
        from leantrade.core.types import DecisionRecord, ForecastAttribution
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
    from leantrade.execution.brokers.csv_import import CsvImportAdapter
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
    from leantrade.attribution.store import AttributionStore
    from leantrade.core.config import get_config
    from leantrade.core.types import DecisionRecord, ForecastAttribution
    from leantrade.data.store.pit import PITStore
    from leantrade.portfolio.allocators import (
        EqualWeightAllocator, HRPAllocator, InverseVolAllocator,
        MeanVarianceAllocator, RiskParityAllocator,
    )
    from leantrade.portfolio.constraints import PortfolioConstraints, apply_constraints

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
            console.print(f"[red]No universe for {market}. Run `leantrade universe import`.[/]")
            raise typer.Exit(1)
        if source == "no_features" or not forecasts:
            console.print("[red]No features/forecasts. Run `leantrade data pull` first.[/]")
            raise typer.Exit(1)
        if source.startswith("raw momentum"):
            console.print(
                f"[yellow]{source} — run `leantrade train --model {model} "
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
    from leantrade.core.types import InstrumentId, Order
    from leantrade.execution.live import LiveExecutionError, check_live_gates
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
    from leantrade.attribution.store import AttributionStore
    from leantrade.core.config import get_config

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
    from leantrade.markets import list_packs
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
    from leantrade.markets.validate import validate_all_packs, validate_pack

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
    from leantrade.markets.loader import _BUILTIN_MARKET_PACKS_DIR

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
    console.print(f"[green]Scaffolded {target}[/] — edit costs and calendar, then run `leantrade market validate {mic.upper()}`")


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
    from leantrade.llm.eval.suite import EvalSuite, EVAL_CASES
    from leantrade.llm.router import LLMRouter

    _setup_logging()
    router = LLMRouter()
    if not router.is_enabled:
        console.print("[red]LLM is disabled. Set [llm] enabled = true in leantrade.toml.[/]")
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
    from leantrade.llm.cache import LLMCache
    from leantrade.core.config import get_config

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
) -> None:
    """Import universe membership from a CSV file."""
    from leantrade.core.config import get_config
    from leantrade.data.store.pit import PITStore
    from leantrade.data.universe import UniverseManager

    _setup_logging()
    cfg = get_config()
    with PITStore(cfg.workspace.data_dir) as store:
        mgr = UniverseManager(store)
        n = mgr.import_csv(mic.upper(), file, symbol_column=symbol_column)
    console.print(f"[green]Imported universe for {mic.upper()}[/] ({n} snapshot(s))")


@universe_app.command("list")
def universe_list(
    mic: str = typer.Option(..., help="Exchange MIC code"),
) -> None:
    """List current universe members for a market."""
    from datetime import timezone
    from leantrade.core.config import get_config
    from leantrade.data.store.pit import PITStore

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
    from leantrade.setup.doctor import run_doctor

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
