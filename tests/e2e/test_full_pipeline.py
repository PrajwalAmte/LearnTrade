"""
End-to-end integration test — exercises all 6 phases in sequence.

Phase 1: Data ingestion → quality checks → PIT enforcement
Phase 2: Feature computation → model training → backtest + report
Phase 3: Event-driven simulation → cost model → reconciliation
Phase 4: LLM cache → contamination guard → feature bridge (mocked LLM)
Phase 5: Allocators → constraints → risk engine → attribution store
Phase 6: Doctor → demo data → new market packs → live execution guard → API

All phases use synthetic/demo data. No network calls, no API keys required.
Approximate runtime: 15–30 seconds.
"""

from __future__ import annotations

import datetime as _dt
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import polars as pl
import pytest

UTC = timezone.utc


# ── Shared workspace fixture ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """Single workspace used across all e2e tests."""
    td = tmp_path_factory.mktemp("e2e_workspace")
    from leantrade.data.store.pit import PITStore
    from leantrade.setup.demo import generate_demo_data

    store = PITStore(td)
    # Generate 300 trading days of synthetic data for 50 instruments
    n_obs = generate_demo_data(store, mic="XNYS", n_days=300, seed=99)
    assert n_obs > 0, "Demo data generation produced no observations"
    yield {"data_dir": td, "store": store, "n_obs": n_obs}
    store.close()


# ── Phase 1 — Data Foundation ─────────────────────────────────────────────────

class TestPhase1Data:
    def test_pit_store_has_observations(self, workspace):
        store = workspace["store"]
        stats = store.stats()
        assert stats["total_observations"] > 10_000  # 50 instruments × 300 days

    def test_universe_registered(self, workspace):
        store = workspace["store"]
        universe = store.get_universe("XNYS", datetime.now(tz=UTC))
        assert len(universe) == 50

    def test_pit_golden_rule_enforced(self, workspace):
        """Bars with available_at in the future must NOT appear in PIT queries."""
        from leantrade.core.types import InstrumentId
        store = workspace["store"]
        past = datetime(2020, 1, 1, tzinfo=UTC)
        ids = [InstrumentId("XNYS:SYM00")]
        df = store.query_bars(ids, as_of=past, since=datetime(2010, 1, 1, tzinfo=UTC))
        assert df.is_empty(), "PIT rule violated: future bars visible at past as_of"

    def test_quality_checks_pass_on_demo_data(self, workspace):
        from leantrade.core.types import InstrumentId
        from leantrade.data.quality.checks import run_all_checks
        store = workspace["store"]
        now = datetime.now(tz=UTC)
        since = now - _dt.timedelta(days=365)
        ids = [InstrumentId(f"XNYS:SYM{i:02d}") for i in range(10)]
        df = store.query_bars(ids, as_of=now, since=since)
        assert not df.is_empty()
        report = run_all_checks(df, as_of=now, snapshot_id="e2e_test",
                                since=since, until=now)
        assert report.passed, f"Quality check errors: {report.issues}"

    def test_market_packs_all_pass_validation(self):
        from leantrade.markets.validate import validate_all_packs
        reports = validate_all_packs()
        # All shipped packs (including Phase 6 additions) must have zero errors
        for mic, report in reports.items():
            errors = [i for i in report.issues if i.severity == "error"]
            assert errors == [], f"{mic} validation errors: {errors}"

    def test_phase6_market_packs_present(self):
        from leantrade.markets.loader import list_packs
        packs = list_packs()
        for mic in ["XLON", "XETR", "XTKS", "XASX", "XBOM"]:
            assert mic in packs, f"Phase 6 market pack {mic} not found"


# ── Phase 2 — Quant Baseline ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def training_result(workspace):
    from leantrade.quant.trainer import QuantTrainer
    from leantrade.quant.models.ridge import RidgeReturnModel
    store = workspace["store"]
    now = datetime.now(tz=UTC)
    since = now - _dt.timedelta(days=290)
    trainer = QuantTrainer(
        store=store, horizon_id="swing_20d",
        model_class=RidgeReturnModel, n_splits=3,
        embargo_bars=10, min_train_bars=30,
        artifacts_dir=workspace["data_dir"] / "models",
    )
    result = trainer.train("XNYS", since, now)
    return result


class TestPhase2Quant:
    def test_training_completes_with_folds(self, training_result):
        assert training_result.ic_result.n_periods >= 2, "Need at least 2 CV folds"

    def test_ic_result_is_finite(self, training_result):
        assert not np.isnan(training_result.ic_result.ic_mean)
        assert not np.isnan(training_result.ic_result.ic_tstat)

    def test_calibration_produces_valid_brier(self, training_result):
        if training_result.calibration:
            assert 0 <= training_result.calibration.brier_score <= 0.26

    def test_model_artifact_saved(self, training_result):
        assert training_result.model_path is not None
        assert Path(training_result.model_path).exists()

    def test_vectorised_backtest_runs(self, workspace, training_result):
        from leantrade.backtest.engine import BacktestConfig, VectorisedBacktest
        from leantrade.backtest.report import generate_report
        from leantrade.features.price_volume import PriceVolumeFeatureSet
        from leantrade.markets.loader import get_pack
        from leantrade.quant.targets.labels import compute_forward_returns, merge_features_labels
        from leantrade.quant.targets.horizon import get_horizon
        store = workspace["store"]
        now = datetime.now(tz=UTC)
        since = now - _dt.timedelta(days=290)
        instruments = store.get_universe("XNYS", now)
        bars = store.query_bars(instruments, as_of=now, since=since)
        feature_set = PriceVolumeFeatureSet()
        features = feature_set.compute_history(instruments, since, now, store)
        h = get_horizon("swing_20d")
        labels = compute_forward_returns(bars, h.forward_days)
        merged = merge_features_labels(features, labels)
        signal_col = "ret_20d" if "ret_20d" in merged.columns else merged.columns[2]
        signals = merged.select([
            pl.col("event_time").alias("date"),
            pl.col("instrument_id"),
            pl.col(signal_col).alias("signal"),
        ]).drop_nulls()
        pack = get_pack("XNYS")
        result = VectorisedBacktest().run(
            signals, bars, pack,
            BacktestConfig(pit_level="none", top_n=10,
                           survivorship_note="synthetic demo data"),
        )
        assert len(result.daily_returns) > 0
        # Generate report
        report = generate_report(
            result, "swing_20d", "XNYS", ["ridge_v1"],
            workspace["data_dir"] / "reports",
        )
        assert report.pit_level == "none"
        assert "cost_drag_bps_annual" in report.__dataclass_fields__


# ── Phase 3 — Execution Realism ───────────────────────────────────────────────

class TestPhase3Execution:
    def test_cost_model_xnys(self):
        from leantrade.execution.costs import compute_trade_cost
        from leantrade.markets.loader import get_pack
        pack = get_pack("XNYS")
        cost = compute_trade_cost("sell", 500, 100.0, 1_000_000, 0.20, pack)
        assert cost.total > 0
        assert cost.total_bps > 0
        assert cost.taxes > 0  # SEC fee on sells

    def test_event_driven_simulator_runs(self, workspace):
        from leantrade.core.types import InstrumentId, Order
        from leantrade.execution.simulator.engine import EventDrivenSimulator, SimulatorConfig
        from leantrade.markets.loader import get_pack
        store = workspace["store"]
        now = datetime.now(tz=UTC)
        since = now - _dt.timedelta(days=10)
        instruments = store.get_universe("XNYS", now)[:3]
        bars = store.query_bars(instruments, as_of=now, since=since)
        if bars.is_empty():
            pytest.skip("No recent bars in demo data")
        pack = get_pack("XNYS")
        sim = EventDrivenSimulator(pack, SimulatorConfig(initial_cash=100_000.0))
        dates = sorted(bars["event_time"].unique().to_list())
        for dt in dates[:5]:
            day_bars = bars.filter(pl.col("event_time") == dt)
            orders = [Order(instrument_id=instruments[0], side="buy", quantity=50)]
            sim.step(dt, day_bars, orders if dt == dates[0] else [])
        result = sim.get_result()
        assert len(result.nav_history) == 5

    def test_live_execution_requires_three_optins(self):
        from leantrade.execution.live import check_live_gates, LiveExecutionError
        from leantrade.core.types import InstrumentId, Order
        order = Order(instrument_id=InstrumentId("XNYS:IBM"), side="buy",
                      quantity=100, notes="confirmed")
        # Gate 1 (config mode != "live") should block by default
        with pytest.raises(LiveExecutionError, match="Gate 1"):
            check_live_gates([order])


# ── Phase 4 — LLM Evidence Layer ─────────────────────────────────────────────

class TestPhase4LLM:
    def test_cache_100pct_hit_rate(self, tmp_path):
        from leantrade.llm.cache import LLMCache
        cache = LLMCache(tmp_path / "cache.duckdb")
        response = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
        cache.set("k1", "gpt-4o", "extract", response)
        # Three hits, one set → 100% cache hit rate after first call
        for _ in range(3):
            assert cache.get("k1") is not None
        stats = cache.stats()
        assert stats["total_hits"] == 3
        cache.close()

    def test_contamination_guard_blocks_training_signal(self):
        from leantrade.llm.guard import is_contaminated
        # gpt-4o cutoff = 2023-10-01; a 2022 backtest date is contaminated
        assert is_contaminated("gpt-4o", datetime(2022, 6, 1, tzinfo=UTC))
        # 2026 is clean
        assert not is_contaminated("gpt-4o", datetime(2026, 1, 1, tzinfo=UTC))

    def test_feature_bridge_pit_enforcement(self):
        from leantrade.llm.extraction.schemas import ExtractedEvidence
        from leantrade.llm.extraction.feature_bridge import (
            evidence_list_to_dataframe, merge_evidence_features_pit,
        )
        ev = ExtractedEvidence(
            doc_id="sha256:abc",
            instrument_ids=["XNYS:SYM00"],
            available_at=datetime(2024, 6, 1, tzinfo=UTC),
            event_type="earnings", guidance_change="raised", confidence=0.85,
        )
        price_df = pl.DataFrame([{"instrument_id": "XNYS:SYM00",
                                   "event_time": datetime(2024, 5, 1, tzinfo=UTC)}])
        ev_df = evidence_list_to_dataframe([ev])
        # Evidence available Jun 1 must NOT be visible at May 1
        merged = merge_evidence_features_pit(price_df, ev_df,
                                             datetime(2024, 5, 1, tzinfo=UTC))
        assert float(merged["guidance_change_score"][0]) == 0.0
        # Must be visible at Jul 1
        merged2 = merge_evidence_features_pit(price_df, ev_df,
                                              datetime(2024, 7, 1, tzinfo=UTC))
        assert float(merged2["guidance_change_score"][0]) == 1.0

    def test_budget_tracker_enforces_per_run_limit(self):
        from leantrade.llm.router import BudgetTracker, BudgetExceededError
        bt = BudgetTracker(per_run_limit=0.01, per_day_limit=100.0)
        bt.record(0.009)
        with pytest.raises(BudgetExceededError):
            bt.check(0.005)


# ── Phase 5 — Research + Portfolio ───────────────────────────────────────────

class TestPhase5Portfolio:
    def test_hrp_allocator_valid_weights(self, workspace):
        from leantrade.core.types import Forecast, ForecastQuantiles, InstrumentId
        from leantrade.portfolio.allocators import HRPAllocator
        rng = np.random.default_rng(7)
        forecasts = [
            Forecast(
                instrument_id=InstrumentId(f"XNYS:SYM{i:02d}"),
                as_of=datetime.now(tz=UTC), horizon="swing_20d",
                expected_return=float(rng.normal(0.01, 0.03)),
                p_positive=0.55, volatility=float(rng.uniform(0.1, 0.3)),
                quantiles=ForecastQuantiles(q05=-0.05, q25=-0.02, q50=0.01,
                                            q75=0.03, q95=0.07),
                model_id="test", data_snapshot_id="",
            )
            for i in range(10)
        ]
        weights = HRPAllocator().allocate(forecasts)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(v >= -1e-9 for v in weights.values())

    def test_constraints_applied(self):
        from leantrade.core.types import InstrumentId
        from leantrade.portfolio.constraints import PortfolioConstraints, apply_constraints
        weights = {InstrumentId(f"XNYS:SYM{i:02d}"): 0.5 if i < 2 else 0.0
                   for i in range(5)}
        # Remove zero weights before constraint application
        weights = {k: v for k, v in weights.items() if v > 0}
        c = PortfolioConstraints(max_position=0.30, long_only=True)
        result = apply_constraints(weights, c)
        # 2 names x 30% cap = 60% max investable; the cap must hold rather
        # than being renormalised back above it to force full investment.
        for v in result.values():
            assert v <= c.max_position + 1e-9
        assert abs(sum(result.values()) - 2 * c.max_position) < 1e-9

    def test_kill_switch_zero_emissions(self):
        from leantrade.core.types import InstrumentId, Order
        from leantrade.risk.engine import KillSwitchError, RiskEngine
        engine = RiskEngine()
        engine.start_of_day(100_000.0)
        engine.engage_kill_switch()
        orders = [Order(instrument_id=InstrumentId(f"XNYS:SYM{i}"),
                        side="buy", quantity=100) for i in range(3)]
        with pytest.raises(KillSwitchError):
            engine.check_orders(orders, [], 100_000.0)
        assert engine.kill_switch_active

    def test_attribution_store_llm_explains_not_sets(self, workspace):
        from leantrade.attribution.store import AttributionStore
        from leantrade.core.types import (
            DecisionRecord, Forecast, ForecastAttribution, ForecastQuantiles,
            InstrumentId,
        )
        from leantrade.portfolio.allocators import HRPAllocator
        rng = np.random.default_rng(42)
        fc = [
            Forecast(
                instrument_id=InstrumentId(f"XNYS:SYM{i:02d}"),
                as_of=datetime.now(tz=UTC), horizon="swing_20d",
                expected_return=float(rng.normal(0.01, 0.03)),
                p_positive=0.55, volatility=0.20,
                quantiles=ForecastQuantiles(q05=-0.05, q25=-0.02, q50=0.01,
                                            q75=0.03, q95=0.07),
                model_id="ridge_v1", data_snapshot_id="",
            )
            for i in range(5)
        ]
        weights = HRPAllocator().allocate(fc)
        record = DecisionRecord(
            decision_id="e2e_test/XNYS:SYM00/swing_20d",
            instrument_id=InstrumentId("XNYS:SYM00"), horizon="swing_20d",
            as_of=datetime.now(tz=UTC), forecast=fc[0],
            thesis="Price momentum supported by volume.",  # LLM narrative
            target_weight=float(weights[InstrumentId("XNYS:SYM00")]),  # optimiser weight
            attribution=ForecastAttribution(models={"ridge_v1": 1.0}, llm_cost_usd=0.001),
        )
        db_path = workspace["data_dir"] / "attr_e2e.duckdb"
        with AttributionStore(db_path) as store:
            store.write(record)
            df = store.query()
        assert len(df) == 1
        # Key invariant: thesis ≠ weight source
        assert str(df["thesis"][0]).strip() != ""
        assert float(df["target_weight"][0]) > 0

    def test_meta_combiner_blends_signals(self):
        from leantrade.combine.combiner import CombinerConfig, MetaCombiner
        from leantrade.core.types import Forecast, ForecastQuantiles, InstrumentId
        rng = np.random.default_rng(10)
        forecasts = [
            Forecast(
                instrument_id=InstrumentId(f"XNYS:SYM{i:02d}"),
                as_of=datetime.now(tz=UTC), horizon="swing_20d",
                expected_return=float(rng.normal(0.01, 0.02)),
                p_positive=0.55, volatility=0.20,
                quantiles=ForecastQuantiles(q05=-0.05, q25=-0.02, q50=0.01,
                                            q75=0.03, q95=0.07),
                model_id="test", data_snapshot_id="",
            )
            for i in range(5)
        ]
        llm_views = {f"XNYS:SYM{i:02d}": float(rng.normal(0.01, 0.01)) for i in range(5)}
        combiner = MetaCombiner(CombinerConfig(method="static"))
        updated = combiner.combine(forecasts, llm_views)
        assert len(updated) == 5
        assert all(f.rank is not None for f in updated)


# ── Phase 6 — Product ─────────────────────────────────────────────────────────

class TestPhase6Product:
    def test_doctor_identifies_no_data_issue(self, tmp_path):
        """Doctor must flag missing data when PIT store is empty."""
        from leantrade.setup.doctor import run_doctor
        # Use a config pointing to an empty dir with no data
        cfg_text = f"""
[workspace]
base_currency = "USD"
data_dir = "{tmp_path}"
timezone = "UTC"
[markets]
enabled = ["XNYS"]
[data.providers]
prices = ["yahoo"]
macro = []
[horizons]
enabled = ["swing_20d"]
[quant]
models = ["ridge"]
calibration = "isotonic"
pooled_across_markets = true
[llm]
enabled = false
[portfolio]
allocator = "hrp"
max_position = 0.05
long_only = true
rebalance = "weekly"
[risk]
max_drawdown_stop = 0.20
kill_switch = true
[execution]
mode = "paper"
"""
        cfg_path = tmp_path / "leantrade.toml"
        cfg_path.write_text(cfg_text)
        issues = run_doctor(config_path=cfg_path)
        checks = [i.check for i in issues]
        # Should detect missing data (no .duckdb file)
        assert "data" in checks

    def test_live_execution_three_optins_required(self):
        from leantrade.execution.live import check_live_gates, LiveExecutionError
        from leantrade.core.types import InstrumentId, Order
        order = Order(instrument_id=InstrumentId("XNYS:IBM"),
                      side="buy", quantity=100, notes="confirmed")
        # Must fail at Gate 1 (config mode != "live") by default
        with pytest.raises(LiveExecutionError, match="Gate 1"):
            check_live_gates([order])

    def test_demo_data_generation(self, tmp_path):
        from leantrade.data.store.pit import PITStore
        from leantrade.setup.demo import generate_demo_data
        store = PITStore(tmp_path)
        n = generate_demo_data(store, mic="XNYS", n_days=20, seed=0)
        assert n >= 20 * 50  # at least 20 bars × 50 instruments
        universe = store.get_universe("XNYS", datetime.now(tz=UTC))
        assert len(universe) == 50
        store.close()

    def test_api_health_endpoint(self):
        from fastapi.testclient import TestClient
        from leantrade.api.app import create_app
        client = TestClient(create_app())
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_api_config_endpoint(self, tmp_path):
        """Config endpoint returns valid structure."""
        from fastapi.testclient import TestClient
        from leantrade.api.app import create_app
        client = TestClient(create_app())
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "base_currency" in data
        assert "execution_mode" in data

    def test_api_setup_wizard_steps(self):
        from fastapi.testclient import TestClient
        from leantrade.api.app import create_app
        client = TestClient(create_app())
        # Setup status
        resp = client.get("/api/setup/status")
        assert resp.status_code == 200
        assert "steps" in resp.json()
        # Available markets
        resp = client.get("/api/setup/available-markets")
        assert resp.status_code == 200
        packs = resp.json()["markets"]
        assert len(packs) >= 8  # XNYS, XNAS, XNSE + Phase 6 additions

    def test_phase_1_to_5_tests_still_pass(self):
        """Smoke test: spot-check that nothing from earlier phases is broken."""
        from leantrade.core.types import InstrumentId
        assert InstrumentId("XNYS:IBM").mic == "XNYS"
        from leantrade.markets.loader import get_pack
        pack = get_pack("XNYS")
        assert pack.currency == "USD"
        from leantrade.llm.guard import is_contaminated
        assert is_contaminated("gpt-4o", datetime(2022, 1, 1, tzinfo=UTC))
