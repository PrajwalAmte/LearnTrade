"""
Fact sheet builder and research roles tests — no LLM calls required.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from learntrade.core.types import Forecast, ForecastQuantiles, InstrumentId
from learntrade.llm.research.fact_sheet import FactSheet, build_fact_sheet
from learntrade.llm.research.roles import ROLES, ResearchRole

UTC = timezone.utc


def _forecast(exp_ret: float = 0.018) -> Forecast:
    return Forecast(
        instrument_id=InstrumentId("XNYS:IBM"),
        as_of=datetime(2024, 1, 15, tzinfo=UTC),
        horizon="swing_20d",
        expected_return=exp_ret,
        p_positive=0.58,
        volatility=0.22,
        quantiles=ForecastQuantiles(q05=-0.05, q25=-0.01, q50=exp_ret, q75=0.04, q95=0.09),
        rank=0.91,
        model_id="ridge_v1", data_snapshot_id="",
    )


# ── Fact sheet ────────────────────────────────────────────────────────────────

def test_build_fact_sheet_basic():
    features = {"ret_1d": 0.003, "ret_5d": 0.012, "ret_20d": 0.025,
                "ewma_vol_20d": 0.22, "vol_surprise_20d": 0.4,
                "high_52w_ratio": 0.87, "close": 153.42}
    sheet = build_fact_sheet("XNYS:IBM", datetime(2024, 1, 15, tzinfo=UTC),
                             "swing_20d", features, _forecast())
    assert sheet.instrument_id == "XNYS:IBM"
    assert abs(sheet.price - 153.42) < 1e-6
    assert sheet.quant_forecast is not None
    assert abs(sheet.quant_forecast["expected_return"] - 0.018) < 1e-6


def test_fact_sheet_to_text_contains_key_info():
    features = {"ret_1d": 0.003, "ret_5d": 0.012, "ret_20d": 0.025,
                "ewma_vol_20d": 0.22, "close": 153.42}
    sheet = build_fact_sheet("XNYS:IBM", datetime(2024, 1, 15, tzinfo=UTC),
                             "swing_20d", features, _forecast())
    text = sheet.to_text()
    assert "XNYS:IBM" in text
    assert "swing_20d" in text
    assert "153.42" in text
    assert "P(positive)" in text


def test_fact_sheet_to_text_no_crash_empty_evidence():
    features = {"ret_20d": 0.01, "ewma_vol_20d": 0.20, "close": 100.0}
    sheet = build_fact_sheet("XNYS:TEST", datetime(2024, 1, 1, tzinfo=UTC),
                             "swing_5d", features)
    text = sheet.to_text()
    assert "XNYS:TEST" in text


def test_fact_sheet_evidence_shown():
    features = {"ret_20d": 0.01, "ewma_vol_20d": 0.20, "close": 100.0}
    sheet = build_fact_sheet(
        "XNYS:IBM", datetime(2024, 1, 15, tzinfo=UTC), "swing_20d", features,
        evidence={"guidance_change_score": 1.0, "n_risk_flags": 2},
    )
    text = sheet.to_text()
    assert "raised" in text.lower() or "guidance" in text.lower()


# ── Research roles ────────────────────────────────────────────────────────────

def test_all_nine_roles_defined():
    expected = {"technical", "fundamental", "macro", "event",
                "bull", "bear", "verifier", "quant_challenger", "synthesiser"}
    assert set(ROLES.keys()) == expected


def test_each_role_has_required_fields():
    for role_id, role in ROLES.items():
        assert isinstance(role, ResearchRole)
        assert len(role.system_prompt) > 50, f"{role_id} has too-short system prompt"
        assert "type" in role.output_schema
        assert "properties" in role.output_schema
        assert len(role.output_schema["properties"]) >= 1


def test_synthesiser_has_required_output_fields():
    schema = ROLES["synthesiser"].output_schema
    required = set(schema.get("required", []))
    assert "thesis" in required
    assert "counter_thesis" in required
    assert "agreement_with_quant" in required


def test_bull_bear_require_escalation():
    assert ROLES["bull"].requires_escalation
    assert ROLES["bear"].requires_escalation


def test_base_roles_do_not_require_escalation():
    for role_id in ["technical", "fundamental", "macro", "event",
                    "verifier", "quant_challenger", "synthesiser"]:
        assert not ROLES[role_id].requires_escalation
