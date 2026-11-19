"""
Evidence extraction tests — schemas, feature bridge, PIT enforcement.
These tests run without an LLM API key (no network calls).
"""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from learntrade.llm.extraction.feature_bridge import (
    FEATURE_NAMES,
    evidence_list_to_dataframe,
    evidence_to_features,
    merge_evidence_features_pit,
)
from learntrade.llm.extraction.schemas import ExtractedEvidence, KeyFigure

UTC = timezone.utc


def _make_evidence(
    guidance: str | None = "raised",
    surprise: str | None = "positive",
    tone: str | None = "positive",
    n_risk_flags: int = 0,
    confidence: float = 0.8,
    instrument_ids: list[str] | None = None,
    available_at: datetime | None = None,
) -> ExtractedEvidence:
    return ExtractedEvidence(
        doc_id="sha256:abc",
        instrument_ids=instrument_ids or ["XNYS:IBM"],
        available_at=available_at or datetime(2024, 1, 15, tzinfo=UTC),
        event_type="earnings",
        guidance_change=guidance,
        surprise_direction=surprise,
        management_tone=tone,
        risk_flags=["fx_headwind"] * n_risk_flags,
        key_figures=[
            KeyFigure(name="eps", value=3.50, unit="USD", quote_span=(10, 25))
        ],
        confidence=confidence,
    )


# ── Schema tests ─────────────────────────────────────────────────────────────

def test_extracted_evidence_valid():
    ev = _make_evidence()
    assert ev.doc_id == "sha256:abc"
    assert ev.guidance_change == "raised"
    assert len(ev.key_figures) == 1


def test_key_figure_with_span():
    fig = KeyFigure(name="revenue", value=5.1, unit="B_USD", quote_span=(0, 10))
    assert fig.quote_span == (0, 10)


def test_key_figure_without_span():
    fig = KeyFigure(name="margin", value=0.42, unit="ratio")
    assert fig.quote_span is None


# ── Feature bridge tests ──────────────────────────────────────────────────────

def test_evidence_to_features_raised_positive():
    ev = _make_evidence(guidance="raised", surprise="positive", tone="positive",
                        n_risk_flags=1, confidence=0.9)
    feats = evidence_to_features(ev)
    assert feats["guidance_change_score"] == 1.0
    assert feats["surprise_score"] == 1.0
    assert feats["management_tone_score"] == 1.0
    assert feats["n_risk_flags"] == 1
    assert feats["evidence_confidence"] == pytest.approx(0.9)
    assert feats["n_key_figures_verified"] == 1  # one figure with quote_span


def test_evidence_to_features_lowered_negative():
    ev = _make_evidence(guidance="lowered", surprise="negative", tone="negative")
    feats = evidence_to_features(ev)
    assert feats["guidance_change_score"] == -1.0
    assert feats["surprise_score"] == -1.0
    assert feats["management_tone_score"] == -1.0


def test_evidence_to_features_null_fields():
    ev = _make_evidence(guidance=None, surprise=None, tone=None)
    feats = evidence_to_features(ev)
    assert feats["guidance_change_score"] == 0.0
    assert feats["surprise_score"] == 0.0
    assert feats["management_tone_score"] == 0.0


def test_evidence_list_to_dataframe():
    ev1 = _make_evidence(instrument_ids=["XNYS:IBM"],
                         available_at=datetime(2024, 1, 15, tzinfo=UTC))
    ev2 = _make_evidence(instrument_ids=["XNYS:AAPL"],
                         available_at=datetime(2024, 1, 20, tzinfo=UTC))
    df = evidence_list_to_dataframe([ev1, ev2])
    assert len(df) == 2
    assert "instrument_id" in df.columns
    assert "available_at" in df.columns
    for col in FEATURE_NAMES:
        assert col in df.columns


def test_evidence_list_to_dataframe_empty():
    df = evidence_list_to_dataframe([])
    assert df.is_empty()


# ── PIT enforcement in merge ──────────────────────────────────────────────────

def test_merge_evidence_features_pit_respects_as_of():
    """Evidence published after as_of must NOT be used."""
    price_df = pl.DataFrame([
        {"instrument_id": "XNYS:IBM", "event_time": datetime(2024, 1, 10, tzinfo=UTC), "ret_1d": 0.01}
    ])
    # Evidence available_at = Jan 20 — FUTURE relative to as_of = Jan 10
    ev_future = _make_evidence(
        guidance="raised",
        available_at=datetime(2024, 1, 20, tzinfo=UTC),
    )
    ev_df = evidence_list_to_dataframe([ev_future])
    as_of = datetime(2024, 1, 10, tzinfo=UTC)
    merged = merge_evidence_features_pit(price_df, ev_df, as_of)
    # Evidence was not available at as_of → feature should be 0
    assert float(merged["guidance_change_score"][0]) == 0.0


def test_merge_evidence_features_pit_uses_visible_evidence():
    """Evidence published before as_of MUST be used."""
    price_df = pl.DataFrame([
        {"instrument_id": "XNYS:IBM", "event_time": datetime(2024, 1, 20, tzinfo=UTC), "ret_1d": 0.01}
    ])
    ev_past = _make_evidence(
        guidance="raised",
        available_at=datetime(2024, 1, 15, tzinfo=UTC),  # 5 days before as_of
    )
    ev_df = evidence_list_to_dataframe([ev_past])
    as_of = datetime(2024, 1, 20, tzinfo=UTC)
    merged = merge_evidence_features_pit(price_df, ev_df, as_of)
    assert float(merged["guidance_change_score"][0]) == 1.0


def test_merge_evidence_pads_with_zeros_when_no_evidence():
    price_df = pl.DataFrame([
        {"instrument_id": "XNYS:IBM", "event_time": datetime(2024, 1, 10, tzinfo=UTC)}
    ])
    merged = merge_evidence_features_pit(price_df, pl.DataFrame(), datetime(2024, 1, 10, tzinfo=UTC))
    for col in FEATURE_NAMES:
        assert col in merged.columns
        assert float(merged[col][0]) == 0.0
