"""
Feature bridge — converts extracted LLM evidence into typed quant features.

Each ExtractedEvidence object contributes one row of features per instrument
per document date. The features respect PIT: ``available_at`` from the evidence
is used as the feature's validity date, never the event_time.

Features produced:
- guidance_change_score   : +1 raised / 0 maintained / -1 lowered
- surprise_score          : +1 positive / 0 in-line / -1 negative
- management_tone_score   : +1 positive / 0 neutral / -1 cautious / -2 negative
- n_risk_flags            : count of risk flags flagged
- evidence_confidence     : model's self-reported confidence [0, 1]
- n_key_figures_verified  : count of verified numeric figures
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from leantrade.llm.extraction.schemas import ExtractedEvidence

_GUIDANCE_MAP = {"raised": 1.0, "maintained": 0.0, "lowered": -1.0}
_SURPRISE_MAP = {"positive": 1.0, "in-line": 0.0, "negative": -1.0}
_TONE_MAP = {"positive": 1.0, "cautious": -0.5, "negative": -1.0}

FEATURE_NAMES = [
    "guidance_change_score",
    "surprise_score",
    "management_tone_score",
    "n_risk_flags",
    "evidence_confidence",
    "n_key_figures_verified",
]


def evidence_to_features(evidence: ExtractedEvidence) -> dict[str, float | int]:
    """Convert one ExtractedEvidence object to a flat feature dict."""
    return {
        "guidance_change_score": _GUIDANCE_MAP.get(evidence.guidance_change or "", 0.0),
        "surprise_score": _SURPRISE_MAP.get(evidence.surprise_direction or "", 0.0),
        "management_tone_score": _TONE_MAP.get(evidence.management_tone or "", 0.0),
        "n_risk_flags": len(evidence.risk_flags),
        "evidence_confidence": evidence.confidence,
        "n_key_figures_verified": sum(
            1 for f in evidence.key_figures if f.quote_span is not None
        ),
    }


def evidence_list_to_dataframe(
    evidence_list: list[ExtractedEvidence],
) -> pl.DataFrame:
    """
    Convert a list of extracted evidence objects to a Polars feature DataFrame.

    Returns columns: instrument_id, available_at, + FEATURE_NAMES.
    One row per (instrument, document). Callers should aggregate to one row
    per (instrument, date) by taking the latest available evidence.
    """
    rows = []
    for ev in evidence_list:
        feats = evidence_to_features(ev)
        for iid in ev.instrument_ids:
            rows.append({
                "instrument_id": iid,
                "available_at": ev.available_at,
                **feats,
            })
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def merge_evidence_features_pit(
    price_features: pl.DataFrame,
    evidence_features: pl.DataFrame,
    as_of: datetime,
) -> pl.DataFrame:
    """
    Left-join price features with the most recent evidence features visible at as_of.

    For each (instrument_id, event_time) in price_features, attach the latest
    evidence features whose available_at <= as_of.
    This enforces PIT: we never use evidence published after the query date.
    """
    if evidence_features.is_empty():
        # Pad with zero-filled evidence columns
        for col in FEATURE_NAMES:
            price_features = price_features.with_columns(pl.lit(0.0).alias(col))
        return price_features

    # Filter evidence to visible-at-as_of only
    ev_pit = evidence_features.filter(pl.col("available_at") <= as_of)
    if ev_pit.is_empty():
        for col in FEATURE_NAMES:
            price_features = price_features.with_columns(pl.lit(0.0).alias(col))
        return price_features

    # Take the latest evidence row per instrument
    ev_latest = (
        ev_pit.sort("available_at", descending=True)
        .group_by("instrument_id")
        .first()
        .select(["instrument_id"] + FEATURE_NAMES)
    )

    return price_features.join(ev_latest, on="instrument_id", how="left").with_columns(
        [pl.col(c).fill_null(0.0) for c in FEATURE_NAMES]
    )
