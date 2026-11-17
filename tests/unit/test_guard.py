"""
Contamination guard tests — cutoff registry, margin enforcement, masking mode.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from leantrade.llm.guard import (
    is_contaminated,
    list_cutoffs,
    mask_document,
    register_cutoff,
)

UTC = timezone.utc


def test_date_before_cutoff_is_contaminated():
    as_of = datetime(2022, 1, 1, tzinfo=UTC)
    assert is_contaminated("gpt-4o", as_of, safety_margin_days=30)


def test_date_well_after_cutoff_is_clean():
    as_of = datetime(2026, 1, 1, tzinfo=UTC)
    assert not is_contaminated("gpt-4o", as_of, safety_margin_days=30)


def test_unknown_model_is_contaminated():
    as_of = datetime(2030, 1, 1, tzinfo=UTC)
    assert is_contaminated("some-unknown-model-xyz", as_of)


def test_safety_margin_boundary():
    # cutoff + 30 days → day 30 is contaminated, day 31 is clean
    assert is_contaminated("gpt-4o", datetime(2023, 10, 30, tzinfo=UTC), safety_margin_days=30)
    assert not is_contaminated("gpt-4o", datetime(2023, 11, 2, tzinfo=UTC), safety_margin_days=30)


def test_register_custom_cutoff():
    register_cutoff("custom-eval-model", datetime(2021, 1, 1, tzinfo=UTC))
    assert not is_contaminated("custom-eval-model", datetime(2023, 1, 1, tzinfo=UTC))
    assert is_contaminated("custom-eval-model", datetime(2020, 6, 1, tzinfo=UTC))


def test_list_cutoffs():
    cutoffs = list_cutoffs()
    assert "gpt-4o" in cutoffs
    assert isinstance(cutoffs["gpt-4o"], str)


def test_mask_document_replaces_dates():
    text = "Revenue grew in Q2 2024 by 20%."
    masked = mask_document(text, replace_tickers=False)
    assert "2024" not in masked


def test_mask_document_scales_prices():
    text = "The stock was at $200.00."
    masked = mask_document(text, price_scale=0.5)
    assert "$100.00" in masked


def test_mask_document_no_crash_on_empty():
    assert mask_document("") == ""
