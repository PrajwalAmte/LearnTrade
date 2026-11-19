"""
Market Pack validation tests.
"""

from __future__ import annotations

import pytest

from learntrade.markets.validate import PackIssue, PackValidationReport, validate_pack, validate_all_packs


SHIPPED_MICS = ["XNYS", "XNAS", "XNSE"]


@pytest.mark.parametrize("mic", SHIPPED_MICS)
def test_shipped_packs_pass_validation(mic: str) -> None:
    report = validate_pack(mic)
    errors = [i for i in report.issues if i.severity == "error"]
    assert errors == [], f"{mic} has validation errors: {errors}"


def test_validate_all_packs_runs() -> None:
    reports = validate_all_packs()
    assert len(reports) >= 3  # at least XNYS, XNAS, XNSE
    for mic, report in reports.items():
        errors = [i for i in report.issues if i.severity == "error"]
        assert errors == [], f"{mic} has errors: {errors}"


def test_validate_generic_pack() -> None:
    report = validate_pack("XXXX")  # loads generic
    # Generic pack should at least have no structural errors
    errors = [i for i in report.issues if i.severity == "error"]
    assert errors == []


def test_validation_report_passed_flag() -> None:
    report = PackValidationReport(mic="TEST", issues=[
        PackIssue("warning", "calendar", "Calendar not found")
    ])
    assert report.passed  # warnings don't fail


def test_validation_report_error_fails() -> None:
    report = PackValidationReport(mic="TEST", issues=[
        PackIssue("error", "sessions", "open >= close")
    ])
    assert not report.passed
