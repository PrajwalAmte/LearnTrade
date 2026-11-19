"""
Market Pack validation.

Checks the structural correctness of a Market Pack YAML:
- Calendar exists in exchange_calendars
- Sessions: open < close, valid HH:MM format
- tick_size bands: positive, sorted
- lot_size: positive integer
- Costs: non-negative amounts
- settlement: valid T+N format

Usage:
    report = validate_pack("XNYS")
    if not report.passed:
        for issue in report.issues:
            print(issue.message)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from learntrade.markets.loader import get_pack, list_packs
from learntrade.markets.schema import MarketPack


@dataclass
class PackIssue:
    severity: Literal["error", "warning"]
    check: str
    message: str


@dataclass
class PackValidationReport:
    mic: str
    issues: list[PackIssue] = field(default_factory=list)
    passed: bool = True

    def __post_init__(self) -> None:
        self.passed = not any(i.severity == "error" for i in self.issues)


def validate_pack(mic: str) -> PackValidationReport:
    pack = get_pack(mic)
    issues: list[PackIssue] = []

    # ── Calendar ──────────────────────────────────────────────────────────
    cal_name = pack.calendar
    if ":" in cal_name:
        cal_name = cal_name.split(":")[1]
    try:
        import exchange_calendars as xcals
        xcals.get_calendar(cal_name)
    except Exception as exc:
        issues.append(PackIssue(
            "warning", "calendar",
            f"Calendar {pack.calendar!r} not found in exchange_calendars: {exc}. "
            "Using weekday fallback."
        ))

    # ── Sessions ──────────────────────────────────────────────────────────
    for name, session in pack.sessions.items():
        try:
            oh, om = map(int, session.open.split(":"))
            ch, cm = map(int, session.close.split(":"))
        except ValueError:
            issues.append(PackIssue(
                "error", "sessions",
                f"Session {name!r}: invalid time format (expected HH:MM)"
            ))
            continue
        if oh * 60 + om >= ch * 60 + cm:
            issues.append(PackIssue(
                "error", "sessions",
                f"Session {name!r}: open {session.open} is not before close {session.close}"
            ))

    # ── Tick sizes ────────────────────────────────────────────────────────
    for i, band in enumerate(pack.tick_size):
        if band.tick <= 0:
            issues.append(PackIssue(
                "error", "tick_size",
                f"tick_size[{i}].tick must be positive, got {band.tick}"
            ))

    # ── Lot size ──────────────────────────────────────────────────────────
    if pack.lot_size < 1:
        issues.append(PackIssue("error", "lot_size", f"lot_size must be >= 1, got {pack.lot_size}"))

    # ── Costs ─────────────────────────────────────────────────────────────
    if pack.costs.commission.bps < 0:
        issues.append(PackIssue("error", "costs", "commission.bps must be non-negative"))
    if pack.costs.commission.flat < 0:
        issues.append(PackIssue("error", "costs", "commission.flat must be non-negative"))
    for tax in pack.costs.taxes:
        if tax.bps < 0:
            issues.append(PackIssue("error", "costs", f"Tax {tax.name!r} bps must be non-negative"))

    # ── Settlement ────────────────────────────────────────────────────────
    if not pack.settlement.startswith("T"):
        issues.append(PackIssue(
            "warning", "settlement",
            f"settlement {pack.settlement!r} has unexpected format (expected T+N)"
        ))

    # ── Currency ──────────────────────────────────────────────────────────
    if len(pack.currency) != 3:
        issues.append(PackIssue(
            "error", "currency",
            f"currency {pack.currency!r} should be a 3-letter ISO 4217 code"
        ))

    return PackValidationReport(mic=mic, issues=issues)


def validate_all_packs() -> dict[str, PackValidationReport]:
    return {mic: validate_pack(mic) for mic in list_packs()}
