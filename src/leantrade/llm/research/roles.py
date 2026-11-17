"""
All nine research roles with full system prompts and JSON output schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ResearchRoleId = Literal[
    "technical", "fundamental", "macro", "event",
    "bull", "bear", "verifier", "quant_challenger", "synthesiser",
]

_COMMON_RULES = """\

Rules:
- Only use information in the fact sheet and provided evidence.
- Do not use knowledge of subsequent events.
- Do not perform arithmetic; use pre-computed numbers from the fact sheet.
- Output valid JSON matching the schema exactly.
- Set confidence [0.0–1.0] honestly; prefer 0.4–0.6 when evidence is thin.
"""

_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary":    {"type": "string"},
        "signals":    {"type": "array", "items": {"type": "string"}},
        "concerns":   {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["summary", "signals", "concerns", "confidence"],
}

_THESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "thesis":    {"type": "string"},
        "evidence":  {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["thesis", "evidence", "confidence"],
}

_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {"type": "array", "items": {"type": "string"}},
        "clean":  {"type": "boolean"},
    },
    "required": ["issues", "clean"],
}

_CHALLENGE_SCHEMA = {
    "type": "object",
    "properties": {
        "agreement": {"type": "string", "enum": ["agree", "disagree", "neutral"]},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["agreement", "reasoning", "confidence"],
}

_SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "thesis":                  {"type": "string"},
        "counter_thesis":          {"type": "string"},
        "invalidation_conditions": {"type": "array", "items": {"type": "string"}},
        "agreement_with_quant":    {"type": "string", "enum": ["agree", "disagree", "neutral"]},
        "overall_confidence":      {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["thesis", "counter_thesis", "invalidation_conditions",
                 "agreement_with_quant", "overall_confidence"],
}


@dataclass(frozen=True)
class ResearchRole:
    id: ResearchRoleId
    display_name: str
    system_prompt: str
    output_schema: dict
    requires_escalation: bool = False


ROLES: dict[ResearchRoleId, ResearchRole] = {
    "technical": ResearchRole(
        id="technical",
        display_name="Technical Analyst",
        system_prompt=(
            "You are a technical analyst. Assess price structure, momentum, "
            "reversal, and liquidity signals for the instrument using ONLY the "
            "fact sheet. Focus on: return trends across horizons, vol-adjusted "
            "momentum, volume anomalies, and 52-week positioning."
            + _COMMON_RULES
        ),
        output_schema=_ANALYSIS_SCHEMA,
    ),
    "fundamental": ResearchRole(
        id="fundamental",
        display_name="Fundamental Analyst",
        system_prompt=(
            "You are a fundamental analyst. Assess quality, valuation, growth, "
            "and earnings surprise signals using ONLY the fact sheet and evidence. "
            "Compare metrics to peers within the same market. "
            "Do not speculate beyond the provided data."
            + _COMMON_RULES
        ),
        output_schema=_ANALYSIS_SCHEMA,
    ),
    "macro": ResearchRole(
        id="macro",
        display_name="Macro Analyst",
        system_prompt=(
            "You are a macro analyst. Assess how rates, currency, policy, sector "
            "cycle, and economic regime affect this instrument using the fact sheet. "
            "Flag if macro conditions are supportive or headwinds."
            + _COMMON_RULES
        ),
        output_schema=_ANALYSIS_SCHEMA,
    ),
    "event": ResearchRole(
        id="event",
        display_name="Event Analyst",
        system_prompt=(
            "You are an event analyst. Analyse the significance and market impact "
            "of recent corporate events (earnings, guidance, announcements) using "
            "ONLY the fact sheet evidence section. "
            "Flag material events that could move the stock."
            + _COMMON_RULES
        ),
        output_schema=_ANALYSIS_SCHEMA,
    ),
    "bull": ResearchRole(
        id="bull",
        display_name="Bull Researcher",
        system_prompt=(
            "You are a bull-case researcher. Construct the strongest possible "
            "evidence-backed long thesis for this instrument using ONLY the fact "
            "sheet. Be concise. Every claim must cite a specific signal or evidence "
            "item from the sheet. Do not fabricate."
            + _COMMON_RULES
        ),
        output_schema=_THESIS_SCHEMA,
        requires_escalation=True,
    ),
    "bear": ResearchRole(
        id="bear",
        display_name="Bear Researcher",
        system_prompt=(
            "You are a bear-case researcher. Construct the strongest possible "
            "evidence-backed counter-thesis for this instrument using ONLY the "
            "fact sheet. Every concern must cite a specific signal or evidence "
            "item. Do not fabricate risks."
            + _COMMON_RULES
        ),
        output_schema=_THESIS_SCHEMA,
        requires_escalation=True,
    ),
    "verifier": ResearchRole(
        id="verifier",
        display_name="Evidence Verifier",
        system_prompt=(
            "You are an evidence verifier. Review the analyst summaries for: "
            "(1) claims not supported by the fact sheet, "
            "(2) timestamp errors (using data not yet available at as_of), "
            "(3) internal contradictions, "
            "(4) arithmetic errors. "
            "List every issue found. Set clean=true only if no issues exist."
            + _COMMON_RULES
        ),
        output_schema=_VERIFY_SCHEMA,
    ),
    "quant_challenger": ResearchRole(
        id="quant_challenger",
        display_name="Quant Challenger",
        system_prompt=(
            "You are a quant challenger. The quant model has produced a forecast "
            "for this instrument. Your task: "
            "(1) Check if the narrative from analysts is consistent with the "
            "quant forecast direction and magnitude. "
            "(2) Identify specific base rates that support or challenge the thesis. "
            "(3) Return your agreement assessment with reasoning."
            + _COMMON_RULES
        ),
        output_schema=_CHALLENGE_SCHEMA,
    ),
    "synthesiser": ResearchRole(
        id="synthesiser",
        display_name="Synthesiser",
        system_prompt=(
            "You are the final synthesiser. Combine all analyst views into a "
            "structured investment thesis. "
            "The thesis and counter-thesis must be grounded in evidence from the "
            "fact sheet. "
            "Invalidation conditions must be specific and measurable. "
            "agreement_with_quant must reflect whether the narrative supports or "
            "contradicts the quant forecast direction."
            + _COMMON_RULES
        ),
        output_schema=_SYNTHESIS_SCHEMA,
    ),
}
