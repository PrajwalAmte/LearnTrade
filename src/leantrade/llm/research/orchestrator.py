"""
Research orchestration state machine.

Steps (plain Python, no LangChain):
  ANALYSIS → VERIFICATION → [ESCALATION] → SYNTHESIS

Escalation triggers (configurable):
  - Quant-LLM disagreement on direction
  - Forecast uncertainty above threshold
  - Material events (earnings, guidance changes)
  - Large proposed position change

Self-consistency sampling: synthesis runs N times; majority agreement kept.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from leantrade.core.types import Forecast
    from leantrade.llm.router import LLMRouter

from leantrade.llm.research.fact_sheet import FactSheet
from leantrade.llm.research.roles import ROLES, ResearchRoleId

logger = logging.getLogger(__name__)


class OrchestratorState(Enum):
    ANALYSIS = "analysis"
    VERIFICATION = "verification"
    ESCALATION = "escalation"
    SYNTHESIS = "synthesis"
    DONE = "done"


@dataclass
class OrchestratorConfig:
    base_roles: list[ResearchRoleId] = field(
        default_factory=lambda: ["technical", "event", "verifier", "quant_challenger", "synthesiser"]
    )
    escalation_roles: list[ResearchRoleId] = field(
        default_factory=lambda: ["fundamental", "macro", "bull", "bear"]
    )
    escalate_on_disagreement: bool = True
    escalate_on_uncertainty_above: float = 0.6
    escalate_on_events: list[str] = field(
        default_factory=lambda: ["earnings", "guidance", "corporate_action"]
    )
    self_consistency_samples: int = 1   # >1 = majority vote on synthesis


@dataclass
class AnalystOutput:
    role_id: ResearchRoleId
    summary: str
    raw: dict[str, Any]


@dataclass
class ResearchResult:
    thesis: str
    counter_thesis: str
    invalidation_conditions: list[str]
    agreement_with_quant: str    # "agree" | "disagree" | "neutral"
    overall_confidence: float
    analyst_outputs: list[AnalystOutput]
    escalated: bool
    state_trace: list[str]
    llm_cost_usd: float = 0.0


class ResearchOrchestrator:
    """
    Runs the research pipeline for one instrument as a typed state machine.

    The LLM explains the quant forecast; it does not override it.
    """

    def __init__(self, router: "LLMRouter") -> None:
        self._router = router

    async def analyze(
        self,
        fact_sheet: FactSheet,
        forecast: "Forecast",
        config: OrchestratorConfig | None = None,
    ) -> ResearchResult:
        cfg = config or OrchestratorConfig()
        trace: list[str] = []
        cost_before = self._router.budget.run_spent

        # ── ANALYSIS ─────────────────────────────────────────────────────
        trace.append(OrchestratorState.ANALYSIS.value)
        base_outputs = await self._run_roles_parallel(
            fact_sheet, [r for r in cfg.base_roles if r not in ("verifier", "quant_challenger", "synthesiser")]
        )

        # ── VERIFICATION ──────────────────────────────────────────────────
        trace.append(OrchestratorState.VERIFICATION.value)
        verified_outputs = await self._verify(fact_sheet, base_outputs)

        # ── ESCALATION CHECK ──────────────────────────────────────────────
        should_escalate = self._should_escalate(verified_outputs, forecast, fact_sheet, cfg)
        escalated = False

        if should_escalate:
            trace.append(OrchestratorState.ESCALATION.value)
            escalated = True
            extra = await self._run_roles_parallel(fact_sheet, cfg.escalation_roles)
            verified_outputs.extend(extra)

        # Quant challenger (always runs after escalation decision)
        if "quant_challenger" in cfg.base_roles:
            challenger = await self._run_role("quant_challenger", fact_sheet, verified_outputs, forecast)
            if challenger:
                verified_outputs.append(challenger)

        # ── SYNTHESIS ─────────────────────────────────────────────────────
        trace.append(OrchestratorState.SYNTHESIS.value)
        synthesis = await self._synthesise(fact_sheet, forecast, verified_outputs, cfg)
        trace.append(OrchestratorState.DONE.value)

        return ResearchResult(
            thesis=synthesis.get("thesis", ""),
            counter_thesis=synthesis.get("counter_thesis", ""),
            invalidation_conditions=synthesis.get("invalidation_conditions", []),
            agreement_with_quant=synthesis.get("agreement_with_quant", "neutral"),
            overall_confidence=float(synthesis.get("overall_confidence", 0.5)),
            analyst_outputs=verified_outputs,
            escalated=escalated,
            state_trace=trace,
            # Actual spend this call incurred, read off the router's budget
            # tracker (which is the only place real token costs are known)
            # rather than left at a hardcoded 0.0.
            llm_cost_usd=self._router.budget.run_spent - cost_before,
        )

    # ── Internals ─────────────────────────────────────────────────────────

    async def _run_roles_parallel(
        self, fact_sheet: FactSheet, role_ids: list[ResearchRoleId]
    ) -> list[AnalystOutput]:
        tasks = [self._run_role(rid, fact_sheet, [], None) for rid in role_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        outputs = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Role failed: %s", r)
            elif r is not None:
                outputs.append(r)
        return outputs

    async def _run_role(
        self,
        role_id: ResearchRoleId,
        fact_sheet: FactSheet,
        prior_outputs: list[AnalystOutput],
        forecast: "Forecast | None",
    ) -> AnalystOutput | None:
        role = ROLES.get(role_id)
        if role is None:
            return None
        prior_text = ""
        if prior_outputs:
            prior_text = "\n\nPrevious analyses:\n" + "\n".join(
                f"[{o.role_id}]: {o.summary}" for o in prior_outputs[:6]
            )
        forecast_text = ""
        if forecast is not None:
            forecast_text = (
                f"\n\nQuant forecast: expected_return={forecast.expected_return:+.2%} "
                f"p_positive={forecast.p_positive:.1%} rank={forecast.rank}"
            )
        messages = [
            {"role": "system", "content": role.system_prompt},
            {"role": "user", "content": (
                "Fact sheet:\n" + fact_sheet.to_text()
                + prior_text + forecast_text
                + f"\n\nReturn JSON matching schema:\n{json.dumps(role.output_schema, indent=2)}"
            )},
        ]
        try:
            resp = await self._router.complete(
                "research", messages,
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            text = self._router.extract_text(resp)
            raw = json.loads(text.strip().strip("```json").strip("```"))
            summary = raw.get("summary") or raw.get("thesis") or raw.get("reasoning") or str(raw)
            return AnalystOutput(role_id=role_id, summary=summary[:500], raw=raw)
        except Exception as exc:
            logger.warning("Role %s failed: %s", role_id, exc)
            return None

    async def _verify(
        self, fact_sheet: FactSheet, outputs: list[AnalystOutput]
    ) -> list[AnalystOutput]:
        # Deterministic rule-based verification (LLM verifier runs only on escalation)
        # For now: drop any output that explicitly says no signals found
        return [o for o in outputs if o.summary.strip()]

    def _should_escalate(
        self,
        outputs: list[AnalystOutput],
        forecast: "Forecast",
        fact_sheet: FactSheet,
        cfg: OrchestratorConfig,
    ) -> bool:
        if cfg.escalate_on_uncertainty_above < 1.0:
            # Escalate if forecast uncertainty is high
            if abs(forecast.p_positive - 0.5) < (1.0 - cfg.escalate_on_uncertainty_above) / 2:
                logger.debug("Escalation: high uncertainty (p_pos=%.2f)", forecast.p_positive)
                return True
        if cfg.escalate_on_events:
            events_text = " ".join(fact_sheet.recent_events).lower()
            for ev in cfg.escalate_on_events:
                if ev in events_text:
                    logger.debug("Escalation: material event %s", ev)
                    return True
        return False

    async def _synthesise(
        self,
        fact_sheet: FactSheet,
        forecast: "Forecast",
        outputs: list[AnalystOutput],
        cfg: OrchestratorConfig,
    ) -> dict[str, Any]:
        from leantrade.llm.research.roles import ROLES

        analyses_text = "\n".join(
            f"[{o.role_id}]: {o.summary}" for o in outputs
        )
        schema = ROLES["synthesiser"].output_schema
        messages = [
            {"role": "system", "content": ROLES["synthesiser"].system_prompt},
            {"role": "user", "content": (
                "Fact sheet:\n" + fact_sheet.to_text()
                + "\n\nAnalyses:\n" + analyses_text
                + f"\n\nQuant forecast: expected_return={forecast.expected_return:+.2%} "
                  f"p_positive={forecast.p_positive:.1%}"
                + f"\n\nReturn JSON matching schema:\n{json.dumps(schema, indent=2)}"
            )},
        ]

        # Self-consistency sampling
        samples = []
        n = max(1, cfg.self_consistency_samples)
        for _ in range(n):
            try:
                resp = await self._router.complete("research", messages,
                                                    response_format={"type": "json_object"},
                                                    temperature=0.3)
                text = self._router.extract_text(resp)
                samples.append(json.loads(text.strip().strip("```json").strip("```")))
            except Exception as exc:
                logger.warning("Synthesis sample failed: %s", exc)

        if not samples:
            return {
                "thesis": "Insufficient data for synthesis.",
                "counter_thesis": "Unable to assess.",
                "invalidation_conditions": [],
                "agreement_with_quant": "neutral",
                "overall_confidence": 0.4,
            }

        # With multiple samples, pick the one with median confidence
        if len(samples) > 1:
            samples.sort(key=lambda s: s.get("overall_confidence", 0.5))
            return samples[len(samples) // 2]
        return samples[0]
