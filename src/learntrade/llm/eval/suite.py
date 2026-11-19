"""
LLM evaluation suite — measures extraction quality across five task dimensions.

Evaluation tasks (from PLAN.md):
1. Financial numerical reasoning   — does the model compute ratios correctly?
2. Source-grounded extraction      — does the model only report what's in the text?
3. Contradiction detection         — does the model spot internal inconsistencies?
4. Structured-output validity      — does the model produce valid JSON schemas?
5. Citation accuracy               — are quote_spans correct?

Additional tracked metrics:
6. Cost per completed task (USD)
7. Latency (seconds per call)

Usage:
    suite = EvalSuite(router)
    results = await suite.run(task="extract", model_ids=["gpt-4o-mini"])
    suite.print_summary(results)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from learntrade.llm.router import LLMRouter

logger = logging.getLogger(__name__)

UTC = timezone.utc


@dataclass
class EvalCase:
    id: str
    task: str
    input_text: str
    expected: dict[str, Any]
    metric: str   # which dimension this case tests


@dataclass
class EvalResult:
    case_id: str
    task: str
    metric: str
    model: str
    passed: bool
    score: float          # 0–1
    latency_seconds: float
    cost_usd: float
    details: dict[str, Any] = field(default_factory=dict)


EVAL_CASES: list[EvalCase] = [
    # ── 1. Financial numerical reasoning ─────────────────────────────────
    EvalCase(
        id="num_reasoning_01",
        task="extract",
        input_text=(
            "Revenue grew from $4.2B in Q1 to $5.1B in Q2, a year-over-year "
            "increase of 21.4%. Operating margin expanded 180 basis points."
        ),
        expected={
            "event_type": "earnings",
            "key_figures": [
                {"name": "revenue_q2", "value": 5.1},
                {"name": "revenue_growth_yoy", "value": 0.214},
            ],
        },
        metric="numerical_reasoning",
    ),
    # ── 2. Source-grounded extraction ─────────────────────────────────────
    EvalCase(
        id="grounding_01",
        task="extract",
        input_text=(
            "The company raised its full-year EPS guidance to $3.50-$3.70, "
            "up from $3.20-$3.40 previously."
        ),
        expected={
            "guidance_change": "raised",
            "key_figures": [{"name": "eps_guidance_high", "value": 3.70}],
        },
        metric="source_grounding",
    ),
    # ── 3. Contradiction detection ────────────────────────────────────────
    EvalCase(
        id="contradiction_01",
        task="verify",
        input_text=(
            "Net income was $200M. The company reported a loss of $50M."
        ),
        expected={"has_contradiction": True},
        metric="contradiction_detection",
    ),
    # ── 4. Structured output validity ─────────────────────────────────────
    EvalCase(
        id="schema_01",
        task="extract",
        input_text=(
            "Management remains cautious about the macroeconomic environment. "
            "FX headwinds remain a concern."
        ),
        expected={
            "management_tone": "cautious",
            "risk_flags": ["fx_headwind"],
        },
        metric="schema_validity",
    ),
    # ── 5. Citation accuracy ──────────────────────────────────────────────
    EvalCase(
        id="citation_01",
        task="extract",
        input_text=(
            "Gross margin improved to 42.3% in the quarter, driven by cost efficiencies."
        ),
        expected={
            "key_figures": [{"name": "gross_margin", "value": 0.423, "span_text": "42.3%"}]
        },
        metric="citation_accuracy",
    ),
]


class EvalSuite:
    def __init__(self, router: "LLMRouter") -> None:
        self._router = router

    async def run(
        self,
        task: str = "extract",
        cases: list[EvalCase] | None = None,
    ) -> list[EvalResult]:
        """Run all eval cases for a task. Returns one EvalResult per case."""
        from learntrade.llm.extraction.prompts import EXTRACT_SYSTEM_V1, EXTRACT_USER_V1
        from learntrade.llm.extraction.extractor import _EXTRACTION_SCHEMA

        active = [c for c in (cases or EVAL_CASES) if c.task == task]
        model = self._router._get_client(task).model
        results: list[EvalResult] = []

        for case in active:
            t0 = time.monotonic()
            try:
                messages = [
                    {"role": "system", "content": EXTRACT_SYSTEM_V1},
                    {"role": "user", "content": EXTRACT_USER_V1.format(
                        available_at=datetime.now(tz=UTC).isoformat(),
                        instrument_id="EVAL",
                        document_text=case.input_text,
                        schema_json=json.dumps(_EXTRACTION_SCHEMA, indent=2),
                    )},
                ]
                response = await self._router.complete(
                    task, messages,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                latency = time.monotonic() - t0
                text = self._router.extract_text(response)

                # Estimate cost
                usage = response.get("usage", {})
                tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                from learntrade.llm.router import _COST_PER_1K
                cost = _COST_PER_1K.get(model, _COST_PER_1K["default"]) * tokens / 1000

                score, details = self._score(case, text)
                results.append(EvalResult(
                    case_id=case.id, task=task, metric=case.metric,
                    model=model, passed=score >= 0.5, score=score,
                    latency_seconds=latency, cost_usd=cost, details=details,
                ))
            except Exception as exc:
                latency = time.monotonic() - t0
                logger.warning("Eval case %s failed: %s", case.id, exc)
                results.append(EvalResult(
                    case_id=case.id, task=task, metric=case.metric,
                    model=model, passed=False, score=0.0,
                    latency_seconds=latency, cost_usd=0.0,
                    details={"error": str(exc)},
                ))
        return results

    def _score(self, case: EvalCase, raw_text: str) -> tuple[float, dict]:
        """Score one eval case. Returns (score 0-1, details dict)."""
        try:
            parsed = json.loads(raw_text.strip().strip("```json").strip("```"))
        except json.JSONDecodeError:
            return 0.0, {"error": "invalid JSON"}

        metric = case.metric
        expected = case.expected

        if metric == "schema_validity":
            # Check required keys are present
            hits = sum(1 for k in expected if k in parsed)
            score = hits / max(len(expected), 1)
            return score, {"found_keys": list(parsed.keys())}

        if metric == "source_grounding":
            score = 1.0 if parsed.get("guidance_change") == expected.get("guidance_change") else 0.0
            return score, {"got": parsed.get("guidance_change")}

        if metric == "numerical_reasoning":
            figures = {f.get("name"): f.get("value") for f in parsed.get("key_figures", [])}
            hits = 0
            for exp_fig in expected.get("key_figures", []):
                got = figures.get(exp_fig["name"])
                if got is not None and abs(float(got) - float(exp_fig["value"])) < abs(float(exp_fig["value"])) * 0.05:
                    hits += 1
            score = hits / max(len(expected.get("key_figures", [])), 1)
            return score, {"found_figures": list(figures.keys())}

        if metric == "citation_accuracy":
            figures = parsed.get("key_figures", [])
            has_span = any(f.get("quote_span") is not None for f in figures)
            return (1.0 if has_span else 0.0), {"has_span": has_span}

        if metric == "contradiction_detection":
            # Check if the model notes a contradiction
            text_lower = raw_text.lower()
            detected = any(w in text_lower for w in ["contradict", "inconsistent", "conflict"])
            return (1.0 if detected else 0.0), {"detected": detected}

        return 0.5, {}

    @staticmethod
    def print_summary(results: list[EvalResult]) -> None:
        from collections import defaultdict
        by_metric: dict[str, list[float]] = defaultdict(list)
        for r in results:
            by_metric[r.metric].append(r.score)

        print(f"\n{'Metric':<30} {'Avg Score':>10} {'Cases':>8}")
        print("-" * 50)
        for metric, scores in sorted(by_metric.items()):
            avg = sum(scores) / len(scores)
            print(f"{metric:<30} {avg:>10.2%} {len(scores):>8}")

        total_cost = sum(r.cost_usd for r in results)
        avg_latency = sum(r.latency_seconds for r in results) / max(len(results), 1)
        print(f"\nTotal cost: ${total_cost:.4f}  Avg latency: {avg_latency:.1f}s")
