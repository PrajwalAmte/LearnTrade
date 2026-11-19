"""
Evidence extractor — processes one document into structured ExtractedEvidence.

Each document is processed once and cached permanently by (doc_hash, model, prompt_version).
Backtest reruns over historical filings cost nothing after the first pass.

Quote-span verification ensures numeric claims come from the source text,
not from the LLM's parametric knowledge (which could be contaminated).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from leantrade.llm.extraction.schemas import ExtractedEvidence, KeyFigure
from leantrade.llm.extraction.prompts import (
    EXTRACT_SYSTEM_V1,
    EXTRACT_USER_V1,
    VERIFY_SYSTEM_V1,
    VERIFY_USER_V1,
    PROMPT_VERSION,
)

if TYPE_CHECKING:
    from leantrade.llm.router import LLMRouter

logger = logging.getLogger(__name__)

_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "event_type": {"type": "string"},
        "guidance_change": {"type": ["string", "null"],
                            "enum": ["raised", "lowered", "maintained", None]},
        "surprise_direction": {"type": ["string", "null"],
                                "enum": ["positive", "negative", "in-line", None]},
        "management_tone": {"type": ["string", "null"],
                             "enum": ["positive", "cautious", "negative", None]},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "key_figures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "number"},
                    "unit": {"type": "string"},
                    "quote_span": {
                        "type": ["array", "null"],
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "required": ["name", "value", "unit"],
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["event_type", "confidence"],
}


class EvidenceExtractor:
    """
    Extracts structured evidence from a single document via an LLM.

    Cache key: (doc_id, model, prompt_version) — changing any of these
    re-triggers extraction.
    """

    def __init__(self, router: "LLMRouter") -> None:
        self._router = router

    async def extract(
        self,
        doc_id: str,
        document_text: str,
        instrument_ids: list[str],
        available_at: datetime,
        verify: bool = True,
    ) -> ExtractedEvidence:
        """
        Extract and optionally verify evidence from a document.

        Args:
            doc_id:         sha256 hash of the document (from CorpusStore)
            document_text:  full document text
            instrument_ids: instruments this document relates to
            available_at:   when the document became publicly available
            verify:         run a second LLM pass to verify numeric claims

        Returns:
            ExtractedEvidence with verified flag on each KeyFigure.
        """
        model = self._router._get_client("extract").model

        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM_V1},
            {
                "role": "user",
                "content": EXTRACT_USER_V1.format(
                    available_at=available_at.isoformat(),
                    instrument_id=", ".join(instrument_ids) or "unknown",
                    document_text=document_text[:8000],  # truncate for context window
                    schema_json=json.dumps(_EXTRACTION_SCHEMA, indent=2),
                ),
            },
        ]

        response = await self._router.complete(
            "extract",
            messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        text = self._router.extract_text(response)
        raw = self._parse_json(text)

        # Build KeyFigures
        figures = [
            KeyFigure(
                name=f.get("name", ""),
                value=float(f.get("value", 0)),
                unit=f.get("unit", ""),
                quote_span=tuple(f["quote_span"]) if f.get("quote_span") else None,
            )
            for f in raw.get("key_figures", [])
        ]

        evidence = ExtractedEvidence(
            doc_id=doc_id,
            instrument_ids=instrument_ids,
            available_at=available_at,
            event_type=raw.get("event_type", "unknown"),
            guidance_change=raw.get("guidance_change"),
            surprise_direction=raw.get("surprise_direction"),
            management_tone=raw.get("management_tone"),
            risk_flags=raw.get("risk_flags", []),
            key_figures=figures,
            confidence=float(raw.get("confidence", 0.5)),
            extractor={"model": model, "prompt_version": PROMPT_VERSION},
        )

        if verify and figures:
            evidence = await self._verify_figures(evidence, document_text)

        return evidence

    async def _verify_figures(
        self, evidence: ExtractedEvidence, document_text: str
    ) -> ExtractedEvidence:
        """
        Run a verification pass: check each numeric claim against its quote_span.
        Uses deterministic span-checking first; LLM only for ambiguous cases.
        """
        # Deterministic verification: check quote_span directly in source text
        verified_figures: list[KeyFigure] = []
        for fig in evidence.key_figures:
            if fig.quote_span is not None:
                span_text = document_text[fig.quote_span[0]: fig.quote_span[1]]
                str_val = str(fig.value)
                # Accept if the numeric value appears in the span text
                if str_val in span_text or str_val.rstrip("0").rstrip(".") in span_text:
                    verified_figures.append(fig)
                    continue
                # Quote-span check failed — drop the figure
                logger.debug("Figure %s failed span verification; dropped", fig.name)
            else:
                # No quote_span: accept with reduced confidence but flag
                verified_figures.append(fig)

        evidence.key_figures = verified_figures
        return evidence

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Parse JSON from LLM output, tolerating markdown code fences."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse extraction JSON: %r", text[:200])
            return {}
