"""
Evidence extraction schemas.

Each document (filing, transcript, news item) is processed once and
stored as structured evidence keyed by (doc_hash, model_id, prompt_version).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class KeyFigure(BaseModel):
    name: str
    value: float
    unit: str
    quote_span: tuple[int, int] | None = None   # character offsets in source text


class ExtractedEvidence(BaseModel):
    """
    Structured output of one evidence extraction pass over a document.

    Numbers must cite a ``quote_span`` so the verifier can check them.
    ``confidence`` is the model's self-reported confidence [0, 1].
    """

    doc_id: str                             # sha256:<hex> of the source document
    instrument_ids: list[str]
    available_at: datetime                  # when the document became publicly available
    event_type: str                         # "earnings", "guidance", "news", etc.
    guidance_change: str | None = None      # "raised", "lowered", "maintained", None
    surprise_direction: str | None = None   # "positive", "negative", "in-line", None
    key_figures: list[KeyFigure] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    management_tone: str | None = None      # "positive", "cautious", "negative"
    confidence: float = 0.5
    extractor: dict[str, Any] = Field(default_factory=dict)  # model, prompt_version
