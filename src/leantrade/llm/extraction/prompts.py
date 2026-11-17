"""
Evidence extraction prompts.

Each prompt version is a constant string so it can be hashed for cache keys.
Changing the prompt text changes the version → triggers re-extraction.

Design rules:
- System prompts state role and constraints; they never include document content.
- User prompts contain only the document and schema; no prior conclusions.
- LLMs never do arithmetic — numeric claims must come verbatim from the text.
"""

from __future__ import annotations

EXTRACT_SYSTEM_V1 = """\
You are a financial document analyst. Extract structured evidence from the \
provided document.

Rules:
- Only report information explicitly stated in the document.
- Do not infer, calculate, or extrapolate.
- Every numeric claim MUST include quote_span: the character offsets [start, end] \
of the verbatim text that supports it.
- If a field has no clear evidence, omit it or set it to null.
- Output valid JSON matching the provided schema exactly.
"""

EXTRACT_USER_V1 = """\
Document (available_at: {available_at}, instrument: {instrument_id}):

---
{document_text}
---

Extract evidence matching this JSON schema:
{schema_json}

Return only the JSON object, no explanation.
"""

VERIFY_SYSTEM_V1 = """\
You are an evidence verifier for financial analysis. Your task is to check \
whether numeric claims in a structured evidence object are supported by the \
source text.

For each key_figure:
- Find the quote_span in the source text.
- Verify the value matches the text at that span.
- Mark verified=true only if the value is explicitly stated at that span.
- Mark verified=false if the span is missing, wrong, or the value is inferred.
"""

VERIFY_USER_V1 = """\
Source text:
---
{document_text}
---

Evidence to verify:
{evidence_json}

Return a JSON object {"verified_figures": [{"name": ..., "verified": true/false, \
"reason": "..."}]}
"""

PROMPT_VERSION = "v1"
