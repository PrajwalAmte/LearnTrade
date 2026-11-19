"""
LLM client protocol.

All provider-specific clients implement ``LLMClient``.
Registered via the ``learntrade.llm_providers`` entry-point group.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """
    Thin wrapper around one LLM backend.

    ``complete`` is the only required method.  All retry, caching and
    budget enforcement is handled by the router (``llm.router``), not here.
    """

    provider: str
    model: str

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Return the raw completion response dict from the provider."""
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings; raise ``NotImplementedError`` if unsupported."""
        ...
