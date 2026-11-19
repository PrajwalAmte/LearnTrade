"""
Anthropic native client (async, prompt caching, batch API support).

Secrets: set LEARNTRADE_ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicClient:
    provider: str = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-5",
        timeout: float = 120.0,
        enable_prompt_caching: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("Anthropic client requires an API key")
        self.model = model
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        if enable_prompt_caching:
            self._headers["anthropic-beta"] = "prompt-caching-2024-07-31"
        self._timeout = timeout

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        # Separate system message from user/assistant turns
        system_content = ""
        conversation: list[dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                conversation.append({"role": msg["role"], "content": msg["content"]})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": conversation,
            "max_tokens": max_tokens or 4096,
            "temperature": temperature,
        }
        if system_content:
            # Mark system prompt for caching (saves cost on repeated calls)
            body["system"] = [{"type": "text", "text": system_content,
                                "cache_control": {"type": "ephemeral"}}]

        async with httpx.AsyncClient(headers=self._headers, timeout=self._timeout) as client:
            resp = await client.post(ANTHROPIC_API_URL, json=body)
            resp.raise_for_status()
            data = resp.json()

        usage = data.get("usage", {})
        logger.debug(
            "Anthropic complete: model=%s input=%s output=%s cache_read=%s",
            self.model,
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            usage.get("cache_read_input_tokens"),
        )
        return data

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Anthropic does not provide an embeddings endpoint")

    @staticmethod
    def extract_text(response: dict[str, Any]) -> str:
        try:
            return response["content"][0]["text"] or ""
        except (KeyError, IndexError):
            return ""

    @staticmethod
    def extract_usage(response: dict[str, Any]) -> dict[str, int]:
        u = response.get("usage", {})
        return {
            "prompt_tokens": u.get("input_tokens", 0),
            "completion_tokens": u.get("output_tokens", 0),
            "cache_read_tokens": u.get("cache_read_input_tokens", 0),
        }
