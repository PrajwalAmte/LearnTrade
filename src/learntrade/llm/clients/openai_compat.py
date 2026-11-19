"""
OpenAI-compatible LLM client (async, retries via tenacity).

Works with any endpoint that follows the OpenAI chat completions API:
OpenAI, Azure OpenAI, OpenRouter, vLLM, LM Studio, llama.cpp server, etc.

Secrets: set LEARNTRADE_OPENAI_API_KEY and optionally LEARNTRADE_OPENAI_BASE_URL.
Never log or store the API key.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class OpenAICompatibleClient:
    provider: str = "openai_compatible"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI-compatible client requires an API key")
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._timeout = timeout

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            body["response_format"] = response_format
        if max_tokens:
            body["max_tokens"] = max_tokens

        async with httpx.AsyncClient(headers=self._headers, timeout=self._timeout) as client:
            resp = await client.post(f"{self._base_url}/chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()

        # Log token usage without logging content
        usage = data.get("usage", {})
        logger.debug(
            "OpenAI complete: model=%s prompt_tokens=%s completion_tokens=%s",
            self.model,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
        )
        return data

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(headers=self._headers, timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/embeddings",
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
        return [item["embedding"] for item in data["data"]]

    @staticmethod
    def extract_text(response: dict[str, Any]) -> str:
        """Pull the first message content from a chat completion response."""
        try:
            return response["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            return ""

    @staticmethod
    def extract_usage(response: dict[str, Any]) -> dict[str, int]:
        return response.get("usage", {})
