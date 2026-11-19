"""
Ollama local model client (async).

Connects to a locally-running Ollama server (default: http://localhost:11434).
No API key required. Models are pulled via `ollama pull <model>`.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)


class OllamaClient:
    provider: str = "ollama"

    def __init__(
        self,
        model: str = "qwen2.5:14b-instruct",
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(2))
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
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            body["options"]["num_predict"] = max_tokens
        if response_format and response_format.get("type") == "json_object":
            body["format"] = "json"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._base_url}/api/chat", json=body)
            resp.raise_for_status()
            data = resp.json()

        logger.debug("Ollama complete: model=%s", self.model)
        return data

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/api/embed",
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            return resp.json().get("embeddings", [])

    @staticmethod
    def extract_text(response: dict[str, Any]) -> str:
        try:
            return response["message"]["content"] or ""
        except (KeyError, TypeError):
            return ""

    @staticmethod
    def extract_usage(response: dict[str, Any]) -> dict[str, int]:
        return {
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0),
        }
