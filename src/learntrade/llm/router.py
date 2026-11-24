"""
LLM router — task → model selection, cache lookup, budget enforcement, cost logging.

Tasks: ``extract``, ``research``, ``verify``, ``escalation``.

The router is the ONLY component that calls LLM clients.
All budget enforcement and cache interaction happens here, not in clients.
API keys are read from AppSecrets; they are never stored on this object.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from learntrade.core.config import LlmModelsConfig, get_config, get_secrets

logger = logging.getLogger(__name__)

LlmTask = Literal["extract", "research", "verify", "escalation"]

# Approximate cost per 1k tokens (USD) — used for budget pre-check.
# These are order-of-magnitude estimates; actual billing may differ.
_COST_PER_1K: dict[str, float] = {
    "gpt-4o": 0.005,
    "gpt-4o-mini": 0.00015,
    "claude-opus-4-5": 0.015,
    "claude-sonnet-4-5": 0.003,
    "default": 0.001,
}


@dataclass
class BudgetTracker:
    """
    Per-run and per-day LLM spending caps.

    The day's spend is kept in ``ledger_path`` (JSON keyed by UTC date) so the daily cap holds
    across restarts and across terminal commands, each of which is a new process.
    """
    per_run_limit: float
    per_day_limit: float
    ledger_path: Path | None = None
    _run_spent: float = field(default=0.0, init=False)
    _day_spent: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._day_spent = self._ledger().get(self._today(), 0.0)

    @staticmethod
    def _today() -> str:
        from datetime import datetime, timezone
        return datetime.now(tz=timezone.utc).date().isoformat()

    def _ledger(self) -> dict[str, float]:
        import json
        if not self.ledger_path:
            return {}
        try:
            return {k: float(v) for k, v in json.loads(Path(self.ledger_path).read_text()).items()}
        except (FileNotFoundError, ValueError):
            return {}

    def _save(self, cost: float) -> None:
        import json
        if not self.ledger_path:
            return
        led = self._ledger()                  # re-read: another process may have spent meanwhile
        today = self._today()
        led[today] = led.get(today, 0.0) + cost
        led = dict(sorted(led.items())[-31:])
        Path(self.ledger_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.ledger_path).write_text(json.dumps(led))
        self._day_spent = led[today]

    def check(self, estimated_cost: float) -> None:
        if self._run_spent + estimated_cost > self.per_run_limit:
            raise BudgetExceededError(
                f"Run budget ${self.per_run_limit:.2f} would be exceeded "
                f"(spent ${self._run_spent:.4f}, requesting ${estimated_cost:.4f})"
            )
        if self.ledger_path:
            self._day_spent = self._ledger().get(self._today(), 0.0)
        if self._day_spent + estimated_cost > self.per_day_limit:
            raise BudgetExceededError(
                f"Daily budget ${self.per_day_limit:.2f} would be exceeded"
            )

    def record(self, cost: float) -> None:
        self._run_spent += cost
        if self.ledger_path:
            self._save(cost)
        else:
            self._day_spent += cost

    @property
    def run_spent(self) -> float:
        return self._run_spent

    @property
    def day_spent(self) -> float:
        return self._day_spent


class BudgetExceededError(RuntimeError):
    pass


class LLMRouter:
    """
    Routes LLM tasks to the configured model, checks cache, enforces budgets.

    Usage:
        router = LLMRouter()
        response = await router.complete("extract", messages)
        text = router.extract_text(response)
    """

    def __init__(self, cache_db_path: Path | None = None) -> None:
        app_cfg = get_config()
        cfg = app_cfg.llm
        data_dir = Path(app_cfg.workspace.data_dir)
        self._enabled = cfg.enabled
        self._budget = BudgetTracker(cfg.budget_usd_per_run, cfg.budget_usd_per_day,
                                     ledger_path=data_dir / "llm_spend.json")
        self._cache_enabled = cfg.cache
        self._model_cfg = cfg.models
        self._clients: dict[LlmTask, Any] = {}

        if self._cache_enabled:
            from learntrade.llm.cache import LLMCache
            path = cache_db_path or data_dir / "llm_cache.duckdb"
            self._cache: LLMCache | None = LLMCache(path)
        else:
            self._cache = None

    # ── Public API ─────────────────────────────────────────────────────────

    async def complete(
        self,
        task: LlmTask,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Route a completion request.

        1. Check if LLM is enabled.
        2. Look up cache; return immediately on hit.
        3. Check budget.
        4. Call the client.
        5. Store in cache, record cost, log latency and tokens.
        """
        if not self._enabled:
            raise RuntimeError(
                "LLM is disabled. Set [llm] enabled = true in learntrade.toml "
                "and configure a provider."
            )

        client = self._get_client(task)
        params = {"temperature": temperature, "max_tokens": max_tokens,
                  "response_format": response_format}
        cache_key = None

        if self._cache is not None:
            cache_key = self._cache.cache_key(client.model, messages, params)
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for task=%s model=%s", task, client.model)
                return cached

        # Rough cost estimate for budget pre-check (assume ~1k tokens)
        est_cost = _COST_PER_1K.get(client.model, _COST_PER_1K["default"]) * 1.0
        self._budget.check(est_cost)

        t0 = time.monotonic()
        response = await client.complete(
            messages,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = time.monotonic() - t0

        # Compute actual cost from token counts
        usage = self._extract_usage(client, response)
        total_tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        actual_cost = _COST_PER_1K.get(client.model, _COST_PER_1K["default"]) * total_tokens / 1000
        self._budget.record(actual_cost)

        logger.info(
            "LLM task=%s model=%s tokens=%d cost=$%.4f latency=%.1fs",
            task, client.model, total_tokens, actual_cost, latency,
        )

        if self._cache is not None and cache_key:
            self._cache.set(cache_key, client.model, task, response)

        return response

    def extract_text(self, response: dict[str, Any]) -> str:
        """Extract the text content from any provider's response format."""
        # Try OpenAI format
        try:
            return response["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            pass
        # Try Anthropic format
        try:
            return response["content"][0]["text"] or ""
        except (KeyError, IndexError, TypeError):
            pass
        # Try Ollama format
        try:
            return response["message"]["content"] or ""
        except (KeyError, TypeError):
            pass
        return ""

    @property
    def budget(self) -> BudgetTracker:
        return self._budget

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    # ── Internals ──────────────────────────────────────────────────────────

    def _get_client(self, task: LlmTask) -> Any:
        if task in self._clients:
            return self._clients[task]
        cfg_map = {
            "extract": self._model_cfg.extract,
            "research": self._model_cfg.research,
            "verify": self._model_cfg.research,
            "escalation": self._model_cfg.escalation,
        }
        model_cfg = cfg_map.get(task, self._model_cfg.extract)
        client = self._build_client(model_cfg)
        self._clients[task] = client
        return client

    def _build_client(self, cfg: Any) -> Any:
        secrets = get_secrets()
        provider = cfg.provider
        model = cfg.model

        if provider in ("openai_compatible", "openai"):
            from learntrade.llm.clients.openai_compat import OpenAICompatibleClient
            return OpenAICompatibleClient(
                api_key=secrets.openai_api_key,
                model=model,
                base_url=getattr(cfg, "base_url", None) or secrets.openai_base_url,
            )
        if provider == "anthropic":
            from learntrade.llm.clients.anthropic import AnthropicClient
            return AnthropicClient(api_key=secrets.anthropic_api_key, model=model)
        if provider == "ollama":
            from learntrade.llm.clients.ollama import OllamaClient
            return OllamaClient(model=model, base_url=getattr(cfg, "base_url", None) or "http://localhost:11434")
        raise ValueError(
            f"Unknown LLM provider {provider!r}. "
            "Supported: openai_compatible, anthropic, ollama"
        )

    @staticmethod
    def _extract_usage(client: Any, response: dict[str, Any]) -> dict[str, int]:
        if hasattr(client, "extract_usage"):
            return client.extract_usage(response)
        return response.get("usage", {})
