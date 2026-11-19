"""
Plugin registry — discovers and vends implementations via entry-point groups.

Usage:
    registry = get_registry()
    provider = registry.get_provider("yahoo")
    adapter  = registry.get_broker("csv_import")
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any


class Registry:
    """Lazy-loading registry backed by setuptools entry points."""

    _GROUPS = {
        "providers": "leantrade.providers",
        "brokers": "leantrade.brokers",
        "markets": "leantrade.markets",
        "features": "leantrade.features",
        "models": "leantrade.models",
        "allocators": "leantrade.allocators",
        "llm_providers": "leantrade.llm_providers",
        "agents": "leantrade.agents",
    }

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}

    def _load_group(self, group_key: str) -> dict[str, Any]:
        if group_key not in self._cache:
            eps = entry_points(group=self._GROUPS[group_key])
            self._cache[group_key] = {ep.name: ep.load() for ep in eps}
        return self._cache[group_key]

    def get_provider(self, name: str) -> Any:
        return self._load_group("providers")[name]

    def list_providers(self) -> list[str]:
        return list(self._load_group("providers").keys())

    def get_broker(self, name: str) -> Any:
        return self._load_group("brokers")[name]

    def list_brokers(self) -> list[str]:
        return list(self._load_group("brokers").keys())

    def get_model(self, name: str) -> Any:
        return self._load_group("models")[name]

    def get_allocator(self, name: str) -> Any:
        return self._load_group("allocators")[name]


_registry: Registry | None = None


def get_registry() -> Registry:
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry
