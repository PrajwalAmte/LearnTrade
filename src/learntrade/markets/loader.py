"""
Market Pack loader.

A Market Pack is a YAML file that describes everything exchange-specific:
calendar, sessions, currency, lot/tick sizes, price limits, settlement,
short-selling rules, and a composable cost/tax model.

The loader searches (in order):
1. ``markets/`` relative to the workspace root
2. Any directory registered via the ``learntrade.markets`` entry-point group
3. The built-in ``markets/`` folder inside the installed package

Callers should use ``get_pack(mic)`` which caches the result.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import]

from learntrade.markets.schema import MarketPack


def _find_builtin_market_packs_dir() -> Path:
    """
    Locate the ``markets/`` directory shipped at the repository root.

    ``__file__`` is ``<repo>/src/learntrade/markets/loader.py``, four levels
    below the repository root, so walk up looking for a ``markets`` sibling.
    An explicit ``LEARNTRADE_MARKETS_DIR`` env var always wins (useful when
    the package is installed from a wheel without the markets/ directory).
    """
    override = os.environ.get("LEARNTRADE_MARKETS_DIR")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / "markets"
        if candidate.is_dir() and (candidate / "generic.yaml").exists():
            return candidate
    # Fall back to the historical (possibly wrong) guess rather than raising
    # at import time — get_pack() will raise a clear error if it's unusable.
    return here.parent.parent.parent.parent / "markets"


# Resolved at import time — the ``markets/`` directory shipped with the package
_BUILTIN_MARKET_PACKS_DIR: Path = _find_builtin_market_packs_dir()

# Workspace-relative override directory (may not exist yet)
_WORKSPACE_MARKET_PACKS_DIR: Path = Path("markets")


def _yaml_load(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)  # type: ignore[return-value]


def _search_dirs() -> list[Path]:
    dirs: list[Path] = []
    if _WORKSPACE_MARKET_PACKS_DIR.exists():
        dirs.append(_WORKSPACE_MARKET_PACKS_DIR)
    if _BUILTIN_MARKET_PACKS_DIR.exists():
        dirs.append(_BUILTIN_MARKET_PACKS_DIR)
    return dirs


@lru_cache(maxsize=64)
def get_pack(mic: str) -> MarketPack:
    """
    Load and cache the Market Pack for the given MIC.

    Falls back to the ``generic`` pack when no specific pack exists.
    """
    mic_upper = mic.upper()
    for directory in _search_dirs():
        candidate = directory / f"{mic_upper}.yaml"
        if candidate.exists():
            return MarketPack.model_validate(_yaml_load(candidate))

    # Fallback to generic
    for directory in _search_dirs():
        generic = directory / "generic.yaml"
        if generic.exists():
            data = _yaml_load(generic)
            data["mic"] = mic_upper  # patch MIC into the generic template
            return MarketPack.model_validate(data)

    raise FileNotFoundError(
        f"No Market Pack found for {mic_upper!r} and no generic.yaml available. "
        f"Run `learntrade market new --mic {mic_upper}` to scaffold one."
    )


def list_packs() -> list[str]:
    """Return the MIC codes of all available Market Packs."""
    seen: set[str] = set()
    for directory in _search_dirs():
        for path in directory.glob("*.yaml"):
            seen.add(path.stem.upper())
    seen.discard("GENERIC")
    return sorted(seen)
