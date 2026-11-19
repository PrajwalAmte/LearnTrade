"""
LearnTrade — local-first, market-agnostic trading intelligence engine.

    from learntrade import Workspace

    ws = Workspace()
    forecasts = ws.rank(market="XNYS", horizon="swing_20d")
"""

from __future__ import annotations

__version__ = "0.1.0-dev"
__all__ = ["__version__"]
