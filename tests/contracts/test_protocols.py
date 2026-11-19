"""
Provider and broker contract tests.

Every registered provider and broker must implement the declared protocol.
"""

from __future__ import annotations

import pytest

from learntrade.data.providers.base import DataProvider
from learntrade.data.providers.yahoo import YahooProvider
from learntrade.data.providers.fred import FREDProvider
from learntrade.data.providers.sec_edgar import SECEDGARProvider
from learntrade.execution.brokers.base import BrokerAdapter
from learntrade.execution.brokers.csv_import import CsvImportAdapter


# ── DataProvider contract ─────────────────────────────────────────────────────

def test_yahoo_satisfies_provider_protocol() -> None:
    provider = YahooProvider()
    assert isinstance(provider, DataProvider)
    assert "bars_daily" in provider.capabilities
    assert provider.markets == "*"
    assert "bars_daily" in provider.pit_level
    assert provider.pit_level["bars_daily"] == "none"
    assert provider.licence


def test_fred_satisfies_provider_protocol() -> None:
    provider = FREDProvider()
    assert isinstance(provider, DataProvider)
    assert "macro" in provider.capabilities
    assert provider.markets == "*"
    assert provider.pit_level["macro"] == "full"


def test_sec_edgar_satisfies_provider_protocol() -> None:
    provider = SECEDGARProvider(user_agent="Test User test@example.com")
    assert isinstance(provider, DataProvider)
    assert "filings" in provider.capabilities
    assert provider.pit_level["filings"] == "full"


def test_sec_edgar_requires_user_agent() -> None:
    with pytest.raises(ValueError, match="User-Agent"):
        SECEDGARProvider(user_agent="")


# ── BrokerAdapter contract ────────────────────────────────────────────────────

def test_csv_import_satisfies_broker_protocol() -> None:
    from pathlib import Path
    adapter = CsvImportAdapter(Path("/dev/null"))
    assert isinstance(adapter, BrokerAdapter)
    assert "holdings" in adapter.capabilities


# ── Protocol field completeness ───────────────────────────────────────────────

@pytest.mark.parametrize("provider_cls,kwargs", [
    (YahooProvider, {}),
    (FREDProvider, {}),
    (SECEDGARProvider, {"user_agent": "Test test@example.com"}),
])
def test_provider_has_all_required_fields(provider_cls, kwargs) -> None:
    provider = provider_cls(**kwargs)
    for field in ("id", "capabilities", "markets", "pit_level", "licence"):
        assert hasattr(provider, field), f"{provider_cls.__name__} missing field {field!r}"
    assert callable(provider.fetch)
