"""
Data engine tests — provider fallback, instrument resolution, pull summary.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from unittest.mock import MagicMock, patch

import pytest

from leantrade.core.config import AppConfig, DataProvidersConfig, DataConfig
from leantrade.core.types import InstrumentId, Observation
from leantrade.data.engine import DataEngine
from leantrade.data.providers.base import DataProvider, FetchRequest, ProviderCapability, PitLevel
from leantrade.data.store.pit import PITStore

UTC = timezone.utc


class StubProvider:
    """A stub data provider that returns a fixed number of observations."""

    def __init__(self, provider_id: str, obs_count: int = 5, should_fail: bool = False) -> None:
        self.id = provider_id
        self.capabilities: set[ProviderCapability] = {"bars_daily"}
        self.markets = "*"
        self.pit_level: dict[ProviderCapability, PitLevel] = {"bars_daily": "none"}
        self.licence = "stub"
        self._obs_count = obs_count
        self._should_fail = should_fail
        self.fetch_called = False

    def fetch(self, request: FetchRequest) -> Iterable[Observation]:
        self.fetch_called = True
        if self._should_fail:
            raise RuntimeError("provider failed")
        now = datetime.now(tz=UTC)
        for instrument_id in request.instruments:
            for i in range(self._obs_count):
                yield Observation(
                    instrument_id=instrument_id,
                    source_id=self.id,
                    kind="bar",
                    event_time=datetime(2024, 1, i + 2, tzinfo=UTC),
                    available_at=now,
                    payload={"close": 100.0 + i, "open": 100.0, "high": 101.0, "low": 99.0,
                             "volume": 1000, "adj_close": 100.0 + i},
                )


@pytest.fixture()
def store(tmp_path: Path) -> PITStore:
    s = PITStore(tmp_path)
    yield s
    s.close()


def _config_with_instruments(*symbols: str, provider: str = "stub") -> AppConfig:
    cfg = AppConfig()
    cfg.markets_enabled = ["XNYS"]
    cfg.data = DataConfig(providers=DataProvidersConfig(prices=[provider]))
    cfg.universe = {
        "XNYS": type("U", (), {
            "source": "explicit",
            "instruments": list(symbols),
            "index": None,
            "n": None,
            "min_adv": None,
            "point_in_time": True,
        })()
    }
    return cfg


def test_pull_uses_first_provider(store: PITStore) -> None:
    provider = StubProvider("stub", obs_count=3)
    cfg = _config_with_instruments("IBM", "AAPL")
    engine = DataEngine(cfg, store)
    engine._registry._cache["stub"] = provider

    summary = engine.pull(
        since=datetime(2024, 1, 1, tzinfo=UTC),
        until=datetime(2024, 1, 10, tzinfo=UTC),
        capabilities={"bars_daily"},
    )

    assert provider.fetch_called
    assert summary.total_added == 6  # 2 instruments × 3 obs each


def test_pull_falls_back_on_provider_failure(store: PITStore) -> None:
    failing = StubProvider("bad_provider", should_fail=True)
    good = StubProvider("good_provider", obs_count=2)

    cfg = AppConfig()
    cfg.markets_enabled = ["XNYS"]
    cfg.data = DataConfig(providers=DataProvidersConfig(prices=["bad_provider", "good_provider"]))
    cfg.universe = {
        "XNYS": type("U", (), {
            "source": "explicit", "instruments": ["IBM"],
            "index": None, "n": None, "min_adv": None, "point_in_time": True,
        })()
    }

    engine = DataEngine(cfg, store)
    engine._registry._cache["bad_provider"] = failing
    engine._registry._cache["good_provider"] = good

    summary = engine.pull(
        since=datetime(2024, 1, 1, tzinfo=UTC),
        until=datetime(2024, 1, 10, tzinfo=UTC),
        capabilities={"bars_daily"},
    )

    assert failing.fetch_called
    assert good.fetch_called
    assert summary.total_added == 2
    assert any(r.error for r in summary.results)
    assert any(r.success for r in summary.results)


def test_pull_empty_universe_adds_nothing(store: PITStore) -> None:
    provider = StubProvider("stub", obs_count=5)
    cfg = AppConfig()
    cfg.markets_enabled = ["XNYS"]
    cfg.data = DataConfig(providers=DataProvidersConfig(prices=["stub"]))
    # No universe configured

    engine = DataEngine(cfg, store)
    engine._registry._cache["stub"] = provider

    summary = engine.pull(
        since=datetime(2024, 1, 1, tzinfo=UTC),
        until=datetime(2024, 1, 10, tzinfo=UTC),
        capabilities={"bars_daily"},
    )
    assert summary.total_added == 0


def test_pull_summary_to_dict(store: PITStore) -> None:
    provider = StubProvider("stub", obs_count=1)
    cfg = _config_with_instruments("IBM")
    engine = DataEngine(cfg, store)
    engine._registry._cache["stub"] = provider

    summary = engine.pull(
        since=datetime(2024, 1, 1, tzinfo=UTC),
        until=datetime(2024, 1, 10, tzinfo=UTC),
        capabilities={"bars_daily"},
    )
    d = summary.to_dict()
    assert "total_observations_added" in d
    assert "results" in d
    assert d["total_observations_added"] == 1
