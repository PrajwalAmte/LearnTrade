"""
Market Pack contract tests.

Each pack must load cleanly and pass basic structural assertions.
"""

from __future__ import annotations

import pytest

from learntrade.markets import get_pack, list_packs
from learntrade.markets.schema import MarketPack


SHIPPED_MICS = ["XNYS", "XNAS", "XNSE"]


@pytest.mark.parametrize("mic", SHIPPED_MICS)
def test_pack_loads(mic: str):
    pack = get_pack(mic)
    assert isinstance(pack, MarketPack)
    assert pack.mic == mic


@pytest.mark.parametrize("mic", SHIPPED_MICS)
def test_pack_has_currency(mic: str):
    pack = get_pack(mic)
    assert len(pack.currency) == 3     # ISO 4217


@pytest.mark.parametrize("mic", SHIPPED_MICS)
def test_pack_has_sessions(mic: str):
    pack = get_pack(mic)
    assert "regular" in pack.sessions


def test_generic_pack_loads():
    pack = get_pack("XXXX")    # unknown MIC → should load generic
    assert pack.mic == "XXXX"


def test_list_packs_includes_shipped():
    packs = list_packs()
    for mic in SHIPPED_MICS:
        assert mic in packs
