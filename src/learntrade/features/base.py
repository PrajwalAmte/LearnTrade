"""
Feature set protocol.

A FeatureSet is a versioned, declarative specification of features
computed point-in-time from the PIT store.
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from typing import Protocol

import polars as pl

from learntrade.core.types import InstrumentId


class FeatureSet(Protocol):
    id: str
    version: str
    requires: list[str]   # provider capability names needed

    @abstractmethod
    def compute(
        self,
        instruments: list[InstrumentId],
        as_of: datetime,
        store: object,     # PITStore — avoids circular import
    ) -> pl.DataFrame:
        """Return a DataFrame with one row per instrument × feature."""
        ...
