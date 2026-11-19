"""
Forecast model protocol and base class.

All quant models implement ``ForecastModel``.
Registered via the ``leantrade.models`` entry-point group.
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import polars as pl

from leantrade.core.types import Forecast, InstrumentId


@runtime_checkable
class ForecastModel(Protocol):
    """
    Contract for all forecast models (linear, GBM, ranking, deep, …).

    ``fit`` is called once per training run on historical in-sample data.
    ``predict_distribution`` is called at decision time on the current feature set.
    """

    id: str
    horizon: str
    feature_set_version: str

    def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> None:
        ...

    def predict_distribution(
        self,
        X: pl.DataFrame,
        instruments: list[InstrumentId],
        as_of: datetime,
    ) -> list[Forecast]:
        ...

    def save(self, path: str) -> None:
        ...

    @classmethod
    def load(cls, path: str) -> "ForecastModel":
        ...
