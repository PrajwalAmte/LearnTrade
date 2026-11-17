"""
CSV / XLSX holdings import adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl

from leantrade.core.types import Holding, Order, OrderAck
from leantrade.execution.brokers.base import BrokerCapability


class CsvImportAdapter:
    """
    Read-only adapter that parses a broker-exported CSV / XLSX holdings file.

    Column detection is permissive — it maps common broker column names to
    the internal schema automatically. Unknown columns are ignored.
    """

    id = "csv_import"
    markets: set[str] = set()       # "*" — works with any market
    capabilities: set[BrokerCapability] = {"holdings"}

    _SYMBOL_ALIASES = ["symbol", "ticker", "instrument", "stock", "security"]
    _QTY_ALIASES = ["quantity", "qty", "shares", "units", "position"]
    _CURRENCY_ALIASES = ["currency", "ccy", "cur"]
    _COST_ALIASES = ["avg_cost", "average_cost", "cost_basis", "avg_price", "cost"]

    def __init__(self, path: Path, mic: str = "XNYS") -> None:
        self._path = path
        self._mic = mic.upper()
        self._df: pl.DataFrame | None = None

    def _load(self) -> pl.DataFrame:
        if self._df is not None:
            return self._df
        if self._path.suffix.lower() in {".xlsx", ".xls"}:
            raise NotImplementedError("Excel import requires openpyxl; install it separately")
        self._df = pl.read_csv(self._path, infer_schema_length=1000)
        return self._df

    def _find_col(self, df: pl.DataFrame, aliases: list[str]) -> str | None:
        lower_cols = {c.lower(): c for c in df.columns}
        for alias in aliases:
            if alias in lower_cols:
                return lower_cols[alias]
        return None

    def place_order(self, order: Order) -> OrderAck:
        raise NotImplementedError("CsvImportAdapter is read-only; it cannot place orders")

    def holdings(self) -> list[Holding]:
        df = self._load()
        sym_col = self._find_col(df, self._SYMBOL_ALIASES)
        qty_col = self._find_col(df, self._QTY_ALIASES)
        cur_col = self._find_col(df, self._CURRENCY_ALIASES)
        cost_col = self._find_col(df, self._COST_ALIASES)

        if not sym_col or not qty_col:
            raise ValueError(
                "Could not detect symbol and quantity columns. "
                f"Available columns: {df.columns}"
            )

        result: list[Holding] = []
        for row in df.iter_rows(named=True):
            from leantrade.core.types import InstrumentId
            instrument_id = InstrumentId(f"{self._mic}:{row[sym_col]}")
            result.append(
                Holding(
                    instrument_id=instrument_id,
                    quantity=float(row[qty_col]),
                    currency=str(row[cur_col]) if cur_col else "USD",
                    average_cost=float(row[cost_col]) if cost_col else None,
                )
            )
        return result
