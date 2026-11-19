"""
FX conversion for multi-currency portfolios.

Uses spot rates for P&L attribution. Rates are sourced from macro observations
(kind="macro", series_id like "USD/EUR") or fall back to 1.0 (same currency).
All portfolio values are converted into the base currency.
"""

from __future__ import annotations

from datetime import datetime


class FXConverter:
    """
    Convert amounts between currencies.

    Rates are stored as {quote_currency: rate_vs_base}, meaning:
        amount_base = amount_foreign / rate
    For example, with base_currency="USD":
        "EUR": 1.08  means 1 EUR = 1.08 USD
        "INR": 0.012 means 1 INR = 0.012 USD
    """

    def __init__(self, base_currency: str = "USD") -> None:
        self._base = base_currency
        # Rates: {from_currency: amount_in_base per 1 unit of from_currency}
        self._rates: dict[str, float] = {base_currency: 1.0}

    def update_rate(self, currency: str, rate_vs_base: float) -> None:
        """Set the spot rate: 1 unit of currency = rate_vs_base units of base."""
        self._rates[currency] = rate_vs_base

    def convert(self, amount: float, from_currency: str, to_currency: str | None = None) -> float:
        """Convert amount from from_currency to base (or to_currency if specified)."""
        to = to_currency or self._base
        if from_currency == to:
            return amount
        rate_from = self._rates.get(from_currency, 1.0)  # 1.0 = assume parity if unknown
        rate_to = self._rates.get(to, 1.0)
        return amount * rate_from / rate_to

    def to_base(self, amount: float, currency: str) -> float:
        return self.convert(amount, currency)

    def portfolio_value_in_base(
        self, positions: dict, prices: dict[str, float]
    ) -> float:
        """
        Compute total portfolio value in base currency.
        positions: {instrument_id: Position}
        prices:    {instrument_id: price_in_instrument_currency}
        """
        total = 0.0
        for iid, pos in positions.items():
            price = prices.get(str(iid), pos.avg_cost)
            value_in_pos_currency = pos.quantity * price
            total += self.to_base(value_in_pos_currency, pos.currency)
        return total

    @property
    def base_currency(self) -> str:
        return self._base

    @property
    def rates(self) -> dict[str, float]:
        return dict(self._rates)
