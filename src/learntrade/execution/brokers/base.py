"""
Broker adapter protocol.

Registered via the ``learntrade.brokers`` entry-point group.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from learntrade.core.types import Holding, Order, OrderAck

BrokerCapability = Literal[
    "holdings", "positions", "orders_read", "quotes", "place_order", "cancel_order"
]


@runtime_checkable
class BrokerAdapter(Protocol):
    id: str
    markets: set[str]
    capabilities: set[BrokerCapability]

    def holdings(self) -> list[Holding]:
        ...

    def place_order(self, order: Order) -> OrderAck:
        """Only available when ``"place_order"`` is in ``capabilities``."""
        ...
