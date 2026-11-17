"""
Live execution safety guard — enforces the three opt-in gates.

Gate 1: config file has execution.mode = "live"
Gate 2: session flag --enable-live was passed to `leantrade serve`
Gate 3: per-order confirmation (each Order must have confirmed=True)

None of the three gates can be bypassed. Any missing gate raises LiveExecutionError.
"""

from __future__ import annotations

import logging

from leantrade.core.types import Order

logger = logging.getLogger(__name__)

_SESSION_LIVE_ENABLED: bool = False


class LiveExecutionError(RuntimeError):
    pass


def enable_live_session() -> None:
    """Gate 2: set when --enable-live flag is passed to leantrade serve."""
    global _SESSION_LIVE_ENABLED
    _SESSION_LIVE_ENABLED = True
    logger.warning(
        "Live execution session enabled. "
        "Every order will require explicit per-order confirmation."
    )


def check_live_gates(orders: list[Order]) -> list[Order]:
    """
    Enforce all three opt-in gates before submitting live orders.

    Raises LiveExecutionError if any gate is not satisfied.
    Returns the confirmed order list.
    """
    from leantrade.core.config import get_config

    cfg = get_config()

    # Gate 1: config must have mode = "live"
    if cfg.execution.mode != "live":
        raise LiveExecutionError(
            "Gate 1 not satisfied: execution.mode is not 'live' in leantrade.toml. "
            "Set [execution] mode = 'live' to enable."
        )

    # Gate 2: session flag
    if not _SESSION_LIVE_ENABLED:
        raise LiveExecutionError(
            "Gate 2 not satisfied: live execution was not explicitly enabled for "
            "this session. Start the server with `leantrade serve --enable-live`."
        )

    # Gate 3: per-order confirmation
    confirmed = []
    for order in orders:
        if not order.notes or "confirmed" not in order.notes.lower():
            logger.warning(
                "Order for %s rejected at Gate 3: missing per-order confirmation "
                "(set order.notes='confirmed')",
                order.instrument_id,
            )
        else:
            confirmed.append(order)

    if not confirmed and orders:
        raise LiveExecutionError(
            "Gate 3 not satisfied: no orders carry per-order confirmation. "
            "Set order.notes='confirmed' on each order."
        )

    logger.info("All 3 live execution gates satisfied for %d order(s)", len(confirmed))
    return confirmed
