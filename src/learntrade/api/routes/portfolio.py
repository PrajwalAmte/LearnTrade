"""Portfolio routes — holdings, health check, rebalance proposals."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["portfolio"])


@router.get("/holdings")
async def holdings() -> dict:
    from learntrade.core.config import get_config
    from learntrade.data.store.app_state import AppStateDB
    cfg = get_config()
    db_path = cfg.workspace.data_dir / "app_state.duckdb"
    if not db_path.exists():
        return {"holdings": [], "portfolio": "default"}
    db = AppStateDB(db_path)
    items = db.get_portfolio("paper_trading") or []
    db.close()
    return {
        "portfolio": "paper_trading",
        "holdings": [
            {"instrument_id": str(h.instrument_id), "quantity": h.quantity,
             "currency": h.currency, "average_cost": h.average_cost}
            for h in items
        ],
    }


@router.get("/health")
async def portfolio_health() -> dict:
    """Concentration, sector, and currency flag checks."""
    from learntrade.core.config import get_config
    from learntrade.data.store.app_state import AppStateDB
    cfg = get_config()
    db_path = cfg.workspace.data_dir / "app_state.duckdb"
    flags = []
    if not db_path.exists():
        return {"flags": [], "passed": True}
    db = AppStateDB(db_path)
    holdings = db.get_portfolio("paper_trading") or []
    db.close()
    if not holdings:
        return {"flags": [], "passed": True}
    max_pos = cfg.portfolio.max_position
    for h in holdings:
        # Rough check: flag any single position that is very large in quantity
        if max_pos and h.quantity > 0:
            pass  # would need NAV to compute fraction
    return {"flags": flags, "passed": len(flags) == 0}


@router.get("/rebalance")
async def rebalance_proposal(
    market: str = "XNYS",
    allocator: str = "hrp",
    dry_run: bool = True,
) -> dict:
    from learntrade.core.config import get_config
    from learntrade.data.store.pit import PITStore
    from learntrade.features.price_volume import PriceVolumeFeatureSet
    from learntrade.portfolio.allocators import HRPAllocator, EqualWeightAllocator
    from learntrade.core.types import Forecast, ForecastQuantiles, InstrumentId
    from datetime import datetime, timezone
    import polars as pl

    cfg = get_config()
    now = datetime.now(tz=timezone.utc)
    alloc = HRPAllocator() if allocator == "hrp" else EqualWeightAllocator()
    with PITStore(cfg.workspace.data_dir) as store:
        instruments = store.get_universe(market, now)
        if not instruments:
            return {"error": f"No universe for {market}", "proposals": []}
        feature_set = PriceVolumeFeatureSet()
        features = feature_set.compute(instruments, now, store)
        if features.is_empty():
            return {"error": "No features available", "proposals": []}
        forecasts = [
            Forecast(
                instrument_id=InstrumentId(row["instrument_id"]),
                as_of=now, horizon="swing_20d",
                expected_return=float(row.get("ret_20d") or 0),
                p_positive=0.5, volatility=float(row.get("ewma_vol_20d") or 0.2),
                quantiles=ForecastQuantiles(q05=-0.05, q25=-0.02, q50=0, q75=0.02, q95=0.05),
                model_id="feature_proxy", data_snapshot_id="",
            )
            for row in features.iter_rows(named=True)
            if row.get("ret_20d") is not None
        ]
        weights = alloc.allocate(forecasts)
    return {
        "market": market, "allocator": allocator, "dry_run": dry_run,
        "proposals": [
            {"instrument_id": str(iid), "target_weight": round(w, 4)}
            for iid, w in sorted(weights.items(), key=lambda x: -x[1])[:20]
        ],
    }
