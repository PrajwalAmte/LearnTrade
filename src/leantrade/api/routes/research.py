"""Research routes — rank, analyze, top ideas."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["research"])


@router.get("/rank")
async def rank(market: str = "XNYS", top: int = 20) -> dict:
    # TODO: call quant engine and return ranked forecasts
    return {"market": market, "top": top, "instruments": []}


@router.get("/analyze/{instrument_id}")
async def analyze(instrument_id: str) -> dict:
    # TODO: return full research report for one instrument
    return {"instrument_id": instrument_id, "report": None}
