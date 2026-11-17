"""Data routes — pull, status, check."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["data"])


@router.get("/status")
async def data_status() -> dict:
    # TODO: return last pull timestamp, row counts, PIT level summary
    return {"status": "not_configured"}
