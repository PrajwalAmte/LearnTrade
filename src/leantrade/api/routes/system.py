"""System routes — health, config, doctor."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["system"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.1.0-dev"}


@router.get("/config")
async def get_config() -> dict:
    from leantrade.core.config import get_config as _cfg
    cfg = _cfg()
    return {
        "base_currency": cfg.workspace.base_currency,
        "markets_enabled": cfg.markets_enabled,
        "horizons": cfg.horizons.enabled,
        "llm_enabled": cfg.llm.enabled,
        "execution_mode": cfg.execution.mode,
    }


@router.get("/doctor")
async def doctor() -> dict:
    from leantrade.setup.doctor import run_doctor
    issues = run_doctor()
    return {
        "passed": not any(i.severity == "error" for i in issues),
        "issues": [
            {"severity": i.severity, "check": i.check,
             "message": i.message, "fix": i.fix}
            for i in issues
        ],
    }
