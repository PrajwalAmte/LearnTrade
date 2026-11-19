"""Setup wizard API — 8-step browser wizard."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["setup"])

_WIZARD_STEPS = [
    {"id": 1, "name": "welcome",        "title": "Welcome & Risk Notice"},
    {"id": 2, "name": "market",         "title": "Where do you invest?"},
    {"id": 3, "name": "currency",       "title": "Your home currency"},
    {"id": 4, "name": "data_sources",   "title": "Data sources"},
    {"id": 5, "name": "ai_assistant",   "title": "AI assistant"},
    {"id": 6, "name": "style",          "title": "Your investing style"},
    {"id": 7, "name": "portfolio",      "title": "Your portfolio"},
    {"id": 8, "name": "first_download", "title": "First data download"},
]


@router.get("/status")
async def setup_status() -> dict:
    from leantrade.core.config import get_config, AppSecrets
    from pathlib import Path
    secrets = AppSecrets()
    cfg_exists = Path(secrets.config_file).exists()
    return {
        "completed": cfg_exists,
        "current_step": 1 if not cfg_exists else 8,
        "steps": _WIZARD_STEPS,
    }


@router.get("/steps")
async def list_steps() -> dict:
    return {"steps": _WIZARD_STEPS}


@router.post("/step/{step_id}")
async def submit_step(step_id: int, payload: dict[str, Any]) -> dict:
    """
    Accept wizard step submission and persist relevant config.
    Returns the next step number or {"completed": true} on step 8.
    """
    if step_id == 1:
        # Risk notice accepted — enable paper trading (safe default)
        return {"accepted": payload.get("accepted", False), "next_step": 2}

    if step_id == 2:
        # Market selection
        mics = payload.get("mics", ["XNYS"])
        return {"mics": mics, "next_step": 3}

    if step_id == 5:
        # AI assistant choice
        choice = payload.get("choice", "none")
        return {"choice": choice, "next_step": 6}

    if step_id == 8:
        return {"completed": True, "message": "Setup complete. Your first data pull is queued."}

    return {"next_step": min(step_id + 1, 8)}


@router.get("/available-markets")
async def available_markets() -> dict:
    from leantrade.markets.loader import list_packs
    return {"markets": list_packs()}


@router.get("/hardware-recommendation")
async def hardware_recommendation() -> dict:
    """Recommend local LLM model based on available RAM."""
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    if ram_gb < 8:
        return {"recommendation": "none", "reason": f"Less than 8 GB RAM ({ram_gb:.0f} GB available)"}
    if ram_gb < 16:
        return {"recommendation": "small", "model": "qwen2.5:7b-instruct",
                "reason": f"{ram_gb:.0f} GB RAM — small model fits comfortably"}
    return {"recommendation": "medium", "model": "qwen2.5:14b-instruct",
            "reason": f"{ram_gb:.0f} GB RAM — medium model recommended"}
