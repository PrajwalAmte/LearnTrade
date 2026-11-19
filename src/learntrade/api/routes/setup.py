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
    from learntrade.core.config import get_config, AppSecrets
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


@router.get("/available-markets")
async def available_markets() -> dict:
    from learntrade.markets.loader import list_packs
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
