"""
FastAPI application factory.

The UI is a single static page, ``api/static/index.html`` (plain HTML/CSS/JS, no build
step and no Node required). Its data comes from ``routes/workspace.py``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from learntrade.api.routes import portfolio, setup, system, workspace


def create_app() -> FastAPI:
    app = FastAPI(
        title="LearnTrade",
        description="Local-first trading intelligence engine",
        version="0.1.0-dev",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # Allow the Vite dev server on port 5173 during development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(system.router,    prefix="/api")
    app.include_router(portfolio.router, prefix="/api/portfolio")
    app.include_router(setup.router,     prefix="/api/setup")
    app.include_router(workspace.router, prefix="/api/app")

    # Serve pre-built UI bundle if it exists
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="ui")

    return app


app = create_app()
