"""
Doctor — diagnoses the five most common setup problems and proposes fixes.

Checks (in order):
1. Config file missing / not initialised
2. No data pulled yet (empty PIT store)
3. Required API keys missing for configured providers
4. No universe configured for any enabled market
5. LLM enabled but API key not set

Each check returns a DoctorIssue with a plain-language message and optional fix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class DoctorIssue:
    severity: Literal["error", "warning", "info"]
    check: str
    message: str
    fix: str | None = None   # CLI command or plain-language instruction


def run_doctor(config_path: Path | None = None) -> list[DoctorIssue]:
    """Run all diagnostic checks. Returns issues sorted by severity."""
    from leantrade.core.config import load_config, AppSecrets

    issues: list[DoctorIssue] = []
    secrets = AppSecrets()
    cfg_file = config_path or Path(secrets.config_file)

    # ── 1. Config file ────────────────────────────────────────────────────
    if not cfg_file.exists():
        issues.append(DoctorIssue(
            severity="error",
            check="config",
            message=(
                f"Config file not found at {cfg_file}. "
                "LearnTrade has not been initialised."
            ),
            fix="leantrade init   (or run `leantrade serve --open` for the browser wizard)",
        ))
        return issues  # remaining checks depend on config

    cfg = load_config(cfg_file)

    # ── 2. Data pulled ────────────────────────────────────────────────────
    data_dir = cfg.workspace.data_dir
    db_path = data_dir / "leantrade.duckdb"
    if not db_path.exists():
        issues.append(DoctorIssue(
            severity="error",
            check="data",
            message="No data found. The PIT store has not been created yet.",
            fix="leantrade data pull --since 2015-01-01",
        ))
    else:
        try:
            import duckdb
            con = duckdb.connect(str(db_path), read_only=True)
            count = con.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
            con.close()
            if count == 0:
                issues.append(DoctorIssue(
                    severity="warning",
                    check="data",
                    message="PIT store exists but contains no observations.",
                    fix="leantrade data pull --since 2015-01-01",
                ))
            else:
                issues.append(DoctorIssue(
                    severity="info",
                    check="data",
                    message=f"PIT store OK: {count:,} observations",
                ))
        except Exception as exc:
            issues.append(DoctorIssue(
                severity="warning",
                check="data",
                message=f"Could not query PIT store: {exc}",
            ))

    # ── 3. API keys ───────────────────────────────────────────────────────
    provider_key_map = {
        "yahoo": None,                      # no key needed
        "fred": "fred_api_key",
        "sec_edgar": "sec_user_agent",
        "fmp": "fmp_api_key",
        "polygon": "polygon_api_key",
        "tiingo": "tiingo_api_key",
    }
    configured_providers = set(
        cfg.data.providers.prices
        + cfg.data.providers.fundamentals
        + cfg.data.providers.macro
        + cfg.data.providers.filings
        + cfg.data.providers.news
    )
    for provider in configured_providers:
        key_attr = provider_key_map.get(provider)
        if key_attr and not getattr(secrets, key_attr, ""):
            issues.append(DoctorIssue(
                severity="warning",
                check="api_keys",
                message=(
                    f"Provider '{provider}' is configured but its API key is not set "
                    f"(LEANTRADE_{key_attr.upper()} in .env)."
                ),
                fix=f"Add LEANTRADE_{key_attr.upper()}=<your-key> to .env",
            ))

    # ── 4. Universe ───────────────────────────────────────────────────────
    if db_path.exists():
        try:
            import duckdb
            from datetime import datetime, timezone
            con = duckdb.connect(str(db_path), read_only=True)
            for mic in cfg.markets_enabled:
                rows = con.execute(
                    "SELECT COUNT(*) FROM observations WHERE kind='universe' "
                    "AND payload->>'mic'=?",
                    [mic],
                ).fetchone()
                if not rows or rows[0] == 0:
                    if mic not in cfg.universe or not cfg.universe[mic].instruments:
                        issues.append(DoctorIssue(
                            severity="error",
                            check="universe",
                            message=(
                                f"No universe configured for {mic}. "
                                "Cannot train without an instrument list."
                            ),
                            fix=(
                                f"leantrade universe import --mic {mic} --file instruments.csv  "
                                "(CSV with a 'symbol' column)"
                            ),
                        ))
            con.close()
        except Exception:
            pass

    # ── 5. LLM keys ───────────────────────────────────────────────────────
    if cfg.llm.enabled:
        provider = cfg.llm.models.extract.provider
        if provider in ("openai_compatible", "openai") and not secrets.openai_api_key:
            issues.append(DoctorIssue(
                severity="error",
                check="llm",
                message="LLM is enabled with OpenAI but LEANTRADE_OPENAI_API_KEY is not set.",
                fix="Add LEANTRADE_OPENAI_API_KEY=<your-key> to .env, or set [llm] enabled=false",
            ))
        elif provider == "anthropic" and not secrets.anthropic_api_key:
            issues.append(DoctorIssue(
                severity="error",
                check="llm",
                message="LLM is enabled with Anthropic but LEANTRADE_ANTHROPIC_API_KEY is not set.",
                fix="Add LEANTRADE_ANTHROPIC_API_KEY=<your-key> to .env",
            ))
        elif provider == "ollama":
            # Check if Ollama is reachable
            try:
                import httpx
                resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
                if resp.status_code != 200:
                    raise ConnectionError
                issues.append(DoctorIssue(severity="info", check="llm",
                                          message="Ollama is running and reachable."))
            except Exception:
                issues.append(DoctorIssue(
                    severity="warning",
                    check="llm",
                    message="LLM is set to Ollama but the server is not reachable at localhost:11434.",
                    fix="Start Ollama: `ollama serve`  then  `ollama pull qwen2.5:14b-instruct`",
                ))

    # Sort: errors first, then warnings, then info
    _order = {"error": 0, "warning": 1, "info": 2}
    issues.sort(key=lambda i: _order[i.severity])
    return issues
