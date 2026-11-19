"""
Pydantic-settings config models for learntrade.toml + environment variables.

Loading order: defaults → learntrade.toml → .env → environment variables.
API keys are NEVER written to learntrade.toml; they go to .env or the OS keychain.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class WorkspaceConfig(BaseModel):
    base_currency: str = "USD"
    data_dir: Path = Path("./data")
    timezone: str = "UTC"


class BrokerCostConfig(BaseModel):
    commission: dict[str, Any] = Field(default_factory=dict)


class MarketConfig(BaseModel):
    broker_costs: BrokerCostConfig = Field(default_factory=BrokerCostConfig)


class UniverseEntry(BaseModel):
    source: Literal["index_members", "liquidity_top_n", "explicit"] = "index_members"
    index: str | None = None
    n: int | None = None
    min_adv: float | None = None
    point_in_time: bool = True
    instruments: list[str] = Field(default_factory=list)


class DataProvidersConfig(BaseModel):
    prices: list[str] = Field(default_factory=lambda: ["yahoo"])
    fundamentals: list[str] = Field(default_factory=list)
    macro: list[str] = Field(default_factory=list)
    news: list[str] = Field(default_factory=list)
    filings: list[str] = Field(default_factory=list)
    corporate_actions: list[str] = Field(default_factory=list)


class DataConfig(BaseModel):
    providers: DataProvidersConfig = Field(default_factory=DataProvidersConfig)
    # RSS URL templates for the `rss` news provider; may use {ticker}, {symbol}, {mic}.
    # Empty = built-in Yahoo Finance headline feed.
    news_feeds: list[str] = Field(default_factory=list)


class HorizonsConfig(BaseModel):
    enabled: list[str] = Field(default_factory=lambda: ["swing_5d", "swing_20d", "positional_60d"])


class QuantConfig(BaseModel):
    models: list[str] = Field(default_factory=lambda: ["ridge", "lgbm"])
    calibration: Literal["platt", "isotonic", "conformal"] = "isotonic"
    pooled_across_markets: bool = True


class LlmModelConfig(BaseModel):
    provider: Literal["openai_compatible", "anthropic", "google", "ollama", "llamacpp"] = (
        "openai_compatible"
    )
    model: str = "gpt-4o-mini"
    base_url: str | None = None


class LlmModelsConfig(BaseModel):
    extract: LlmModelConfig = Field(default_factory=LlmModelConfig)
    research: LlmModelConfig = Field(default_factory=LlmModelConfig)
    escalation: LlmModelConfig = Field(default_factory=LlmModelConfig)


class LlmEscalationConfig(BaseModel):
    on_quant_disagreement: bool = True
    on_uncertainty_above: float = 0.6
    on_events: list[str] = Field(
        default_factory=lambda: ["earnings", "guidance", "corporate_action"]
    )
    self_consistency_samples: int = 3


class LlmConfig(BaseModel):
    enabled: bool = False
    budget_usd_per_run: float = 5.0
    budget_usd_per_day: float = 20.0
    cache: bool = True
    contamination_guard: bool = True
    safety_margin_days: int = 30
    models: LlmModelsConfig = Field(default_factory=LlmModelsConfig)
    escalation: LlmEscalationConfig = Field(default_factory=LlmEscalationConfig)


class PortfolioConfig(BaseModel):
    allocator: Literal[
        "equal_weight", "inverse_vol", "mean_variance", "hrp", "risk_parity", "cvar"
    ] = "hrp"
    max_position: float = 0.05
    max_sector: float = 0.25
    max_currency_exposure: dict[str, float] = Field(default_factory=dict)
    long_only: bool = True
    rebalance: Literal["daily", "weekly", "monthly", "manual"] = "weekly"


class RiskConfig(BaseModel):
    max_drawdown_stop: float = 0.20
    daily_loss_stop: float = 0.03
    kill_switch: bool = True


class ExecutionConfig(BaseModel):
    mode: Literal["backtest", "paper", "live"] = "paper"
    live_confirm_each_order: bool = True


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """
    Parsed representation of config/learntrade.toml.
    Loaded once at startup by ``load_config``; treated as immutable thereafter.
    """

    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    markets: dict[str, MarketConfig] = Field(default_factory=dict)
    markets_enabled: list[str] = Field(default_factory=lambda: ["XNYS"])
    universe: dict[str, UniverseEntry] = Field(default_factory=dict)
    data: DataConfig = Field(default_factory=DataConfig)
    horizons: HorizonsConfig = Field(default_factory=HorizonsConfig)
    quant: QuantConfig = Field(default_factory=QuantConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)


# ---------------------------------------------------------------------------
# Environment variables (secrets only)
# ---------------------------------------------------------------------------


class AppSecrets(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LEARNTRADE_", env_file=".env")

    fmp_api_key: str = ""
    polygon_api_key: str = ""
    tiingo_api_key: str = ""
    eodhd_api_key: str = ""
    alpha_vantage_api_key: str = ""
    fred_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str = ""
    sec_user_agent: str = ""
    data_dir: str = "./data"
    config_file: str = "./config/learntrade.toml"
    port: int = 8787
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _migrate_legacy_names() -> None:
    """
    Accept settings written when the project was called "leantrade":
    LEANTRADE_* environment variables and LEANTRADE_* lines in .env are mapped to LEARNTRADE_*
    (an explicitly set LEARNTRADE_* value always wins).
    """
    import os

    for key, value in list(os.environ.items()):
        if key.startswith("LEANTRADE_"):
            os.environ.setdefault("LEARNTRADE_" + key[len("LEANTRADE_"):], value)
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("LEANTRADE_") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault("LEARNTRADE_" + k[len("LEANTRADE_"):], v.strip().strip("\"'"))


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load and validate learntrade.toml; fall back to defaults if absent."""
    _migrate_legacy_names()
    secrets = AppSecrets()
    path = config_path or Path(secrets.config_file)
    if not path.exists():
        legacy = path.with_name("leantrade.toml")   # pre-rename config file name
        if legacy.exists():
            path = legacy

    if not path.exists():
        return AppConfig()

    with open(path, "rb") as fh:
        raw: dict[str, Any] = tomllib.load(fh)

    # Promote top-level markets dict and enabled list
    markets_raw: dict[str, Any] = raw.pop("markets", {})
    markets_enabled: list[str] = markets_raw.pop("enabled", [])
    markets: dict[str, MarketConfig] = {
        mic: MarketConfig.model_validate(cfg) for mic, cfg in markets_raw.items()
    }

    return AppConfig(
        markets=markets,
        markets_enabled=markets_enabled,
        **{k: v for k, v in raw.items() if k != "markets"},
    )


_config: AppConfig | None = None
_secrets: AppSecrets | None = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_secrets() -> AppSecrets:
    global _secrets
    if _secrets is None:
        _secrets = AppSecrets()
    return _secrets
