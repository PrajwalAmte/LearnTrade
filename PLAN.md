# LearnTrade — Implementation Plan

Six phases, sequenced so each is validated before the next adds complexity.
No phase is complete until its exit criteria are met and checked into the repository.

---

## Phase 1 — Foundation (M0 + M1)

**Goal:** Every subsequent phase can build on a correct, leakage-free data layer.

### Deliverables

| Area | Tasks |
|---|---|
| **Domain contracts** | Finalise `InstrumentId`, `Instrument`, `Observation`, `Forecast`, `Order`, `DecisionRecord` Pydantic models. Freeze `FetchRequest` and `DataProvider` protocol. Write contract test suite (`tests/contracts/`). |
| **Market Packs** | Complete `generic.yaml`. Validate and ship `XNYS`, `XNAS`, `XNSE`. Add `leantrade market validate` logic: calendar vs. price-history check, tick/lot rounding test, cost-model sanity assertions. |
| **PIT store** | Full DuckDB + Parquet schema: observations table, raw store (append-only Parquet partitioned by source/year/month), app-state DB (watchlists, runs, attribution). Enforce `available_at ≤ as_of` on every query path. |
| **Data providers** | Yahoo Finance adapter (EOD bars, `pit_level = "none"`). SEC EDGAR adapter (filings, `pit_level = "full"`). FRED macro adapter (`pit_level = "full"` via ALFRED vintages). Provider fallback and retry logic in the data engine. |
| **Quality checks** | Implement all checks in `data/quality/checks.py`: leakage, missing bars vs. calendar, duplicate observations, TZ/session-boundary errors, survivorship flag. `leantrade data check` command wired up end-to-end. |
| **Corpus** | Parquet + DuckDB FTS store for filings and news. Provenance fields: `doc_hash`, `available_at`, `licence_tag`. |

### Exit criteria
- Leakage test suite passes on a pulled dataset for at least one market.
- Survivorship-aware universe for XNYS (S&P 500 point-in-time membership).
- `leantrade data pull && leantrade data check` runs clean.
- All contract tests green.

---

## Phase 2 — Quant Baseline (M2)

**Goal:** A working, calibrated forecast pipeline with honest out-of-sample evaluation.

### Deliverables

| Area | Tasks |
|---|---|
| **Feature engine** | Price/volume features (returns, momentum, reversal, volatility, Amihud liquidity, volume surprise). Cross-sectional ranks and z-scores within market and sector. Fundamental features with `available_at` lag enforcement. Event features (days to/since earnings, dividends, index changes). Feature-set versioning so every model records the exact version it was trained on. |
| **Horizon targets** | Horizon label generator for `swing_5d`, `swing_20d`, `positional_60d`. Forward-return computation with calendar-aware bar shifts. Excess-return over configurable benchmark. |
| **Models** | Ridge regression (baseline). LightGBM return model. LightGBM LambdaRank cross-sectional ranker. EWMA volatility model. Plug `ForecastModel` protocol; register via entry points. |
| **Calibration** | Isotonic regression calibration for `p_positive`. Platt scaling alternative. Conformal prediction intervals for quantiles. Calibration-curve and Brier-score reporting. |
| **Validation** | Purged walk-forward splitter with configurable embargo. Multiple random seeds. Block-bootstrap confidence intervals on IC and Sharpe. Deflated Sharpe ratio for multiple-testing correction. |
| **Vectorised backtest** | Fast vectorised engine sharing Market Pack cost functions. IC, rank IC, directional accuracy, ROC-AUC, calibration curves. HTML + Markdown report output to `research/reports/`. |
| **CLI wiring** | `leantrade train --horizon`, `leantrade backtest`, `leantrade rank` fully functional. |

### Exit criteria
- Out-of-sample IC and calibration reports published for XNYS swing and positional horizons.
- Backtest report includes full cost drag, survivorship note, PIT-level annotation.
- Ridge baseline IC is statistically significant (t-stat > 2) on the held-out period.

---

## Phase 3 — Execution Realism (M3)

**Goal:** One event-driven simulator shared by backtest, paper trading, and (eventually) live.

### Deliverables

| Area | Tasks |
|---|---|
| **Event-driven engine** | Next-bar fills (never same-bar). Spread model: quotes when available, volatility-based otherwise. Square-root market impact. Participation limits. Partial fills, halts, price-limit days (no fill). |
| **Cost model** | Full Market Pack cost pipeline: commission (flat / bps / flat-or-bps), side-specific taxes, exchange and regulatory fees, VAT/GST on fees. Broker cost overrides from `leantrade.toml`. |
| **Market microstructure** | Lot and tick rounding per Market Pack. Settlement cycles and cash availability modelling (T+0/T+1/T+2). Half-days and holiday calendar integration via `exchange_calendars`. |
| **Corporate actions** | Split and dividend adjustment. Merger and delisting handling. Survivorship-safe universe at each step. |
| **Multi-currency** | FX conversion into base currency. Per-position currency tracking. P&L attribution by currency. |
| **Reconciliation** | Automated check that event-driven and vectorised results agree within configurable tolerance on the same input. Any discrepancy fails CI. |
| **Paper trading mode** | Live data feed → event-driven engine → proposed orders (no submission). Portfolio state persisted in app-state DB. |

### Exit criteria
- Event-driven and vectorised results reconcile within 5 bps CAGR on the XNYS backtest from Phase 2.
- `leantrade backtest` report labels results as "event-driven" or "vectorised" and shows cost breakdown.
- Paper trading starts cleanly on `leantrade serve` with no live data yet configured.

---

## Phase 4 — LLM Evidence Layer (M4)

**Goal:** Optional, cached LLM evidence features that the quant engine can consume; contamination-safe.

### Deliverables

| Area | Tasks |
|---|---|
| **LLM clients** | OpenAI-compatible thin client (async, retries via `tenacity`). Anthropic native client (prompt caching, batch API). Ollama / llama.cpp local client. Provider selection and fallback chain via `llm.router`. |
| **Response cache** | SQLite/DuckDB cache keyed on `(prompt_hash, model, parameters)`. Cache hit returns immediately; zero cost on reruns. TTL and manual invalidation. |
| **Evidence extraction** | `ExtractedEvidence` schema with quote-span citations. Deterministic verifier: each numeric claim checked against source-text span. Extraction keyed on `(doc_hash, model_id, prompt_version)`. Multilingual support where the model supports it. |
| **Contamination guard** | Model training-cutoff registry. `is_contaminated(model_id, as_of)` enforced on every LLM signal before it enters a training row. Masking mode: anonymise ticker, name, date, rescaled prices. Backtest report flags contaminated signals and excludes them from "validated" metrics. |
| **LLM eval suite** | Evaluation tasks: financial numerical reasoning, extraction accuracy, contradiction detection, structured-output validity, citation accuracy, cost per task, latency. `leantrade llm eval --task extract --models ...` wired up. |
| **Feature bridge** | `guidance_change`, `event_type`, `management_tone`, `risk_flags`, `surprise_direction` as typed features fed into the quant feature engine with `available_at` from the source document. |
| **Budget enforcement** | Per-run and per-day hard limits. Clear log message when a budget is hit; quant engine continues without LLM signals. |

### Exit criteria
- Ablation report shows whether LLM evidence features add statistically significant IC versus Phase 2 quant-only baseline.
- Cache hit rate on a re-run of the same backtest is 100%.
- Contamination guard blocks all LLM signals for dates within `cutoff + safety_margin` in backtest mode.

---

## Phase 5 — Research Engine + Meta-Combiner + Portfolio (M5 + M6)

**Goal:** Full decision pipeline — from ranked candidates through LLM research, signal combination, and risk-constrained allocation.

### Deliverables

| Area | Tasks |
|---|---|
| **Fact sheet builder** | Deterministic Python builder: indicators, ratios, deltas, recent extracted evidence. LLMs receive only this sheet, never raw data. |
| **Research roles** | Implement all nine roles: technical, fundamental, macro, event, bull, bear, verifier, quant challenger, synthesiser. Each is a `ResearchRole` with system prompt, evidence scope, and JSON output schema. |
| **Orchestration state machine** | Typed state machine (plain Python, no LangChain). Steps: fact sheet → independent analysis (parallel) → verification → conditional escalation → synthesis. Escalation triggers: quant disagreement, uncertainty above threshold, material events, large position changes. Self-consistency sampling (N=3 by default). |
| **Meta-combiner** | Static weights (baseline). Rolling out-of-sample IC weighting. Regime-conditioned weighting. Stacked meta-model on out-of-fold predictions. Contextual bandits for routing (optional, ablation-gated). Combiner trains only on data available at each historical decision time. |
| **Allocators** | Complete `MeanVarianceAllocator` (shrinkage covariance). `HRPAllocator` (hierarchical risk parity). `RiskParityAllocator`. `CVaRAllocator` (via `cvxpy`). All benchmarked against equal-weight baseline. |
| **Constraints** | Maximum position, sector, country, currency, liquidity (% of ADV), turnover, minimum trade size. Lot and tick rounding. Long-only and long-short per-market rules. Horizon sleeves with separate risk budgets. |
| **Risk engine** | Maximum drawdown stop. Daily loss stop. Exposure and leverage caps. Concentration and correlation checks. Kill switch: `RiskEngine.engage_kill_switch()` stops all order generation. |
| **Attribution store** | Write `DecisionRecord` after each decision. Append realised return once the horizon elapses. `leantrade decision-history` command. Memory retrieval (ablation-gated): similarity search over past records and verified evidence. |
| **Ablation ladder** | Automated runner for all eight rungs. Each rung must add incremental IC or net Sharpe to remain enabled by default. Report published to `research/reports/`. |

### Exit criteria
- Ablation ladder results published for XNYS swing horizon.
- Portfolio-level net CAGR, Sharpe, max drawdown reported with block-bootstrap confidence intervals.
- Kill switch tested: engaging it during a paper-trading run produces zero further order emissions.
- LLMs explain allocations in `DecisionRecord.thesis`; weights are set only by the optimiser.

---

## Phase 6 — Product & Expansion (M7 + M8)

**Goal:** A non-technical user goes from Download ZIP to their first ideas in under 15 minutes, on any platform.

### Deliverables

| Area | Tasks |
|---|---|
| **Launchers** | `Start LeanTrade.bat` (Windows). `Start LeanTrade.command` (macOS). `start.sh` (Linux). Each: installs `uv` if absent (no admin rights), syncs deps, starts server, opens browser. Progress displayed in a terminal window; errors logged to `logs/launcher.log`. |
| **Browser setup wizard** | 8-step wizard served by the FastAPI app. Step 1: risk notice. Step 2: country/exchange picker (auto-fills Market Pack). Step 3: currency. Step 4: data sources with Test button. Step 5: AI choice (None / Local / Cloud) with RAM check and Ollama install offer. Step 6: investor profile and risk level picker backed by `presets/`. Step 7: holdings import (CSV/XLSX with column-detection preview). Step 8: first data pull with progress bar; demo mode available while pulling. |
| **Simple mode UI** | Home (top idea cards with plain-language confidence, reasons, risks, invalidation). My portfolio (health check: concentration, sector, currency flags). Suggested changes (proposed trades with estimated fees, taxes, plain explanation). Watchlist with change alerts. Research report (bull/bear/evidence). Practice trading (paper portfolio). Learn (glossary + how-it-works). "What does this mean?" button on every number. |
| **Advanced mode UI** | Strategy lab (horizons, models, features, allocators, constraints). Backtests (run + download report). AI settings (roles, models, escalation rules, budgets). Model reports (IC, calibration, ablation). Decision history (attribution explorer). Broker connections (read-only). Live trading toggle (behind confirmation + kill switch UI). |
| **Doctor / updater / backups** | `leantrade doctor`: checks install, internet, data sources, API keys, disk space, AI. Each issue has a plain-language explanation and a Fix button where automatable. `leantrade update`: `git pull` for clones, ZIP download for ZIP installs; backs up settings/data before updating. Backup/restore: single `.zip` of settings, portfolios, watchlists, decision history. API keys excluded by default (opt-in, encrypted with password). |
| **Demo mode** | Bundled sample data (1 year, 50 instruments, synthetic). Fully functional without any data pull. Clearly watermarked "Demo data — not real". |
| **Intraday horizon** | Extend feature engine for intraday bars (liquidity, spread, session effects). Train intraday model. Add to execution engine and UI (Advanced mode only). |
| **Additional markets** | Community-contributed packs: `XLON`, `XETR`, `XTKS`, `XASX`, `XBOM`. Each validated with `leantrade market validate`. |
| **Additional providers** | Tiingo, Polygon, EODHD price adapters. FMP fundamentals. Stooq (EOD, free). DBnomics / FRED macro. GDELT news. |
| **Additional brokers** | Interactive Brokers read-only adapter. Zerodha read-only adapter. Generic CSV/XLSX import wizard with custom column mapping. |
| **Optional live execution** | `execution.mode = "live"` gated behind explicit config flag + UI toggle. Per-order confirmation dialog. Notional and position limits before submission. Active kill switch required. Audit log of every submitted order. |
| **i18n** | Interface translation infrastructure. First translations: English (default), Hindi, German. Number / currency / date locale formatting. |
| **Desktop shell (optional)** | Tauri 2 wrapper around the same React UI. Produces `.exe`, `.dmg`, `.AppImage` installers. No new features; same engine, different distribution format. |

### Exit criteria
- Usability test: a non-technical tester completes setup (Download ZIP → first ideas) in under 15 minutes on Windows, macOS, and Linux, without opening a terminal.
- `leantrade doctor` correctly identifies and proposes fixes for the five most common setup problems.
- Live execution requires three explicit opt-ins (config flag + UI toggle + per-order confirmation) and cannot be triggered accidentally.
- All Phase 1–5 test suites remain green after Phase 6 changes.
