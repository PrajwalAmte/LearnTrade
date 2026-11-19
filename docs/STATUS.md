# What exists, what is experimental, what is only a plan

[docs/TECHNICAL.md](TECHNICAL.md) describes the **target design**. This page says which parts of that design
actually exist today. If the two disagree, this page is right.

How to read the columns:

- **Implemented and tested**: code exists, has automated tests, and has been run on real data (or, for
  pure logic, covered by tests).
- **Implemented, experimental**: code exists but has only been tested with fakes, has not been run end to
  end on real data, or is not connected to the main flow yet.
- **Designed, not implemented**: described in the design document; no working code.
- **Planned**: ideas, not designed in detail.

Last reviewed against the code on 2026-11-19 (updated after point-in-time membership).

## Implemented and tested

| Area | What works | How it was checked |
|---|---|---|
| Point-in-time data store | Data is stored with the time it became known; queries cannot see the future | Unit and leakage tests |
| Prices | Yahoo Finance daily prices for any market with a profile; the market's benchmark index is downloaded too | Live run on NSE (30 stocks); unit tests |
| Universe | Import a stock list from a CSV or typed list | Unit tests, live run |
| Market Packs | Exchange profiles (calendar, currency, lot and tick size, cost model) for NSE, BSE, NYSE, NASDAQ, LSE, XETRA, TSE, ASX | Pack validation tests |
| Features | Returns, momentum, volatility, volume surprise, liquidity, distance from 52-week high; each normalised across stocks on the same date so models learn "stronger than the others today" | Unit tests; better out-of-sample ranking on the NSE test |
| Quant models | Ridge regression (live-tested), LightGBM regressor and LightGBM ranker (live-tested; on macOS they need `brew install libomp`) | Unit tests; live training |
| Validation | Purged walk-forward cross-validation, an untouched final test period, per-date rank IC, overlap-corrected t-stat, block-bootstrap ranges, isotonic calibration | Unit tests; live runs |
| Out-of-sample evaluation | `learntrade evaluate` compares momentum, ridge, LightGBM and the ranker on the same stocks, dates, costs and trading schedule | Unit tests; live run (ridge and momentum) |
| Cost model | Commission, taxes, exchange and regulatory fees, spread, market impact per exchange | Unit tests |
| Backtest engines | Vectorised and event-driven; both use the same cost model and trade on the forecast horizon | Unit tests, reconciliation test |
| Single-stock check | Score any stock, even outside your list, and see where it would rank | Live run (NSE) |
| Reliability report | Each trained model writes a report; the UI turns it into a plain-language verdict with reasons | Unit tests; live run |
| News headlines | RSS headlines stored point-in-time and shown per stock | Live run (NSE); unit tests |
| Index membership over time | Training and testing use "who was in the index on each date" when a membership history exists; S&P 500 join dates are built in (`universe fetch-index SP500`), any market via CSV (`--date-column` / `--added-column`) | Unit tests; live run on the S&P 500 |
| Web page | Demo mode, real-stock setup, ranking, stock page, update data, reliability panel, single-stock lookup | Tried in a browser |

## Implemented, experimental

| Area | State |
|---|---|
| GDELT news source | Written, unit-tested only; never run live |
| LLM evidence extraction (`news extract`) | Written and cached; tested with fake model responses only, never against a real model |
| LLM research (`analyze` when the LLM is switched on) | Orchestrator, fact sheet, roles; tested with fakes only |
| Evidence features in model training | Feature code exists but training does not use it yet (news is context only) |
| Portfolio construction and `rebalance` | Allocators (equal weight, inverse vol, HRP, mean-variance, risk parity), position caps; CLI only, not in the web page |
| Risk engine | Drawdown stop, daily loss stop, kill switch, position limits; unit-tested, not part of the web page |
| Decision history | Forecasts and decisions saved and queryable (`decision-history`); CLI only |
| Meta-combiner | Unit-tested; not connected to ranking or `analyze` |
| Live-order safety gates | Three-gate check exists; there is **no broker adapter**, so nothing can place a real order |
| SEC EDGAR filings | Provider written; needs a contact string; not used in any model |
| FRED macro data | Provider written; not used in any model |
| Terminal setup wizard (`learntrade init`) | Exists; not the main path |

## Designed, not implemented

- Browser setup wizard and Settings pages (the old `/api/setup/*` routes are placeholders)
- Fundamentals providers (FMP, Polygon are stubs that return nothing)
- Broker connections for placing real orders
- Auto-download of full membership history (including companies that LEFT an index) for markets other than via CSV; the S&P 500 download lists current members with join dates only
- Intraday data and intraday models
- Plugin marketplace beyond the provider entry-point hook
- Models trained across several markets at once

## Planned

- Social-media signals (only if they prove their worth in testing)
- Portfolio, backtest and decision-history screens in the web page
- A separate frontend workspace (React or similar) once the page outgrows one file
- Cloud or multi-user hosting (the tool is deliberately local-first)

## Known limits

- **Companies that left an index.** Join dates are built in for the S&P 500, but companies that already dropped out are not in that public table. Import a full membership CSV to include them.
- **One test window.** The final test is a single recent period; a different one could rank models differently.
- **Free prices.** Yahoo data can be late, adjusted differently, or missing for delisted symbols.
- **Removed placeholder routes.** The empty `/api/research/*` and `/api/data/*` routes and the fake wizard step handler were deleted; the web page uses `/api/app/*`.
