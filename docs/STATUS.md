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

Last reviewed against the code on 2026-11-20 (updated after fundamentals).

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
| Company financials (US filings) | SEC EDGAR company facts for US stocks (needs `LEARNTRADE_SEC_USER_AGENT`): true filing dates, ten or more years of annual and quarterly statements, first-reported values; stock splits from Yahoo keep per-share figures in the same units as prices. Preferred over Yahoo for any stock it covers | Unit tests; live run (see note below) |
| Company financials | Yahoo annual and quarterly statements (revenue, profit, EPS, equity, debt, cash flow) stored point-in-time: a statement counts as public 60 days (quarterly) or 90 days (annual) after its period ends, because the real filing date is not provided. Nine features: earnings yield, book/price, sales/price, return on equity, gross margin, cash flow/assets, debt/assets, revenue growth, profit growth. Shown on each stock's page and used automatically in training once stored | Unit tests; live run on 30 NSE stocks (see note below) |
| Index membership over time | Training and testing use "who was in the index on each date" when a membership history exists; S&P 500 join dates are built in (`universe fetch-index SP500`), any market via CSV (`--date-column` / `--added-column`) | Unit tests; live run on the S&P 500 |
| MCP server (`learntrade mcp`) | Read-only tools (`status`, `rank`, `stock`, `reliability`, `evaluate`) over stdio for Claude Code, Codex and other MCP clients | Unit tests over a real stdio connection; tried with the MCP client library on real NSE and S&P 500 data |
| Evidence gate | Outlook, probability and "would rank" are withheld when testing found no edge, halved for an early signal; funds/ETFs/indices and stocks unlike the training list are never scored (price facts only). Same rules in the web page, MCP tools and terminal | Unit tests; live run on NSE with small caps and gold/silver ETFs |
| Portfolio import and analysis | Zerodha/Groww/generic CSV and Excel holdings (header auto-detection, several sheets and files); holdings resolved via Yahoo search (ISIN, symbol, name) and AMFI scheme list; prices from Yahoo, fund NAV history from mfapi.in; value, P&L, allocation by asset class/sector/size, concentration, last-year risk of the current mix, correlated pairs, split/bonus adjustment of stale files. Web page, CLI and MCP tool | Unit tests; live run on a 10-holding sample (stocks, gold/silver ETFs, two funds) |
| Nifty 500 and size groups | `universe fetch-index NIFTY500` (also NIFTY100, NIFTYMIDCAP150, NIFTYSMALLCAP250) from NSE's free lists, today's members only; "Use the Nifty 500" button | Live download (501 members); unit tests |
| Company size feature | log market value (shares x price) added to the financial features (feature set v4) | Unit tests |
| Evaluation sweep (`learntrade sweep`) | Every group x horizon x feature set x model judged by one fixed rule (CV rank IC >= 0.02, adjusted t >= 2, untouched period > 0), with the number of passes expected by chance; never replaces saved models | Unit tests; live run on the Nifty 500 (see note below) |
| Model file safety | Model and calibrator files record library versions; a mismatch is treated as "needs retraining" instead of loading a possibly broken model | Unit tests |
| LLM spending cap | The daily cap is kept in `data/llm_spend.json`, so it holds across restarts and terminal commands | Unit tests |
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

- Looking inside mutual funds (their monthly stock holdings), so sector and size cover fund holdings too
- Historical profit/loss from a tradebook (only current holdings are imported)

- Browser setup wizard and Settings pages (the old `/api/setup/*` routes are placeholders)
- Fundamentals from FMP or Polygon (stubs that return nothing); a source with full filing history and true filing dates for non-US markets (non-US stocks use Yahoo, about three years)
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

## Note on fundamentals (measured 2026-11-20)

Yahoo's free statements go back about 4 annual and 5 quarterly periods, so a stock has roughly three
years of usable history. On 30 NSE stocks (rank IC in cross-validation / on the untouched final test):

| Features | Ridge | LightGBM |
|---|---|---|
| Price only | 0.00 / +0.14 | 0.00 / +0.03 |
| Price + all nine financial features | +0.01 / +0.07 | +0.03 / +0.03 |

The difference is inside the noise on both models, so treat fundamentals as **context plus a small
possible lift, not a proven edge**. Statements change only once or twice a year, so with few years
of history the model sees them mostly as a fixed fingerprint of each company. A deeper history
(US filings via SEC EDGAR company facts carry true filing dates back decades) is the natural next step.

On the full S&P 500 (503 stocks, SEC filings with true filing dates, 2023-09 to 2026-10; untouched
test window 2026-05-04 to 2026-10-22) rank IC moved from price-only to price plus financials:
ridge -0.078 to -0.071, LightGBM -0.053 to -0.046, LightGBM ranker -0.049 to -0.024. All three
shift toward zero but stay inside the error range, so there is still no measured edge on this
group from these inputs.

## Nifty 500 sweep (measured 2026-11-23)

499 Nifty 500 stocks, prices from June 2018, today's index lists (companies that left are
missing). 20 configurations: 4 groups (whole list, Nifty 100, Midcap 150, Smallcap 250) x
2 horizons (1 and 3 months) x price-only / price + financials, ridge; plus LightGBM on the whole
list. Reports: `research/reports/20261123_173725_XNSE_sweep.md` and `..._174927_...`.

**0 of 20 passed** the fixed rule (about 0.5 would pass by chance).

- Adding financials raised the cross-validation rank IC in 9 of 10 pairs, and three
  configurations cleared the CV bar on their own (Smallcap 250, 1 month: +0.052, t 3.05;
  Midcap 150, 1 month: +0.041, t 2.22). But all three were negative on the untouched last
  six months (-0.074 and -0.017), so none passes.
- Price-only signals show nothing in any group.
- Financials from the free source start around 2022, so the earliest test periods ran without
  them. A longer financial history, and repeating the untouched test on several periods instead
  of one, are what it would take to tell a real effect from a lucky stretch.
