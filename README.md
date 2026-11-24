# LearnTrade

**A free tool that looks at a list of stocks and tells you which ones look relatively stronger or weaker right now, and how sure it is.**

It runs on your own computer. It does not buy or sell anything. It does not need an account, a subscription or a credit card.

*LearnTrade gives information, not financial advice.*

---

## What it does, in plain words

1. You give it a list of stocks you care about (for example, 10 to 50 companies).
2. It downloads their past prices (for free) and recent news headlines.
3. It also reads each company's financials (earnings, valuation, profitability, growth) when they are available.
4. It looks for patterns in the price history and financials, tests whether those patterns would have worked in the past, and learns from them.
5. It shows you a ranked list: strongest outlook at the top, weakest at the bottom, with a short plain-English explanation for each stock.

It works for **any country's stock market** that has a market profile (India, USA, UK, Germany, Japan, Australia and more are included).

## What it is not

- It does not trade for you or touch your broker account. You stay in control.
- It does not read social media. News is headlines only.
- It does not send your data anywhere. Everything stays in the `data` folder on your computer.

**What is finished and what is only planned?** See [docs/STATUS.md](docs/STATUS.md). It lists every feature as tested, experimental or not built yet, so you are never promised something that does not exist.

---

## Install (3 steps, no coding)

You need a Windows, Mac or Linux computer, about 2 GB of free space and an internet connection the first time.

### Step 1: Download

On the project's GitHub page, click the green **Code** button, then **Download ZIP**. Unzip it anywhere (your Documents folder is fine).

### Step 2: Double-click the starter for your computer

| Your computer | Double-click this file | If it says "blocked" or "unknown developer" |
|---|---|---|
| Windows | `Start LearnTrade.bat` | Click **More info**, then **Run anyway** |
| Mac | `Start LearnTrade.command` | Right-click the file, choose **Open**, then **Open** again (only the first time) |
| Linux | `start.sh` | Right-click, Properties, allow running as a program |

The first start takes a few minutes while it sets itself up. You will see a window with text scrolling; that is normal. Later starts take a few seconds.

### Step 3: Your web browser opens by itself

If it does not, type `http://localhost:8787` in your browser's address bar.

To stop LearnTrade, close the black starter window.

---

## Using it

### First time: pick one of two options

**Try the demo.** Click **Start the demo**. It creates 50 made-up stocks so you can look around safely. It takes about a minute and needs no internet. The page shows a "Demo" badge so you never mistake it for real data.

**Use real stocks.**
1. Choose your country's stock market.
2. Type the stock symbols you care about, at least 8, separated by commas. Use the short code, like `RELIANCE, TCS, INFY, HDFCBANK` for India or `AAPL, MSFT, GOOGL, AMZN` for the USA.
3. Click **Download data and set up** and wait. The first time takes a few minutes (it downloads three years of prices and trains itself).

In India you can skip typing: click **Use the Nifty 500**. It covers large, mid and small companies, so the model learns from companies like the ones most people actually hold, not only the biggest 30 or 50. It takes 10 to 15 minutes the first time.

In the USA you can skip typing: click **Use the whole S&P 500**. It loads all ~500 companies together with the date each one joined the index (a few minutes longer the first time).

### The main screen

A ranked list of your stocks. Click any stock to see its details, key financials (P/E, return on equity, growth) and recent news.

| Word on screen | What it means |
|---|---|
| **Outlook** | How much better or worse than the overall market this stock is expected to do over the chosen period. `+2%` means "expected to beat the market by about 2%". |
| **Chance it beats the market** | The estimated probability that the stock does better than the overall market (the benchmark index) over the period. 53% is only slightly better than a coin flip, and that is typical. In a falling market a stock can beat the market and still lose money. |
| **Bumpiness** | How much the stock tends to jump around. Higher means a rougher ride. |
| **Typical range** | Where most outcomes are expected to land. It is usually wide. That is the honest picture. |
| **Looking ahead** | Choose about 1 week, 1 month or 3 months. |

### My portfolio

Click **My portfolio** (top right), or **Analyse my portfolio** on the first screen, and upload the holdings file from your broker. You do not need to set up a stock list first.

- **Which files:** Zerodha (Kite holdings CSV, or the Console holdings Excel file with its stocks and mutual-funds sheets), Groww (stocks and mutual funds statements), or any CSV or Excel file with a name, symbol or ISIN column and a quantity column. You can upload a stocks file and a funds file together.
- **What it does:** finds each holding (NSE/BSE symbols by ISIN, symbol or name; mutual funds through AMFI's official scheme list), downloads prices and daily fund NAVs, and shows:
  - value today, what you paid, and profit or loss, per holding and in total
  - what you own: Indian stocks, equity funds, gold, silver, debt, international
  - your directly held stocks by sector and by company size (large, mid, small)
  - concentration: your biggest holding, your top five, and how many equal-sized holdings the portfolio behaves like
  - the last year for your current mix: return, how much it swings, its worst fall, and holdings that move almost in step
  - **Worth knowing:** plain facts such as "Financial Services is 65% of your stocks" or "this holding is 46% below what you paid". These describe your portfolio; they are not advice to buy or sell.
- **Splits and bonuses:** if your file was exported before a split or bonus (for example HDFC Bank's 1:1 bonus in August 2025), the quantity and cost are put on today's basis automatically, and the page says so.
- **Not yet:** LearnTrade cannot see inside mutual funds, so the sector and size breakdown covers stocks you hold directly. Click **Update prices** to refresh.

From a terminal: `uv run learntrade portfolio import holdings.csv funds.xlsx`, then `uv run learntrade portfolio show`.

### Check any single stock

Not in your list? Type its short symbol into **Check any single stock**. LearnTrade fetches its prices and shows where it would rank compared with your list (for example "would rank 12 of 31").

### How well it has worked

Above the list, a box says how the model did when it was tested on past data, in plain words:

| Label | What it means |
|---|---|
| **Looks useful** | The ranking beat chance in testing, and the edge is bigger than the noise |
| **Early signal** | The ranking leans the right way; more history will show how solid it is |
| **Not working here** | For this group of stocks the ranking did not beat chance. More stocks or a longer history often helps |
| **Quick estimate** | No trained model yet; the order reflects recent price moves. Tap **Update data** to train one |

**Numbers only appear when they are earned.** If testing found no edge ("Not working here" or "Quick estimate"), LearnTrade does not show an outlook, a "chance it beats the market" or a "would rank N of M". You still see your list in the model's score order, how much each stock moves, its financials and news, and the page says plainly why the forecast is hidden. With an "Early signal" the numbers are shown but toned down by half; with "Looks useful" they are shown in full.

Two more cases never get a score:

- **Funds and indices** (gold or silver ETFs, index funds): they are not companies, so they get measured facts only (returns, volatility, worst fall from a high).
- **Stocks unlike your list**: when you check a stock that is much smaller or much more volatile than every stock the model learned from (for example a small-cap checked against a list of large caps), it says so and does not score it.

Open **Why, and the numbers behind it** to see the details. The most useful number is the **final untouched test**: the most recent months are kept aside and never used for learning, so it shows how the model did on data it had not seen.

### Stock lists and history

When a stock list is today's list, companies that dropped out of an index in the past are missing from the history, which can make old results look a little rosier than they were. LearnTrade handles this for you where it can:

- **S&P 500:** the **Use the whole S&P 500** button (or `uv run learntrade universe fetch-index SP500`) loads every member with its join date, so a stock is only used from the day it actually joined.
- **Any market:** if you have a file of index membership over time, import it with `uv run learntrade universe import file.csv --mic XNSE --date-column date` (one row per date and symbol), or with `--added-column` if it lists the date each stock joined. LearnTrade then uses "who was in the index on that date" for training and testing.
- The box above the list tells you which basis was used.

### Keeping it fresh

Click **Update data** now and then (for example once a week). It fetches the latest prices and news and re-trains. The demo cannot be updated because its data is made up.

---

## Getting the most from it

- **Use it to compare.** The ranking shows which stocks look stronger than the others in your list, which is a good way to build a shortlist.
- **Use 30 or more stocks.** Comparisons get steadier with a bigger list, and the "How well it has worked" box will tell you when it is on solid ground.
- **Look at the typical range** as well as the outlook. It shows how much a stock can swing either way.
- **Refresh weekly.** **Update data** keeps prices and news current and re-trains the model.

---

## Company financials

When you set up real stocks, LearnTrade downloads each company's annual and quarterly results (free source) and uses valuation, profitability and growth alongside the price patterns. Each stock's page shows the main numbers. A result only counts from a safe delay after the period ends (60 to 90 days), so the tool never uses a number before the market could have known it. Outside the USA the free source covers about three years, so there it adds context and a small boost, not a guarantee.

**Deeper history for US stocks (recommended).** The US regulator, the SEC, publishes every company's filings for free, with the exact date each one came out and ten or more years of history. To use it, tell the SEC who is asking (they require a name and email so they can contact you if your computer sends too many requests). Add this line to your `.env` file (copy `.env.example` to `.env` first):

```
LEARNTRADE_SEC_USER_AGENT=Your Name your.email@example.com
```

Then press **Update data** (or run `uv run learntrade data pull --market XNYS --capabilities fundamentals,corporate_actions`). Your email is only sent to the SEC in the request header, nowhere else. Without this line LearnTrade quietly uses the three-year free source instead. Stock splits are taken into account, so per-share numbers line up with prices.

## News

When you set up real stocks, LearnTrade also fetches recent headlines for each one (free sources). They appear on each stock's page. Headlines mentioning earnings, dividends or similar events are flagged for extra attention in the optional AI research step (see [Connect an AI model](#connect-an-ai-model-optional)). News is not yet part of the numbers behind the ranking; it is there to give you context.

---

## Connect an AI model (optional)

The ranking, the numbers and the web page work **without any AI account**. Connecting an AI model adds two extras that you run from a terminal (the web page does not use it yet):

- **A written second opinion on one stock:** `uv run learntrade analyze TCS --horizon swing_20d` (use your own symbol). The AI reads the numbers and recent headlines and writes a short case for and against.
- **Reading headlines for key events** (earnings, dividends and similar): `uv run learntrade news extract`

This part is newer than the rest and has only been tested with stand-in responses, not a live AI service, so treat it as a preview.

### Step 1: Get a key (or use a local model)

| You have | What to use |
|---|---|
| An OpenAI key, or any service that copies OpenAI's format (OpenRouter, Azure, LM Studio, vLLM) | provider `openai_compatible` |
| An Anthropic (Claude) key | provider `anthropic` |
| Nothing to pay for, and a computer that can run [Ollama](https://ollama.com) | provider `ollama` (no key needed; Ollama must be running) |

### Step 2: Put the key in the `.env` file

In the LearnTrade folder, copy `.env.example` to a new file named `.env` (the dot at the start is part of the name), open it in any text editor, and fill in the line for your service:

```
LEARNTRADE_OPENAI_API_KEY=sk-...your key...
# Only for services other than OpenAI itself, for example OpenRouter:
LEARNTRADE_OPENAI_BASE_URL=https://openrouter.ai/api/v1

LEARNTRADE_ANTHROPIC_API_KEY=...your key...
```

Keys stay on your computer. Never put them in `learntrade.toml`.

### Step 3: Switch it on and choose the model

Copy `config/learntrade.example.toml` to `config/learntrade.toml`, then edit the `[llm]` part:

```toml
[llm]
enabled = true
budget_usd_per_run = 1.00     # a spending cap per run
budget_usd_per_day = 5.00     # a spending cap per day

[llm.models.extract]          # reads headlines (cheap model is fine)
provider = "openai_compatible"
model    = "gpt-4o-mini"

[llm.models.research]         # writes the for/against case
provider = "openai_compatible"
model    = "gpt-4o"

[llm.models.escalation]       # a second look when the first one is unsure
provider = "openai_compatible"
model    = "gpt-4o"
```

Use the same provider for all three parts if you only have one key. The example file points the third part at Anthropic, so change it unless you have an Anthropic key too. For a local model, use `provider = "ollama"` and `model = "llama3.1"` (any model you have pulled).

### Step 4: Try it

```bash
uv run learntrade analyze TCS
```

If it says the AI is switched off, check that `enabled = true` is saved in `config/learntrade.toml`. If it reports a missing or rejected key, re-check `.env` (no spaces or quotes around the key). The budget caps stop it from spending more than you set.

---

## Use LearnTrade from Claude Code, Codex or another AI tool (MCP)

LearnTrade can run as an **MCP server**, which lets an AI coding assistant ask it questions directly ("which stocks in my list look strongest, and how reliable is that ranking?") instead of reading terminal text. It is read-only: it cannot place orders and cannot change your stock lists or models.

What the assistant can ask:

| Tool | What it returns |
|---|---|
| `status` | Which markets have data (and how fresh), which models are trained |
| `rank` | The ranked list for a market, strongest first, with the reliability verdict |
| `stock` | One company: forecast, key financials, recent headlines, reliability. A stock outside your list is only fetched if the assistant passes `download=True` |
| `reliability` | How well the ranking has worked in testing |
| `evaluate` | The out-of-sample comparison of all models (can take a few minutes on big lists) |
| `portfolio` | Your imported holdings: value, profit/loss, allocation, concentration, last-year risk and "worth knowing" facts |

Set up your data first with the web page (so there is something to ask about), then connect the assistant. Replace `/path/to/LearnTrade` with the folder you unzipped.

**Claude Code**

```bash
claude mcp add learntrade -- uv run --directory /path/to/LearnTrade learntrade mcp
```

**Codex**

```bash
codex mcp add learntrade -- uv run --directory /path/to/LearnTrade learntrade mcp
```

or add this to `~/.codex/config.toml`:

```toml
[mcp_servers.learntrade]
command = "uv"
args = ["run", "--directory", "/path/to/LearnTrade", "learntrade", "mcp"]
```

**Other MCP tools** (Cursor, Claude Desktop, and so on) take the same command: `uv run --directory /path/to/LearnTrade learntrade mcp`.

By default it reads the `data` folder inside the LearnTrade folder. To point it somewhere else, add `--data-dir /some/folder` after `learntrade mcp`. Then ask your assistant something like *"Use LearnTrade to show the ranking for XNSE and tell me how reliable it is."* If the web page is in the middle of updating data, a tool may say the data is busy; ask again in a minute.

---

## Something went wrong?

| Problem | What to do |
|---|---|
| The starter window closes instantly | Open the `logs` folder and read `launcher.log`, or ask for help (see below) and include that file |
| The browser did not open | Type `http://localhost:8787` in the address bar |
| It says the port is already in use | LearnTrade may already be running in another window. Close that one, or restart your computer |
| "Download data" fails | Check your internet connection and that the symbols are correct. Free data sources sometimes limit how fast you can download, so try again in a few minutes |
| It says it is busy updating | Wait a minute and refresh the page. It does one job at a time |
| A stock is missing | The symbol may be wrong. Check it on your broker's app or a finance website |
| Mac: "LightGBM needs the OpenMP library" | Open Terminal and run `brew install libomp`. The default model works without it |

For a health check, open a terminal in the folder and run `uv run learntrade doctor`. It explains any problems in plain language.

---

## Your data and privacy

- Everything is stored in the `data` folder next to the program. Delete the folder to erase everything.
- Prices and headlines are downloaded from free public sources. Only the stock symbols you typed are used to request them.
- Nothing about you or your stocks is sent to the developers.

---

## Words you may meet

| Term | Plain meaning |
|---|---|
| Ticker / symbol | The short code of a company on an exchange, like `TCS` |
| Market / exchange | The place where stocks trade, like the NSE in India |
| Benchmark | A reference index (such as Nifty 50). "Outlook vs. the market" is measured against it |
| Probability | A chance between 0% and 100% |
| Training | The tool studying past prices to learn patterns |

---

## For developers and the technically curious

The full technical documentation (architecture, command-line usage, adding a new country, configuration, testing methodology, roadmap) is in [docs/TECHNICAL.md](docs/TECHNICAL.md). A few useful commands:

```bash
uv run learntrade --help          # all commands
uv run learntrade news pull       # fetch headlines
uv run learntrade rank            # ranked list in the terminal
uv run learntrade evaluate --market XNSE   # fair, out-of-sample comparison of all models
uv run learntrade sweep --market XNSE --groups all,NIFTY100,NIFTYMIDCAP150,NIFTYSMALLCAP250
                                           # test many setups against one fixed pass rule
uv run pytest                    # run the test suite
```

Note that parts of the technical document describe the long-term plan; the browser screens described above are what is available today.

## License

Apache License 2.0 is proposed. A `LICENSE` file has not been added to the project yet.
