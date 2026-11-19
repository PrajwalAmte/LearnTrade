# LearnTrade

**A free tool that looks at a list of stocks and tells you which ones look relatively stronger or weaker right now, and how sure it is.**

It runs on your own computer. It does not buy or sell anything. It does not need an account, a subscription or a credit card.

*LearnTrade gives information, not financial advice.*

---

## What it does, in plain words

1. You give it a list of stocks you care about (for example, 10 to 50 companies).
2. It downloads their past prices (for free) and recent news headlines.
3. It looks for patterns in the price history, tests whether those patterns would have worked in the past, and learns from them.
4. It shows you a ranked list: strongest outlook at the top, weakest at the bottom, with a short plain-English explanation for each stock.

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

In the USA you can skip typing: click **Use the whole S&P 500**. It loads all ~500 companies together with the date each one joined the index (a few minutes longer the first time).

### The main screen

A ranked list of your stocks. Click any stock to see its details and recent news.

| Word on screen | What it means |
|---|---|
| **Outlook** | How much better or worse than the overall market this stock is expected to do over the chosen period. `+2%` means "expected to beat the market by about 2%". |
| **Chance it rises** | The estimated probability that the stock ends higher. 53% is only slightly better than a coin flip, and that is typical. Markets are hard. |
| **Bumpiness** | How much the stock tends to jump around. Higher means a rougher ride. |
| **Typical range** | Where most outcomes are expected to land. It is usually wide. That is the honest picture. |
| **Looking ahead** | Choose about 1 week, 1 month or 3 months. |

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

Open **Why, and the numbers behind it** to see the details. The most useful number is the **final untouched test**: the most recent months are kept aside and never used for learning, so it shows how the model did on data it had not seen.

### Stock lists and history

When a stock list is today's list, companies that dropped out of an index in the past are missing from the history, which can make old results look a little rosier than they were. LearnTrade handles this for you where it can:

- **S&P 500:** the **Use the whole S&P 500** button (or `uv run leantrade universe fetch-index SP500`) loads every member with its join date, so a stock is only used from the day it actually joined.
- **Any market:** if you have a file of index membership over time, import it with `uv run leantrade universe import file.csv --mic XNSE --date-column date` (one row per date and symbol), or with `--added-column` if it lists the date each stock joined. LearnTrade then uses "who was in the index on that date" for training and testing.
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

## News

When you set up real stocks, LearnTrade also fetches recent headlines for each one (free sources). They appear on each stock's page. Headlines mentioning earnings, dividends or similar events are flagged for extra attention in the optional AI research step. News is not yet part of the numbers behind the ranking; it is there to give you context.

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

For a health check, open a terminal in the folder and run `uv run leantrade doctor`. It explains any problems in plain language.

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
uv run leantrade --help          # all commands
uv run leantrade news pull       # fetch headlines
uv run leantrade rank            # ranked list in the terminal
uv run leantrade evaluate --market XNSE   # fair, out-of-sample comparison of all models
uv run pytest                    # run the test suite
```

Note that parts of the technical document describe the long-term plan; the browser screens described above are what is available today.

## License

Apache License 2.0 is proposed. A `LICENSE` file has not been added to the project yet.
