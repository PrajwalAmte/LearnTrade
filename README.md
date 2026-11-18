# LeanTrade

**A free tool that looks at a list of stocks and tells you which ones look relatively stronger or weaker right now, and how sure it is.**

It runs on your own computer. It does not buy or sell anything. It does not need an account, a subscription or a credit card.

> **Please read this first.** LeanTrade gives information, not financial advice. Nobody can predict markets reliably, and this tool can be wrong, sometimes for long stretches. Never invest money you cannot afford to lose.

---

## What it does, in plain words

1. You give it a list of stocks you care about (for example, 10 to 50 companies).
2. It downloads their past prices (for free) and recent news headlines.
3. It looks for patterns in the price history, tests whether those patterns would have worked in the past, and learns from them.
4. It shows you a ranked list: strongest outlook at the top, weakest at the bottom, with a short plain-English explanation for each stock.

It works for **any country's stock market** that has a market profile (India, USA, UK, Germany, Japan, Australia and more are included).

## What it does not do

- It does **not** place trades or touch your broker account.
- It does **not** promise profit. A ranking is a hint, not a guarantee.
- It does **not** read social media. News is headlines only.
- It does **not** send your data anywhere. Everything stays in the `data` folder on your computer.

---

## Install (3 steps, no coding)

You need a Windows, Mac or Linux computer, about 2 GB of free space and an internet connection the first time.

### Step 1: Download

On the project's GitHub page, click the green **Code** button, then **Download ZIP**. Unzip it anywhere (your Documents folder is fine).

### Step 2: Double-click the starter for your computer

| Your computer | Double-click this file | If it says "blocked" or "unknown developer" |
|---|---|---|
| Windows | `Start LeanTrade.bat` | Click **More info**, then **Run anyway** |
| Mac | `Start LeanTrade.command` | Right-click the file, choose **Open**, then **Open** again (only the first time) |
| Linux | `start.sh` | Right-click, Properties, allow running as a program |

The first start takes a few minutes while it sets itself up. You will see a window with text scrolling; that is normal. Later starts take a few seconds.

### Step 3: Your web browser opens by itself

If it does not, type `http://localhost:8787` in your browser's address bar.

To stop LeanTrade, close the black starter window.

---

## Using it

### First time: pick one of two options

**Try the demo.** Click **Start the demo**. It creates 50 made-up stocks so you can look around safely. It takes about a minute and needs no internet. The page shows a "Demo" badge so you never mistake it for real data.

**Use real stocks.**
1. Choose your country's stock market.
2. Type the stock symbols you care about, at least 8, separated by commas. Use the short code, like `RELIANCE, TCS, INFY, HDFCBANK` for India or `AAPL, MSFT, GOOGL, AMZN` for the USA.
3. Click **Download data and set up** and wait. The first time takes a few minutes (it downloads three years of prices and trains itself).

### The main screen

A ranked list of your stocks. Click any stock to see its details and recent news.

| Word on screen | What it means |
|---|---|
| **Outlook** | How much better or worse than the overall market this stock is expected to do over the chosen period. `+2%` means "expected to beat the market by about 2%". |
| **Chance it rises** | The estimated probability that the stock ends higher. 53% is only slightly better than a coin flip, and that is typical. Markets are hard. |
| **Bumpiness** | How much the stock tends to jump around. Higher means a rougher ride. |
| **Typical range** | Where most outcomes are expected to land. It is usually wide. That is the honest picture. |
| **Looking ahead** | Choose about 1 week, 1 month or 3 months. |

### Keeping it fresh

Click **Update data** now and then (for example once a week). It fetches the latest prices and news and re-trains. The demo cannot be updated because its data is made up.

---

## How to read the results sensibly

- **Compare, don't obey.** The ranking shows which stocks look stronger than the others in your list. It does not say you should buy.
- **Watch for the yellow warnings.** If the tool says it did not find a reliable pattern, believe it. Every stock then gets nearly the same chance and the order means very little. Adding more stocks to your list usually helps.
- **A confident-looking number is not a certain one.** Look at the "typical range" too.
- **Use it as one input** next to your own research and your own risk limits.

---

## News

When you set up real stocks, LeanTrade also fetches recent headlines for each one (free sources). They appear on each stock's page. Headlines mentioning earnings, dividends or similar events are flagged for extra attention in the optional AI research step. News is not yet part of the numbers behind the ranking; it is there to give you context.

---

## Something went wrong?

| Problem | What to do |
|---|---|
| The starter window closes instantly | Open the `logs` folder and read `launcher.log`, or ask for help (see below) and include that file |
| The browser did not open | Type `http://localhost:8787` in the address bar |
| It says the port is already in use | LeanTrade may already be running in another window. Close that one, or restart your computer |
| "Download data" fails | Check your internet connection and that the symbols are correct. Free data sources sometimes limit how fast you can download, so try again in a few minutes |
| It says it is busy updating | Wait a minute and refresh the page. It does one job at a time |
| A stock is missing | The symbol may be wrong. Check it on your broker's app or a finance website |

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
| Probability | A chance between 0% and 100%. It is not a promise |
| Training | The tool studying past prices to learn patterns |

---

## For developers and the technically curious

The full technical documentation (architecture, command-line usage, adding a new country, configuration, testing methodology, roadmap) is in [docs/TECHNICAL.md](docs/TECHNICAL.md). A few useful commands:

```bash
uv run leantrade --help          # all commands
uv run leantrade news pull       # fetch headlines
uv run leantrade rank            # ranked list in the terminal
uv run pytest                    # run the test suite
```

Note that parts of the technical document describe the long-term plan; the browser screens described above are what is available today.

## License

Apache License 2.0 is proposed. A `LICENSE` file has not been added to the project yet.
