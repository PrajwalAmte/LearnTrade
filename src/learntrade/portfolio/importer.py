"""
Read a holdings export from a broker (CSV or Excel) into plain rows.

Brokers name their columns differently and Excel exports often start with a few lines of
account details before the real table, so this module:

  1. reads every sheet (Zerodha Console puts stocks and mutual funds on separate sheets),
  2. finds the header row by looking for known column names in the first rows,
  3. maps columns to roles (name, symbol, ISIN, quantity, average price, ...) using synonyms.

Known layouts (matched by synonyms, not by exact position):
  Zerodha Kite:     Instrument, Qty., Avg. cost, LTP, Invested, Cur. val, P&L
  Zerodha Console:  Symbol, ISIN, Sector, Quantity Available, ..., Average Price, Previous Closing Price
  Groww stocks:     Stock Name, ISIN, Quantity, Average buy price, Buy value, Closing price, Closing value
  Groww funds:      Scheme Name, AMC, Category, Sub-category, Folio No., Units, Invested Value, Current Value
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# role -> header names (lower-case, punctuation removed) that mean it
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "symbol": ("symbol", "instrument", "tradingsymbol", "trading symbol", "nse symbol", "ticker",
               "scrip", "scrip code", "security id"),
    "name": ("stock name", "company name", "scheme name", "scheme", "fund name", "security name",
             "name", "security", "company"),
    "isin": ("isin", "isin code", "isin no"),
    "quantity": ("quantity", "qty", "quantity available", "units", "balance units", "no of shares",
                 "shares", "holding quantity", "total quantity", "available quantity", "free quantity"),
    "avg_price": ("avg cost", "average price", "avg price", "average buy price", "avg buy price",
                  "average cost", "buy avg", "avg nav", "average nav", "purchase nav", "cost price"),
    "invested": ("invested", "buy value", "invested value", "invested amount", "cost value",
                 "purchase value", "total cost", "amount invested"),
    "price": ("ltp", "closing price", "previous closing price", "current price", "last price",
              "market price", "nav", "current nav", "close price"),
    "value": ("cur val", "current value", "closing value", "market value", "present value",
              "value at market"),
    "sector": ("sector", "industry"),
    "category": ("category", "sub category", "subcategory", "instrument type", "asset class",
                 "scheme category", "type"),
}
# roles that must be present for a table to count as holdings
_REQUIRED_ANY_ID = ("symbol", "name", "isin")


def _norm(text: Any) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _num(text: Any) -> float | None:
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).strip().replace(",", "").replace("₹", "").replace("Rs.", "").replace("Rs", "")
    s = s.replace("%", "").strip()
    if s in ("", "-", "--", "NA", "N/A"):
        return None
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return None


@dataclass
class RawHolding:
    sheet: str
    row: int
    name: str | None = None
    symbol: str | None = None
    isin: str | None = None
    quantity: float | None = None
    avg_price: float | None = None
    invested: float | None = None
    price: float | None = None
    value: float | None = None
    sector: str | None = None
    category: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImportResult:
    holdings: list[RawHolding] = field(default_factory=list)
    columns: dict[str, dict[str, str]] = field(default_factory=dict)   # sheet -> role -> header
    skipped: list[str] = field(default_factory=list)                   # why rows/sheets were skipped


def _read_sheets(data: bytes, filename: str) -> dict[str, list[list[Any]]]:
    if filename.lower().endswith((".xlsx", ".xlsm")):
        from openpyxl import load_workbook
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        return {ws.title: [list(r) for r in ws.iter_rows(values_only=True)] for ws in wb.worksheets}
    text = data.decode("utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    return {"csv": [row for row in csv.reader(io.StringIO(text), dialect)]}


def _match_header(cells: list[Any]) -> dict[str, int]:
    """role -> column index, using the most specific synonym that matches each header."""
    roles: dict[str, int] = {}
    names = [_norm(c) for c in cells]
    for role, syns in _SYNONYMS.items():
        best: tuple[int, int] | None = None       # (rank of synonym, column)
        for col, h in enumerate(names):
            if not h:
                continue
            for rank, syn in enumerate(syns):
                if h == syn or (len(syn) > 3 and h.startswith(syn + " ")):
                    if best is None or rank < best[0]:
                        best = (rank, col)
                    break
        if best is not None:
            roles[role] = best[1]
    # One column must not serve two roles (e.g. "Scheme Name" as both name and symbol).
    if "symbol" in roles and "name" in roles and roles["symbol"] == roles["name"]:
        del roles["symbol"]
    return roles


def parse_holdings(data: bytes, filename: str) -> ImportResult:
    """Parse a broker export. Never raises on odd rows; reasons go to ``skipped``."""
    out = ImportResult()
    for sheet, rows in _read_sheets(data, filename).items():
        header_at, roles = None, {}
        for i, cells in enumerate(rows[:40]):
            r = _match_header(cells)
            if "quantity" in r and any(k in r for k in _REQUIRED_ANY_ID):
                header_at, roles = i, r
                break
        if header_at is None:
            out.skipped.append(f"{sheet}: no holdings table found (needs a quantity column and a "
                               "name, symbol or ISIN column)")
            continue
        header = rows[header_at]
        out.columns[sheet] = {role: str(header[c]) for role, c in roles.items()}
        blank_run = 0
        for j, cells in enumerate(rows[header_at + 1:], start=header_at + 2):
            get = lambda role: cells[roles[role]] if role in roles and roles[role] < len(cells) else None
            ident = [str(get(k)).strip() for k in _REQUIRED_ANY_ID if get(k) not in (None, "")]
            qty = _num(get("quantity"))
            if not ident:
                blank_run += 1
                if blank_run >= 3:
                    break                                  # end of the table (footer follows)
                continue
            blank_run = 0
            first = _norm(ident[0])
            if first.startswith(("total", "grand total", "net")):
                continue
            if not qty or qty <= 0:
                out.skipped.append(f"{sheet} row {j}: {ident[0]} has no quantity")
                continue
            h = RawHolding(
                sheet=sheet, row=j,
                name=(str(get("name")).strip() if get("name") not in (None, "") else None),
                symbol=(str(get("symbol")).strip().upper() if get("symbol") not in (None, "") else None),
                isin=(str(get("isin")).strip().upper() if get("isin") not in (None, "") else None),
                quantity=qty, avg_price=_num(get("avg_price")), invested=_num(get("invested")),
                price=_num(get("price")), value=_num(get("value")),
                sector=(str(get("sector")).strip() if get("sector") not in (None, "") else None),
                category=(str(get("category")).strip() if get("category") not in (None, "") else None),
            )
            if h.symbol:
                h.symbol = re.sub(r"[-.](EQ|BE|SM|ST|BZ)$", "", h.symbol)   # NSE series suffixes
            if h.avg_price is None and h.invested:
                h.avg_price = h.invested / qty
            if h.isin and not re.fullmatch(r"IN[EF][A-Z0-9]{9}", h.isin):
                h.isin = None
            out.holdings.append(h)
    return out


def parse_file(path: Path) -> ImportResult:
    return parse_holdings(Path(path).read_bytes(), Path(path).name)
