#!/usr/bin/env python3
"""Build universe.txt from saved ETF holdings files, tagged by index membership.

Source of truth is each ETF's own holdings export - the actual funds, not a
scraped index page:

    data/holdings/spy.xlsx   SPDR S&P 500 ETF Trust        (State Street)
    data/holdings/qqq.csv    Invesco QQQ Trust, Series 1   (Invesco)
    data/holdings/iwm.csv    iShares Russell 2000 ETF      (BlackRock)

Download fresh exports from the fund pages and drop them in that directory
with those names, then:

    python scripts/build_universe.py

The Russell 2000 is where the filtering matters. Its holdings file contains
~1,950 lines, of which a few hundred are cash sweeps, index futures, rights,
and unlisted private positions, and several hundred more are names too thin
to trade at any size. Membership alone is not a tradability test, so the
script prices every candidate and drops anything below the liquidity floor
(--min-price / --min-dollar-volume). That pass needs yfinance; pass
--no-liquidity to skip it and write the raw membership list instead.

Output format keeps the existing one-ticker-per-line contract - the index
tags live in a trailing comment, which both agents' loaders already strip:

    AAPL    # SPX,QQQ
    TWST    # IWM
"""

import argparse
import csv
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOLDINGS = os.path.join(ROOT, "data", "holdings")
UNIVERSE = os.path.join(ROOT, "universe.txt")

# Liquidity floor. A momentum signal is only worth publishing if the position
# can actually be entered and, more to the point, exited on the stop. $5 keeps
# out the sub-penny-spread names; $10M median dollar volume means a few hundred
# shares is a rounding error on the day's tape.
MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 10_000_000

# Rows the funds hold that are not tradable common stock.
NON_EQUITY_EXCHANGES = ("NO MARKET", "NON-LISTED", "-")


def norm(ticker: str) -> str:
    """ETF files write class shares three different ways; Yahoo wants one.

    SSGA says BRK.B, BlackRock says "MOG A", Invesco says BRK.B. All of them
    become BRK-B / MOG-A, which is what yfinance resolves.
    """
    t = ticker.strip().upper()
    t = re.sub(r"[\s.]+", "-", t)
    return t


def tradable(sym: str) -> bool:
    """A plain US listing, optionally with a single class suffix."""
    return bool(re.fullmatch(r"[A-Z]{1,5}(-[A-Z])?", sym))


# ── Per-fund parsers ----------------------------------------------------------

XL_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def parse_spy(path: str):
    """SSGA .xlsx. An xlsx is a zip of XML, so this needs no openpyxl for a
    file we parse a few times a year.

    Deliberately not shared with fetch_sp500.py's parser: that one keys off a
    Sector column to drop cash and contra rows, and SSGA's non-US exports
    (holdings-daily-au-en-spy.xlsx) ship that column empty, so every row would
    look like cash. Here the ticker shape is the filter instead - SSGA writes
    cash as "-" and contingent-value rows as "2602335D", neither of which is a
    valid listing - which works on both the US and AU layouts.
    """
    z = zipfile.ZipFile(path)

    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(XL_NS + "si"):
            shared.append("".join(t.text or "" for t in si.iter(XL_NS + "t")))

    def cell_text(c):
        v = c.find(XL_NS + "v")
        if v is None or v.text is None:
            inline = c.find(XL_NS + "is")
            return "".join(t.text or "" for t in inline.iter(XL_NS + "t")) if inline is not None else ""
        return shared[int(v.text)] if c.get("t") == "s" else v.text

    rows = []
    for r in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(XL_NS + "row"):
        cells = {}
        for c in r.findall(XL_NS + "c"):
            col = re.match(r"([A-Z]+)", c.get("r") or "A").group(1)
            cells[col] = cell_text(c).strip()
        rows.append(cells)

    # The preamble varies between exports, so find the header by its labels.
    header_at, cols = None, {}
    for i, r in enumerate(rows):
        labels = {v.lower(): k for k, v in r.items() if v}
        if "ticker" in labels and "name" in labels:
            header_at, cols = i, labels
            break
    if header_at is None:
        sys.exit(f"ERROR: no holdings header in {path} - SSGA may have changed the layout")

    as_of = ""
    for r in rows[:header_at]:
        for v in r.values():
            if v.lower().startswith("as of"):
                as_of = v
                break

    t_col, n_col = cols["ticker"], cols["name"]
    symbols, seen = [], set()
    for r in rows[header_at + 1:]:
        sym = norm(r.get(t_col, ""))
        name = r.get(n_col, "").strip()
        if not sym or not name or not tradable(sym) or sym in seen:
            continue
        seen.add(sym)
        symbols.append(sym)

    return symbols, as_of


def parse_qqq(path: str):
    """Invesco CSV: Ticker,Company,Share/ Par,% TNA,Class of shares,CUSIP,...

    The file is UTF-8 with a BOM. Invesco includes non-exchange-traded
    positions (private placements the trust holds via special purpose
    vehicles), which carry a ticker but cannot be bought; the liquidity pass
    is what removes those, since the file itself does not flag them.
    """
    symbols, seen = [], set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sym = norm(row.get("Ticker") or "")
            cls = (row.get("Class of shares") or "").strip().lower()
            if not sym or not tradable(sym) or sym in seen:
                continue
            if cls and "common stock" not in cls:
                continue
            seen.add(sym)
            symbols.append(sym)
    return symbols, ""


def parse_iwm(path: str):
    """iShares CSV: ~9 preamble lines, then a header row, then holdings, then
    a trailing disclaimer block. Find the header by its labels rather than by
    line number, and stop at the first row that no longer looks like a holding.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()

    as_of = ""
    header_at = None
    for i, line in enumerate(lines):
        if line.lower().startswith("fund holdings as of"):
            as_of = "As of " + line.split(",", 1)[1].strip().strip('"')
        if line.startswith("Ticker,") and "Asset Class" in line:
            header_at = i
            break
    if header_at is None:
        sys.exit(f"ERROR: no holdings header in {path} - iShares may have changed the layout")

    reader = csv.DictReader(lines[header_at:])
    symbols, seen = [], set()
    for row in reader:
        sym = norm(row.get("Ticker") or "")
        asset = (row.get("Asset Class") or "").strip().lower()
        exch = (row.get("Exchange") or "").strip().upper()
        if not sym or sym in seen:
            continue
        if asset != "equity":
            continue  # cash, money market, futures
        if any(exch.startswith(x) for x in NON_EQUITY_EXCHANGES):
            continue  # unlisted private positions and vesting rights
        if not tradable(sym):
            continue
        seen.add(sym)
        symbols.append(sym)
    return symbols, as_of


FUNDS = [
    # tag,   filename,    parser
    ("SPX", "spy.xlsx", parse_spy),
    ("QQQ", "qqq.csv", parse_qqq),
    ("IWM", "iwm.csv", parse_iwm),
]

# A broken parse would silently shrink the universe, so each fund declares the
# floor below which the result is treated as a failed read rather than a real
# change in the index.
MIN_EXPECTED = {"SPX": 450, "QQQ": 90, "IWM": 1500}


# ── Liquidity ----------------------------------------------------------------

def liquidity_screen(symbols, min_price, min_dollar_volume, chunk=200):
    """Keep symbols that priced and clear the floor. Returns (kept, stats).

    Chunked so one bad ticker cannot poison the whole download, and because
    yfinance's batch endpoint starts dropping columns on very wide requests.
    """
    import pandas as pd
    import yfinance as yf

    kept, failed, thin = [], [], []
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        print(f"  pricing {i + 1}-{i + len(batch)} of {len(symbols)}...", flush=True)
        try:
            df = yf.download(batch, period="3mo", interval="1d", group_by="ticker",
                             auto_adjust=True, progress=False, threads=True)
        except Exception as e:                       # network, rate limit, bad batch
            print(f"    batch failed ({e}) - retrying one by one")
            df = None

        for sym in batch:
            try:
                d = df[sym].dropna() if df is not None and sym in df.columns.get_level_values(0) else None
            except Exception:
                d = None
            if d is None or len(d) < 20:
                failed.append(sym)
                continue
            price = float(d["Close"].iloc[-1])
            dollar_vol = float((d["Close"] * d["Volume"]).median())
            if price < min_price or dollar_vol < min_dollar_volume:
                thin.append((sym, price, dollar_vol))
                continue
            kept.append(sym)

    return kept, {"failed": failed, "thin": thin}


# ── Main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-price", type=float, default=MIN_PRICE)
    ap.add_argument("--min-dollar-volume", type=float, default=MIN_DOLLAR_VOLUME)
    ap.add_argument("--no-liquidity", action="store_true",
                    help="skip the pricing pass (writes raw index membership)")
    ap.add_argument("--out", default=UNIVERSE)
    args = ap.parse_args()

    # symbol -> ordered list of index tags it belongs to
    membership, as_ofs = {}, {}
    for tag, filename, parser in FUNDS:
        path = os.path.join(HOLDINGS, filename)
        if not os.path.exists(path):
            sys.exit(f"ERROR: {path} not found.\n"
                     f"  Save the {tag} holdings export there and re-run.")
        symbols, as_of = parser(path)
        if not as_of:
            # Invesco's export carries no as-of row. The download date is the
            # next best thing, and labelling it as such is better than "unknown".
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            as_of = f"Saved {mtime.strftime('%d-%b-%Y')}"
        if len(symbols) < MIN_EXPECTED[tag]:
            sys.exit(f"ERROR: only parsed {len(symbols)} tickers from {filename} "
                     f"(expected at least {MIN_EXPECTED[tag]}) - refusing to overwrite "
                     f"{os.path.basename(args.out)} with what looks like a broken read")
        as_ofs[tag] = as_of
        for s in symbols:
            membership.setdefault(s, []).append(tag)
        print(f"{tag}: {len(symbols)} tickers  ({as_of or 'date unknown'})")

    candidates = sorted(membership)
    print(f"\nUnion: {len(candidates)} unique tickers")

    stats = {"failed": [], "thin": []}
    if args.no_liquidity:
        kept = candidates
        print("Skipping the liquidity pass (--no-liquidity)")
    else:
        print(f"\nLiquidity screen: price >= ${args.min_price:g}, "
              f"median dollar volume >= ${args.min_dollar_volume / 1e6:g}M")
        kept, stats = liquidity_screen(candidates, args.min_price, args.min_dollar_volume)

    kept_set = set(kept)
    # Membership counts AFTER the screen - the honest numbers to put in the
    # header, since the per-fund counts above are pre-filter.
    final_counts = {}
    for tag, _, _ in FUNDS:
        final_counts[tag] = sum(1 for s in kept if tag in membership[s])

    screen_note = ("#          no liquidity screen applied (--no-liquidity)\n"
                   if args.no_liquidity else
                   f"#          price >= ${args.min_price:g} and median 3-month dollar volume "
                   f">= ${args.min_dollar_volume / 1e6:g}M\n")

    header = (
        "# Scan universe for Harry Trading Desk. Tagged by index membership.\n"
        "#\n"
        "# Sources: each fund's own daily holdings export (the fund, not a scraped index)\n"
        f"#   SPX  SPDR S&P 500 ETF Trust (SPY), State Street   {as_ofs.get('SPX') or 'date unknown'}\n"
        f"#   QQQ  Invesco QQQ Trust Series 1, Invesco          {as_ofs.get('QQQ') or 'date unknown'}\n"
        f"#   IWM  iShares Russell 2000 ETF, BlackRock          {as_ofs.get('IWM') or 'date unknown'}\n"
        "#\n"
        f"# Built  : {datetime.now().strftime('%Y-%m-%d')} by scripts/build_universe.py\n"
        f"# Count  : {len(kept)} tickers  "
        f"(SPX {final_counts.get('SPX', 0)}, QQQ {final_counts.get('QQQ', 0)}, "
        f"IWM {final_counts.get('IWM', 0)}; SPX and QQQ overlap heavily)\n"
        "#\n"
        "# Filtered:\n"
        "#          cash, money market, futures and unlisted private positions removed\n"
        + screen_note +
        f"#          {len(stats['thin'])} too thin to trade, {len(stats['failed'])} did not price\n"
        "#\n"
        "# Class shares use Yahoo notation: BRK-B, MOG-A  (funds write BRK.B, \"MOG A\")\n"
        "#\n"
        "# One ticker per line; the trailing comment is the index membership.\n"
        "# Blank lines and # comments are ignored by the loaders, so an agent that\n"
        "# does not care about membership reads this file unchanged.\n"
        "#\n"
        "# Regenerate: refresh data/holdings/, then python scripts/build_universe.py\n"
        "\n"
    )

    width = max((len(s) for s in kept), default=6)
    lines = [f"{s:<{width}}  # {','.join(membership[s])}" for s in kept]

    with open(args.out, "w") as f:
        f.write(header + "\n".join(lines) + "\n")

    print(f"\nWrote {args.out}")
    print(f"  {len(kept)} tickers: SPX {final_counts.get('SPX', 0)}, "
          f"QQQ {final_counts.get('QQQ', 0)}, IWM {final_counts.get('IWM', 0)}")
    if stats["thin"]:
        worst = sorted(stats["thin"], key=lambda t: t[2])[:5]
        print(f"  dropped {len(stats['thin'])} as too thin, e.g. "
              + ", ".join(f"{s} (${p:.2f}, ${v / 1e6:.1f}M)" for s, p, v in worst))
    if stats["failed"]:
        print(f"  dropped {len(stats['failed'])} that did not price: "
              f"{stats['failed'][:8]}{' ...' if len(stats['failed']) > 8 else ''}")


if __name__ == "__main__":
    main()
