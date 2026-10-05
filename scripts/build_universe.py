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
from typing import Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOLDINGS = os.path.join(ROOT, "data", "holdings")
SECTORS_DIR = os.path.join(HOLDINGS, "sectors")
UNIVERSE = os.path.join(ROOT, "universe.txt")

# The eleven sector SPDRs, which between them hold every S&P 500 constituent
# exactly once. Tagging a ticker with its sector ETF rather than a sector NAME
# is the point: the agents gate on whether that ETF is below its 20-day
# average, so the tag has to be the thing they can actually price.
SSGA_URL = ("https://www.ssga.com/us/en/intermediary/library-content/products/"
            "fund-data/etfs/us/holdings-daily-us-en-{}.xlsx")
SECTOR_ETFS = {
    "XLB":  "Materials",
    "XLC":  "Communication Services",
    "XLE":  "Energy",
    "XLF":  "Financials",
    "XLI":  "Industrials",
    "XLK":  "Technology",
    "XLP":  "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU":  "Utilities",
    "XLV":  "Health Care",
    "XLY":  "Consumer Discretionary",
}

# BlackRock's own sector names in iwm.csv, mapped onto the same tags. Russell
# 2000 names are not in any sector SPDR, so this is the only sector source for
# two thirds of the universe. The GICS names line up one-for-one except that
# BlackRock still writes "Information Technology" for what SSGA calls
# Technology, and splits nothing else differently.
IWM_SECTOR_MAP = {
    "materials": "XLB",
    "communication": "XLC",
    "energy": "XLE",
    "financials": "XLF",
    "financial services": "XLF",
    "industrials": "XLI",
    "information technology": "XLK",
    "technology": "XLK",
    "consumer staples": "XLP",
    "real estate": "XLRE",
    "utilities": "XLU",
    "health care": "XLV",
    "healthcare": "XLV",
    "consumer discretionary": "XLY",
}

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


def fetch_sectors(refresh: bool = True) -> Dict[str, str]:
    """symbol -> sector ETF tag, from the eleven sector SPDRs' own holdings.

    Downloaded rather than hand-saved because SSGA, unlike Invesco and
    BlackRock, serves these at a stable URL that works without a browser
    session - the same one fetch_sp500.py already uses for SPY. Cached under
    data/holdings/sectors/ so a later build works offline and so the file that
    produced a given universe.txt is in the repo.
    """
    os.makedirs(SECTORS_DIR, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"}
    sector_of = {}

    for etf in SECTOR_ETFS:
        path = os.path.join(SECTORS_DIR, f"{etf.lower()}.xlsx")
        if refresh or not os.path.exists(path):
            try:
                import requests
                r = requests.get(SSGA_URL.format(etf.lower()), headers=headers, timeout=60)
                # An SSGA error page is still bytes; only a real xlsx starts PK.
                if r.status_code == 200 and r.content[:2] == b"PK":
                    with open(path, "wb") as f:
                        f.write(r.content)
                else:
                    print(f"  {etf}: HTTP {r.status_code}, not an xlsx - using cache if present")
            except Exception as e:
                print(f"  {etf}: download failed ({e}) - using cache if present")

        if not os.path.exists(path):
            print(f"  {etf}: no file, sector will be missing for its holdings")
            continue
        try:
            symbols, _ = parse_spy(path)       # identical SSGA layout
        except Exception as e:
            print(f"  {etf}: parse failed ({e})")
            continue
        for s in symbols:
            # First fund wins. The sector SPDRs do not overlap by design, so a
            # collision means SSGA reclassified something mid-rebalance; taking
            # the first keeps the build deterministic either way.
            sector_of.setdefault(s, etf)
        print(f"  {etf}: {len(symbols)} holdings ({SECTOR_ETFS[etf]})")

    return sector_of


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


def iwm_sectors(path: str) -> Dict[str, str]:
    """symbol -> sector ETF tag, from BlackRock's own Sector column.

    The only sector source for the Russell 2000 names, which are in none of
    the sector SPDRs. Unmapped sectors (BlackRock occasionally emits blanks
    and one-off labels) are simply left untagged rather than guessed at.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    header_at = next((i for i, l in enumerate(lines)
                      if l.startswith("Ticker,") and "Asset Class" in l), None)
    if header_at is None:
        return {}

    out, unmapped = {}, set()
    for row in csv.DictReader(lines[header_at:]):
        sym = norm(row.get("Ticker") or "")
        sec = (row.get("Sector") or "").strip().lower()
        if not sym or not sec or (row.get("Asset Class") or "").strip().lower() != "equity":
            continue
        tag = next((v for k, v in IWM_SECTOR_MAP.items() if sec.startswith(k)), None)
        if tag:
            out[sym] = tag
        else:
            unmapped.add(sec)
    if unmapped:
        print(f"  IWM sectors not mapped to an ETF: {sorted(unmapped)}")
    return out


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
    ap.add_argument("--no-sectors", action="store_true",
                    help="skip sector ETF tagging entirely")
    ap.add_argument("--cached-sectors", action="store_true",
                    help="use the sector files already in data/holdings/sectors/")
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

    # Sector ETF per symbol. The eleven sector SPDRs cover the S&P 500
    # exactly; BlackRock's own Sector column covers the Russell names, which
    # are in none of them.
    sector_of = {}
    if not args.no_sectors:
        print("\nSector SPDR holdings:")
        sector_of = fetch_sectors(refresh=not args.cached_sectors)
    iwm_path = os.path.join(HOLDINGS, "iwm.csv")
    if os.path.exists(iwm_path):
        for s, tag in iwm_sectors(iwm_path).items():
            sector_of.setdefault(s, tag)      # SPDR membership wins where both exist

    for s, tag in sector_of.items():
        if s in membership and tag not in membership[s]:
            membership[s].append(tag)

    candidates = sorted(membership)
    tagged = sum(1 for s in candidates if any(t in SECTOR_ETFS for t in membership[s]))
    print(f"\nUnion: {len(candidates)} unique tickers, {tagged} with a sector tag")

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

    sector_counts = {e: sum(1 for s in kept if e in membership[s]) for e in SECTOR_ETFS}
    sector_tagged = sum(1 for s in kept if any(t in SECTOR_ETFS for t in membership[s]))

    header = (
        "# Scan universe for Harry Trading Desk. Tagged by index and sector.\n"
        "#\n"
        "# Sources: each fund's own daily holdings export (the fund, not a scraped index)\n"
        f"#   SPX  SPDR S&P 500 ETF Trust (SPY), State Street   {as_ofs.get('SPX') or 'date unknown'}\n"
        f"#   QQQ  Invesco QQQ Trust Series 1, Invesco          {as_ofs.get('QQQ') or 'date unknown'}\n"
        f"#   IWM  iShares Russell 2000 ETF, BlackRock          {as_ofs.get('IWM') or 'date unknown'}\n"
        "#   XL*  the eleven sector SPDRs, State Street        downloaded at build time\n"
        "#        (Russell names take their sector from BlackRock's own column)\n"
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
        f"# Sector : {sector_tagged} of {len(kept)} tagged  ("
        + ", ".join(f"{e} {n}" for e, n in sorted(sector_counts.items()) if n) + ")\n"
        "#\n"
        "# One ticker per line; the trailing comment is the index membership plus\n"
        "# the sector SPDR that holds it. Sector tags are ETF symbols, not sector\n"
        "# names, because the agents gate on whether that ETF is below its 20-day\n"
        "# average - the tag has to be something they can price.\n"
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
