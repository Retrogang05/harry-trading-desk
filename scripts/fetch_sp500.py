#!/usr/bin/env python3
"""Download State Street's daily SPY holdings export into data/holdings/.

    python scripts/fetch_sp500.py

The S&P 500 is only one of the three indexes in the scan universe now, so this
no longer writes universe.txt - it refreshes the SPY input and leaves the
building to scripts/build_universe.py, which combines all three and applies
the liquidity screen. The full refresh is:

    python scripts/fetch_sp500.py      # SPY, automated
    # download QQQ and IWM by hand into data/holdings/ (see build_universe.py)
    python scripts/build_universe.py

Only Invesco and BlackRock are manual: neither serves its holdings file at a
stable URL that survives without a browser session, while SSGA's does.

Uses only the standard library: an .xlsx is a zip of XML, so there is no need
to add openpyxl to requirements.txt for a file we parse a few times a year.
The parse here is a sanity check on the download, not the thing that produces
the universe.
"""

import os
import re
import ssl
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from io import BytesIO

SPY_HOLDINGS_URL = (
    "https://www.ssga.com/us/en/intermediary/library-content/products/"
    "fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
)
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOLDINGS = os.path.join(ROOT, "data", "holdings")
SPY_FILE = os.path.join(HOLDINGS, "spy.xlsx")


def ssl_context():
    """Trust store for the request.

    A python.org install on macOS ships without CA certificates unless
    "Install Certificates.command" has been run, so the stdlib default context
    fails where curl succeeds. certifi comes along with requests/yfinance, so
    prefer it when it is importable.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download():
    # SSGA returns 403 to the default urllib agent.
    req = urllib.request.Request(SPY_HOLDINGS_URL, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,*/*",
    })
    try:
        with urllib.request.urlopen(req, timeout=60, context=ssl_context()) as r:
            return r.read()
    except urllib.error.URLError as e:
        if isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
            sys.exit(
                "ERROR: TLS certificate verification failed.\n"
                "  Your Python has no CA bundle. Either:\n"
                "    pip install certifi\n"
                "  or, on a python.org macOS install, run:\n"
                "    /Applications/Python\\ 3.x/Install\\ Certificates.command"
            )
        sys.exit(f"ERROR: download failed - {e}")


def parse(blob):
    z = zipfile.ZipFile(BytesIO(blob))

    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(NS + "si"):
            shared.append("".join(t.text or "" for t in si.iter(NS + "t")))

    def text(c):
        v = c.find(NS + "v")
        if v is None or v.text is None:
            inline = c.find(NS + "is")
            return "".join(t.text or "" for t in inline.iter(NS + "t")) if inline is not None else ""
        return shared[int(v.text)] if c.get("t") == "s" else v.text

    rows = []
    for r in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(NS + "row"):
        cells = {}
        for c in r.findall(NS + "c"):
            col = re.match(r"([A-Z]+)", c.get("r") or "A").group(1)
            cells[col] = text(c).strip()
        rows.append(cells)

    # The preamble rows vary, so find the header by its labels rather than index.
    header_at, cols = None, {}
    for i, r in enumerate(rows):
        labels = {v.lower(): k for k, v in r.items() if v}
        if "ticker" in labels and "name" in labels:
            header_at, cols = i, labels
            break
    if header_at is None:
        sys.exit("ERROR: could not find the holdings header row - SSGA may have changed the layout")

    as_of = ""
    for r in rows[:header_at]:
        for v in r.values():
            if v.lower().startswith("as of"):
                as_of = v
                break

    t_col, n_col = cols["ticker"], cols["name"]
    s_col = cols.get("sector")

    symbols, seen, dropped = [], set(), []
    for r in rows[header_at + 1:]:
        ticker, name = r.get(t_col, "").strip(), r.get(n_col, "").strip()
        if not ticker or not name:
            continue
        sector = (r.get(s_col, "") if s_col else "").strip()

        # Cash sweeps, FX and contra rows are holdings but not tradable equities.
        if sector.lower() in ("cash or derivatives/other", "unassigned", "") or ticker in ("-", "--"):
            dropped.append((ticker, name))
            continue
        if not re.fullmatch(r"[A-Z][A-Z.]{0,6}", ticker):
            dropped.append((ticker, name))
            continue

        # SSGA writes class shares as BRK.B; Yahoo expects BRK-B.
        sym = ticker.replace(".", "-")
        if sym not in seen:
            seen.add(sym)
            symbols.append(sym)

    return symbols, as_of, dropped


def main():
    print(f"Downloading {SPY_HOLDINGS_URL}")
    blob = download()

    # Parse before writing. A 403 page or a truncated transfer is still bytes;
    # only a file that yields ~500 tickers is worth putting on disk, since
    # build_universe.py would otherwise fail on a file that looks downloaded.
    symbols, as_of, dropped = parse(blob)
    if len(symbols) < 400:
        sys.exit(f"ERROR: only parsed {len(symbols)} tickers - refusing to overwrite "
                 f"{os.path.basename(SPY_FILE)} with what looks like a broken download")

    os.makedirs(HOLDINGS, exist_ok=True)
    with open(SPY_FILE, "wb") as f:
        f.write(blob)

    print(f"Wrote {SPY_FILE}")
    print(f"  {len(symbols)} tickers  ({as_of or 'date unknown'})")
    print(f"  dropped {len(dropped)} non-equity rows: {[d[0] for d in dropped][:5]}")
    print("\nNext: refresh data/holdings/qqq.csv and iwm.csv by hand, then")
    print("      python scripts/build_universe.py")


if __name__ == "__main__":
    main()
