# Vendored from ~/Documents/Claude/Options Screener/cboe.py (that folder is not a git
# repo, so this copy is what CI runs). Sync manually if the source changes.

"""CBOE delayed-quote option chains -- a drop-in replacement for the yfinance
option chain, which stopped returning usable data.

WHY THIS EXISTS
    yfinance's option_chain() still returns rows, but as of Aug 2026 the fields the
    screeners actually filter on are dead:

        bid > 0           :   6 / 298   (SPY Sep monthly)
        ask > 0           :   6 / 298
        openInterest > 0  :   0 / 298
        impliedVolatility : 0.250007 for EVERY strike -- a hardcoded placeholder,
                            not a volatility smile

    Zero open interest on the most liquid option chain in the world is not a
    market-hours or rate-limit artifact; a constant IV across all strikes proves
    the field is synthetic. Every options filter in condor.py / spreads.py /
    leaps.py depends on bid/ask (mid price + spread%), open interest (liquidity)
    and IV (IV/HV ratio, delta) -- so all three returned zero candidates.

    CBOE publishes the delayed chain that powers its own website. No API key, no
    account, no signup. One request returns every expiry for an underlying, with
    real bid/ask/OI/IV *and* full greeks.

WHAT CHANGES FOR CALLERS
    Nothing. get_options_list() and get_option_chain() keep their signatures and
    yfinance's column names, so the strategy modules are untouched.

    Bonus: delta/gamma/theta/vega/rho now arrive from the source. core.bs_delta()
    was only ever a workaround for Yahoo having no greeks -- prefer the 'delta'
    column when it is present and fall back to bs_delta() when it is not.

CAVEAT
    Quotes are ~15 minutes delayed. Irrelevant for an end-of-day screener; do not
    build an execution path on them.
"""

import re
import threading
import time
from collections import namedtuple

import pandas as pd
import requests

CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
}

# Cash-settled index products are served under an underscore-prefixed name.
INDEX_SYMBOLS = {"SPX", "SPXW", "VIX", "NDX", "RUT", "DJX", "XSP", "OEX"}

# OCC symbol: AAPL260918P00310000 -> root + YYMMDD + C/P + strike * 1000 (8 digits)
OCC_PATTERN = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")

# One fetch per underlying serves every expiry, so the cache is keyed by symbol
# alone. Lives for one process run; the lock guards ThreadPoolExecutor workers.
_cache_lock = threading.Lock()
_chain_cache = {}

OptionChain = namedtuple("OptionChain", ["calls", "puts", "underlying_price"])

# yfinance column name -> CBOE field name. Keeping yfinance's names is what makes
# this a drop-in; the greeks have no yfinance equivalent and keep CBOE's names.
_FIELD_MAP = {
    "bid": "bid",
    "ask": "ask",
    "lastPrice": "last_trade_price",
    "volume": "volume",
    "openInterest": "open_interest",
    "impliedVolatility": "iv",
}
_GREEKS = ["delta", "gamma", "theta", "vega", "rho"]


def _cboe_symbol(symbol):
    return f"_{symbol}" if symbol.upper() in INDEX_SYMBOLS else symbol.upper()


def _fetch_raw(symbol, retries=3):
    """Fetch one underlying's full chain.

    Uses requests rather than urllib deliberately: requests ships certifi, while
    urllib falls back to the system trust store, which a python.org macOS install
    leaves empty (SSL: CERTIFICATE_VERIFY_FAILED). core.py already depends on
    requests for the SSGA universe fetch, so this adds nothing.
    """
    url = CBOE_URL.format(symbol=_cboe_symbol(symbol))
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=30)
            # 404 means CBOE has no chain for this root - permanent, don't retry.
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
        if attempt < retries - 1:
            time.sleep(1.5 * (attempt + 1))  # linear backoff; CBOE's CDN is generous
    raise RuntimeError(f"CBOE fetch failed for {symbol}: {last_err}")


def _build_frame(symbol):
    """Fetch and parse the full chain for one underlying into a tidy DataFrame."""
    raw = _fetch_raw(symbol)
    if not raw:
        return None, None

    data = raw.get("data") or {}
    options = data.get("options") or []
    if not options:
        return None, None

    # 'close' is the delayed underlying print; 'current_price' appears intraday.
    spot = data.get("current_price") or data.get("close")
    spot = float(spot) if spot else None

    rows = []
    for o in options:
        m = OCC_PATTERN.match(o.get("option", ""))
        if not m:
            continue  # non-standard root (adjusted/flex contracts) - skip
        _root, ymd, cp, strike_raw = m.groups()
        row = {
            "contractSymbol": o["option"],
            "expiry": f"20{ymd[0:2]}-{ymd[2:4]}-{ymd[4:6]}",
            "type": "call" if cp == "C" else "put",
            "strike": int(strike_raw) / 1000.0,
        }
        for yf_name, cboe_name in _FIELD_MAP.items():
            row[yf_name] = o.get(cboe_name)
        for g in _GREEKS:
            row[g] = o.get(g)
        rows.append(row)

    if not rows:
        return None, None

    df = pd.DataFrame(rows)
    numeric = list(_FIELD_MAP) + _GREEKS + ["strike"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    # Downstream code does `openInterest.fillna(0)`; volume/OI as floats is what
    # yfinance produced too, so leave the dtype alone and only fix the obvious NaNs.
    return df, spot


def _get_cached(symbol):
    key = symbol.upper()
    with _cache_lock:
        if key in _chain_cache:
            return _chain_cache[key]
    # Fetch outside the lock so concurrent workers on *different* symbols don't
    # serialise behind each other. A duplicate fetch of the same symbol is a
    # tolerable waste; holding the lock across a 30s network call is not.
    value = _build_frame(symbol)
    with _cache_lock:
        _chain_cache.setdefault(key, value)
        return _chain_cache[key]


# ── Public API: mirrors core.get_options_list / core.get_option_chain ──────────

def get_options_list(ticker_obj):
    """Expiry strings ('YYYY-MM-DD'), ascending. Accepts a yf.Ticker or a str."""
    symbol = getattr(ticker_obj, "ticker", ticker_obj)
    df, _ = _get_cached(symbol)
    if df is None or df.empty:
        return ()
    return tuple(sorted(df["expiry"].unique()))


def get_option_chain(ticker_obj, expiry):
    """OptionChain(calls, puts, underlying_price) for one expiry.

    Columns match yfinance (strike, bid, ask, lastPrice, volume, openInterest,
    impliedVolatility) plus delta/gamma/theta/vega/rho.
    """
    symbol = getattr(ticker_obj, "ticker", ticker_obj)
    df, spot = _get_cached(symbol)
    if df is None or df.empty:
        empty = pd.DataFrame(columns=["strike", *_FIELD_MAP, *_GREEKS])
        return OptionChain(empty, empty.copy(), None)

    sel = df[df["expiry"] == expiry]
    calls = sel[sel["type"] == "call"].sort_values("strike").reset_index(drop=True)
    puts = sel[sel["type"] == "put"].sort_values("strike").reset_index(drop=True)
    return OptionChain(calls, puts, spot)


def get_underlying_price(ticker_obj):
    """Delayed underlying print from the same payload - saves a separate quote call."""
    symbol = getattr(ticker_obj, "ticker", ticker_obj)
    _, spot = _get_cached(symbol)
    return spot


def clear_cache():
    with _cache_lock:
        _chain_cache.clear()
