# Vendored from ~/Documents/Claude/Options Screener/core.py (that folder is not a git
# repo, so this copy is what CI runs). Sync manually if the source changes.

"""Shared building blocks for the options screeners (iron condor, credit spreads):
universe sourcing, bulk price/trend metrics, earnings/expiry helpers, and the
Black-Scholes delta used for strike selection. Strategy-specific filtering and
scoring lives in condor.py and spreads.py -- this module has no opinion on what
makes a good trade, only on how to fetch and measure things cheaply and once.
"""

import json
import logging
import math
import re
import sys
import threading
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import cboe

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Option chains come from CBOE; price history, earnings and sector still come from
# Yahoo (those endpoints are fine -- it is specifically the option chain that
# degraded). Set False to fall back to the old yfinance chain path for comparison.
USE_CBOE_CHAINS = True

# Per-run cache for per-ticker Yahoo fetches (earnings, expiry list, option chains,
# sector). The iron condor and credit spread pipelines run as independent fetch-then-
# filter passes with no shared memory, but their stage-1 candidate lists overlap
# meaningfully (a stock can be both "range-bound" and "mildly bullish" at once) --
# measured ~20% of stage-2 candidate-slots are a ticker already fetched by an earlier
# pipeline in the same run. Caching here means the second pipeline reuses that result
# instead of re-fetching. Only needs to live for one process run, so a plain dict is
# enough; the lock guards concurrent access from ThreadPoolExecutor workers, not
# cross-run persistence.
_cache_lock = threading.Lock()
_earnings_cache = {}
_options_list_cache = {}
_option_chain_cache = {}
_sector_cache = {}

MAJOR_ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLV", "XLY", "XLP", "GLD", "EEM"]

# Primary source: State Street's own daily SPY holdings file -- SPY *is* the S&P 500,
# so this is the fund manager's ground truth, updated daily, not a scraped HTML table.
SPY_HOLDINGS_URL = ("https://www.ssga.com/us/en/intermediary/etfs/library-content/"
                     "products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx")
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                                 "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")
UNIVERSE_CACHE_DAYS = 30  # S&P 500 constituents rarely change; refetch roughly monthly

RISK_FREE_RATE = 0.04  # rough current short-term T-bill yield; delta is not very sensitive to this

# Barchart defines Expected Move as ~85% of the ATM straddle's value -- a straddle's raw
# price slightly overstates the 1-SD move implied by a lognormal terminal distribution.
EXPECTED_MOVE_FACTOR = 0.85


def _fetch_sp500_from_ssga():
    resp = requests.get(SPY_HOLDINGS_URL, headers=BROWSER_HEADERS, timeout=15)
    resp.raise_for_status()
    raw_bytes = resp.content

    # Locate the header row instead of hardcoding a skiprows count -- SSGA's file has a
    # few title/date rows above the real table that could shift without notice.
    preview = pd.read_excel(BytesIO(raw_bytes), header=None, nrows=10)
    header_row = next(
        (i for i in range(len(preview)) if preview.iloc[i].astype(str).str.strip().eq("Ticker").any()),
        None,
    )
    if header_row is None:
        raise ValueError("couldn't find the holdings table header in the SSGA file")

    df = pd.read_excel(BytesIO(raw_bytes), skiprows=header_row)
    tickers = df["Ticker"].dropna().astype(str).str.strip()
    tickers = tickers[tickers.str.match(TICKER_PATTERN)]  # drops cash lines ("-") and CVR/corp-action rows
    return sorted(set(tickers.str.replace(".", "-", regex=False)))


def _fetch_sp500_from_wikipedia():
    resp = requests.get(SP500_WIKI_URL, headers=BROWSER_HEADERS, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    return sorted(set(tables[0]["Symbol"].str.replace(".", "-", regex=False)))


def get_universe(cache_dir="data"):
    cache_path = Path(cache_dir) / "universe_cache.json"
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
            age_days = (datetime.now() - datetime.fromisoformat(cache["fetched_at"])).days
            if age_days < UNIVERSE_CACHE_DAYS and cache.get("tickers"):
                return sorted(set(cache["tickers"]) | set(MAJOR_ETFS))
        except Exception:
            pass  # corrupt/unreadable cache -- fall through and refetch

    sp500, source = [], None
    for name, fetch in [("SSGA", _fetch_sp500_from_ssga), ("Wikipedia", _fetch_sp500_from_wikipedia)]:
        try:
            sp500 = fetch()
            source = name
            break
        except Exception as e:
            print(f"Warning: S&P 500 list from {name} failed ({e})", file=sys.stderr)

    if sp500:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({
                "fetched_at": datetime.now().isoformat(),
                "source": source,
                "tickers": sp500,
            }))
        except Exception:
            pass  # caching is an optimization, not a requirement
    else:
        print("Warning: both S&P 500 sources failed, using ETFs only", file=sys.stderr)

    return sorted(set(sp500) | set(MAJOR_ETFS))


def _rsi14(close, period=14):
    """Wilder's RSI (the original 1978 formula, and what Yahoo Finance/TradingView/
    StockCharts all show by default) -- NOT a flat average of the last 14 days
    (that's "Cutler's RSI", a different, less common variant that reads meaningfully
    differently after a sharp recent move). Wilder's version exponentially smooths
    gains/losses over the whole history (alpha = 1/period) rather than a flat window,
    so it needs the full series, not just the tail."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    last_gain, last_loss = avg_gain.iloc[-1], avg_loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    rs = last_gain / last_loss
    return 100 - (100 / (1 + rs))


def compute_price_metrics(tickers, batch_size=100):
    """Bulk price/volatility/trend metrics for every ticker with enough history. This is
    the one (slow) network round-trip both the iron condor and credit spread screeners
    read from -- strategy-specific filtering happens afterward on this shared table, so
    running both strategies in one session doesn't pay for the download twice."""
    results = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(batch, period="1y", group_by="ticker", threads=True,
                                auto_adjust=True, progress=False)
        except Exception as e:
            print(f"  batch {i}-{i+len(batch)} download failed: {e}", file=sys.stderr)
            continue

        for t in batch:
            try:
                df = data[t] if len(batch) > 1 else data
                df = df.dropna()
                if len(df) < 60:
                    continue
                close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

                daily_ret = close.pct_change().dropna()
                # 30 trading days -- matches the common vendor convention (e.g. Barchart's
                # "30D His Vol") rather than 20, which was an arbitrary implementation
                # default with no basis in the strategy research.
                hv30 = daily_ret.tail(30).std() * np.sqrt(252) * 100
                avg_dollar_vol = (close.tail(20) * vol.tail(20)).mean()
                daily_range_pct = ((high - low) / close).tail(20).mean() * 100

                wk52_high, wk52_low, last = close.tail(252).max(), close.tail(252).min(), close.iloc[-1]
                range_pos = (last - wk52_low) / (wk52_high - wk52_low) if wk52_high != wk52_low else np.nan

                sma20 = close.tail(20).mean()
                sma50 = close.tail(50).mean() if len(close) >= 50 else np.nan
                sma200 = close.tail(200).mean() if len(close) >= 200 else np.nan
                rsi14 = _rsi14(close)

                results.append({
                    "ticker": t,
                    "last_price": round(float(last), 2),
                    "hv30": round(float(hv30), 1),
                    "avg_dollar_volume_m": round(float(avg_dollar_vol) / 1e6, 1),
                    "daily_range_pct": round(float(daily_range_pct), 2),
                    "range_position_52w": round(float(range_pos), 2) if range_pos == range_pos else None,
                    "sma20": round(float(sma20), 2),
                    "sma50": round(float(sma50), 2) if sma50 == sma50 else None,
                    "sma200": round(float(sma200), 2) if sma200 == sma200 else None,
                    "rsi14": round(float(rsi14), 1),
                })
            except Exception:
                continue
    return pd.DataFrame(results)


def get_next_earnings_days(ticker_obj):
    symbol = ticker_obj.ticker
    with _cache_lock:
        if symbol in _earnings_cache:
            return _earnings_cache[symbol]

    result = None
    try:
        cal = ticker_obj.calendar
        ed = None
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if isinstance(ed, list) and ed:
                ed = ed[0]
        elif isinstance(cal, pd.DataFrame) and not cal.empty and "Earnings Date" in cal.index:
            ed = cal.loc["Earnings Date"].iloc[0]
        if ed:
            result = (pd.Timestamp(ed) - pd.Timestamp.today()).days
    except Exception:
        pass

    with _cache_lock:
        _earnings_cache[symbol] = result
    return result


def get_options_list(ticker_obj):
    """Expiry list for an underlying.

    Sourced from CBOE, not Yahoo: yfinance's chain still returns rows but its
    bid/ask/openInterest are zeroed and impliedVolatility is a constant
    placeholder, which silently starved every options filter in this project.
    See cboe.py for the measurements. cboe does its own per-symbol caching (one
    request covers all expiries), so the local caches here are now redundant for
    the chain path and kept only for earnings/sector.
    """
    if not USE_CBOE_CHAINS:
        symbol = ticker_obj.ticker
        with _cache_lock:
            if symbol in _options_list_cache:
                return _options_list_cache[symbol]
        result = ticker_obj.options
        with _cache_lock:
            _options_list_cache[symbol] = result
        return result

    return cboe.get_options_list(ticker_obj)


def get_option_chain(ticker_obj, expiry):
    """Option chain for one expiry, with yfinance-compatible column names.

    CBOE additionally supplies delta/gamma/theta/vega/rho. Prefer the chain's own
    'delta' column where present -- bs_delta() below was only ever a workaround
    for Yahoo shipping no greeks.
    """
    if not USE_CBOE_CHAINS:
        key = (ticker_obj.ticker, expiry)
        with _cache_lock:
            if key in _option_chain_cache:
                return _option_chain_cache[key]
        result = ticker_obj.option_chain(expiry)
        with _cache_lock:
            _option_chain_cache[key] = result
        return result

    return cboe.get_option_chain(ticker_obj, expiry)


def _third_friday(year, month):
    d = datetime(year, month, 1)
    first_friday = d + timedelta(days=(4 - d.weekday()) % 7)
    return first_friday + timedelta(days=14)


def pick_expiry(options, dte_min, dte_max, dte_floor=21, monthly_horizon=75):
    """Nearest standard monthly (3rd Friday) expiry to the target window -- single-stock
    open interest concentrates on the monthly cycle, so an off-cycle weekly a few days
    closer to the target is usually a trap (looks right on DTE, fails the liquidity
    filters). Never returns anything below dte_floor, even if that means reaching past
    dte_max to the next monthly out -- a too-short DTE trades gamma risk for a DTE
    number that only looks closer to target. Falls back to nearest expiry of any kind
    inside [max(dte_min, dte_floor), dte_max] if no monthly is found in range."""
    today = datetime.today()
    mid = (dte_min + dte_max) / 2

    best_monthly, best_monthly_diff = None, None
    for exp in options:
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d")
        except ValueError:
            continue
        if exp_date.date() != _third_friday(exp_date.year, exp_date.month).date():
            continue
        dte = (exp_date - today).days
        if dte < dte_floor or dte > monthly_horizon:
            continue
        diff = abs(dte - mid)
        if best_monthly_diff is None or diff < best_monthly_diff:
            best_monthly_diff, best_monthly = diff, exp
    if best_monthly is not None:
        return best_monthly

    best, best_diff, lo = None, None, max(dte_min, dte_floor)
    for exp in options:
        try:
            dte = (datetime.strptime(exp, "%Y-%m-%d") - today).days
        except ValueError:
            continue
        if lo <= dte <= dte_max:
            diff = abs(dte - mid)
            if best_diff is None or diff < best_diff:
                best_diff, best = diff, exp
    return best


def get_sector(ticker_obj, ticker_symbol):
    if ticker_symbol in MAJOR_ETFS:
        return "ETF"
    with _cache_lock:
        if ticker_symbol in _sector_cache:
            return _sector_cache[ticker_symbol]

    try:
        sector = ticker_obj.info.get("sector")
        result = sector if sector else "Unknown"
    except Exception:
        result = "Unknown"

    with _cache_lock:
        _sector_cache[ticker_symbol] = result
    return result


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_delta(spot, strike, dte_days, iv, option_type, r=RISK_FREE_RATE):
    """Black-Scholes delta. Yahoo's free option chain has no Greeks, so this is computed
    locally from spot/strike/DTE/IV -- the same inputs the chain already gives us. Used
    to pick the short strike at a target delta (the industry-standard way to size a
    credit spread's probability of profit) rather than eyeballing a fixed OTM %."""
    T = dte_days / 365.0
    if T <= 0 or iv is None or iv <= 0 or spot <= 0 or strike <= 0:
        return None
    try:
        d1 = (math.log(spot / strike) + (r + 0.5 * iv ** 2) * T) / (iv * math.sqrt(T))
    except (ValueError, ZeroDivisionError):
        return None
    return _norm_cdf(d1) if option_type == "call" else _norm_cdf(d1) - 1
