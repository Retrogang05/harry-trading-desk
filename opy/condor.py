# Vendored from ~/Documents/Claude/Options Screener/condor.py (that folder is not a git
# repo, so this copy is what CI runs). Sync manually if the source changes.

"""Iron condor candidate screening: range-bound, liquid names with option premium
priced rich to realized volatility. See core.py for the shared universe/metrics
fetch this reads from.
"""

import concurrent.futures
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from core import EXPECTED_MOVE_FACTOR, get_next_earnings_days, get_option_chain, get_options_list, get_sector, pick_expiry

# Barchart defines Expected Move as ~85% of the ATM straddle's value -- a straddle's raw
# price slightly overstates the 1-SD move implied by a lognormal terminal distribution.
# Computed locally from the option chain we already pull; no external dependency.


def filter_stage1(metrics, min_dollar_volume, min_daily_range_pct, max_daily_range_pct, range_position_band):
    """Range-bound thesis: liquid, moderate daily movement, sitting mid-range (not
    trending toward a 52-week high/low) -- the opposite profile from the credit spread
    strategies, which is why this filters the *same* shared metrics table differently."""
    df = metrics.dropna(subset=["range_position_52w"]).copy()
    lo, hi = range_position_band
    return df[
        (df["avg_dollar_volume_m"] * 1e6 >= min_dollar_volume)
        & df["daily_range_pct"].between(min_daily_range_pct, max_daily_range_pct)
        & df["range_position_52w"].between(lo, hi)
    ].copy()


def _screen_one_candidate(row, dte_min, dte_max, dte_floor, min_earnings_days,
                           max_spread_pct, min_open_interest, min_iv_hv_ratio):
    """Fetch + filter a single ticker's options data. Runs on a worker thread -- pure
    I/O wait on network calls, so this is the unit of work ThreadPoolExecutor fans out.
    Returns (result_row_or_None, iv_hv_ratio_or_None). The ratio is returned whenever it
    was computable, independent of whether the row itself passed every later filter, so
    the caller can build an unbiased regime sample even from candidates that didn't make
    the final shortlist."""
    t = row["ticker"]
    try:
        tk = yf.Ticker(t)
        earnings_days = get_next_earnings_days(tk)
        if earnings_days is not None and earnings_days < min_earnings_days:
            return None, None

        options = get_options_list(tk)
        if not options:
            return None, None
        expiry = pick_expiry(options, dte_min, dte_max, dte_floor=dte_floor)
        if expiry is None:
            return None, None

        chain = get_option_chain(tk, expiry)
        calls, puts = chain.calls, chain.puts
        if calls.empty or puts.empty:
            return None, None

        last = row["last_price"]
        calls = calls.assign(dist=(calls["strike"] - last).abs())
        puts = puts.assign(dist=(puts["strike"] - last).abs())
        atm_calls, atm_puts = calls.nsmallest(2, "dist"), puts.nsmallest(2, "dist")

        atm_iv = pd.concat([atm_calls["impliedVolatility"], atm_puts["impliedVolatility"]]).mean() * 100
        if not atm_iv or np.isnan(atm_iv) or atm_iv <= 0:
            return None, None

        iv_hv_ratio = atm_iv / row["hv30"] if row["hv30"] > 0 else np.nan
        if np.isnan(iv_hv_ratio):
            return None, None
        ratio = float(iv_hv_ratio)
        if iv_hv_ratio < min_iv_hv_ratio:
            return None, ratio

        near = pd.concat([atm_calls, atm_puts])
        near = near[(near["bid"] > 0) & (near["ask"] > 0)]
        if near.empty:
            return None, ratio
        mid_px = (near["bid"] + near["ask"]) / 2
        spread_pct = float(((near["ask"] - near["bid"]) / mid_px).mean() * 100)
        if spread_pct > max_spread_pct:
            return None, ratio

        min_oi = int(near["openInterest"].fillna(0).min())
        if min_oi < min_open_interest:
            return None, ratio

        atm_call, atm_put = atm_calls.iloc[0], atm_puts.iloc[0]
        expected_move_dollar, expected_move_pct = None, None
        if atm_call["bid"] > 0 and atm_call["ask"] > 0 and atm_put["bid"] > 0 and atm_put["ask"] > 0:
            straddle_price = (atm_call["bid"] + atm_call["ask"]) / 2 + (atm_put["bid"] + atm_put["ask"]) / 2
            expected_move_dollar = round(float(straddle_price) * EXPECTED_MOVE_FACTOR, 2)
            expected_move_pct = round(expected_move_dollar / last * 100, 1)

        result = {
            **row.to_dict(),
            "expiry": expiry,
            "dte": (datetime.strptime(expiry, "%Y-%m-%d") - datetime.today()).days,
            "earnings_days_away": earnings_days,
            "atm_iv_pct": round(float(atm_iv), 1),
            "iv_hv_ratio": round(ratio, 2),
            "avg_spread_pct": round(spread_pct, 1),
            "min_open_interest": min_oi,
            "expected_move_dollar": expected_move_dollar,
            "expected_move_pct": expected_move_pct,
            "sector": get_sector(tk, t),
        }
        return result, ratio
    except Exception:
        return None, None


def stage2_options_screen(candidates_df, dte_min, dte_max, dte_floor, min_earnings_days,
                           max_spread_pct, min_open_interest, min_iv_hv_ratio, max_workers=8):
    """Parallelized over tickers with threads -- each candidate is 3-4 blocking network
    calls (calendar, options list, option chain, sometimes company info) with negligible
    CPU work in between, so this is textbook I/O-bound fan-out. Keep max_workers modest;
    too high risks Yahoo's unofficial endpoint rate-limiting the whole run."""
    rows, regime_ratios = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_screen_one_candidate, row, dte_min, dte_max, dte_floor,
                             min_earnings_days, max_spread_pct, min_open_interest, min_iv_hv_ratio)
            for _, row in candidates_df.iterrows()
        ]
        for future in concurrent.futures.as_completed(futures):
            result, ratio = future.result()
            if ratio is not None:
                regime_ratios.append(ratio)
            if result is not None:
                rows.append(result)
    return pd.DataFrame(rows), regime_ratios


def score_and_rank(df):
    if df.empty:
        return df
    df = df.copy()
    df["score"] = (
        df["iv_hv_ratio"].rank(pct=True) * 40
        + (1 - (df["range_position_52w"] - 0.5).abs() * 2).clip(lower=0) * 20
        + df["avg_dollar_volume_m"].rank(pct=True) * 15
        + (1 - df["avg_spread_pct"].rank(pct=True)) * 15
        + df["min_open_interest"].rank(pct=True) * 10
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)
