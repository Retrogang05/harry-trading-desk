# Vendored from ~/Documents/Claude/Options Screener/rsi.py (that folder is not a git
# repo, so this copy is what CI runs). Sync manually if the source changes.

"""RSI momentum screener: four non-overlapping bands (oversold, approaching oversold,
approaching overbought, overbought), stock-level only -- flags candidates and gives
supporting context (trend, liquidity, earnings, ATM IV/HV), but does not pick a
specific options structure the way the other three screeners do. No strategy was
named to pair with each zone, so this stays a signal list you act on yourself,
however you decide to (bull put, bear call, LEAPS, wheel, or nothing at all).

Standard Wilder thresholds are 30/70 for oversold/overbought; the 30-40 and 60-70
bands are an early-warning tier (building weakness or strength, not yet extreme).
Worth knowing: in a genuine trend, RSI often lives in a shifted range (uptrends
tend to hold 40-80 with 40 acting as support; downtrends tend to hold 20-60 with 60
acting as resistance) -- so "60-70" on a strongly trending stock isn't automatically
a warning the way it is on a range-bound one. That context is shown, not filtered on.

Stage 1 costs zero extra network calls -- RSI, trend, and liquidity are already in
the shared metrics table. Stage 2 only adds earnings proximity and ATM IV (at the
same front-month DTE window the iron condor/spread screeners use) as context; unlike
every other screener here, it never drops a candidate for thin/unusable options data
since the point is the stock-level RSI signal, not a specific trade.
"""

import concurrent.futures
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from core import get_next_earnings_days, get_option_chain, get_options_list, get_sector, pick_expiry


def filter_oversold_stage1(metrics, min_dollar_volume, oversold_max):
    df = metrics.dropna(subset=["rsi14"]).copy()
    liquid = df["avg_dollar_volume_m"] * 1e6 >= min_dollar_volume
    return df[(df["rsi14"] < oversold_max) & liquid].copy()


def filter_approaching_oversold_stage1(metrics, min_dollar_volume, oversold_max, approaching_oversold_max):
    df = metrics.dropna(subset=["rsi14"]).copy()
    liquid = df["avg_dollar_volume_m"] * 1e6 >= min_dollar_volume
    band = (df["rsi14"] >= oversold_max) & (df["rsi14"] <= approaching_oversold_max)
    return df[band & liquid].copy()


def filter_approaching_overbought_stage1(metrics, min_dollar_volume, approaching_overbought_min, overbought_min):
    df = metrics.dropna(subset=["rsi14"]).copy()
    liquid = df["avg_dollar_volume_m"] * 1e6 >= min_dollar_volume
    band = (df["rsi14"] >= approaching_overbought_min) & (df["rsi14"] <= overbought_min)
    return df[band & liquid].copy()


def filter_overbought_stage1(metrics, min_dollar_volume, overbought_min):
    df = metrics.dropna(subset=["rsi14"]).copy()
    liquid = df["avg_dollar_volume_m"] * 1e6 >= min_dollar_volume
    return df[(df["rsi14"] > overbought_min) & liquid].copy()


def _screen_one_rsi_candidate(row, dte_min, dte_max):
    """Enrichment only -- earnings proximity and ATM IV/HV for context. Never returns
    None for options-data reasons; a stock is still a valid RSI signal even if its
    chain is thin or unreadable, it just shows '--' for the IV fields in that case."""
    t = row["ticker"]
    try:
        tk = yf.Ticker(t)
        earnings_days = get_next_earnings_days(tk)
        sector = get_sector(tk, t)

        atm_iv_pct, iv_hv_ratio, expiry, dte = None, None, None, None
        options = get_options_list(tk)
        if options:
            expiry = pick_expiry(options, dte_min, dte_max, dte_floor=dte_min)
            if expiry is not None:
                dte = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.today()).days
                chain = get_option_chain(tk, expiry)
                calls, puts = chain.calls, chain.puts
                if not calls.empty and not puts.empty:
                    last = row["last_price"]
                    calls = calls.assign(dist=(calls["strike"] - last).abs())
                    puts = puts.assign(dist=(puts["strike"] - last).abs())
                    atm_call = calls.nsmallest(1, "dist").iloc[0]
                    atm_put = puts.nsmallest(1, "dist").iloc[0]
                    ivs = [v for v in (atm_call["impliedVolatility"], atm_put["impliedVolatility"])
                           if v is not None and v == v and v > 0]
                    if ivs:
                        atm_iv_pct = float(np.mean(ivs)) * 100
                        if row["hv30"] > 0:
                            iv_hv_ratio = atm_iv_pct / row["hv30"]

        return {
            **row.to_dict(),
            "sector": sector,
            "earnings_days_away": earnings_days,
            "expiry": expiry,
            "dte": dte,
            "atm_iv_pct": round(atm_iv_pct, 1) if atm_iv_pct is not None else None,
            "iv_hv_ratio": round(iv_hv_ratio, 2) if iv_hv_ratio is not None else None,
        }
    except Exception:
        return {**row.to_dict(), "sector": "Unknown", "earnings_days_away": None,
                "expiry": None, "dte": None, "atm_iv_pct": None, "iv_hv_ratio": None}


def stage2_rsi_screen(candidates_df, dte_min, dte_max, max_workers=8):
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_screen_one_rsi_candidate, row, dte_min, dte_max)
            for _, row in candidates_df.iterrows()
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                rows.append(result)
    return pd.DataFrame(rows)


def score_and_rank(df, direction):
    """direction: 'bullish' (oversold family -- lower RSI is a more extreme, more
    notable signal) or 'bearish' (overbought family -- higher RSI is more extreme)."""
    if df.empty:
        return df
    df = df.copy()
    extremity = (1 - df["rsi14"].rank(pct=True)) if direction == "bullish" else df["rsi14"].rank(pct=True)
    df["score"] = extremity * 60 + df["avg_dollar_volume_m"].rank(pct=True) * 40
    return df.sort_values("score", ascending=False).reset_index(drop=True)
