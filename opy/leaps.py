# Vendored from ~/Documents/Claude/Options Screener/leaps.py (that folder is not a git
# repo, so this copy is what CI runs). Sync manually if the source changes.

"""LEAPS (Long-term Equity AnticiPation Securities) call-buying screener.

Directionally and mechanically the opposite of every other strategy in this project:
those sell premium and want it rich (high IV relative to realized vol); this *buys*
premium (a deep ITM call, 9+ months out, as a leveraged stock substitute) and wants
it cheap. Research synthesized from OptionsPlay, Option Alpha, TradingBlock, SoFi,
Fidelity, Charles Schwab, and Merrill Edge converges on:

  - Delta 0.70-0.85 (deep ITM) -- tracks the stock almost 1:1 with low extrinsic
    value, versus a shallower/OTM LEAPS which is a more leveraged speculative bet.
  - 9+ months to expiration, commonly 12-18 months; roll with 3-6 months left.
  - Buy when IV is LOW relative to its own realized vol (IV/HV ratio low, ideally
    below or near 1.0) -- the inverse of the other screeners' "rich premium" filter,
    since you're now paying for vega instead of collecting it. Fidelity and Days to
    Expiry both independently make this same point.
  - Strike ~20-30% ITM, premium typically 20-30% of the stock's price (OptionsPlay).

Two entry theses, run side by side rather than picking one:

  - Trend Following: price above a rising 50-day average (same thesis as the bull
    put screener) -- ride an uptrend already in motion.
  - RSI Reversal: RSI <= 30 (oversold) within a still-intact long-term uptrend
    (price above its 200-day average) -- buy the dip, not the breakout, matching
    OptionsPlay's "Bullish Counter Trend" RSI-based scan concept. Deliberately does
    NOT also require the short-term trend (20d > 50d average): a pullback sharp
    enough to hit RSI 30 will usually have already inverted that crossover, so
    requiring both at once would be close to contradictory.

Both feed the same Stage 2 (delta/IV/liquidity/earnings) and share the general
"cheap IV, deep ITM, avoid earnings" LEAPS principles -- only the entry-timing
thesis differs, so only Stage 1 and the score's "quality" weight differ between them.

Earnings avoidance uses a much shorter buffer than the 30-day one used for the
30-45 day premium-selling trades -- you're not protecting the whole multi-month
hold (it'll span several earnings regardless), just avoiding buying into an
immediate pre-earnings IV runup / post-earnings crush right at entry.

Merrill Edge also flags interest-rate risk as unusually large for LEAPS (more of a
macro/market-timing factor than a per-stock filter, so not encoded here).
"""

import concurrent.futures
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from core import bs_delta, get_next_earnings_days, get_option_chain, get_options_list, get_sector, pick_expiry


def filter_leaps_stage1_trend(metrics, min_dollar_volume):
    """Same bullish-trend thesis as the bull put screener (price above a rising
    50-day average) -- ride an uptrend already in motion."""
    df = metrics.dropna(subset=["sma50"]).copy()
    trend = (df["last_price"] > df["sma50"]) & (df["sma20"] > df["sma50"])
    liquid = df["avg_dollar_volume_m"] * 1e6 >= min_dollar_volume
    df = df[trend & liquid].copy()
    df["trend_strength"] = (df["sma20"] / df["sma50"] - 1) * 100
    return df


def filter_leaps_stage1_reversal(metrics, min_dollar_volume, max_rsi=30):
    """Oversold pullback (RSI <= max_rsi) within a still-intact long-term uptrend
    (price above its 200-day average) -- buy the dip, not the breakout."""
    df = metrics.dropna(subset=["sma200", "rsi14"]).copy()
    long_term_uptrend = df["last_price"] > df["sma200"]
    oversold = df["rsi14"] <= max_rsi
    liquid = df["avg_dollar_volume_m"] * 1e6 >= min_dollar_volume
    return df[long_term_uptrend & oversold & liquid].copy()


def _screen_one_leaps_candidate(row, target_delta, dte_min, dte_max, monthly_horizon,
                                 max_iv_hv_ratio, max_spread_pct, min_open_interest, min_earnings_days):
    t = row["ticker"]
    try:
        tk = yf.Ticker(t)
        earnings_days = get_next_earnings_days(tk)
        if earnings_days is not None and earnings_days < min_earnings_days:
            return None

        options = get_options_list(tk)
        if not options:
            return None
        expiry = pick_expiry(options, dte_min, dte_max, dte_floor=dte_min, monthly_horizon=monthly_horizon)
        if expiry is None:
            return None
        dte = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.today()).days

        chain = get_option_chain(tk, expiry)
        calls = chain.calls.copy()
        if calls.empty:
            return None

        last = row["last_price"]
        calls["delta"] = calls.apply(
            lambda r: bs_delta(last, r["strike"], dte, r["impliedVolatility"], "call"), axis=1
        )
        calls = calls.dropna(subset=["delta"])
        itm = calls[calls["strike"] < last]  # deep ITM only -- delta 0.7-0.85 lives here
        if itm.empty:
            return None

        leg = itm.loc[(itm["delta"] - target_delta).abs().idxmin()]
        if leg["bid"] <= 0 or leg["ask"] <= 0:
            return None

        mid = (leg["bid"] + leg["ask"]) / 2
        spread_pct = (leg["ask"] - leg["bid"]) / mid * 100
        if spread_pct > max_spread_pct:
            return None

        min_oi = int(leg["openInterest"] or 0)
        if min_oi < min_open_interest:
            return None

        iv_pct = float(leg["impliedVolatility"]) * 100
        iv_hv_ratio = iv_pct / row["hv30"] if row["hv30"] > 0 else np.nan
        if np.isnan(iv_hv_ratio) or iv_hv_ratio > max_iv_hv_ratio:
            return None  # too rich -- you're overpaying for vega on a buy

        premium = float(leg["ask"])  # buying, so the conservative fill is the ask
        strike = float(leg["strike"])
        itm_pct = (last - strike) / last * 100
        premium_pct_of_stock = premium / last * 100
        breakeven = strike + premium
        leverage = last / premium if premium > 0 else None

        return {
            **row.to_dict(),
            "expiry": expiry,
            "dte": dte,
            "earnings_days_away": earnings_days,
            "strike": round(strike, 2),
            "delta": round(float(leg["delta"]), 2),
            "premium": round(premium, 2),
            "premium_pct_of_stock": round(premium_pct_of_stock, 1),
            "itm_pct": round(itm_pct, 1),
            "breakeven": round(breakeven, 2),
            "leverage": round(leverage, 1) if leverage else None,
            "iv_pct": round(iv_pct, 1),
            "iv_hv_ratio": round(float(iv_hv_ratio), 2),
            "avg_spread_pct": round(spread_pct, 1),
            "min_open_interest": min_oi,
            "sector": get_sector(tk, t),
        }
    except Exception:
        return None


def stage2_leaps_screen(candidates_df, target_delta, dte_min, dte_max, monthly_horizon,
                         max_iv_hv_ratio, max_spread_pct, min_open_interest, min_earnings_days,
                         max_workers=8):
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_screen_one_leaps_candidate, row, target_delta, dte_min, dte_max,
                             monthly_horizon, max_iv_hv_ratio, max_spread_pct, min_open_interest,
                             min_earnings_days)
            for _, row in candidates_df.iterrows()
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                rows.append(result)
    return pd.DataFrame(rows)


def score_and_rank(df, style):
    """style: 'trend' weights trend_strength as the quality signal (stronger
    momentum is better); 'reversal' weights RSI (lower/more oversold is better).
    Everything else -- cheap IV, liquidity, tight spreads -- is identical, since
    those are strategy-wide LEAPS principles, not entry-thesis-specific."""
    if df.empty:
        return df
    df = df.copy()
    quality = df["trend_strength"].rank(pct=True) if style == "trend" else (1 - df["rsi14"].rank(pct=True))
    df["score"] = (
        (1 - df["iv_hv_ratio"].rank(pct=True)) * 30      # cheaper IV relative to HV is better (buying vega)
        + quality * 20
        + df["avg_dollar_volume_m"].rank(pct=True) * 20
        + df["min_open_interest"].rank(pct=True) * 15
        + (1 - df["avg_spread_pct"].rank(pct=True)) * 15
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)
