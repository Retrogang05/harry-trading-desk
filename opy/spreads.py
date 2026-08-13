# Vendored from ~/Documents/Claude/Options Screener/spreads.py (that folder is not a git
# repo, so this copy is what CI runs). Sync manually if the source changes.

"""Bull put / bear call credit spread screening.

Unlike the iron condor (wants range-bound), a single-side credit spread wants a
*directional* lean: bull put spreads want an uptrend (sell OTM puts below a rising
stock), bear call spreads want a downtrend (sell OTM calls above a falling stock).
Research consistently points to: price vs. its moving averages for trend, a short
strike around 20-30 delta (~70-80% probability of profit), and a minimum credit of
25-33% of the spread's width (below that, the risk/reward doesn't compensate for
the trade). RSI shows up as a secondary confirmation signal, not a primary filter,
so it's surfaced here rather than gated on.

Strike selection needs delta, which Yahoo's free chain doesn't include -- computed
locally via Black-Scholes (core.bs_delta) from the same spot/strike/IV/DTE the chain
already gives us.
"""

import concurrent.futures
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from core import bs_delta, get_next_earnings_days, get_option_chain, get_options_list, get_sector, pick_expiry

# OptionsPlay's published convention is notably more aggressive than tastytrade/Option
# Alpha's 20-30 delta default: sell near-ATM (~50 delta), buy ~25 delta -- more credit,
# lower probability of profit, more assignment risk. Surfaced as an extra comparison
# column on candidates the *primary* (target_delta) methodology already selected, not
# as an alternate filter -- the shortlist itself stays driven by the configured params.
OPTIONSPLAY_SHORT_DELTA = 0.50
OPTIONSPLAY_LONG_DELTA = 0.25


def _find_leg_by_delta(side, target_delta):
    if side.empty:
        return None
    idx = (side["delta"].abs() - target_delta).abs().idxmin()
    return side.loc[idx]


def filter_bull_put_stage1(metrics, min_dollar_volume, min_daily_range_pct, max_daily_range_pct):
    df = metrics.dropna(subset=["sma50"]).copy()
    trend = (df["last_price"] > df["sma50"]) & (df["sma20"] > df["sma50"])
    liquid = (
        (df["avg_dollar_volume_m"] * 1e6 >= min_dollar_volume)
        & df["daily_range_pct"].between(min_daily_range_pct, max_daily_range_pct)
    )
    return df[trend & liquid].copy()


def filter_bear_call_stage1(metrics, min_dollar_volume, min_daily_range_pct, max_daily_range_pct):
    df = metrics.dropna(subset=["sma50"]).copy()
    trend = (df["last_price"] < df["sma50"]) & (df["sma20"] < df["sma50"])
    liquid = (
        (df["avg_dollar_volume_m"] * 1e6 >= min_dollar_volume)
        & df["daily_range_pct"].between(min_daily_range_pct, max_daily_range_pct)
    )
    return df[trend & liquid].copy()


def _leg_spread_pct(leg):
    mid = (leg["bid"] + leg["ask"]) / 2
    return (leg["ask"] - leg["bid"]) / mid * 100 if mid > 0 else None


def _screen_one_spread_candidate(row, direction, dte_min, dte_max, dte_floor, min_earnings_days,
                                  target_delta, max_width_pct, min_credit_width_pct,
                                  max_spread_pct, min_open_interest, min_iv_hv_ratio):
    """direction: 'bull_put' (sell OTM puts below spot) or 'bear_call' (sell OTM calls
    above spot). Returns a result row or None -- same fetch-then-filter shape as the
    iron condor's per-candidate worker, run on a thread pool for the same reason
    (I/O-bound network calls, negligible CPU)."""
    t = row["ticker"]
    try:
        tk = yf.Ticker(t)
        earnings_days = get_next_earnings_days(tk)
        if earnings_days is not None and earnings_days < min_earnings_days:
            return None

        options = get_options_list(tk)
        if not options:
            return None
        expiry = pick_expiry(options, dte_min, dte_max, dte_floor=dte_floor)
        if expiry is None:
            return None
        dte = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.today()).days

        chain = get_option_chain(tk, expiry)
        last = row["last_price"]
        opt_type = "put" if direction == "bull_put" else "call"
        side = (chain.puts if direction == "bull_put" else chain.calls).copy()
        if side.empty:
            return None

        side["delta"] = side.apply(
            lambda r: bs_delta(last, r["strike"], dte, r["impliedVolatility"], opt_type), axis=1
        )
        side = side.dropna(subset=["delta"])
        side = side[side["strike"] < last] if direction == "bull_put" else side[side["strike"] > last]
        if side.empty:
            return None

        short_leg = _find_leg_by_delta(side, target_delta)

        # Credit/width is NOT maximized by a fixed target width -- extrinsic value decays
        # fast moving away from the short strike, so a wide spread mostly buys expensive
        # (relative to its own tiny value) far-OTM padding that dilutes the ratio rather
        # than helping it. Search strikes near-to-far and take the narrowest width that
        # clears the credit/width bar -- the most capital-efficient spread that still
        # meets the research-backed threshold, capped so the search doesn't wander into
        # illiquid strikes miles from the money.
        max_width = last * max_width_pct / 100
        if direction == "bull_put":
            long_pool = side[(side["strike"] < short_leg["strike"]) & (side["strike"] >= short_leg["strike"] - max_width)]
            long_pool = long_pool.sort_values("strike", ascending=False)
        else:
            long_pool = side[(side["strike"] > short_leg["strike"]) & (side["strike"] <= short_leg["strike"] + max_width)]
            long_pool = long_pool.sort_values("strike", ascending=True)
        if long_pool.empty or short_leg["bid"] <= 0:
            return None

        long_leg, net_credit, width, credit_width_pct = None, None, None, None
        for _, candidate in long_pool.iterrows():
            if candidate["ask"] <= 0:
                continue
            c_credit = float(short_leg["bid"] - candidate["ask"])
            c_width = abs(float(short_leg["strike"] - candidate["strike"]))
            if c_width <= 0 or c_credit <= 0:
                continue
            c_ratio = c_credit / c_width * 100
            if c_ratio >= min_credit_width_pct:
                long_leg, net_credit, width, credit_width_pct = candidate, c_credit, c_width, c_ratio
                break
        if long_leg is None:
            return None  # no width (up to max_width_pct) clears the credit/width bar

        short_spread_pct = _leg_spread_pct(short_leg)
        long_spread_pct = _leg_spread_pct(long_leg)
        if short_spread_pct is None or long_spread_pct is None:
            return None
        avg_leg_spread_pct = (short_spread_pct + long_spread_pct) / 2
        if avg_leg_spread_pct > max_spread_pct:
            return None

        min_oi = int(min(short_leg["openInterest"] or 0, long_leg["openInterest"] or 0))
        if min_oi < min_open_interest:
            return None

        atm_iv = float(side.loc[(side["strike"] - last).abs().idxmin(), "impliedVolatility"]) * 100
        iv_hv_ratio = atm_iv / row["hv30"] if row["hv30"] > 0 else np.nan
        if np.isnan(iv_hv_ratio) or iv_hv_ratio < min_iv_hv_ratio:
            return None

        approx_pop = (1 - abs(float(short_leg["delta"]))) * 100

        # Return on capital: for a cash-secured defined-risk spread, capital at risk is
        # the width minus the credit already collected (max loss), not the width itself.
        # SMB Capital's stated heuristic -- "the spread with the lowest dollar credit per
        # lot often has the highest ROC" -- is just this ratio in disguise: a narrower,
        # cheaper spread ties up less capital per dollar of credit than a wider one with
        # proportionally more credit. Informational, not part of the score.
        capital_at_risk = width - net_credit
        roc_pct = round(net_credit / capital_at_risk * 100, 1) if capital_at_risk > 0 else None

        # OptionsPlay-style comparison leg (informational only -- see module docstring).
        op_short = _find_leg_by_delta(side, OPTIONSPLAY_SHORT_DELTA)
        op_long_pool = side[side["strike"] != op_short["strike"]] if op_short is not None else side.iloc[0:0]
        op_long = _find_leg_by_delta(op_long_pool, OPTIONSPLAY_LONG_DELTA) if not op_long_pool.empty else None
        op_net_credit = op_width = op_credit_width_pct = None
        if op_short is not None and op_long is not None and op_short["bid"] > 0 and op_long["ask"] > 0:
            op_width_val = abs(float(op_short["strike"] - op_long["strike"]))
            op_credit_val = float(op_short["bid"] - op_long["ask"])
            if op_width_val > 0 and op_credit_val > 0:
                op_net_credit = round(op_credit_val, 2)
                op_width = round(op_width_val, 2)
                op_credit_width_pct = round(op_credit_val / op_width_val * 100, 1)

        return {
            **row.to_dict(),
            "direction": "Bull Put" if direction == "bull_put" else "Bear Call",
            "expiry": expiry,
            "dte": dte,
            "earnings_days_away": earnings_days,
            "short_strike": round(float(short_leg["strike"]), 2),
            "long_strike": round(float(long_leg["strike"]), 2),
            "width": round(width, 2),
            "net_credit": round(net_credit, 2),
            "credit_width_pct": round(credit_width_pct, 1),
            "roc_pct": roc_pct,
            "short_delta": round(float(short_leg["delta"]), 2),
            "approx_pop": round(approx_pop, 1),
            "atm_iv_pct": round(atm_iv, 1),
            "iv_hv_ratio": round(float(iv_hv_ratio), 2),
            "avg_spread_pct": round(avg_leg_spread_pct, 1),
            "min_open_interest": min_oi,
            "sector": get_sector(tk, t),
            "op_short_strike": round(float(op_short["strike"]), 2) if op_short is not None else None,
            "op_long_strike": round(float(op_long["strike"]), 2) if op_long is not None else None,
            "op_net_credit": op_net_credit,
            "op_credit_width_pct": op_credit_width_pct,
        }
    except Exception:
        return None


def stage2_spread_screen(candidates_df, direction, dte_min, dte_max, dte_floor, min_earnings_days,
                          target_delta, max_width_pct, min_credit_width_pct, max_spread_pct,
                          min_open_interest, min_iv_hv_ratio, max_workers=8):
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_screen_one_spread_candidate, row, direction, dte_min, dte_max, dte_floor,
                             min_earnings_days, target_delta, max_width_pct, min_credit_width_pct,
                             max_spread_pct, min_open_interest, min_iv_hv_ratio)
            for _, row in candidates_df.iterrows()
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                rows.append(result)
    return pd.DataFrame(rows)


def score_and_rank(df):
    if df.empty:
        return df
    df = df.copy()
    df["score"] = (
        df["credit_width_pct"].rank(pct=True) * 30
        + df["approx_pop"].rank(pct=True) * 25
        + df["iv_hv_ratio"].rank(pct=True) * 20
        + df["avg_dollar_volume_m"].rank(pct=True) * 15
        + df["min_open_interest"].rank(pct=True) * 10
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)
