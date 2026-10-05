#!/usr/bin/env python3
"""Goldy (GOLD) - moving-average cross scanner.

Finds two upward crosses across the shared universe:

    20/50    the 20-day SMA crossing above the 50-day
    50/200   the 50-day crossing above the 200-day - the classic golden cross

Both are reported as events with a date, not as states: a cross that happened
three weeks ago is not news, so only crosses inside LOOKBACK sessions are
published, and how fresh one is scores into its rank.

WHAT THE BACKTEST SAYS, because it should be read before the list is traded.
13,173 crosses across 580 names, 2019-2026, forward returns measured against
SPY over 60 sessions:

  * Neither cross beats a random day in the same stocks. Random days in this
    universe returned +1.91% alpha over 60 sessions; 20/50 crosses returned
    +1.62% (t=-0.47) and 50/200 crosses +2.28% (t=+0.44). Both differences are
    indistinguishable from zero. The headline "+1.6%/+2.3% after a cross" is
    the universe's own drift, not the signal's.

  * What looked like the strongest filter - wide separation between the two
    averages - is a volatility proxy. Split by separation alone it looks
    excellent (50/200: +7.11% widest quartile vs -0.02% narrowest). Hold
    volatility constant and it collapses, and the sign flips between buckets.
    Normalising separation by ATR does not rescue it either.

  * Volatility itself is what orders the outcomes, monotonically and by a
    wide margin: calmest ADR quartile -1.16%, wildest +9.03% (50/200). That
    is why ADR is published on every row rather than buried in the score.

  * Two filters that sound right actively hurt. Requiring price above the
    200-day: 20/50 crosses above it returned +1.23% vs +2.23% below it.
    Requiring a rising 200-day: +0.88% vs +2.22% flat-or-falling. A cross is
    worth more when it marks a turn than when it confirms a trend already
    running, so neither is applied here.

So the score below describes the cross - how fresh, how decisive, how clean,
how confirmed. It is deliberately NOT presented as a forecast, and nothing in
it was shown to rank forward returns. Goldy is a discovery tool: it tells you
which names just crossed, so they can be looked at. Monu is the agent with
filters that survived validation.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import anthropic
from anthropic import Anthropic

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-haiku-4-5"

AGENT = {
    "id": "GOLD",
    "name": "Goldy",
    "strategy": "MA crosses",
    "description": "Finds 20/50 and 50/200 moving-average crosses as they happen. "
                   "A discovery list, not a ranked forecast - see the README.",
    # Same muted register as the others (Monu #c8974a, Opy #8a7fa8, Trey #6f8fae).
    "accent": "#7fa88a",
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DATA_DIR = os.path.join(REPO_ROOT, "docs", "data")
UNIVERSE_FILE = os.path.join(REPO_ROOT, "universe.txt")

# ── Parameters ───────────────────────────────────────────────────────────

FAST, MID, SLOW = 20, 50, 200

# A cross is only news while it is recent. Ten sessions is two trading weeks:
# long enough that a scan running daily never misses one, short enough that
# the list is still actionable rather than a history of the quarter.
LOOKBACK = 10

# Whipsaw window: how far back to count crosses of the same pair. Six months
# of sessions. A pair that has crossed five times in that window is chopping,
# not trending, whichever way it last went.
WHIPSAW_WINDOW = 126

# SMA200 needs 200 sessions, and the whipsaw count wants WHIPSAW_WINDOW more
# on top of that. 600 calendar days is ~410 sessions, comfortably above both.
CALENDAR_DAYS = 600
MIN_ROWS = SLOW + 30          # enough for the cross itself; whipsaw degrades gracefully

# Tradability floor, matched to scripts/build_universe.py and Monu. Only bites
# on the Russell names - every S&P 500 constituent clears it easily.
MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 10_000_000
LIQUIDITY_WINDOW = 60

# Published rows per index, deduplicated across the two cross types. Matches
# Monu's shape so the two agents' lists are the same length and the index
# chips on the dashboard mean the same thing on both.
PER_INDEX = 6
# Projections get their own smaller allocation on top - see the quota comment
# in scan(). Six confirmed plus three projected per index, minus the SPX/QQQ
# overlap, lands near twenty rows, the same length as Monu's list.
PER_INDEX_APPROACH = 3

INDEX_TAGS = ["SPX", "QQQ", "IWM"]
UNTAGGED = "OTHER"

# A pair's two directions get their own names because the bearish 50/200 has
# one in common usage and the bullish 20/50 does not.
CROSS_TYPES = [
    # key,      fast, slow, bullish label,   bearish label
    ("20_50",   FAST, MID,  "20/50 Cross",   "20/50 Breakdown"),
    ("50_200",  MID,  SLOW, "Golden Cross",  "Death Cross"),
]

# ── Approaching crosses ──────────────────────────────────────────────────
# A cross that has not happened yet, but is converging toward one. The point
# is lead time: by the time the averages actually touch, the move that pulled
# them together has largely happened, so a watchlist entry a week early is
# worth more than the cross itself.
#
# Detection is a linear extrapolation of the gap. Take gap = fast - slow, its
# average daily change over APPROACH_SLOPE_DAYS, and project when the gap
# reaches zero. Linear rather than anything cleverer because the gap between
# two moving averages is already heavily smoothed - fitting a curve to it
# would be fitting to the smoothing.
APPROACH_MAX_DAYS = 10        # project no further than this; beyond it is noise
APPROACH_SLOPE_DAYS = 5       # window the convergence rate is measured over
APPROACH_MAX_GAP_ATR = 2.0    # and the gap must already be inside this many ATR

# ...but not INSIDE this many. A pair sitting 0.02 ATR apart is not about to
# cross, it is entangled: the two averages are running along each other and
# will touch, separate and touch again for weeks. The first run of this scan
# published 21 rows of exactly that, every one scoring 93-100 because "crosses
# in 0.1 days" and "gap of 0.01 ATR" max out imminence and convergence
# together. Those are the worst signals here, not the best - the lead time an
# approaching signal exists to provide only means something when there is
# still a gap to close.
APPROACH_MIN_GAP_ATR = 0.25

# Score axes. Every one of these describes the cross; none of them was shown
# to predict the forward return (see the module docstring). They exist so the
# list has a defensible order, not so the number can be read as an edge.
CROSS_DIMENSIONS = [
    {"key": "freshness",     "label": "Freshness",     "max": 30},
    {"key": "decisiveness",  "label": "Decisiveness",  "max": 25},
    {"key": "cleanliness",   "label": "Cleanliness",   "max": 25},
    {"key": "confirmation",  "label": "Confirmation",  "max": 20},
]

# Approaching rows are scored on different axes: freshness is meaningless for
# something that has not happened, and imminence is the whole point.
#
# These sum to 75, not 100, and that ceiling is the point. The dashboard sorts
# every row on the desk by score, so two scales that both top out at 100 put a
# projection above a confirmed cross - which is backwards, because a cross
# happened and an approach might not. On the first run that is exactly what
# occurred: nine projections scoring 84-96 sat above every real cross at
# 69-77. Capping the weaker claim lower makes one comparable scale out of two
# different axis sets, without the totals drifting from the parts.
APPROACH_DIMENSIONS = [
    {"key": "imminence",     "label": "Imminence",     "max": 28},
    {"key": "convergence",   "label": "Convergence",   "max": 19},
    {"key": "cleanliness",   "label": "Cleanliness",   "max": 18},
    {"key": "confirmation",  "label": "Confirmation",  "max": 10},
]
APPROACH_MAX_SCORE = sum(d["max"] for d in APPROACH_DIMENSIONS)   # 75

DIMENSIONS = CROSS_DIMENSIONS   # the feed-level default the dashboard falls back to


# ── Universe ─────────────────────────────────────────────────────────────

def load_universe() -> Tuple[List[str], Dict[str, List[str]]]:
    """Symbols plus index membership, from the shared universe.txt.

    Same format as Monu reads: "TICKER  # SPX,QQQ", membership in a trailing
    comment. A file written before the tags existed loads untagged.
    """
    if not os.path.exists(UNIVERSE_FILE):
        sys.exit(f"ERROR: {UNIVERSE_FILE} not found - run scripts/build_universe.py")

    symbols, membership = [], {}
    with open(UNIVERSE_FILE) as f:
        for line in f:
            ticker, _, comment = line.partition("#")
            sym = ticker.strip().upper()
            if not sym or sym in membership:
                continue
            tags = [t.strip().upper() for t in comment.split(",") if t.strip()]
            symbols.append(sym)
            membership[sym] = [t for t in tags if t in INDEX_TAGS] or [UNTAGGED]

    counts = {t: sum(1 for v in membership.values() if t in v) for t in INDEX_TAGS + [UNTAGGED]}
    logger.info(f"Loaded {len(symbols)} symbols  ("
                + ", ".join(f"{t} {n}" for t, n in counts.items() if n) + ")")
    return symbols, membership


# ── Data ─────────────────────────────────────────────────────────────────

def warm_tz_cache() -> None:
    """Create yfinance's SQLite timezone cache with one unthreaded request.

    That cache lives under ~/.cache/py-yfinance and is created lazily on first
    use. A CI runner starts cold, so the first THREADED batch has every worker
    racing to create the same database and the losers come back as
    OperationalError('database is locked') - silently empty frames. This is
    what killed Trey's runs #12 and #13. One single-ticker request first means
    the threads only ever read.
    """
    try:
        yf.download("SPY", period="5d", threads=False, progress=False, auto_adjust=True)
    except Exception as e:
        logger.warning(f"tz cache warm-up failed ({e}) - continuing")


def fetch_many(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Batched download. Chunked so one bad ticker cannot poison the universe."""
    warm_tz_cache()

    end = datetime.now()
    start = end - timedelta(days=CALENDAR_DAYS)
    out, CHUNK = {}, 100

    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i:i + CHUNK]
        logger.info(f"Fetching {len(chunk)} symbols ({i + 1}-{i + len(chunk)} of {len(symbols)})...")
        try:
            raw = yf.download(chunk, start=start, end=end, progress=False,
                              group_by="ticker", auto_adjust=True, threads=True)
        except Exception as e:
            logger.error(f"Batch {i // CHUNK} failed: {e}")
            continue

        for sym in chunk:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if sym not in raw.columns.get_level_values(0):
                        continue
                    df = raw[sym]
                else:
                    df = raw
                df = df.dropna()
                if len(df) >= MIN_ROWS:
                    out[sym] = df
            except Exception as e:
                logger.warning(f"{sym}: could not extract from batch ({e})")

    logger.info(f"{len(out)}/{len(symbols)} symbols have enough history")
    return out


def liquid_enough(df: pd.DataFrame) -> bool:
    """Can the position be entered and, more to the point, exited on the stop?"""
    try:
        w = df.tail(LIQUIDITY_WINDOW)
        if float(df["Close"].iloc[-1]) < MIN_PRICE:
            return False
        return float((w["Close"] * w["Volume"]).median()) >= MIN_DOLLAR_VOLUME
    except (KeyError, IndexError, ValueError, TypeError):
        return False


# A single session cannot legitimately move this far. Beyond it the series has
# an unadjusted corporate action in it - a split, a spin-off, a reverse split -
# and every average spanning the gap is arithmetic on two different securities.
MAX_SESSION_MOVE = 0.40


def continuous(df: pd.DataFrame, window: int) -> bool:
    """Reject a price series with an unadjusted corporate action in it.

    yfinance's auto_adjust handles dividends and ordinary splits, but not
    everything: CTVA came back with 2026-09-30 at $77.65 and 2026-10-01 at
    $14.44, an 82% overnight "fall" that is a spin-off nobody adjusted. Its
    50-day average was then the mean of twenty $78 bars and two $12 ones,
    which crosses whatever you like, and its ADR - mean range over current
    price - read 18.5% for a large-cap agriculture name.

    Checked across the whole window the averages are computed over, not just
    recent bars: a gap 100 sessions back still poisons a 200-day mean. Real
    stocks do gap 40% on news, but a moving-average scanner cannot say
    anything useful about one that has, so skipping is right either way.
    """
    try:
        close = df["Close"].tail(window + 1)
        if len(close) < 2:
            return False
        step = (close / close.shift(1)).dropna()
        return bool(((step - 1).abs() < MAX_SESSION_MOVE).all())
    except (KeyError, IndexError, ValueError, TypeError):
        return False


# ── Cross detection ──────────────────────────────────────────────────────

def _flips(above: pd.Series, i: int) -> int:
    """How many times the pair changed sides in the six months up to bar i,
    so the count describes what was knowable at that point.

    Same dtype trap as find_cross: compare against an explicitly-bool shift,
    and seed fill_value from the first bar rather than counting "True != NaN"
    as a flip.
    """
    window = above.iloc[max(0, i - WHIPSAW_WINDOW):i + 1]
    if window.empty:
        return 0
    return int((window != window.shift(1, fill_value=window.iloc[0])).sum())


def find_cross(df: pd.DataFrame, fast_w: int, slow_w: int) -> Dict:
    """The most recent cross of fast through slow, either way, if it is inside
    LOOKBACK sessions. Returns None otherwise.

    Direction comes back as "bullish" (fast crossed above slow) or "bearish"
    (fast crossed below). Both tests are strict about the prior bar: fast must
    have been on the other side of slow on the previous session, or the signal
    re-fires every day the two averages run along touching each other.
    """
    close = df["Close"]
    fast = close.rolling(fast_w).mean()
    slow = close.rolling(slow_w).mean()

    above = fast > slow
    # shift(fill_value=False).astype(bool) is not belt-and-braces, it is the
    # whole correctness of this function. In pandas 3.x, .shift() on a bool
    # Series returns OBJECT dtype (bool values plus NaN), and ~ on an object
    # Series is bitwise NOT on the underlying ints: ~True is -2, ~False is -1,
    # and BOTH are truthy. So `above & ~above.shift(1).fillna(False)` silently
    # reduces to `above` - every bar with fast above slow reads as a fresh
    # cross, and the scanner reports one for every stock in an uptrend.
    prev = above.shift(1, fill_value=False).astype(bool)
    up = above & ~prev
    # The bearish side needs its own guard: fast must be strictly below slow
    # now and have been above before. Writing it as ~up would also fire on
    # every bar where the two are merely equal.
    down = (~above) & prev

    crossed = up | down
    recent = crossed.tail(LOOKBACK)
    if not recent.any():
        return None

    when = recent[recent].index[-1]
    i = close.index.get_loc(when)

    # Separation now, as a fraction. Reported but weighted lightly - on the
    # backtest this was a volatility proxy, not a quality measure.
    sep = float(fast.iloc[-1] / slow.iloc[-1] - 1) if slow.iloc[-1] else 0.0

    return {
        "state": "crossed",
        "bias": "bullish" if bool(up.loc[when]) else "bearish",
        "date": when.strftime("%Y-%m-%d"),
        "bars_ago": int(len(close) - 1 - i),
        "flips_6m": _flips(above, i),
        "separation_pct": sep * 100,
        "fast_ma": float(fast.iloc[-1]),
        "slow_ma": float(slow.iloc[-1]),
        "price_at_cross": float(close.iloc[i]),
    }


def find_approach(df: pd.DataFrame, fast_w: int, slow_w: int) -> Dict:
    """A cross that has not happened yet but is converging toward one.

    The lead time is the point. By the time two averages actually touch, the
    move that dragged them together has already happened; seeing the gap close
    a week out is what makes an early entry possible at all.

    Projection is deliberately linear: gap = fast - slow, measure its average
    daily change over APPROACH_SLOPE_DAYS, and solve for zero. The gap between
    two moving averages is already heavily smoothed, so fitting anything more
    elaborate would be fitting to the smoothing rather than to the price.

    Three conditions, all required:
      * the gap is closing, not widening
      * it reaches zero inside APPROACH_MAX_DAYS
      * the gap sits between APPROACH_MIN_GAP_ATR and APPROACH_MAX_GAP_ATR -
        far enough out that there is real lead time, close enough that the
        projection is not extrapolating across a chasm
    """
    close = df["Close"]
    fast = close.rolling(fast_w).mean()
    slow = close.rolling(slow_w).mean()
    if len(close) < slow_w + APPROACH_SLOPE_DAYS + 1:
        return None

    gap_now = float(fast.iloc[-1] - slow.iloc[-1])
    gap_then = float(fast.iloc[-1 - APPROACH_SLOPE_DAYS] - slow.iloc[-1 - APPROACH_SLOPE_DAYS])
    if np.isnan(gap_now) or np.isnan(gap_then) or gap_now == 0:
        return None

    rate = (gap_now - gap_then) / APPROACH_SLOPE_DAYS      # gap change per session
    if rate == 0:
        return None

    # Closing means the gap is moving toward zero: a negative gap must be
    # rising, a positive gap must be falling.
    if (gap_now < 0 and rate <= 0) or (gap_now > 0 and rate >= 0):
        return None

    days = -gap_now / rate
    if not (0 < days <= APPROACH_MAX_DAYS):
        return None

    atr = float((df["High"] - df["Low"]).tail(20).mean())
    if not atr or np.isnan(atr):
        return None
    gap_atr = abs(gap_now) / atr
    if not (APPROACH_MIN_GAP_ATR <= gap_atr <= APPROACH_MAX_GAP_ATR):
        return None

    above = fast > slow
    return {
        "state": "approaching",
        # Crossing UP from below is the bullish case, so the bias is the side
        # it is heading to, not the side it is on now.
        "bias": "bullish" if gap_now < 0 else "bearish",
        "days_to_cross": round(days, 1),
        "gap_atr": round(gap_atr, 2),
        "bars_ago": 0,
        "flips_6m": _flips(above, len(close) - 1),
        "separation_pct": float(fast.iloc[-1] / slow.iloc[-1] - 1) * 100 if slow.iloc[-1] else 0.0,
        "fast_ma": float(fast.iloc[-1]),
        "slow_ma": float(slow.iloc[-1]),
    }


def adr_pct(df: pd.DataFrame, window: int = 20) -> float:
    """Average daily range as a percentage of price.

    Published on every row because it is the one measure in this whole study
    that ordered forward returns monotonically - calmest quartile -1.2%,
    wildest +9.0% over 60 sessions. It is a risk measure, not a buy signal.
    """
    try:
        rng = (df["High"] - df["Low"]).tail(window).mean()
        return float(rng / df["Close"].iloc[-1] * 100)
    except (KeyError, IndexError, ValueError, ZeroDivisionError):
        return 0.0


def score_signal(df: pd.DataFrame, sig: Dict) -> Dict:
    """Describe the signal. See the module docstring: these axes order the
    list, they do not forecast it.

    Two axis sets, because a cross that has happened and one that has not are
    not describable on the same terms - freshness is meaningless before the
    event, and imminence is meaningless after it. Confirmation is read in the
    direction of the signal's own bias, so a bearish cross scores for price
    sitting BELOW its averages, not above.
    """
    close = df["Close"]
    price = float(close.iloc[-1])
    adr = adr_pct(df) or 1.0
    bullish = sig["bias"] == "bullish"

    # Axis maxima come from the dimension lists so the parts can never drift
    # from the total the dashboard draws them against.
    dims = {d["key"]: d["max"] for d in
            (CROSS_DIMENSIONS if sig["state"] == "crossed" else APPROACH_DIMENSIONS)}

    # Cleanliness is common to both: a pair that has flipped repeatedly is
    # chopping. 2 or fewer flips in six months is a clean trend change, 8+
    # is noise, whichever way it last went.
    cleanliness = int(round(dims["cleanliness"]
                            * max(0.0, min(1.0, (8 - sig["flips_6m"]) / 6))))

    # Confirmation - price on the signal's own side of both averages, plus
    # unusual volume. Built on a 0-1 scale, then taken to this axis's maximum.
    conf_frac = 0.0
    if (price > sig["fast_ma"]) == bullish:
        conf_frac += 0.35
    if (price > sig["slow_ma"]) == bullish:
        conf_frac += 0.35
    try:
        vol_ratio = float(df["Volume"].tail(LOOKBACK).max() / df["Volume"].tail(60).mean())
        conf_frac += 0.30 if vol_ratio > 1.5 else 0.15 if vol_ratio > 1.2 else 0.0
    except (KeyError, ZeroDivisionError, ValueError):
        vol_ratio = 0.0
    confirmation = int(round(dims["confirmation"] * min(1.0, conf_frac)))

    if sig["state"] == "crossed":
        # Freshness - today's cross is worth more attention than one from a
        # fortnight ago, purely because it is still actionable.
        freshness = int(round(dims["freshness"]
                              * max(0.0, 1 - sig["bars_ago"] / LOOKBACK)))
        # Decisiveness - how far the averages have separated since crossing,
        # in the stock's own daily ranges so a volatile name does not win by
        # default. Capped: past ~3 ADR the cross is old news whatever the
        # calendar says.
        decisiveness = int(round(dims["decisiveness"]
                                 * min(1.0, abs(sig["separation_pct"]) / (adr * 3))))
        parts = {"freshness": freshness, "decisiveness": decisiveness,
                 "cleanliness": cleanliness, "confirmation": confirmation}
    else:
        # Imminence - how few sessions until the projected cross. The whole
        # value of an approaching signal is lead time, so it carries the most
        # weight of any axis here.
        imminence = int(round(dims["imminence"]
                              * max(0.0, 1 - sig["days_to_cross"] / APPROACH_MAX_DAYS)))
        # Convergence - how far through the approach band the gap has come.
        # Measured across the band rather than from zero, because zero is
        # excluded as entanglement: a pair at the MIN edge is as converged as
        # this agent will call anything.
        span = APPROACH_MAX_GAP_ATR - APPROACH_MIN_GAP_ATR
        through = (APPROACH_MAX_GAP_ATR - sig["gap_atr"]) / span if span else 1.0
        convergence = int(round(dims["convergence"] * max(0.0, min(1.0, through))))
        parts = {"imminence": imminence, "convergence": convergence,
                 "cleanliness": cleanliness, "confirmation": confirmation}

    return {**parts, "total": sum(parts.values()), "adr": adr, "vol_ratio": vol_ratio}


def levels(df: pd.DataFrame, cross: Dict) -> Dict:
    """Entry, stop and target for a confirmed BULLISH cross. Returns the
    context fields only for anything else.

    Nothing long is published for a bearish cross or an approaching one, and
    that is deliberate rather than unfinished. A death cross with an entry and
    a target underneath it reads as a buy on a dashboard whose other agents
    all publish long setups, and an approaching cross has not happened - any
    entry quoted for it is a guess at a price that may never trade. Both still
    publish their cross detail, which is the actionable part.

    The horizon here is ~60 sessions, not Monu's 20: a cross is a position
    signal and the backtest measured it over three months. Two stops are
    plausible - just under the slow average that defined the cross (the
    premise is gone if price closes back below it) or a volatility stop at
    2 ATR - and this takes whichever is TIGHTER.

    max(), not min(). Taking the lower of the two always produces the widest
    possible stop, which is catastrophic on a name that has already run far
    past its slow average: KOD crossed at $94.45 with its 50-day at $43.65,
    and the structural stop alone implied risking 55% of the position. When
    the average is miles below, the volatility stop is the only usable one;
    when it sits just under price, it is the structural level that matters.
    above_slow_ma_pct says which case this is.
    """
    price = float(df["Close"].iloc[-1])
    context = {
        # How far price sits from the average involved. Large values mean a
        # late cross - the move largely happened before the averages caught
        # up, and the structural stop is unusable.
        "above_slow_ma_pct": round((price / cross["slow_ma"] - 1) * 100, 1)
                             if cross["slow_ma"] else 0.0,
    }
    if cross["state"] != "crossed" or cross["bias"] != "bullish":
        return context

    rng = (df["High"] - df["Low"]).tail(20).mean()
    atr = float(rng) if rng and not np.isnan(rng) else price * 0.02

    structural = cross["slow_ma"] * 0.98
    volatility = price - atr * 2
    stop = max(structural, volatility)

    # A stop can still land unusably wide on a very volatile name; floor the
    # risk at 15% of price so the published R:R stays meaningful.
    stop = max(stop, price * 0.85)

    risk = max(price - stop, atr * 0.5)
    return {
        **context,
        "entry": price,
        "stop_loss": stop,
        "take_profit": price + risk * 2.0,     # 2:1 over a 3-month hold
        "risk_reward_ratio": 2.0,
        "stop_basis": "structure" if structural >= volatility else "volatility",
    }


# ── Reasoning ────────────────────────────────────────────────────────────

class Reasoner:
    def __init__(self):
        # Same contract as the other agents: reasoning is optional, every
        # number above is pure price maths and must publish without a key.
        # The SDK raises a bare TypeError when no credential resolves, which
        # is not an anthropic.* exception, so gate on the key up front.
        self.enabled = bool(os.environ.get("ANTHROPIC_API_KEY"))
        self.client = Anthropic() if self.enabled else None
        if not self.enabled:
            logger.warning("ANTHROPIC_API_KEY not set - signals publish without reasoning")

    def write(self, o: Dict) -> str:
        """Narrate one row. The prompt differs by state and direction, because
        'what would invalidate this' means something different for a cross
        that has happened than for one that is still a projection."""
        if not self.enabled:
            return "[reasoning unavailable: ANTHROPIC_API_KEY not configured]"

        approaching = o["signal_state"] == "approaching"
        common = (
            f"Fast MA ${o['fast_ma']:.2f} vs slow MA ${o['slow_ma']:.2f} "
            f"(separation {o['separation_pct']:+.2f}%)\n"
            f"This pair has changed sides {o['flips_6m']} times in the last 6 months.\n"
            f"Average daily range: {o['adr_pct']:.1f}% of price.\n"
            f"Price is {o['above_slow_ma_pct']:+.1f}% from the slow average."
        )
        if approaching:
            prompt = (
                f"{o['symbol']} at ${o['price']:.2f} has NOT yet crossed. Its moving "
                f"averages are converging and project to cross in roughly "
                f"{o['days_to_cross']:g} sessions, in the "
                f"{'bullish' if o['bias'] == 'bullish' else 'bearish'} direction.\n\n{common}\n\n"
                "In 2-3 sentences: how reliable is this projection, and what would "
                "have to happen in price for the averages to separate again instead "
                "of crossing? Be concrete and sceptical. The projection is a straight-"
                "line extrapolation, so say if that looks fragile here. Do not give a "
                "price target and do not recommend a trade."
            )
        else:
            prompt = (
                f"{o['symbol']} at ${o['price']:.2f} just completed a {o['strategy']} "
                f"({'bullish' if o['bias'] == 'bullish' else 'bearish'}).\n\n"
                f"Cross date: {o['cross_date']} ({o['bars_ago']} sessions ago)\n{common}\n\n"
                "In 2-3 sentences, assess this specific cross: is it a clean trend "
                "change or part of a choppy sequence, and what would invalidate it? "
                "Be concrete and sceptical. Do not give a price target and do not "
                "recommend a trade."
            )
        try:
            r = self.client.messages.create(
                model=CLAUDE_MODEL, max_tokens=250,
                messages=[{"role": "user", "content": prompt}])
            return r.content[0].text.strip()
        except anthropic.RateLimitError:
            return "[reasoning unavailable: rate limited]"
        except anthropic.APIStatusError as e:
            return f"[reasoning unavailable: API error {e.status_code}]"
        except anthropic.APIConnectionError:
            return "[reasoning unavailable: connection error]"
        except Exception as e:
            return f"[reasoning unavailable: {type(e).__name__}]"


# ── Scan ─────────────────────────────────────────────────────────────────

def scan(symbols: List[str], membership: Dict[str, List[str]],
         per_index: int = PER_INDEX) -> Tuple[List[Dict], Dict]:
    """Find every recent cross, then publish the best `per_index` per index.

    Deliberately NO market-regime gate, unlike Monu. Momentum breakouts fail
    in a falling market, which is why Monu refuses to publish in one. A 50/200
    cross is the opposite case: it is itself a regime signal, and the backtest
    found crosses arriving under a flat or falling 200-day did BETTER than
    those confirming a rising one (+2.22% vs +0.88% over 60 sessions). Gating
    on an uptrend would suppress exactly the turns this agent exists to catch.
    """
    frames = fetch_many(symbols)

    found, stats = [], {"scanned": len(frames), "illiquid": 0, "discontinuous": 0,
                        "by_type": {}}
    for sym, df in frames.items():
        if not liquid_enough(df):
            stats["illiquid"] += 1
            continue
        if not continuous(df, SLOW):
            stats["discontinuous"] += 1
            continue

        # A confirmed cross beats an approaching one for the same pair: once
        # it has happened, the projection is no longer the news.
        hits = {}
        for key, fw, sw, bull_label, bear_label in CROSS_TYPES:
            sig = find_cross(df, fw, sw) or find_approach(df, fw, sw)
            if sig:
                hits[key] = (bull_label if sig["bias"] == "bullish" else bear_label, sig)

        for key, (label, c) in hits.items():
            s = score_signal(df, c)
            lv = levels(df, c)
            approaching = c["state"] == "approaching"
            bullish = c["bias"] == "bullish"

            strategy = f"Nearing {label}" if approaching else label
            stats["by_type"][strategy] = stats["by_type"].get(strategy, 0) + 1

            dims = APPROACH_DIMENSIONS if approaching else CROSS_DIMENSIONS
            fields = ([{"label": "Projected", "value": f"~{c['days_to_cross']:g} sessions",
                        "tone": "pos" if c["days_to_cross"] <= 4 else None},
                       {"label": "Gap", "value": f"{c['gap_atr']:.2f} ATR"}]
                      if approaching else
                      [{"label": "Cross date", "value": c["date"]},
                       {"label": "Sessions ago", "value": str(c["bars_ago"])}])

            found.append({
                "symbol": sym,
                "price": float(df["Close"].iloc[-1]),
                "score": s["total"],
                "strategy": strategy,
                "cross_type": key,
                # State and direction, published so the dashboard can label and
                # colour the row instead of assuming every signal is a buy.
                "signal_state": c["state"],
                "bias": c["bias"],
                "signal_label": (("Nearing " if approaching else "")
                                 + ("Bullish" if bullish else "Bearish")),
                "signal_tone": "pos" if bullish else "neg",
                "dimensions": dims,
                "breakdown": {d["key"]: s[d["key"]] for d in dims},
                "indexes": membership.get(sym, [UNTAGGED]),
                "cross_date": c.get("date"),
                "bars_ago": c["bars_ago"],
                "days_to_cross": c.get("days_to_cross"),
                "flips_6m": c["flips_6m"],
                "separation_pct": round(c["separation_pct"], 2),
                "adr_pct": round(s["adr"], 2),
                "fast_ma": round(c["fast_ma"], 2),
                "slow_ma": round(c["slow_ma"], 2),
                # Both pairs signalling at once is worth seeing: the 20/50
                # confirming a fresh golden cross is the textbook stacked
                # setup. Flagged, not scored - n was far too small to test
                # whether it means anything.
                "stacked": len(hits) > 1,
                **lv,
                "setup": {
                    # Named, because a bullish cross also renders the standard
                    # entry/stop/target panel, which is already called Setup.
                    "title": "Cross detail",
                    "label": (f"{strategy} · {c['flips_6m']} flips/6m · ADR {s['adr']:.1f}%"
                              + (f" · ~{c['days_to_cross']:g}d out" if approaching
                                 else f" · {c['date']}, {c['bars_ago']}d ago")),
                    "fields": fields + [
                        {"label": "Direction", "value": "Bullish" if bullish else "Bearish",
                         "tone": "pos" if bullish else "neg"},
                        {"label": "Fast MA", "value": f"${c['fast_ma']:.2f}"},
                        {"label": "Slow MA", "value": f"${c['slow_ma']:.2f}"},
                        {"label": "Separation", "value": f"{c['separation_pct']:+.2f}%"},
                        {"label": "Price vs slow MA", "value": f"{lv['above_slow_ma_pct']:+.1f}%",
                         "tone": "neg" if abs(lv["above_slow_ma_pct"]) > 25 else None},
                        {"label": "Flips (6m)", "value": str(c["flips_6m"]),
                         "tone": "pos" if c["flips_6m"] <= 2 else "neg" if c["flips_6m"] > 6 else None},
                        {"label": "ADR", "value": f"{s['adr']:.1f}%"},
                        {"label": "Both pairs", "value": "yes" if len(hits) > 1 else "no",
                         "tone": "pos" if len(hits) > 1 else None},
                    ],
                },
            })

    logger.info(f"{len(found)} signals found: "
                + ", ".join(f"{k} {v}" for k, v in sorted(stats["by_type"].items())))

    # Quota per index, then dedupe - same rule as Monu. The Russell 2000 is
    # two thirds of the universe and would otherwise take most of the list on
    # candidate count alone.
    buckets = {t: [] for t in INDEX_TAGS + [UNTAGGED]}
    for o in found:
        for tag in o["indexes"]:
            if tag in buckets:
                buckets[tag].append(o)

    # Confirmed crosses and projections get SEPARATE quotas rather than
    # competing on one score. They are not the same kind of claim - a cross
    # happened, an approach might - and their scores are not comparable: the
    # approach axes top out whenever the gap is small, so on a shared ranking
    # the projections took all 21 slots on the first run and pushed out every
    # actual cross. Splitting the allocation is honest about the difference
    # and guarantees both are visible.
    picked, selected = {}, []
    for tag in INDEX_TAGS + [UNTAGGED]:
        if not buckets[tag]:
            continue
        crossed = [o for o in buckets[tag] if o["signal_state"] == "crossed"]
        nearing = [o for o in buckets[tag] if o["signal_state"] == "approaching"]
        group = (sorted(crossed, key=lambda x: -x["score"])[:per_index]
                 + sorted(nearing, key=lambda x: -x["score"])[:PER_INDEX_APPROACH])
        logger.info(f"  {tag}: {len(crossed)} crossed / {len(nearing)} nearing "
                    f"-> {len(group)} published")
        for o in group:
            # Key on symbol+pair: a name signalling on both pairs is two rows.
            k = (o["symbol"], o["cross_type"])
            if k not in picked:
                picked[k] = o
                selected.append(o)

    # Confirmed crosses first within the list, then projections, each by score.
    selected.sort(key=lambda x: (x["signal_state"] != "crossed", -x["score"]))
    for i, o in enumerate(selected, 1):
        o["rank"] = i

    stats["published"] = len(selected)
    return selected, stats


# ── Publish ──────────────────────────────────────────────────────────────

def publish(opportunities: List[Dict], membership: Dict[str, List[str]], stats: Dict) -> None:
    """Additive-manifest contract, same as the other three agents: this agent
    writes only its own file plus its own manifest entry, so all of them can
    run on independent schedules without clobbering one another."""
    os.makedirs(DOCS_DATA_DIR, exist_ok=True)

    counts = {t: sum(1 for v in membership.values() if t in v) for t in INDEX_TAGS + [UNTAGGED]}
    by_type = ", ".join(f"{k} {v}" for k, v in stats["by_type"].items()) or "none"

    payload = {
        "agent": AGENT,
        "scan_date": datetime.now().isoformat(),
        "dimensions": DIMENSIONS,
        "groups": {
            "key": "indexes",
            "label": "Index",
            "values": [t for t in INDEX_TAGS + [UNTAGGED]
                       if any(t in o.get("indexes", []) for o in opportunities)],
        },
        "context": [
            {"label": "Universe", "value": f"{len(membership)} symbols  ("
                                           + " / ".join(f"{t} {n}" for t, n in counts.items() if n) + ")"},
            {"label": "Scanned", "value": f"{stats['scanned']} with history"},
            {"label": "Skipped", "value": f"{stats['illiquid']} illiquid, "
                                          f"{stats['discontinuous']} price gaps"},
            {"label": "Lookback", "value": f"{LOOKBACK} sessions"},
            {"label": "Crosses found", "value": by_type},
            {"label": "Published", "value": str(stats["published"])},
            {"label": "Model", "value": CLAUDE_MODEL},
        ],
        "opportunities": opportunities,
    }
    if not opportunities:
        payload["empty_message"] = (
            f"No 20/50 or 50/200 cross in the last {LOOKBACK} sessions, and none "
            f"projected inside {APPROACH_MAX_DAYS}. That is a normal result - "
            f"crosses are events, not a daily occurrence."
        )

    agent_file = os.path.join(DOCS_DATA_DIR, f"{AGENT['id']}.json")
    # Serialize before writing: json.dump streams into the file, so a
    # non-serializable value halfway through leaves a truncated, invalid file.
    blob = json.dumps(payload, indent=2, default=str)
    with open(agent_file, "w") as f:
        f.write(blob)

    manifest_path = os.path.join(DOCS_DATA_DIR, "agents.json")
    try:
        with open(manifest_path) as f:
            registered = json.load(f).get("agents", [])
    except (FileNotFoundError, json.JSONDecodeError):
        registered = []
    if AGENT["id"] not in registered:
        registered.append(AGENT["id"])
        with open(manifest_path, "w") as f:
            json.dump({"agents": registered}, f, indent=2)

    logger.info(f"Published {len(opportunities)} crosses -> {agent_file}")


def main():
    logger.info("Starting Goldy - moving-average cross scanner...")
    symbols, membership = load_universe()
    opportunities, stats = scan(symbols, membership)

    reasoner = Reasoner()
    for o in opportunities:
        o["reasoning"] = reasoner.write(o)

    publish(opportunities, membership, stats)

    with open("goldy_results.json", "w") as f:
        json.dump({"scan_date": datetime.now().isoformat(),
                   "stats": stats, "opportunities": opportunities},
                  f, indent=2, default=str)

    for o in opportunities:
        logger.info(f"  #{o['rank']:>2} {o['symbol']:<6} {o['strategy']:<14} "
                    f"score {o['score']:>3}  {o['cross_date']} ({o['bars_ago']}d)  "
                    f"flips {o['flips_6m']}  ADR {o['adr_pct']}%")

    return opportunities


if __name__ == "__main__":
    main()
