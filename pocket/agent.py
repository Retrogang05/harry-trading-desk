#!/usr/bin/env python3
"""Pocket (PCKT) - the Finviz relative-volume screen, with the level map drawn.

Automates a manual workflow: run a volume screen, then eyeball each chart for
a "pocket" - a gap between major pivot levels with nothing in between, where
price is expected to travel freely. This agent runs the screen and draws the
map, so the eyeballing starts from a list rather than a blank chart.

THE SCREEN (the user's Finviz settings, on the data we fetch)
    average volume  > 500K   (63-session mean, excluding today - Finviz's 3 months)
    current volume  > 500K
    relative volume > 1.5    (today / that average)
    Market cap Mid+ is not applied: we do not fetch market cap. Every S&P 500
    and Nasdaq-100 name clears it anyway; it would only thin the Russell side.

THE MAP, per screen hit
    major levels    swing highs/lows that are the extreme of +/-10 sessions,
                    clustered when within 0.75 ATR - the setting that reproduced
                    the pockets drawn by hand on AAOI, GOOG, CRWV and IREN
    floor / ceiling the nearest major level below and above price
    pocket          the gap between them, in ATR and %
    state           broke UP through a level today, broke DOWN, or sitting inside

WHAT THE BACKTEST SAID - read before trading the list. 2019-2026, 250-400
names, strictly no lookahead:

  * A target with no pivot in the way is reached about as often as one with a
    pivot in the way: 0-3 points more, across 219,232 tests at 1-3 ATR. The
    "price travels freely through a pocket" premise is mostly not visible in
    the data. The small advantage is on the DOWNSIDE.
  * Low volume inside the pocket made things WORSE, not better: thin pockets
    were reached less often at every distance. So thinness is shown on each
    row but not scored.
  * On the screen's own days (rel vol > 1.5), a break UP into a pocket
    returned +1.12% over 20 sessions against +2.04% for a random high-volume
    day - the pocket subtracted. A break DOWN underperformed a random day by
    0.68% (p=0.03) - the one supportive result, from one cell of four, and the
    stock still rose 0.69% on average, so it reads as relative weakness rather
    than a profitable short.
  * Relative volume itself carried information: a random liquid day with rel
    vol > 1.5 was followed by +2.04% against +1.29% otherwise.

Tested again on 32,799 point-in-time rows (2020-26) with this file's own
map, simulating the trades it publishes:

  * As first published, breakouts averaged +0.007R and breakdowns -0.124R.
  * 36 confluence tests (18 conditions x both directions), each required to
    win in 2020-22 AND 2023-26. One survived: a breakdown within 10% of the
    52-week high (-0.124R -> +0.031R). Shorting already-weak stocks lost
    -0.161R. Stacking conditions made results worse, not better.
  * The exit mattered more than any entry filter. The far-side target capped
    breakout winners; holding 20 sessions with a stop 1 ATR past the level
    gave +0.210R (+1.50%), positive in both eras. Hence the trade rules in
    trade_levels().
  * Even so, every layer of pocket logic returned LESS than simply holding
    every name that passed the volume screen (+3.23% over 20 sessions vs
    +1.50% for the breakout trade). The screen is where the return is; the
    map is best used for stops and context. All long-side figures are
    flattered by survivorship (today's universe, delisted names absent);
    the comparisons between them are not.

So the score below orders the list - fresh break, volume, room to the far
side - and is not a forecast. The agent's value is time: it replaces scanning
charts one by one, and its floors and ceilings are natural stop and target
locations for trades found here or elsewhere.
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import anthropic
from anthropic import Anthropic

# Repo root, for the universe.txt parser shared with the other agents.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import universe
import history

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-haiku-4-5"

AGENT = {
    "id": "PCKT",
    "name": "Pocket",
    "strategy": "Volume screen + level map",
    "description": "Your Finviz relative-volume screen, run on the whole universe, with the "
                   "major levels and pockets drawn for every hit. A map, not a forecast.",
    # Same muted register as Monu #c8974a, Opy #8a7fa8, Trey #6f8fae, Goldy #7fa88a.
    "accent": "#a8857f",
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DATA_DIR = os.path.join(REPO_ROOT, "docs", "data")

INDEX_TAGS = universe.INDEX_TAGS
UNTAGGED = universe.UNTAGGED

# ── The screen ───────────────────────────────────────────────────────────

MIN_AVG_VOLUME = 500_000
MIN_CURRENT_VOLUME = 500_000
MIN_RELATIVE_VOLUME = 1.5
AVG_VOLUME_WINDOW = 63        # ~3 months, Finviz's definition; today excluded

# ── The map ──────────────────────────────────────────────────────────────

# Two years of daily bars: long enough that a major level from last year's
# range still exists on the map, short enough that 2019's prices do not.
CALENDAR_DAYS = 750
MIN_ROWS = 260

PIVOT_BARS = 10       # a major pivot is the extreme of +/- this many sessions
# ...except the most recent swing, which needs only this many sessions after
# it. The full 10-after rule can never confirm anything in the last 10 bars,
# and the latest swing is usually the level that matters most: DECK's 88.40
# high on 2026-10-19 - the line drawn by hand as its ceiling - was invisible,
# so the agent mapped the ceiling at 92.60 instead. Still no lookahead: the
# bars after the pivot must have printed. Shortening confirmation for ALL
# history instead found every hand-drawn line but promoted months-old minor
# swings too, collapsing PHM's map into a 3-point zone.
RECENT_RIGHT_BARS = 3
CLUSTER_ATR = 0.75    # pivots within this many ATR are one level
MIN_POCKET_ATR = 1.2  # narrower than this, two levels are just a congestion zone
STOP_BUFFER_ATR = 0.25      # breakdowns: stop this far past the crossed level

# Breakouts carry NO fixed target - they are held to a stop or a time exit.
# Tested on 4,207 historical breakouts (2020-26, point-in-time, using this
# file's own map): the far-side target capped the winners, and the trade as
# first published averaged +0.007R. Dropping the target and holding 20
# sessions lifted that to about +0.23R, positive in both 2020-22 and
# 2023-26; a stop 1 ATR past the level beat 0.25 ATR on % return in both
# eras. The win rate falls to roughly 40% - fewer, larger winners.
BREAKOUT_STOP_ATR = 1.0
BREAKOUT_HOLD_SESSIONS = 20

# Breakdowns carry short levels only from strength. Of 18 confluences tested
# on 4,157 historical breakdowns, the only one that improved them in BOTH
# eras (p=0.001, p=0.027) was being within 10% of the 52-week high:
# -0.124R overall became +0.031R. Shorting stocks already weak - below their
# averages, weak sector, oversold, SPY in a downtrend - was where the losses
# were (-0.161R across 81% of breakdowns). The rest still publish as rows,
# labelled bearish, but with no trade attached.
SHORT_MAX_OFF_HIGH = 0.10

# A close counts as a BREAK only if it clears the level by this much. Inside
# that band price is testing the level, not through it: PHM closed 121.19
# over a level at 120.93 - 0.07 ATR - and was labelled "broke up", while the
# same zone drawn by hand was resistance overhead. Levels are zones, not
# lines.
BREAK_ATR = 0.25

# A stock whose average daily range is under this fraction of its price is
# not trading freely - typically pinned at a takeover price while the deal
# closes. BZH on the first run: closes of 33.43-33.48 for weeks, ATR 0.07 on
# a $33 stock, so every ATR-denominated number blew up and its "pocket"
# scored 43.8 ATR, the widest on the list. The calmest freely-trading large
# caps sit around 1%, so 0.5% only catches the pinned.
MIN_ATR_PCT = 0.005

# A single session cannot legitimately move this far; beyond it the series has
# an unadjusted corporate action and every level across the gap is arithmetic
# on two different securities. Same guard as Monu and Goldy.
MAX_SESSION_MOVE = 0.40

# Published rows per index, deduplicated across indexes - same shape as the
# other agents, so the index chips mean the same thing everywhere.
PER_INDEX = 8

# Descriptive axes. None was shown to forecast; see the module docstring.
DIMENSIONS = [
    {"key": "fresh_break", "label": "Fresh break",     "max": 20},
    {"key": "rel_volume",  "label": "Relative volume", "max": 40},
    {"key": "room",        "label": "Room to far side", "max": 40},
]


# ── Data ─────────────────────────────────────────────────────────────────

def warm_tz_cache() -> None:
    """Create yfinance's SQLite timezone cache with one unthreaded request
    before any threaded batch touches it. A cold CI cache plus a thread pool
    is the race that killed Trey's runs #12 and #13 (OperationalError:
    database is locked, surfacing as silently empty frames)."""
    try:
        yf.download("SPY", period="5d", threads=False, progress=False, auto_adjust=True)
    except Exception as e:
        logger.warning(f"tz cache warm-up failed ({e}) - continuing")


def fetch_many(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Batched download, chunked so one bad ticker cannot poison the universe."""
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
        if not isinstance(raw.columns, pd.MultiIndex):
            # A collapsed frame cannot be attributed to a ticker - skip the
            # batch rather than guess, as sectors.py does.
            logger.error(f"Batch {i // CHUNK} came back unlabelled - skipped")
            continue
        present = set(raw.columns.get_level_values(0))
        for sym in chunk:
            if sym not in present:
                continue
            df = raw[sym].dropna()
            if len(df) >= MIN_ROWS:
                out[sym] = df
    logger.info(f"{len(out)}/{len(symbols)} symbols have enough history")
    return out


def continuous(df: pd.DataFrame) -> bool:
    """Reject a series with an unadjusted split or spin-off anywhere in it."""
    try:
        close = df["Close"]
        step = (close / close.shift(1)).dropna()
        return len(step) > 0 and bool(((step - 1).abs() < MAX_SESSION_MOVE).all())
    except (KeyError, ValueError, TypeError):
        return False


def screen(df: pd.DataFrame) -> Dict:
    """The Finviz gate. Returns the volume figures if the name passes, else None.

    Average volume excludes today, as Finviz's does - otherwise a 5x volume
    day inflates its own denominator and understates its relative volume.
    """
    v = df["Volume"]
    if len(v) < AVG_VOLUME_WINDOW + 1:
        return None
    avg = float(v.iloc[-AVG_VOLUME_WINDOW - 1:-1].mean())
    cur = float(v.iloc[-1])
    if avg <= 0:
        return None
    rv = cur / avg
    if avg < MIN_AVG_VOLUME or cur < MIN_CURRENT_VOLUME or rv < MIN_RELATIVE_VOLUME:
        return None
    return {"avg_volume": avg, "volume": cur, "rel_volume": rv}


# ── The map ──────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, n: int = 20) -> float:
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def major_levels(df: pd.DataFrame, a: float) -> List[Dict]:
    """Confirmed major pivots clustered into horizontal levels, lowest first.

    A pivot is the extreme of PIVOT_BARS sessions before it and PIVOT_BARS
    after - except inside the last PIVOT_BARS sessions, where only
    RECENT_RIGHT_BARS after are required (see that constant). Either way the
    sessions after must already have printed, so a pivot only exists in
    hindsight and the map never looks ahead.
    """
    H, L, idx = df["High"].values, df["Low"].values, df.index
    n = len(H)
    pts = []
    for i in range(PIVOT_BARS, n - RECENT_RIGHT_BARS):
        right = PIVOT_BARS if i < n - PIVOT_BARS else RECENT_RIGHT_BARS
        if i + right >= n:
            continue
        if H[i] == H[i - PIVOT_BARS:i + right + 1].max():
            pts.append((float(H[i]), idx[i]))
        if L[i] == L[i - PIVOT_BARS:i + right + 1].min():
            pts.append((float(L[i]), idx[i]))
    pts.sort()
    if not pts:
        return []
    groups, cur = [], [pts[0]]
    for p in pts[1:]:
        if p[0] - cur[0][0] > a * CLUSTER_ATR:
            groups.append(cur)
            cur = []
        cur.append(p)
    groups.append(cur)
    return [{"level": float(np.mean([g[0] for g in grp])),
             "touches": len(grp),
             "last": max(g[1] for g in grp)} for grp in groups]


def zone_thinness(df: pd.DataFrame, lo: float, hi: float) -> float:
    """Volume traded inside [lo, hi] relative to what an even spread across the
    window's range would put there. 1.0 = ordinary, 0.3 = vacant air.

    Daily bars only, so each bar's volume is spread evenly over its range - an
    approximation of a volume profile, not tick data. Shown, not scored: the
    backtest found thin pockets were crossed LESS often, not more.
    """
    rng_lo, rng_hi = float(df["Low"].min()), float(df["High"].max())
    if rng_hi <= rng_lo or hi <= lo:
        return float("nan")
    H, L, V = df["High"].values, df["Low"].values, df["Volume"].values
    span = np.maximum(H - L, 1e-9)
    overlap = np.clip(np.minimum(H, hi) - np.maximum(L, lo), 0, None)
    inside = float((V * overlap / span).sum())
    total = float(V.sum())
    expected = (hi - lo) / (rng_hi - rng_lo)
    return (inside / total) / expected if total and expected else float("nan")


def map_levels(df: pd.DataFrame) -> Dict:
    """Floor, ceiling, and what price did to them today."""
    a = atr(df)
    if not a or np.isnan(a):
        return None
    lv = major_levels(df, a)
    price = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2])

    below = [x for x in lv if x["level"] < price]
    above = [x for x in lv if x["level"] > price]
    floor = below[-1] if below else None
    ceiling = above[0] if above else None

    # Did today's close cross a major level? A break up means yesterday closed
    # at or below a level that today closed CLEARLY above - by BREAK_ATR - and
    # that level is now the floor. A break down is the mirror. A close within
    # BREAK_ATR of either level is a test of it, not a break.
    state = "inside"
    margin = BREAK_ATR * a
    if floor and prev <= floor["level"] and price >= floor["level"] + margin:
        state = "broke_up"
    elif ceiling and prev >= ceiling["level"] and price <= ceiling["level"] - margin:
        state = "broke_down"
    elif (floor and price - floor["level"] < margin) or \
         (ceiling and ceiling["level"] - price < margin):
        state = "testing"

    width = (ceiling["level"] - floor["level"]) if floor and ceiling else None
    return {
        "atr": a, "price": price,
        # Today's close-to-close move. "Inside" says no level was crossed; it
        # says nothing about direction, and DECK was flagged Inside on the
        # day it gapped down 4.9% on 3x volume - the most important fact on
        # the row was missing from it.
        "day_move_pct": (price / prev - 1) * 100,
        "floor": floor, "ceiling": ceiling, "state": state,
        "width_atr": width / a if width else None,
        "width_pct": width / floor["level"] * 100 if width else None,
        "room_up_pct": (ceiling["level"] / price - 1) * 100 if ceiling else None,
        "room_down_pct": (1 - floor["level"] / price) * 100 if floor else None,
        "thinness": zone_thinness(df, floor["level"], ceiling["level"]) if width else float("nan"),
        # The next level out on each side, so a trader can see past the pocket.
        "next_above": above[1]["level"] if len(above) > 1 else None,
        "next_below": below[-2]["level"] if len(below) > 1 else None,
        # Distance below the 52-week high, which decides whether a breakdown
        # gets a trade (SHORT_MAX_OFF_HIGH).
        "off_high_pct": (price / float(df["High"].tail(252).max()) - 1) * 100,
    }


# ── Score and levels ─────────────────────────────────────────────────────

def score(m: Dict, vol: Dict) -> Dict:
    """Descriptive ordering only. See the module docstring."""
    fresh = 20 if m["state"] in ("broke_up", "broke_down") else 0
    # Rel vol 1.5 scores 0, 3.0 and above scores full. Scaled over the range
    # screen hits actually occupy: a 1.5-4.0 scale gave a 1.9x name 5 of 40
    # points and left most of the published list greyed out on the dashboard.
    rv = int(round(40 * max(0.0, min(1.0, (vol["rel_volume"] - MIN_RELATIVE_VOLUME) / 1.5))))
    # Room in the direction of travel, in ATR; 4 ATR or more is full. An
    # inside row has no direction, so it gets the larger side.
    a = m["atr"]
    up_atr = (m["ceiling"]["level"] - m["price"]) / a if m["ceiling"] else 4.0
    dn_atr = (m["price"] - m["floor"]["level"]) / a if m["floor"] else 4.0
    room_atr = up_atr if m["state"] == "broke_up" else dn_atr if m["state"] == "broke_down" \
        else max(up_atr, dn_atr)
    room = int(round(40 * max(0.0, min(1.0, room_atr / 4.0))))
    return {"fresh_break": fresh, "rel_volume": rv, "room": room,
            "total": fresh + rv + room, "room_atr": room_atr}


def short_from_strength(m: Dict) -> bool:
    return m["state"] == "broke_down" and m["off_high_pct"] > -SHORT_MAX_OFF_HIGH * 100


def trade_levels(m: Dict) -> Dict:
    """The trade a row carries, in the form that tested best. See the
    BREAKOUT_* and SHORT_MAX_OFF_HIGH constants for the evidence.

    Break up: entry at the close, stop 1 ATR under the level it crossed, no
    target - exit after BREAKOUT_HOLD_SESSIONS sessions or at the stop.

    Break down from strength (within 10% of the 52-week high): the short-side
    trade as originally tested - stop 0.25 ATR over the crossed level, target
    the floor (3 ATR if open air) - published bearish so the dashboard never
    words or colours it as a buy. Any other breakdown gets no trade.

    Inside and testing rows get none: they have not chosen a direction.
    """
    a, px = m["atr"], m["price"]
    if m["state"] == "broke_up":
        stop = m["floor"]["level"] - BREAKOUT_STOP_ATR * a
        if px - stop <= 0:
            return {}
        return {"entry": px, "stop_loss": stop,
                "exit_rule": f"{BREAKOUT_HOLD_SESSIONS} sessions or stop"}
    if short_from_strength(m):
        stop = m["ceiling"]["level"] + STOP_BUFFER_ATR * a
        target = m["floor"]["level"] if m["floor"] else px - 3 * a
        risk = stop - px
        if risk <= 0:
            return {}
        return {"entry": px, "stop_loss": stop, "take_profit": target,
                "risk_reward_ratio": round((px - target) / risk, 2),
                "exit_rule": "target or stop"}
    return {}


# ── Reasoning ────────────────────────────────────────────────────────────

class Reasoner:
    def __init__(self):
        # Reasoning is optional: everything above is price arithmetic and must
        # publish without a key. The SDK raises a bare TypeError when no
        # credential resolves, so gate on the key up front.
        self.enabled = bool(os.environ.get("ANTHROPIC_API_KEY"))
        self.client = Anthropic() if self.enabled else None
        if not self.enabled:
            logger.warning("ANTHROPIC_API_KEY not set - rows publish without reasoning")

    def write(self, o: Dict) -> str:
        if not self.enabled:
            return "[reasoning unavailable: ANTHROPIC_API_KEY not configured]"
        m = o["map"]
        fl = f"${m['floor']:.2f}" if m["floor"] is not None else "none in 2 years"
        cl = f"${m['ceiling']:.2f}" if m["ceiling"] is not None else "none in 2 years"
        prompt = (
            f"{o['symbol']} closed at ${o['price']:.2f}, {o['day_move_pct']:+.1f}% on the day, "
            f"on {o['rel_volume']:.1f}x its normal volume.\n"
            f"State: {o['strategy']}.\n"
            f"Nearest major level below: {fl}. Nearest above: {cl}.\n"
            f"Average daily range (ATR) ${m['atr']:.2f}.\n\n"
            "In 2-3 sentences: what is the volume telling you here, and which of the "
            "two levels is more likely to matter next? Name the price that would show "
            "the move has failed. Be concrete and sceptical - the backtest behind this "
            "screen found little edge in the levels themselves. Do not recommend a trade."
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

STATE_LABEL = {
    "broke_up":   ("Broke up into pocket",   "bullish", "Broke up",   "pos"),
    "broke_down": ("Broke down into pocket", "bearish", "Broke down", "neg"),
    "inside":     ("Inside pocket",          None,      "Inside",     None),
    "testing":    ("Testing a level",        None,      "Testing",    None),
}


def _f(x, fmt="${:.2f}", none="—"):
    return fmt.format(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else none


def scan(symbols: List[str], membership: Dict[str, List[str]],
         per_index: int = PER_INDEX) -> Tuple[List[Dict], Dict]:
    """Screen, map, then publish the best `per_index` per index.

    No market-regime gate. The screen is a volume event, not a directional
    bet, and the one result the backtest supported - breakdowns underperform -
    is most useful precisely when the market is weak.
    """
    frames = fetch_many(symbols)
    # The session being screened is the latest bar anyone has. A symbol whose
    # own last bar is older stopped trading or stopped updating - BZH came
    # back ending 2026-10-19 on a 2026-10-23 run and was screened on a bar
    # four sessions stale. Relative volume on an old bar is not today's.
    as_of = max((df.index[-1] for df in frames.values()), default=None)
    stats = {"fetched": len(frames), "discontinuous": 0, "stale": 0, "pinned": 0,
             "screened": 0, "no_map": 0, "by_state": {}, "as_of": as_of}
    found = []

    for sym, df in frames.items():
        if df.index[-1] < as_of:
            stats["stale"] += 1
            continue
        if not continuous(df):
            stats["discontinuous"] += 1
            continue
        vol = screen(df)
        if not vol:
            continue
        stats["screened"] += 1
        m = map_levels(df)
        if not m or (m["floor"] is None and m["ceiling"] is None):
            stats["no_map"] += 1
            continue
        if m["atr"] / m["price"] < MIN_ATR_PCT:
            stats["pinned"] += 1
            continue

        # Only names that are actually in a pocket: two major levels at least
        # MIN_POCKET_ATR apart, or no level at all on one side (open air,
        # the widest pocket there is). Two levels closer than that are a
        # congestion zone, and labelling a break into one "into a pocket"
        # was wrong on the first run - CCI's "pocket" was $83.60-$86.71.
        open_air = m["floor"] is None or m["ceiling"] is None
        if not open_air and m["width_atr"] < MIN_POCKET_ATR:
            stats["tight"] = stats.get("tight", 0) + 1
            continue

        s = score(m, vol)
        lv = trade_levels(m)
        strategy, bias, label, tone = STATE_LABEL[m["state"]]
        if short_from_strength(m):
            strategy = "Broke down from strength"
        stats["by_state"][strategy] = stats["by_state"].get(strategy, 0) + 1

        fl = m["floor"]["level"] if m["floor"] else None
        cl = m["ceiling"]["level"] if m["ceiling"] else None
        row = {
            "symbol": sym,
            "price": m["price"],
            "score": s["total"],
            "strategy": strategy,
            "signal_state": m["state"],
            "signal_label": label,
            "dimensions": DIMENSIONS,
            "breakdown": {d["key"]: s[d["key"]] for d in DIMENSIONS},
            "indexes": membership.get(sym, [UNTAGGED]),
            "rel_volume": round(vol["rel_volume"], 2),
            "day_move_pct": round(m["day_move_pct"], 2),
            "volume": int(vol["volume"]),
            "avg_volume": int(vol["avg_volume"]),
            "map": {"floor": fl, "ceiling": cl, "atr": round(m["atr"], 4),
                    "width_atr": m["width_atr"], "width_pct": m["width_pct"],
                    "thinness": None if np.isnan(m["thinness"]) else round(m["thinness"], 2),
                    "next_above": m["next_above"], "next_below": m["next_below"],
                    "floor_touches": m["floor"]["touches"] if m["floor"] else None,
                    "ceiling_touches": m["ceiling"]["touches"] if m["ceiling"] else None},
            **lv,
            "setup": {
                # Named, because a break also renders the standard entry/stop/
                # target panel, which is already called Setup.
                "title": "Level map",
                "label": (f"{strategy} · {m['day_move_pct']:+.1f}% today on "
                          f"{vol['rel_volume']:.1f}x volume · "
                          + (f"pocket {m['width_atr']:.1f} ATR / {m['width_pct']:.1f}%"
                             if m["width_atr"] else "open on one side")),
                "fields": [
                    {"label": "Exit", "value": lv.get("exit_rule", "no trade on this state")},
                    {"label": "Off 52w high", "value": f"{m['off_high_pct']:.1f}%"},
                    {"label": "Today", "value": f"{m['day_move_pct']:+.1f}%",
                     "tone": "pos" if m["day_move_pct"] > 0 else "neg" if m["day_move_pct"] < 0 else None},
                    {"label": "Ceiling", "value": _f(cl, none="open air")},
                    {"label": "Floor", "value": _f(fl, none="open air")},
                    {"label": "Room up", "value": _f(m["room_up_pct"], "{:+.1f}%")},
                    {"label": "Room down", "value": _f(m["room_down_pct"], "-{:.1f}%")},
                    {"label": "Pocket", "value": _f(m["width_atr"], "{:.1f} ATR")},
                    {"label": "Rel volume", "value": f"{vol['rel_volume']:.1f}x",
                     "tone": "pos" if vol["rel_volume"] >= 3 else None},
                    {"label": "Volume in zone", "value": _f(m["thinness"], "{:.2f}x even")},
                    {"label": "Next above", "value": _f(m["next_above"])},
                    {"label": "Next below", "value": _f(m["next_below"])},
                ],
            },
        }
        if bias:
            row["bias"] = bias
            row["signal_tone"] = tone
        found.append(row)

    logger.info(f"{stats['screened']} passed the volume screen of {stats['fetched']}; "
                + ", ".join(f"{k} {v}" for k, v in sorted(stats["by_state"].items())))

    # Quota per index, then dedupe. Ranked on score alone: a fresh break is
    # worth 20 of the 100 there, which is enough to lift it without letting
    # breaks crowd out every name already sitting inside a wide pocket - the
    # state most of the hand-drawn examples were in. Forcing breaks first did
    # exactly that on the first run (17 breaks, 1 inside row published).
    buckets = {t: [] for t in INDEX_TAGS + [UNTAGGED]}
    for o in found:
        for tag in o["indexes"]:
            if tag in buckets:
                buckets[tag].append(o)
    key = lambda o: -o["score"]
    picked, selected = set(), []
    for tag in INDEX_TAGS + [UNTAGGED]:
        for o in sorted(buckets[tag], key=key)[:per_index]:
            if o["symbol"] not in picked:
                picked.add(o["symbol"])
                selected.append(o)
    selected.sort(key=key)
    for i, o in enumerate(selected, 1):
        o["rank"] = i
    history.attach(selected, frames)
    stats["published"] = len(selected)
    return selected, stats


# ── Publish ──────────────────────────────────────────────────────────────

def publish(opportunities: List[Dict], membership: Dict[str, List[str]], stats: Dict) -> None:
    """Additive-manifest contract shared by every agent: write only this
    agent's file plus its own manifest entry."""
    os.makedirs(DOCS_DATA_DIR, exist_ok=True)
    counts = {t: sum(1 for v in membership.values() if t in v) for t in INDEX_TAGS + [UNTAGGED]}
    states = ", ".join(f"{k} {v}" for k, v in sorted(stats["by_state"].items())) or "none"
    as_of = stats["as_of"].strftime("%Y-%m-%d") if stats.get("as_of") is not None else "unknown"

    payload = {
        "agent": AGENT,
        "scan_date": datetime.now().isoformat(),
        "dimensions": DIMENSIONS,
        "groups": {
            "key": "indexes", "label": "Index",
            "values": [t for t in INDEX_TAGS + [UNTAGGED]
                       if any(t in o.get("indexes", []) for o in opportunities)],
        },
        "context": [
            {"label": "Screen", "value": f"avg vol >{MIN_AVG_VOLUME // 1000}K, "
                                         f"vol >{MIN_CURRENT_VOLUME // 1000}K, "
                                         f"rel vol >{MIN_RELATIVE_VOLUME}"},
            # The bar the screen was run on. Yahoo's daily bar can land hours
            # after the close, so this is the honest "as of", not the run time.
            {"label": "Data as of", "value": as_of},
            {"label": "Universe", "value": f"{len(membership)} symbols  ("
                                           + " / ".join(f"{t} {n}" for t, n in counts.items() if n) + ")"},
            {"label": "Passed screen", "value": f"{stats['screened']} of {stats['fetched']}"
                                                 f"  ({stats['stale']} stale, {stats['pinned']} pinned dropped)"},
            {"label": "In a pocket", "value": f"{sum(stats['by_state'].values())} "
                                              f"({stats.get('tight', 0)} between close levels dropped)"},
            {"label": "States", "value": states},
            {"label": "Published", "value": str(stats["published"])},
            {"label": "Model", "value": CLAUDE_MODEL},
        ],
        "opportunities": opportunities,
    }
    if not opportunities:
        payload["empty_message"] = (
            f"Nothing passed the volume screen today (rel vol > {MIN_RELATIVE_VOLUME} on "
            f"liquid names). That happens on quiet sessions."
        )

    agent_file = os.path.join(DOCS_DATA_DIR, f"{AGENT['id']}.json")
    # Serialise first: json.dump streams, so a bad value mid-write would leave
    # a truncated file behind.
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
    logger.info(f"Published {len(opportunities)} rows -> {agent_file}")


def load_universe() -> Tuple[List[str], Dict[str, List[str]]]:
    try:
        symbols, membership, sector_map = universe.load()
    except FileNotFoundError:
        sys.exit(f"ERROR: {universe.UNIVERSE_FILE} not found - run scripts/build_universe.py")
    logger.info(universe.describe(symbols, membership, sector_map))
    return symbols, membership


def main():
    logger.info("Starting Pocket - volume screen + level map...")
    symbols, membership = load_universe()
    opportunities, stats = scan(symbols, membership)

    reasoner = Reasoner()
    for o in opportunities:
        o["reasoning"] = reasoner.write(o)

    publish(opportunities, membership, stats)

    with open("pocket_results.json", "w") as f:
        json.dump({"scan_date": datetime.now().isoformat(), "stats": stats,
                   "opportunities": opportunities}, f, indent=2, default=str)

    for o in opportunities:
        m = o["map"]
        logger.info(f"  #{o['rank']:>2} {o['symbol']:<6} {o['strategy']:<24} "
                    f"rv {o['rel_volume']:>4.1f}x  floor {_f(m['floor'])}  ceiling {_f(m['ceiling'])}")
    return opportunities


if __name__ == "__main__":
    main()
