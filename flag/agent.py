#!/usr/bin/env python3
"""Flag (FLAG) - bull flags: a sharp run-up, a quiet drift down, a breakout.

THE PATTERN, as explicit rules (from the "bull flag breakout" setup card)
    trend     close above the 21 EMA, 9 EMA above the 21 EMA
    pole      a run-up of at least 4 ATR and 10% into a high that nothing
              since has exceeded
    flag      the 5-20 sessions after that high: highs sloping down, the
              pullback giving back at most half the pole, and volume drying
              up - lower than during the pole, and falling
    breakout  a close above the flag's upper trendline on at least 1.5x the
              20-day average volume
    trade     entry at the breakout close, stop at the flag's low, held 20
              sessions or until stopped

Published rows are either FORMING (a well-formed flag, not yet broken out -
a watchlist with the trigger price and the stop) or BREAKOUT (broke out
today - carries the trade).

WHAT THE BACKTEST SAID - read before trading the list. Point-in-time, no
lookahead, 2019-2026. Built on a 385-stock sample, then re-run unchanged on
the other 1,098 stocks in the universe, none of which had been looked at:

  * A forming flag alone is not a signal. On the 20 sessions after it was
    first seen it did no better than any uptrend day (+1.4% vs +1.3%), and
    more flags broke their low within 10 sessions (54%) than broke out (24%).
    The forming rows are a watchlist of where a breakout would be, nothing
    more.
  * The breakout is where the result is - from WELL-FORMED flags only, and
    only with the volume confirmation. Well-formed means volume in the flag
    at most 70% of the pole's, a pullback of at most 40% of the pole, and a
    flag of 5-8 sessions. Held 20 sessions or to the stop, in R:

                                   2019-22   2023-26
        sample, well-formed         -0.02     +0.70
        unseen stocks, well-formed  +0.00     +0.54   (n=499, t=2.4 overall)
        unseen stocks, loose flags  -0.00     +0.21
        plain 20-day-high breakout  +0.06     +0.12

    The well-formed filter beat loose flags in both samples. WITHOUT the
    volume rule it did not replicate on the unseen stocks (-0.03 / +0.15),
    so the card's volume rule stays.
  * The edge is recent: in 2019-22 these breakouts broke even. Treat it as a
    regime-dependent edge, not a constant one.
  * Exits: a 20-session hold beat a 2R or 3R target, and exiting on the first
    close below the 9 EMA - the card's suggestion - was the weakest exit.
  * Survivorship flatters every long-side figure (today's index members,
    delisted names absent). Costs are not included. Average stop distance on
    a breakout was ~8% of price, so size accordingly.

The score orders the list; it is not a forecast.
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import anthropic
from anthropic import Anthropic

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import universe
import history

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-haiku-4-5"

AGENT = {
    "id": "FLAG",
    "name": "Flag",
    "strategy": "Bull flag: pole, quiet drift, breakout",
    "description": "Well-formed bull flags across the universe: forming flags with their "
                   "trigger and stop, and breakouts on volume with the trade attached.",
    # Same muted register as the other agents' accents.
    "accent": "#a87f9a",
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DATA_DIR = os.path.join(REPO_ROOT, "docs", "data")
INDEX_TAGS = universe.INDEX_TAGS
UNTAGGED = universe.UNTAGGED

# ~275 sessions: the 200-day SMA on the chart needs 262, and the detector
# needs well under 100.
CALENDAR_DAYS = 400
MIN_ROWS = 120

# ── The pattern. Every number here is the one that was backtested. ────────

POLE_LOOK = 15          # the pole's low is the lowest low in this many sessions before its top
POLE_ATR = 4.0          # pole height at least this many ATR...
POLE_PCT = 0.10         # ...and at least this fraction of price
FLAG_MIN, FLAG_MAX = 5, 20   # flag length searched, shortest valid wins
MAX_RETRACE = 0.5       # flag low may give back at most this much of the pole
MAX_VOL_RATIO = 0.8     # flag volume at most this fraction of the pole's
BREAKOUT_VOL = 1.5      # breakout volume vs the 20-day average, today excluded
EPISODE_DAYS = 10       # a forming flag is followed this long for its breakout

# "Well-formed": the stricter cut that held up on unseen stocks. A flag must
# pass these to be published at all.
WELL_VOL_RATIO = 0.70
WELL_RETRACE = 0.40
WELL_FLAG_MAX = 8

HOLD_SESSIONS = 20
MAX_RISK_PCT = 0.25     # a stop further than this is not a flag trade
MIN_ATR_PCT = 0.005     # pinned (takeover) stocks, as in Pocket
MAX_SESSION_MOVE = 0.40 # unadjusted corporate action guard, as in the other agents
PER_INDEX = 8

DIMENSIONS = [
    {"key": "breakout", "label": "Broke out",       "max": 25},
    {"key": "dry",      "label": "Volume dry-up",   "max": 30},
    {"key": "shallow",  "label": "Shallow pullback", "max": 25},
    {"key": "pole",     "label": "Pole strength",   "max": 20},
]


# ── Data ─────────────────────────────────────────────────────────────────

def warm_tz_cache() -> None:
    """One unthreaded request creates yfinance's timezone cache before any
    threaded batch races to (see Pocket and Trey)."""
    try:
        yf.download("SPY", period="5d", threads=False, progress=False, auto_adjust=True)
    except Exception as e:
        logger.warning(f"tz cache warm-up failed ({e}) - continuing")


def fetch_many(symbols: List[str]) -> Dict[str, pd.DataFrame]:
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
            logger.error(f"Batch {i // CHUNK} came back unlabelled - skipped")
            continue
        present = set(raw.columns.get_level_values(0))
        for sym in chunk:
            if sym in present:
                df = raw[sym].dropna()
                if len(df) >= MIN_ROWS:
                    out[sym] = df
    logger.info(f"{len(out)}/{len(symbols)} symbols have enough history")
    return out


def continuous(df: pd.DataFrame) -> bool:
    try:
        step = (df["Close"] / df["Close"].shift(1)).dropna()
        return len(step) > 0 and bool(((step - 1).abs() < MAX_SESSION_MOVE).all())
    except (KeyError, ValueError, TypeError):
        return False


# ── Detection ────────────────────────────────────────────────────────────

def arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Everything the detector reads, as arrays aligned to df's rows."""
    c = df["Close"]
    h, l, v = df["High"].values, df["Low"].values, df["Volume"].values.astype(float)
    pc = np.r_[np.nan, c.values[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return {
        "h": h, "l": l, "c": c.values, "v": v,
        "e9": c.ewm(span=9, adjust=False).mean().values,
        "e21": c.ewm(span=21, adjust=False).mean().values,
        "atr": pd.Series(tr).rolling(20).mean().values,
        "vavg": pd.Series(v).rolling(20).mean().shift(1).values,
    }


def flag_at(a: Dict[str, np.ndarray], i: int) -> Optional[Dict]:
    """The flag visible at session i, using sessions <= i only, or None.

    The shortest flag length that satisfies every rule wins. The upper
    trendline is fitted through the flag's highs excluding session i, lifted
    so no high sits above it, and projected to i.
    """
    h, l, c, v = a["h"], a["l"], a["c"], a["v"]
    if not (c[i] > a["e21"][i] and a["e9"][i] > a["e21"][i]):
        return None
    for L in range(FLAG_MIN, FLAG_MAX + 1):
        p = i - L
        if p - POLE_LOOK < 0:
            return None
        top = h[p]
        if top < h[p - POLE_LOOK:i + 1].max():
            continue
        pole_low = l[p - POLE_LOOK:p].min()
        pole = top - pole_low
        if not (pole >= POLE_ATR * a["atr"][p] and pole / pole_low >= POLE_PCT):
            continue
        flag_low = l[p + 1:i + 1].min()
        retrace = (top - flag_low) / pole
        if retrace > MAX_RETRACE:
            continue
        pole_start = p - POLE_LOOK + int(np.argmin(l[p - POLE_LOOK:p]))
        vol_ratio = v[p + 1:i].mean() / v[pole_start:p + 1].mean()
        if vol_ratio > MAX_VOL_RATIO:
            continue
        xs = np.arange(p + 1, i)
        if len(xs) < FLAG_MIN - 1:
            continue
        fit = np.polyfit(xs, h[p + 1:i], 1)
        if fit[0] >= 0:                                   # highs must slope down
            continue
        if np.polyfit(xs, v[p + 1:i], 1)[0] >= 0:         # volume must be falling
            continue
        lift = (h[p + 1:i] - np.polyval(fit, xs)).max()
        return {"length": L, "top_idx": p, "top": float(top), "pole_low": float(pole_low),
                "pole_pct": float(pole / pole_low), "pole_atr": float(pole / a["atr"][p]),
                "flag_low": float(flag_low), "retrace": float(retrace),
                "vol_ratio": float(vol_ratio),
                "trigger": float(np.polyval(fit, i) + lift)}
    return None


def well_formed(f: Dict) -> bool:
    return (f["vol_ratio"] <= WELL_VOL_RATIO and f["retrace"] <= WELL_RETRACE
            and f["length"] <= WELL_FLAG_MAX)


def read_flag(df: pd.DataFrame) -> Optional[Dict]:
    """Today's state for one stock: a forming flag, a breakout today, or None.

    Mirrors the backtest. A flag "episode" starts on the first session it is
    visible; its trigger and stop are fixed then. It is followed for
    EPISODE_DAYS sessions: a close below the flag low ends it, a close above
    the trigger on BREAKOUT_VOL volume is the breakout.
    """
    a = arrays(df)
    T = len(a["c"]) - 1
    for i in range(max(0, T - EPISODE_DAYS), T + 1):
        f = flag_at(a, i)
        if f is None:
            continue
        # Earliest visible session in the window = the episode start. Earlier
        # sessions in the window had no flag, so this is where it began.
        if not well_formed(f):
            return None
        state = "forming"
        for j in range(i, T + 1):
            if a["c"][j] < f["flag_low"]:
                return None                                # failed before breaking out
            broke = a["c"][j] > f["trigger"] and a["v"][j] >= BREAKOUT_VOL * a["vavg"][j]
            if broke:
                if j < T:
                    return None                            # broke out earlier - old news
                state = "breakout"
                break
        f.update(state=state, started=df.index[i], sessions=T - i,
                 price=float(a["c"][T]), atr=float(a["atr"][T]),
                 vol_x=float(a["v"][T] / a["vavg"][T]) if a["vavg"][T] > 0 else None,
                 day_move_pct=float((a["c"][T] / a["c"][T - 1] - 1) * 100),
                 top_date=df.index[f["top_idx"]])
        return f
    return None


# ── Score and trade ──────────────────────────────────────────────────────

def score(f: Dict) -> Dict:
    """Descriptive ordering. Dry-up and depth are scored across the
    well-formed range (0.70 -> 0.40 volume ratio, 40% -> 0% retrace)."""
    brk = 25 if f["state"] == "breakout" else 0
    dry = int(round(30 * max(0.0, min(1.0, (WELL_VOL_RATIO - f["vol_ratio"]) / 0.30))))
    shallow = int(round(25 * max(0.0, min(1.0, (WELL_RETRACE - f["retrace"]) / WELL_RETRACE))))
    pole = int(round(20 * max(0.0, min(1.0, (f["pole_atr"] - POLE_ATR) / 6.0))))
    return {"breakout": brk, "dry": dry, "shallow": shallow, "pole": pole,
            "total": brk + dry + shallow + pole}


def trade_levels(f: Dict) -> Dict:
    """Breakouts only: entry at the close, stop at the flag low, held
    HOLD_SESSIONS sessions or until stopped. Forming rows carry none."""
    if f["state"] != "breakout":
        return {}
    risk = f["price"] - f["flag_low"]
    if risk <= 0 or risk / f["price"] > MAX_RISK_PCT:
        return {}
    return {"entry": f["price"], "stop_loss": f["flag_low"],
            "exit_rule": f"{HOLD_SESSIONS} sessions or stop"}


# ── Reasoning ────────────────────────────────────────────────────────────

class Reasoner:
    def __init__(self):
        self.enabled = bool(os.environ.get("ANTHROPIC_API_KEY"))
        self.client = Anthropic() if self.enabled else None
        if not self.enabled:
            logger.warning("ANTHROPIC_API_KEY not set - rows publish without reasoning")

    def write(self, o: Dict) -> str:
        if not self.enabled:
            return "[reasoning unavailable: ANTHROPIC_API_KEY not configured]"
        f = o["flag"]
        prompt = (
            f"{o['symbol']} closed at ${o['price']:.2f} ({o['day_move_pct']:+.1f}% today).\n"
            f"Bull flag: a {f['pole_pct']:.0f}% pole into ${f['pole_top']:.2f}, then a "
            f"{f['length']}-session flag down to ${f['flag_low']:.2f} ({f['retrace']:.0f}% of the pole), "
            f"on {f['vol_ratio']:.0f}% of the pole's volume.\n"
            f"State: {o['strategy']}. Trigger ${f['trigger']:.2f}, stop ${f['flag_low']:.2f}.\n\n"
            "In 2-3 sentences: is this a clean flag or a stock losing momentum, and what "
            "would confirm or kill it? Be concrete and sceptical - flags that are still "
            "forming showed no edge in testing, and more broke down than broke out. "
            "Do not recommend a trade."
        )
        try:
            r = self.client.messages.create(model=CLAUDE_MODEL, max_tokens=250,
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

STATE = {
    "breakout": ("Flag breakout", "Breakout", "pos"),
    "forming":  ("Flag forming",  "Forming",  None),
}


def build_row(sym: str, f: Dict, membership: Dict[str, List[str]]) -> Dict:
    s = score(f)
    lv = trade_levels(f)
    strategy, label, tone = STATE[f["state"]]
    trig_gap = (f["trigger"] / f["price"] - 1) * 100
    row = {
        "symbol": sym,
        "price": f["price"],
        "score": s["total"],
        "strategy": strategy,
        "signal_state": f["state"],
        "signal_label": label,
        "dimensions": DIMENSIONS,
        "breakdown": {d["key"]: s[d["key"]] for d in DIMENSIONS},
        "indexes": membership.get(sym, [UNTAGGED]),
        "day_move_pct": round(f["day_move_pct"], 2),
        "flag": {
            "pole_top": f["top"], "pole_low": f["pole_low"],
            "pole_pct": round(f["pole_pct"] * 100, 1), "length": f["length"],
            "flag_low": f["flag_low"], "retrace": round(f["retrace"] * 100, 1),
            "vol_ratio": round(f["vol_ratio"] * 100, 1), "trigger": f["trigger"],
            "started": f["started"].strftime("%Y-%m-%d"),
            "top_date": f["top_date"].strftime("%Y-%m-%d"),
        },
        # Drawn on the detail-panel chart for every row, so a forming flag shows
        # its trigger and stop without carrying a trade.
        "chart_levels": [
            {"label": "Trigger", "value": f["trigger"], "tone": "neutral"},
            {"label": "Flag low", "value": f["flag_low"], "tone": "neg"},
        ],
        **lv,
        "setup": {
            "title": "Flag",
            "label": (f"{f['pole_pct'] * 100:.0f}% pole · {f['length']}-session flag · "
                      f"{f['retrace'] * 100:.0f}% pullback on {f['vol_ratio'] * 100:.0f}% of pole volume"),
            "fields": [
                {"label": "Exit", "value": lv.get("exit_rule", "watch: no trade until breakout")},
                {"label": "Trigger", "value": f"${f['trigger']:.2f}"},
                {"label": "To trigger", "value": f"{trig_gap:+.1f}%" if f["state"] == "forming" else "cleared",
                 "tone": "pos" if f["state"] == "breakout" else None},
                {"label": "Stop (flag low)", "value": f"${f['flag_low']:.2f}", "tone": "neg"},
                {"label": "Pole", "value": f"+{f['pole_pct'] * 100:.0f}% · {f['pole_atr']:.1f} ATR"},
                {"label": "Pole top", "value": f"${f['top']:.2f}"},
                {"label": "Flag", "value": f"{f['length']} sessions"},
                {"label": "Pullback", "value": f"{f['retrace'] * 100:.0f}% of pole"},
                {"label": "Flag volume", "value": f"{f['vol_ratio'] * 100:.0f}% of pole"},
                {"label": "Today volume", "value": f"{f['vol_x']:.1f}x avg" if f["vol_x"] else "—",
                 "tone": "pos" if f["vol_x"] and f["vol_x"] >= BREAKOUT_VOL else None},
                {"label": "Risk", "value": f"{(f['price'] - f['flag_low']) / f['price'] * 100:.1f}% to stop"},
                {"label": "Today", "value": f"{f['day_move_pct']:+.1f}%",
                 "tone": "pos" if f["day_move_pct"] > 0 else "neg" if f["day_move_pct"] < 0 else None},
            ],
        },
    }
    if f["state"] == "breakout":
        row["bias"] = "bullish"
        row["signal_tone"] = tone
    return row


def scan(symbols: List[str], membership: Dict[str, List[str]],
         per_index: int = PER_INDEX) -> Tuple[List[Dict], Dict]:
    frames = fetch_many(symbols)
    as_of = max((df.index[-1] for df in frames.values()), default=None)
    stats = {"fetched": len(frames), "stale": 0, "discontinuous": 0, "pinned": 0,
             "forming": 0, "breakout": 0, "as_of": as_of}
    found = []
    for sym, df in frames.items():
        if df.index[-1] < as_of:
            stats["stale"] += 1
            continue
        if not continuous(df):
            stats["discontinuous"] += 1
            continue
        f = read_flag(df)
        if f is None:
            continue
        if not f["atr"] or f["atr"] / f["price"] < MIN_ATR_PCT:
            stats["pinned"] += 1
            continue
        stats[f["state"]] += 1
        found.append(build_row(sym, f, membership))

    logger.info(f"{stats['breakout']} breakouts, {stats['forming']} forming, of {stats['fetched']}")

    # Every breakout publishes; forming flags fill each index's quota by score.
    selected = [o for o in found if o["signal_state"] == "breakout"]
    picked = {o["symbol"] for o in selected}
    for tag in INDEX_TAGS + [UNTAGGED]:
        pool = sorted((o for o in found if tag in o["indexes"] and o["symbol"] not in picked),
                      key=lambda o: -o["score"])
        room = max(0, per_index - sum(1 for o in selected if tag in o["indexes"]))
        for o in pool[:room]:
            picked.add(o["symbol"])
            selected.append(o)
    selected.sort(key=lambda o: (o["signal_state"] != "breakout", -o["score"]))
    for i, o in enumerate(selected, 1):
        o["rank"] = i
    history.attach(selected, frames)
    stats["published"] = len(selected)
    return selected, stats


# ── Publish ──────────────────────────────────────────────────────────────

def publish(opportunities: List[Dict], membership: Dict[str, List[str]], stats: Dict) -> None:
    os.makedirs(DOCS_DATA_DIR, exist_ok=True)
    counts = {t: sum(1 for v in membership.values() if t in v) for t in INDEX_TAGS + [UNTAGGED]}
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
            {"label": "Data as of", "value": as_of},
            {"label": "Universe", "value": f"{len(membership)} symbols  ("
                                           + " / ".join(f"{t} {n}" for t, n in counts.items() if n) + ")"},
            {"label": "Found", "value": f"{stats['breakout']} breakouts, {stats['forming']} forming"},
            {"label": "Well-formed", "value": f"flag vol <= {WELL_VOL_RATIO:.0%} of pole, pullback "
                                              f"<= {WELL_RETRACE:.0%}, {FLAG_MIN}-{WELL_FLAG_MAX} sessions"},
            {"label": "Breakout", "value": f"close > trigger on >= {BREAKOUT_VOL}x volume"},
            {"label": "Dropped", "value": f"{stats['stale']} stale, {stats['pinned']} pinned, "
                                          f"{stats['discontinuous']} discontinuous"},
            {"label": "Published", "value": str(stats["published"])},
            {"label": "Model", "value": CLAUDE_MODEL},
        ],
        "opportunities": opportunities,
    }
    if not opportunities:
        payload["empty_message"] = "No well-formed bull flags today."
    agent_file = os.path.join(DOCS_DATA_DIR, f"{AGENT['id']}.json")
    blob = json.dumps(payload, indent=2, default=str)
    with open(agent_file, "w") as fh:
        fh.write(blob)

    manifest_path = os.path.join(DOCS_DATA_DIR, "agents.json")
    try:
        with open(manifest_path) as fh:
            registered = json.load(fh).get("agents", [])
    except (FileNotFoundError, json.JSONDecodeError):
        registered = []
    if AGENT["id"] not in registered:
        registered.append(AGENT["id"])
        with open(manifest_path, "w") as fh:
            json.dump({"agents": registered}, fh, indent=2)
    logger.info(f"Published {len(opportunities)} rows -> {agent_file}")


def main():
    logger.info("Starting Flag - bull flag scanner...")
    try:
        symbols, membership, sector_map = universe.load()
    except FileNotFoundError:
        sys.exit("ERROR: universe.txt not found - run scripts/build_universe.py")
    opportunities, stats = scan(symbols, membership)
    reasoner = Reasoner()
    for o in opportunities:
        o["reasoning"] = reasoner.write(o)
    publish(opportunities, membership, stats)
    with open("flag_results.json", "w") as fh:
        json.dump({"scan_date": datetime.now().isoformat(), "stats": stats,
                   "opportunities": opportunities}, fh, indent=2, default=str)
    for o in opportunities:
        f = o["flag"]
        logger.info(f"  #{o['rank']:>2} {o['symbol']:<6} {o['strategy']:<14} trigger "
                    f"${f['trigger']:.2f}  stop ${f['flag_low']:.2f}  score {o['score']}")
    return opportunities


if __name__ == "__main__":
    main()
