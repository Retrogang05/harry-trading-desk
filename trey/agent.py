"""Trey: a TQQQ position-state signal, computed on QQQ.

Not a screener. Monu and Opy rank candidates out of a 500-name universe;
Trey watches one instrument and publishes one row whose "score" is really a
state. Three states, all decided on the unlevered index so that TQQQ's 3x
noise never generates a false crossover:

    OUT   QQQ close below its 200-day SMA            -> hold cash
    HALF  above 200-day, fast trend not confirmed    -> half position
    FULL  above 200-day, 10d > 20d and close > 50d   -> full position

Signals are read at the close and meant to be acted on the next session.
The scheduled run fires after the US close so the published state is
"tomorrow's" position.

WHY THESE RULES AND NOT THE BOOK'S
    Derived from Masonson's QQQ/TQQQ system, then backtested on real data
    before any of it was written down here. What survived:
      - the 200/225-day gate on QQQ: on real TQQQ over the last ten years the
        SMA200 gate matched buy-and-hold's return (2889% vs 2877%) while
        cutting max drawdown from -82% to -56%. One 2022 save (-47% vs -79%)
        paid for every whipsaw around it.
      - the fast 10>20/price>50 signal as a SIZE modifier, not a gate: best
        Sharpe of every variant tested (0.86 vs 0.61 buy-and-hold) and it
        caught the COVID crash a 200-day system is too slow for
        (-42% vs -55%).
    What did not survive:
      - the seasonal "Jul-Oct flat" rule: worse in every single variant.
      - "all conditions must align" as the entry rule: worst in every test.
        More filters just meant later entries.
      - 225 vs 200 days: regime-dependent. 225 wins on 25 years of QQQ (one
        dot-com-era event), 200 wins on TQQQ's whole life and on the last
        decade. It's a constant below; neither is a robust edge.

Same resilience shape as the other two agents: reasoning is gated on the
API key up front with a last-resort except, so a narration failure can never
take down the signal itself.
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
import anthropic
from anthropic import Anthropic

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-haiku-4-5"

AGENT = {
    "id": "TREY",
    "name": "Trey",
    "strategy": "TQQQ trend gate",
    "description": "Holds TQQQ only while QQQ is above its 200-day average; sizes up when "
                    "the short-term trend confirms. Signals on QQQ, execution on TQQQ.",
    "accent": "#6f8fae",   # muted blue - the register the README asks for
}

SIGNAL = "QQQ"    # where every rule is evaluated
TRADED = "TQQQ"   # what the state applies to

# The gate. 200 vs 225 is the one parameter the backtest could not settle -
# see the module docstring. Change here and nowhere else.
GATE_SMA = 200
FAST_SMA, SLOW_SMA, MID_SMA = 10, 20, 50

# The raw 10>20 crossover flips size roughly every two weeks, and about a
# quarter of those flips reverse within a single day. Requiring the new
# value to hold for CONFIRM_DAYS before it changes the position was the one
# refinement that improved BOTH Sharpe (0.86 -> 0.88) and return on the
# last ten years of TQQQ while cutting size trades by a third. Longer is
# worse: at 5 days the signal is too slow to catch a COVID-speed crash,
# which is the reason the sizing layer exists at all. Applies to the fast
# trend only - the 200-day gate is slow enough to not need it.
CONFIRM_DAYS = 3

# 200 trading days of warm-up plus a year of history so the published
# "since" date and change-count are real rather than truncated by the fetch.
CALENDAR_DAYS = 730

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DATA_DIR = os.path.join(REPO_ROOT, "docs", "data")

# The score IS the state. Bands on the dashboard are shared across agents
# (80+ Strong / 70-79 Buy / 60-69 Watch / <60 Weak), so the weights below
# are chosen to land each state in the band that reads correctly:
#   OUT  -> 0    Weak
#   HALF -> 60   Watch
#   FULL -> 100  Strong
# The fast trend is scored as one all-or-nothing dimension because FULL
# requires both halves; splitting it would let HALF read as 80/Strong.
DIMENSIONS = [
    {"key": "gate", "label": f"QQQ > {GATE_SMA}d", "max": 60},
    {"key": "fast", "label": f"{FAST_SMA}>{SLOW_SMA} & P>{MID_SMA}d", "max": 40},
]


# ── Data ─────────────────────────────────────────────────────────────────

def fetch() -> Dict[str, pd.Series]:
    end = datetime.now()
    start = end - timedelta(days=CALENDAR_DAYS)
    raw = yf.download([SIGNAL, TRADED], start=start, end=end, auto_adjust=True,
                      progress=False, group_by="ticker", threads=True)
    out = {}
    for sym in (SIGNAL, TRADED):
        frame = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
        close = frame["Close"].dropna()
        if len(close) < GATE_SMA + 5:
            raise RuntimeError(f"{sym}: only {len(close)} closes, need > {GATE_SMA} for the gate")
        out[sym] = close
    return out


# ── Signal ───────────────────────────────────────────────────────────────

def confirmed(sig: pd.Series, n: int) -> pd.Series:
    """Hysteresis: the output only takes the input's new value once the input
    has held it for n consecutive days. Identical to the function the
    backtest used, so the live signal matches what was tested."""
    if n <= 1:
        return sig.astype(bool)
    vals = sig.astype(bool).values
    out = vals.copy()
    cur, run = bool(vals[0]), 0
    for i, v in enumerate(vals):
        if v != cur:
            run += 1
            if run >= n:
                cur, run = v, 0
        else:
            run = 0
        out[i] = cur
    return pd.Series(out, index=sig.index)


def states(qqq: pd.Series) -> pd.Series:
    """Daily state over the whole fetched window, so we can say since-when."""
    sma = lambda n: qqq.rolling(n).mean()
    gate = qqq > sma(GATE_SMA)
    fast = confirmed((sma(FAST_SMA) > sma(SLOW_SMA)) & (qqq > sma(MID_SMA)), CONFIRM_DAYS)
    s = pd.Series("OUT", index=qqq.index)
    s[gate & ~fast] = "HALF"
    s[gate & fast] = "FULL"
    # Rows before the gate SMA exists aren't a real OUT, they're unknown.
    s[sma(GATE_SMA).isna()] = None
    return s.dropna()


def current(qqq: pd.Series, tqqq: pd.Series) -> Dict:
    st = states(qqq)
    today = st.index[-1]
    state = st.iloc[-1]

    # Walk back to the first day of the current run of this state.
    since = today
    for d in reversed(st.index[:-1]):
        if st[d] != state:
            break
        since = d

    # Transitions in the trailing 12 months, split two ways because they are
    # very different costs: a gate change is a full round trip (rare, ~2/yr);
    # a size change is a half-position trade (frequent, ~8/yr even after
    # confirmation). Reporting one blended number hid that the first time.
    year_ago = today - pd.Timedelta(days=365)
    recent = st[st.index >= year_ago]
    inout = recent.map(lambda x: "OUT" if x == "OUT" else "IN")
    changed = recent != recent.shift(1)
    gate_changes = int((inout != inout.shift(1)).sum() - 1) if len(recent) > 1 else 0
    size_changes = int((changed & (inout == inout.shift(1))).sum())
    prev_state = None
    for d in reversed(st.index):
        if st[d] != state:
            prev_state = st[d]
            break

    sma = lambda n: qqq.rolling(n).mean().iloc[-1]
    q = float(qqq.iloc[-1])
    return {
        "date": today.date().isoformat(),
        "state": state,
        "since": since.date().isoformat(),
        "prev_state": prev_state,
        "gate_changes_12m": gate_changes,
        "size_changes_12m": size_changes,
        "qqq": q,
        "tqqq": float(tqqq.iloc[-1]),
        "gate_sma": float(sma(GATE_SMA)),
        "gate_pct": (q / float(sma(GATE_SMA)) - 1) * 100,
        "fast_sma": float(sma(FAST_SMA)),
        "slow_sma": float(sma(SLOW_SMA)),
        "mid_sma": float(sma(MID_SMA)),
        "mid_pct": (q / float(sma(MID_SMA)) - 1) * 100,
        "fast_ok": bool(sma(FAST_SMA) > sma(SLOW_SMA)),
        "mid_ok": bool(q > sma(MID_SMA)),
    }


# ── Reasoning ────────────────────────────────────────────────────────────

class Reasoner:
    def __init__(self):
        self.enabled = bool(os.environ.get("ANTHROPIC_API_KEY"))
        self.client = Anthropic() if self.enabled else None
        if not self.enabled:
            logger.warning("ANTHROPIC_API_KEY not set - signal will publish without written reasoning")

    def explain(self, c: Dict) -> str:
        if not self.enabled:
            return "[reasoning unavailable: ANTHROPIC_API_KEY not configured]"
        prompt = (
            f"TQQQ position signal, computed on QQQ. Today's state: {c['state']}, held since "
            f"{c['since']}" + (f" (previously {c['prev_state']})" if c['prev_state'] else "") + ".\n"
            f"QQQ {c['qqq']:.2f} is {c['gate_pct']:+.1f}% vs its {GATE_SMA}-day average "
            f"({c['gate_sma']:.2f}). Fast trend: {FAST_SMA}d {'above' if c['fast_ok'] else 'below'} "
            f"{SLOW_SMA}d; price {c['mid_pct']:+.1f}% vs {MID_SMA}d. {c['gate_changes_12m']} gate changes and "
            f"{c['size_changes_12m']} size changes in the last 12 months.\n\n"
            "Rules: OUT below the gate average; HALF above it; FULL when the fast trend also "
            "confirms. Write 2-3 sentences: what the current state means for a TQQQ holder, "
            "how close the gate or fast trend is to flipping, and the risk of acting on it. "
            "Professional, concise, do not restate the numbers verbatim, do not recommend a trade."
        )
        try:
            msg = self.client.messages.create(model=CLAUDE_MODEL, max_tokens=220,
                                              messages=[{"role": "user", "content": prompt}])
            text = "".join(b.text for b in msg.content if b.type == "text").strip()
            return text or "[reasoning unavailable: empty response]"
        except anthropic.RateLimitError:
            return "[reasoning unavailable: rate limited]"
        except anthropic.APIStatusError as e:
            return f"[reasoning unavailable: API error {e.status_code}]"
        except anthropic.APIConnectionError:
            return "[reasoning unavailable: connection error]"
        except Exception as e:
            logger.error(f"unexpected reasoning failure: {type(e).__name__}: {e}")
            return f"[reasoning unavailable: {type(e).__name__}]"


# ── Publish ──────────────────────────────────────────────────────────────

def build_row(c: Dict, reasoning: str) -> Dict:
    gate_pts = 60 if c["state"] != "OUT" else 0
    fast_pts = 40 if c["state"] == "FULL" else 0
    tone = {"FULL": "pos", "HALF": None, "OUT": "neg"}[c["state"]]
    yn = lambda ok: "yes" if ok else "no"
    return {
        "rank": 1,
        "symbol": TRADED,
        "price": round(c["tqqq"], 2),
        "score": gate_pts + fast_pts,
        "strategy": f"Trend gate · {c['state']}",
        "dimensions": DIMENSIONS,
        "breakdown": {"gate": gate_pts, "fast": fast_pts},
        "setup": {
            "label": f"{c['state']} since {c['since']} · QQQ {c['qqq']:.2f}",
            "fields": [
                {"label": "State", "value": c["state"], "tone": tone},
                {"label": f"QQQ vs {GATE_SMA}d", "value": f"{c['gate_pct']:+.1f}%",
                 "tone": "pos" if c["gate_pct"] > 0 else "neg"},
                {"label": f"{FAST_SMA}d > {SLOW_SMA}d", "value": yn(c["fast_ok"])},
                {"label": f"Price vs {MID_SMA}d", "value": f"{c['mid_pct']:+.1f}%"},
            ],
        },
        "reasoning": reasoning,
    }


def publish(row: Dict, c: Dict) -> None:
    """Same additive-manifest contract as Monu and Opy: this agent writes only
    its own file plus its own manifest entry, so all three can run on
    independent schedules without clobbering one another."""
    os.makedirs(DOCS_DATA_DIR, exist_ok=True)
    payload = {
        "agent": AGENT,
        "scan_date": datetime.now().isoformat(),
        "context": [
            {"label": "State", "value": c["state"]},
            {"label": "Since", "value": c["since"]},
            {"label": "Gate changes 12m", "value": str(c["gate_changes_12m"])},
            {"label": "Size changes 12m", "value": str(c["size_changes_12m"])},
            {"label": "Signal on", "value": SIGNAL},
            {"label": "Gate", "value": f"{GATE_SMA}-day SMA"},
        ],
        "opportunities": [row],
    }
    with open(os.path.join(DOCS_DATA_DIR, f"{AGENT['id']}.json"), "w") as f:
        json.dump(payload, f, indent=2)

    manifest = os.path.join(DOCS_DATA_DIR, "agents.json")
    try:
        with open(manifest) as f:
            registered = json.load(f).get("agents", [])
    except (FileNotFoundError, json.JSONDecodeError):
        registered = []
    if AGENT["id"] not in registered:
        registered.append(AGENT["id"])
        with open(manifest, "w") as f:
            json.dump({"agents": registered}, f, indent=2)
    logger.info(f"Published {AGENT['id']}: {c['state']} since {c['since']}")


def main():
    logger.info(f"Trey: {TRADED} state from {SIGNAL} vs {GATE_SMA}-day gate")
    reasoner = Reasoner()
    px = fetch()
    c = current(px[SIGNAL], px[TRADED])
    logger.info(
        f"{c['date']}: {c['state']} (since {c['since']}) - QQQ {c['qqq']:.2f} "
        f"{c['gate_pct']:+.1f}% vs {GATE_SMA}d, fast={'ok' if c['fast_ok'] else 'no'}, "
        f"P>{MID_SMA}={'ok' if c['mid_ok'] else 'no'}, {c['gate_changes_12m']} gate / {c['size_changes_12m']} size changes 12m"
    )
    row = build_row(c, reasoner.explain(c))
    publish(row, c)
    with open("trey_results.json", "w") as f:
        json.dump({"scan_date": datetime.now().isoformat(), "signal": c, "row": row}, f, indent=2)
    return c


if __name__ == "__main__":
    main()
