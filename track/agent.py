#!/usr/bin/env python3
"""Results - what happened to every signal the agents published.

Replays each agent's dated archive (results/, goldy/results/, ...) against
the prices that came after, using the trade each row published, and writes
docs/data/results.json for the dashboard's Results tab.

The archives were written before their outcomes existed, so this is a
forward test, not a backtest. It is rebuilt from scratch on every run.

Fill and exit rules, the same for every agent:
    signal day  the session the row's price came from
    entry       next session's open. A Monu PULLBACK is a limit at its entry:
                filled at min(open, entry) if a low reaches it within 5
                sessions, otherwise "not filled"
    skipped     the open is already through the stop
    stop        hit when a low (long) or high (short) touches it; a gap
                through it exits at the open
    target      likewise. Stop and target on the same day counts as the stop
    time exit   Pocket and Flag 20 sessions, Goldy 63 (its "3-month hold").
                Monu has none: open until stop or target
    R           (exit - fill) / (fill - stop), signed for shorts
    repeats     a signal for a symbol the agent already holds (or is waiting
                to fill) is ignored, as you would not buy it twice

Opy:
    credit spreads  held to expiry, settled on the expiry close:
                    R = (credit - loss at expiry) / max loss. Credit is the
                    published mid, so real fills would be a little worse
    iron condors    no strikes are published, so only whether the stock
                    finished inside its published expected move
    LEAPS, RSI      not tracked (LEAPS run 400+ days; RSI rows have no trade)
"""
import glob
import json
import logging
import os
import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "data", "results.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("results")

# Agents tracked, in tab order. `backtest` is the expectancy the agent's own
# backtest found for the trades it publishes, shown beside the live figure.
AGENTS = [
    {"id": "MNTM", "name": "Monu", "dir": "results", "accent": "#c8974a",
     "backtest": None, "backtest_note": "Only the A grade's two filters were tested"},
    {"id": "PCKT", "name": "Pocket", "dir": "pocket/results", "accent": "#a8857f",
     "backtest": 0.21, "backtest_note": "Breakouts, 2020-26"},
    {"id": "FLAG", "name": "Flag", "dir": "flag/results", "accent": "#a87f9a",
     "backtest": 0.54, "backtest_note": "Well-formed breakouts, unseen stocks 2023-26"},
    {"id": "GOLD", "name": "Goldy", "dir": "goldy/results", "accent": "#7fa88a",
     "backtest": None, "backtest_note": "Crosses did no better than random days"},
    {"id": "OPY", "name": "Opy", "dir": "opy/results", "accent": "#8a7fa8",
     "backtest": None, "backtest_note": "Not backtested"},
]

FILL_WINDOW = 5          # sessions a Monu pullback limit stays working
TIME_EXIT = {"PCKT": 20, "FLAG": 20, "GOLD": 63}
SPLIT_LIKE = 0.15        # published price this far from the bar = split since


# ── Signals ────────────────────────────────────────────────────────────────

def archives(d: str):
    for f in sorted(glob.glob(os.path.join(ROOT, d, "scan_*.json"))):
        try:
            j = json.load(open(f))
        except (OSError, ValueError):
            logger.warning(f"unreadable archive {f}")
            continue
        yield pd.Timestamp(os.path.basename(f)[5:13]), j


def group_of(agent: str, o: Dict) -> str:
    short = o.get("bias") == "bearish"
    if agent == "MNTM":
        if short:
            return "Breakdown"
        g = o.get("structure")
        return f"Long · grade {g}" if g else "Long · before grades"
    if agent == "PCKT":
        return "Breakdown" if short else "Breakout"
    if agent == "FLAG":
        return "Breakout"
    if agent == "GOLD":
        return o.get("strategy") or "Cross"
    return o.get("strategy") or ""


def stock_signals() -> List[Dict]:
    out = []
    for a in AGENTS:
        if a["id"] == "OPY":
            continue
        for day, j in archives(a["dir"]):
            for o in j.get("opportunities", []):
                if o.get("entry") is None or o.get("stop_loss") is None:
                    continue
                out.append({
                    "agent": a["id"], "symbol": o["symbol"], "file_date": day,
                    "price": float(o["price"]), "entry": float(o["entry"]),
                    "stop": float(o["stop_loss"]),
                    "target": float(o["take_profit"]) if o.get("take_profit") is not None else None,
                    "side": -1 if o.get("bias") == "bearish" else 1,
                    "limit": o.get("setup_type") == "PULLBACK",
                    "strategy": o.get("strategy") or o.get("setup_type") or "",
                    "group": group_of(a["id"], o), "score": o.get("score"),
                })
    return out


SPREAD = re.compile(r"(Bull Put|Bear Call)\s+([\d.]+)/([\d.]+)\s+·\s+(\d+)d")
CONDOR = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _field(o: Dict, label: str) -> Optional[float]:
    for f in (o.get("setup") or {}).get("fields", []):
        if f.get("label") == label:
            m = re.search(r"-?[\d.]+", str(f.get("value", "")).replace(",", ""))
            return float(m.group()) if m else None
    return None


def _friday(d: date) -> date:
    """Option expiries are Fridays; DTE was counted from the run date, so
    snap to the nearest one."""
    return d + timedelta(days=((4 - d.weekday() + 3) % 7) - 3)


def option_signals():
    spreads, condors = [], []
    for day, j in archives("opy/results"):
        run = pd.Timestamp(j.get("scan_date") or day).date()
        for o in j.get("opportunities", []):
            label = (o.get("setup") or {}).get("label", "")
            if "Spread" in (o.get("strategy") or ""):
                m = SPREAD.search(label)
                credit = _field(o, "Credit")
                if not m or credit is None:
                    continue
                kind, k1, k2, dte = m.group(1), float(m.group(2)), float(m.group(3)), int(m.group(4))
                width = abs(k1 - k2)
                if width <= credit or credit <= 0:
                    continue
                spreads.append({
                    "symbol": o["symbol"], "file_date": day, "price": float(o["price"]),
                    "kind": kind, "short": k1, "long": k2, "credit": credit, "width": width,
                    "expiry": pd.Timestamp(_friday(run + timedelta(days=dte))),
                })
            elif o.get("strategy") == "Iron Condor":
                m = CONDOR.search(label)
                em = _field(o, "Exp. Move")
                if not m or not em:
                    continue
                condors.append({"symbol": o["symbol"], "file_date": day, "price": float(o["price"]),
                                "expiry": pd.Timestamp(m.group(1)), "move": em / 100})
    return spreads, condors


# ── Prices ─────────────────────────────────────────────────────────────────

def fetch(symbols: List[str], start: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    # auto_adjust=False: split-adjusted but not dividend-adjusted, the closest
    # to the prices the agents saw on the day.
    raw = yf.download(sorted(symbols), start=(start - timedelta(days=10)).strftime("%Y-%m-%d"),
                      auto_adjust=False, progress=False, threads=True, group_by="ticker")
    out = {}
    for s in symbols:
        try:
            df = raw[s] if len(symbols) > 1 or s in raw.columns.get_level_values(0) else raw
            df = df[["Open", "High", "Low", "Close"]].dropna()
        except Exception:
            continue
        if len(df):
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            out[s] = df
    logger.info(f"prices for {len(out)}/{len(symbols)} symbols")
    return out


def signal_day(df: pd.DataFrame, file_date: pd.Timestamp, price: float):
    """The session the published price came from: whichever of the last
    three sessions up to the file date closed nearest it. Older archives were
    named by run date, not session. Returns (index position, scale) where
    scale corrects the levels for a split since."""
    pos = df.index.searchsorted(file_date, side="right") - 1
    if pos < 0:
        return None, 1.0
    cands = range(max(0, pos - 2), pos + 1)
    best = min(cands, key=lambda i: abs(df["Close"].iloc[i] / price - 1))
    ratio = df["Close"].iloc[best] / price
    if abs(ratio - 1) > 0.02:
        best = pos                                   # intraday price: use the file's session
    return best, (ratio if abs(ratio - 1) > SPLIT_LIKE else 1.0)


# ── Replay ─────────────────────────────────────────────────────────────────

def replay(s: Dict, df: pd.DataFrame) -> Optional[Dict]:
    i, scale = signal_day(df, s["file_date"], s["price"])
    if i is None:
        return None
    side = s["side"]
    entry, stop = s["entry"] * scale, s["stop"] * scale
    target = s["target"] * scale if s["target"] else None
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    dates = df.index
    row = {"agent": s["agent"], "symbol": s["symbol"], "group": s["group"],
           "strategy": s["strategy"], "side": "short" if side < 0 else "long",
           "score": s["score"], "signal_date": dates[i].strftime("%Y-%m-%d"),
           "stop": round(stop, 4), "target": round(target, 4) if target else None}
    n = len(df)
    if i + 1 >= n:
        return {**row, "status": "pending"}

    # Entry
    k = i + 1
    if s["limit"]:
        fill = None
        for j in range(k, min(n, k + FILL_WINDOW)):
            if (side > 0 and l[j] <= entry) or (side < 0 and h[j] >= entry):
                fill, k = (min(o[j], entry) if side > 0 else max(o[j], entry)), j
                break
        if fill is None:
            if k + FILL_WINDOW <= n:
                return {**row, "status": "skipped", "reason": "Limit not filled"}
            return {**row, "status": "pending"}
    else:
        fill = o[k]
    if (fill - stop) * side <= 0:
        return {**row, "status": "skipped", "reason": "Opened through the stop",
                "entry_date": dates[k].strftime("%Y-%m-%d")}
    risk = (fill - stop) * side
    row.update(entry=round(float(fill), 4), entry_date=dates[k].strftime("%Y-%m-%d"))

    # Exit. The entry bar counts for stops and targets only after a market
    # open fill (the whole bar is then after the fill); a limit fill could
    # come at the bar's low, so its own bar is skipped.
    hold = TIME_EXIT.get(s["agent"])
    first = k if not s["limit"] else k + 1
    for j in range(first, n):
        lo, hi, op = l[j], h[j], o[j]
        adverse = lo if side > 0 else hi
        favour = hi if side > 0 else lo
        if (adverse - stop) * side <= 0:
            px = op if (op - stop) * side <= 0 and j > k else stop
            return _close(row, px, fill, risk, side, dates[j], "Stop", j - k)
        if target is not None and (favour - target) * side >= 0:
            px = op if (op - target) * side >= 0 and j > k else target
            return _close(row, px, fill, risk, side, dates[j], "Target", j - k)
        if hold and j - k + 1 >= hold:
            return _close(row, c[j], fill, risk, side, dates[j], f"{hold} sessions", j - k)
    last = n - 1
    return {**row, "status": "open", "last": round(float(c[last]), 4),
            "r": round(float((c[last] - fill) * side / risk), 2), "sessions": int(last - k + 1)}


def _close(row, px, fill, risk, side, d, why, held):
    return {**row, "status": "closed", "exit": round(float(px), 4),
            "exit_date": d.strftime("%Y-%m-%d"), "reason": why, "sessions": int(held + 1),
            "r": round(float((px - fill) * side / risk), 2)}


def replay_stocks(signals: List[Dict], px: Dict[str, pd.DataFrame]) -> List[Dict]:
    trades, busy = [], {}            # (agent, symbol, side) -> date the last trade frees up
    for s in sorted(signals, key=lambda s: (s["file_date"], s["agent"])):
        df = px.get(s["symbol"])
        if df is None:
            continue
        key = (s["agent"], s["symbol"], s["side"])
        if key in busy and s["file_date"] <= busy[key]:
            continue
        t = replay(s, df)
        if not t:
            continue
        trades.append(t)
        free = t.get("exit_date") or (None if t["status"] in ("open", "pending") else t["signal_date"])
        busy[key] = pd.Timestamp(free) if free else pd.Timestamp.max
    return trades


def settle_spread(s: Dict, df: pd.DataFrame) -> Optional[Dict]:
    maxloss = s["width"] - s["credit"]
    row = {"agent": "OPY", "symbol": s["symbol"], "group": s["kind"] + " spread",
           "strategy": f"{s['kind']} {s['short']:g}/{s['long']:g}",
           "side": "short" if s["kind"] == "Bear Call" else "long",
           "signal_date": s["file_date"].strftime("%Y-%m-%d"),
           "expiry": s["expiry"].strftime("%Y-%m-%d"), "credit": s["credit"],
           "entry": s["price"]}

    def loss_at(S):
        over = (s["short"] - S) if s["kind"] == "Bull Put" else (S - s["short"])
        return min(max(over, 0.0), s["width"])

    if df.index[-1] >= s["expiry"]:
        pos = df.index.searchsorted(s["expiry"], side="right") - 1
        S = float(df["Close"].iloc[pos])
        r = (s["credit"] - loss_at(S)) / maxloss
        return {**row, "status": "closed", "exit": round(S, 2),
                "exit_date": df.index[pos].strftime("%Y-%m-%d"), "reason": "Expiry", "r": round(r, 2)}
    S = float(df["Close"].iloc[-1])
    return {**row, "status": "open", "last": round(S, 2),
            "r": round((s["credit"] - loss_at(S)) / maxloss, 2)}


def replay_options(spreads, condors, px):
    trades, seen = [], set()
    for s in spreads:
        key = (s["symbol"], s["kind"], s["expiry"])     # one spread per expiry
        if key in seen or s["symbol"] not in px:
            continue
        seen.add(key)
        t = settle_spread(s, px[s["symbol"]])
        if t:
            trades.append(t)
    rows, seen = [], set()
    for s in condors:
        key = (s["symbol"], s["expiry"])
        if key in seen or s["symbol"] not in px or px[s["symbol"]].index[-1] < s["expiry"]:
            continue
        seen.add(key)
        df = px[s["symbol"]]
        pos = df.index.searchsorted(s["expiry"], side="right") - 1
        S = float(df["Close"].iloc[pos])
        move = S / s["price"] - 1
        rows.append({"symbol": s["symbol"], "signal_date": s["file_date"].strftime("%Y-%m-%d"),
                     "expiry": s["expiry"].strftime("%Y-%m-%d"), "move": round(move * 100, 1),
                     "allowed": round(s["move"] * 100, 1), "inside": bool(abs(move) <= s["move"])})
    return trades, rows


# ── Summary ────────────────────────────────────────────────────────────────

def stats(trades: List[Dict]) -> Dict:
    closed = [t for t in trades if t["status"] == "closed"]
    open_ = [t for t in trades if t["status"] == "open"]
    rs = [t["r"] for t in sorted(closed, key=lambda t: t["exit_date"])]
    out = {"closed": len(rs), "open": len(open_),
           "skipped": sum(t["status"] == "skipped" for t in trades),
           "open_r": round(sum(t["r"] for t in open_), 2)}
    # Closed trades alone lean to losers early on: stops close fast, time
    # exits and targets take weeks. Marking open trades at the last close
    # gives the fairer running figure.
    marked = rs_all = [t["r"] for t in closed + open_]
    if marked:
        out["marked_r"] = round(float(np.mean(rs_all)), 2)
    if not rs:
        return out
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    streak = run = 0
    for r in rs:
        run = run + 1 if r <= 0 else 0
        streak = max(streak, run)
    out.update(win_rate=round(len(wins) / len(rs) * 100, 1), avg_r=round(float(np.mean(rs)), 2),
               total_r=round(float(np.sum(rs)), 2),
               avg_win=round(float(np.mean(wins)), 2) if wins else None,
               avg_loss=round(float(np.mean(losses)), 2) if losses else None,
               profit_factor=(round(sum(wins) / -sum(losses), 2) if losses and sum(losses) < 0 else None),
               worst_streak=streak)
    return out


def curve(trades: List[Dict]) -> Dict:
    closed = sorted((t for t in trades if t["status"] == "closed"), key=lambda t: t["exit_date"])
    dates, total, run = [], [], 0.0
    for t in closed:
        run += t["r"]
        if dates and dates[-1] == t["exit_date"]:
            total[-1] = round(run, 2)
        else:
            dates.append(t["exit_date"])
            total.append(round(run, 2))
    return {"dates": dates, "r": total}


def main():
    stock = stock_signals()
    spreads, condors = option_signals()
    syms = {s["symbol"] for s in stock} | {s["symbol"] for s in spreads} | {s["symbol"] for s in condors}
    starts = [s["file_date"] for s in stock + spreads + condors]
    if not starts:
        logger.error("no archived signals found")
        return
    px = fetch(sorted(syms), min(starts))
    through = max(df.index[-1] for df in px.values())

    trades = replay_stocks(stock, px)
    opt_trades, condor_rows = replay_options(spreads, condors, px)
    trades += opt_trades
    trades = [t for t in trades if t["status"] != "pending"]

    agents = []
    for a in AGENTS:
        mine = [t for t in trades if t["agent"] == a["id"]]
        groups = []
        for g in sorted({t["group"] for t in mine}):
            groups.append({"name": g, **stats([t for t in mine if t["group"] == g])})
        first = next(archives(a["dir"]), (None, None))[0]
        entry = {"id": a["id"], "name": a["name"], "accent": a["accent"],
                 "since": first.strftime("%Y-%m-%d") if first is not None else None,
                 "backtest_r": a["backtest"], "backtest_note": a["backtest_note"],
                 **stats(mine), "groups": groups, "curve": curve(mine)}
        if a["id"] == "OPY":
            entry["condors"] = {"n": len(condor_rows), "inside": sum(r["inside"] for r in condor_rows),
                                "rows": sorted(condor_rows, key=lambda r: r["expiry"], reverse=True)}
        agents.append(entry)
        logger.info(f"{a['name']}: {entry['closed']} closed, {entry['open']} open, "
                    f"avg R {entry.get('avg_r')}, win {entry.get('win_rate')}%")

    trades.sort(key=lambda t: (t.get("exit_date") or "9999", t["signal_date"]), reverse=True)
    out = {"generated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
           "data_through": through.strftime("%Y-%m-%d"),
           "agents": agents, "trades": trades}
    with open(OUT, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    logger.info(f"wrote {OUT}: {len(trades)} trades through {out['data_through']}")


if __name__ == "__main__":
    main()
