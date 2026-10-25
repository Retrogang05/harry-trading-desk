#!/usr/bin/env python3
"""Leaders (LDRS) - Top 25 momentum leaderboard per index, Stocktwits-style.

Ranks every member of the S&P 500, Nasdaq-100 and (liquid) Russell 2000 by
YEAR-TO-DATE price return, and publishes the top 25 of each with last week's
rank, week-to-date return, sector, and the names that fell off the list.

Checked against Stocktwits' own "Top 25 Momentum" for the week of
2026-10-23: YTD and WTD match to 0.02-0.03 percentage points. 21 of 25 S&P
names matched; ours also had BE, CRWD, ILMN and P, all in State Street's SPY
holdings as of 2026-10-01. Sentiment and watcher counts are Stocktwits' own
social data and are not reproduced.

WHAT THE BACKTEST SAID - read before trading the list. Every Friday
2019-2026, top 25 by YTD per index, equal-weight forward return against the
equal-weight average of ALL that index's members, significance on
non-overlapping dates only:

                 20 days    50 days    100 days
    S&P 500      +1.72%     +4.16%     +9.62%     (t = 3.5 / 3.0 / 2.3)
    Nasdaq-100   +0.71%     +1.51%     +3.69%     (not significant at 50/100)
    Russell 2000 +1.69%     +3.13%     +7.42%     (not significant at 50/100)

  * The S&P result is significant only in 2023-26 (t 2.7-3.0); 2019-22 is
    positive but not significant. Half of the 2023-26 excess came from a
    single stock per window (+7.61% at 50 days -> +3.98% without it).
  * In the Russell 2000 momentum REVERSES: the bottom 25 beat the top 25 by
    5.8% over 50 days.
  * The deciles are U-shaped - the worst YTD stocks also beat the average -
    which is the volatility effect seen in every test on this desk, not
    momentum.
  * The test used TODAY's index members. Stocks added to the S&P because
    they ran up count in years before they joined, flattering the top 25;
    Russell losers that went bust are absent, flattering the bottom 25. The
    free data cannot correct for either.

So this is a watchlist. As a signal it is weak, era-dependent and S&P-only.
It makes no Claude calls - a leaderboard needs no narration.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import universe
import sectors
import history

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AGENT = {
    "id": "LDRS",
    "name": "Leaders",
    "strategy": "Top 25 momentum (YTD)",
    "description": "Top 25 by year-to-date return in each index, with last week's rank, "
                   "week-to-date move and who fell off. A watchlist, not a signal.",
    "accent": "#9a9a7f",
}

DOCS_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "data")
INDEX_TAGS = universe.INDEX_TAGS
LIST_NAME = {"SPX": "S&P 500", "QQQ": "Nasdaq-100", "IWM": "Russell 2000"}
INDEX_ETF = {"SPX": "SPY", "QQQ": "QQQ", "IWM": "IWM"}
TOP_N = 25

# 400 calendar days covers last year's final close even when the PREVIOUS
# week's ranking falls in the prior year (the first week of January).
CALENDAR_DAYS = 400

# A split-shaped one-day jump (ratio near 2:1, 3:1, 1:10 ...) is an
# unadjusted corporate action, and on a ranking by YTD return it would put a
# stock straight into the top 25 on an artefact. 54 of 1,517 symbols carried
# one in the 2019-26 history used for the backtest. A real 40% move on news
# rarely lands on one of these ratios, so genuine momentum names are kept.
SPLIT_RATIOS = np.array([2, 3, 4, 5, 10, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10])


def warm_tz_cache() -> None:
    """One unthreaded request before any threaded batch - see Trey #12/#13."""
    try:
        yf.download("SPY", period="5d", threads=False, progress=False)
    except Exception as e:
        logger.warning(f"tz cache warm-up failed ({e}) - continuing")


def fetch_closes(symbols: List[str]) -> pd.DataFrame:
    """Unadjusted closes: Stocktwits ranks on price return, and matching it to
    0.02 points needed auto_adjust=False."""
    warm_tz_cache()
    end = datetime.now()
    start = end - timedelta(days=CALENDAR_DAYS)
    cols = {}
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]
        logger.info(f"Fetching {len(chunk)} symbols ({i + 1}-{i + len(chunk)} of {len(symbols)})...")
        try:
            raw = yf.download(chunk, start=start, end=end, auto_adjust=False,
                              progress=False, threads=True, group_by="ticker")
        except Exception as e:
            logger.error(f"batch failed: {e}")
            continue
        if not isinstance(raw.columns, pd.MultiIndex):
            logger.error("batch came back unlabelled - skipped")
            continue
        present = set(raw.columns.get_level_values(0))
        for s in chunk:
            if s in present:
                c = raw[s]["Close"].dropna()
                if len(c) > 20:
                    cols[s] = c
    return pd.DataFrame(cols).sort_index()


def split_shaped(c: pd.Series) -> bool:
    step = (c / c.shift(1)).dropna().values
    big = step[np.abs(step - 1) >= 0.4]
    if not len(big):
        return False
    return bool((np.min(np.abs(big[:, None] / SPLIT_RATIOS - 1), axis=1) < 0.03).any())


def ytd_at(C: pd.DataFrame, when: pd.Timestamp) -> pd.Series:
    """YTD return of every column as of `when`, from the last close of the
    prior calendar year."""
    prior = C.index[C.index.year < when.year]
    if not len(prior):
        return pd.Series(dtype=float)
    base = C.loc[prior.max()]
    return (C.loc[when] / base - 1).dropna()


def build(C: pd.DataFrame, membership: Dict[str, List[str]],
          sector_map: Dict[str, str], etf: pd.DataFrame) -> Tuple[List[Dict], Dict]:
    """The three leaderboards from a close panel. Pure function of its inputs
    so it can be tested without a network."""
    as_of = C.index.max()
    # Drop symbols that stopped updating (BZH-style) and split artefacts.
    last = C.apply(lambda s: s.last_valid_index())
    stale = [s for s in C.columns if last[s] < as_of]
    splits = [s for s in C.columns if s not in stale and split_shaped(C[s].dropna())]
    C = C.drop(columns=stale + splits)

    # "Week" is the calendar week: WTD is measured from the last close of the
    # previous week, and last week's ranking is taken at that same close.
    wk = as_of.isocalendar()
    prev_week = C.index[[(d.isocalendar().year, d.isocalendar().week) < (wk.year, wk.week)
                         for d in C.index]]
    prev_close_day = prev_week.max() if len(prev_week) else None

    ytd_now = ytd_at(C, as_of)
    ytd_prev = ytd_at(C, prev_close_day) if prev_close_day is not None else pd.Series(dtype=float)
    wtd = (C.loc[as_of] / C.loc[prev_close_day] - 1).dropna() if prev_close_day is not None \
        else pd.Series(dtype=float)

    rows, context = [], {"as_of": as_of, "stale": len(stale), "splits": len(splits), "lists": {}}
    for g in INDEX_TAGS:
        members = [s for s in C.columns if g in membership.get(s, [])]
        now = ytd_now.reindex(members).dropna().sort_values(ascending=False)
        prev = ytd_prev.reindex(members).dropna().sort_values(ascending=False)
        rank_now = {s: i + 1 for i, s in enumerate(now.index)}
        rank_prev = {s: i + 1 for i, s in enumerate(prev.index)}
        top = list(now.index[:TOP_N])
        fell = [s for s in prev.index[:TOP_N] if s not in top]

        top_wtd = float(wtd.reindex(top).mean()) if len(top) else float("nan")
        e = etf.get(INDEX_ETF[g])
        etf_wtd = (float(e.loc[as_of] / e.loc[prev_close_day] - 1)
                   if e is not None and prev_close_day is not None
                   and as_of in e.index and prev_close_day in e.index else float("nan"))
        context["lists"][g] = {"members": len(now), "top_wtd": top_wtd, "etf_wtd": etf_wtd}

        for s in top + fell:
            r, pr = rank_now.get(s), rank_prev.get(s)
            on_list = s in top
            if not on_list:
                label, tone = "Fell off", "neg"
            elif pr is None or pr > TOP_N:
                label, tone = "New", "pos"
            elif pr > r:
                label, tone = f"▲{pr - r}", "pos"
            elif pr < r:
                label, tone = f"▼{r - pr}", "neg"
            else:
                label, tone = "—", None
            sec = sector_map.get(s)
            w = wtd.get(s)
            rows.append({
                "symbol": s,
                "price": float(C.loc[as_of, s]),
                # Rank shown as a score so the desk-wide sort orders the list:
                # #1 = 100 down to #25 = 52. Fell-off rows sit below.
                "score": max(0, 102 - 2 * r) if on_list else max(0, 50 - (r - TOP_N)),
                "strategy": f"{LIST_NAME[g]} Top {TOP_N}" + ("" if on_list else " · fell off"),
                "signal_label": label,
                "signal_tone": tone,
                # The list this row belongs to, so the index chips show each
                # leaderboard exactly. A name on two lists is two rows.
                "indexes": [g],
                "rank": r,
                "prev_rank": pr,
                "ytd_pct": round(float(ytd_now[s]) * 100, 2),
                "wtd_pct": None if w is None or np.isnan(w) else round(float(w) * 100, 2),
                "sector": sec,
                "dimensions": [],
                "setup": {
                    "title": "Leaderboard",
                    "label": (f"{LIST_NAME[g]} #{r}" + (f" (last week #{pr})" if pr else " (new)")
                              + f" · YTD {ytd_now[s] * 100:+.1f}%"),
                    "fields": [
                        {"label": "Rank", "value": f"#{r}"},
                        {"label": "Last week", "value": f"#{pr}" if pr else "—"},
                        {"label": "YTD", "value": f"{ytd_now[s] * 100:+.1f}%",
                         "tone": "pos" if ytd_now[s] > 0 else "neg"},
                        {"label": "Week to date", "value": "—" if w is None or np.isnan(w) else f"{w * 100:+.1f}%",
                         "tone": None if w is None or np.isnan(w) else ("pos" if w > 0 else "neg")},
                        {"label": "Sector", "value": sectors.SECTOR_ETFS.get(sec, "—")},
                        {"label": "List vs index WTD",
                         "value": f"{top_wtd * 100:+.1f}% vs {INDEX_ETF[g]} {etf_wtd * 100:+.1f}%"
                         if not (np.isnan(top_wtd) or np.isnan(etf_wtd)) else "—"},
                    ],
                },
            })
    return rows, context


def publish(rows: List[Dict], context: Dict, membership: Dict[str, List[str]]) -> None:
    os.makedirs(DOCS_DATA_DIR, exist_ok=True)
    as_of = context["as_of"].strftime("%Y-%m-%d")
    ctx = [{"label": "Data as of", "value": as_of},
           {"label": "Ranked by", "value": "YTD price return, per index"}]
    for g in INDEX_TAGS:
        L = context["lists"].get(g)
        if L:
            ctx.append({"label": f"{LIST_NAME[g]} top {TOP_N} WTD",
                        "value": f"{L['top_wtd'] * 100:+.1f}% vs {INDEX_ETF[g]} {L['etf_wtd'] * 100:+.1f}%"
                        if not (np.isnan(L['top_wtd']) or np.isnan(L['etf_wtd'])) else "—"})
    ctx.append({"label": "Dropped", "value": f"{context['stale']} stale, {context['splits']} split artefacts"})
    payload = {
        "agent": AGENT,
        "scan_date": datetime.now().isoformat(),
        "dimensions": [],
        "groups": {"key": "indexes", "label": "Index",
                   "values": [g for g in INDEX_TAGS if any(g in r["indexes"] for r in rows)]},
        "context": ctx,
        "opportunities": rows,
    }
    path = os.path.join(DOCS_DATA_DIR, f"{AGENT['id']}.json")
    blob = json.dumps(payload, indent=2, default=str)
    with open(path, "w") as f:
        f.write(blob)
    manifest = os.path.join(DOCS_DATA_DIR, "agents.json")
    try:
        with open(manifest) as f:
            reg = json.load(f).get("agents", [])
    except (FileNotFoundError, json.JSONDecodeError):
        reg = []
    if AGENT["id"] not in reg:
        reg.append(AGENT["id"])
        with open(manifest, "w") as f:
            json.dump({"agents": reg}, f, indent=2)
    logger.info(f"Published {len(rows)} rows -> {path}")


def main():
    logger.info("Starting Leaders - Top 25 momentum leaderboard...")
    try:
        symbols, membership, sector_map = universe.load()
    except FileNotFoundError:
        sys.exit("ERROR: universe.txt not found - run scripts/build_universe.py")
    C = fetch_closes(symbols)
    etf = fetch_closes(list(INDEX_ETF.values()))
    rows, context = build(C, membership, sector_map, etf)
    history.attach(rows, {r["symbol"]: C[r["symbol"]] for r in rows})
    publish(rows, context, membership)
    with open("leaders_results.json", "w") as f:
        json.dump({"scan_date": datetime.now().isoformat(), "rows": rows,
                   "context": {k: v for k, v in context.items() if k != "as_of"}}, f, indent=2, default=str)
    for g in INDEX_TAGS:
        top = [r for r in rows if r["indexes"] == [g] and "fell off" not in r["strategy"]]
        logger.info(f"  {LIST_NAME[g]}: " + ", ".join(f"{r['symbol']} {r['ytd_pct']:+.0f}%" for r in top[:5]) + " ...")
    return rows


if __name__ == "__main__":
    main()
