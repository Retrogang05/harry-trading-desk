#!/usr/bin/env python3
"""Sector regime: which sector SPDRs are below their 20-day average.

Shared by Monu and Opy so both gate their bearish ideas on the same reading
of the same eleven ETFs, rather than each deciding "weak" its own way.

The premise is breadth. A desk whose only gate is SPY's 50/200 keeps
publishing long setups through exactly the periods when the index is held up
by one or two sectors and everything else is rolling over - on 2026-10-05,
ten of the eleven sector SPDRs were below their 20-day average while SPY's
50-day was still above its 200-day and Monu's gate still read UPTREND.

A sector below its 20-day SMA is a short-term weakness signal, not a bear
market call. That is the right sensitivity for the job: it decides whether to
LOOK for bearish ideas in a sector, and the per-name filters downstream decide
whether any are worth publishing.
"""

import logging
from typing import Dict, List

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ETF -> the sector it represents. Keys double as the tags in universe.txt,
# which is why these are symbols rather than sector names.
SECTOR_ETFS = {
    "XLB":  "Materials",
    "XLC":  "Communication Services",
    "XLE":  "Energy",
    "XLF":  "Financials",
    "XLI":  "Industrials",
    "XLK":  "Technology",
    "XLP":  "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU":  "Utilities",
    "XLV":  "Health Care",
    "XLY":  "Consumer Discretionary",
}

SMA = 20
# Enough calendar days for 20 sessions plus slack for holidays.
CALENDAR_DAYS = 90


def read() -> Dict[str, Dict]:
    """Current state of every sector ETF: {tag: {weak, price, sma, pct, name}}.

    A failed fetch leaves that sector OUT of the result rather than defaulting
    it either way. Defaulting to weak would invent bearish signals from a
    network error; defaulting to strong would hide real ones. Absent means the
    caller publishes nothing for that sector, which is the honest answer.
    """
    tags = list(SECTOR_ETFS)
    try:
        # threads=False: this is the first yfinance call in Monu's bearish
        # path and in Opy's, and a cold CI cache plus a thread pool is the
        # race that killed Trey's runs #12 and #13. Eleven symbols do not
        # need parallelism.
        raw = yf.download(tags, period=f"{CALENDAR_DAYS}d", auto_adjust=True,
                          progress=False, threads=False, group_by="ticker")
    except Exception as e:
        logger.error(f"sector fetch failed ({e}) - no sector gating this run")
        return {}

    # A flat frame means yfinance collapsed an 11-ticker request to one
    # unlabelled series, and nothing in it says WHICH ticker survived. The
    # previous version fell through to raw["Close"] inside the per-tag loop,
    # which handed every one of the eleven sectors the same price and SMA -
    # so one surviving ETF could mark all eleven weak, or all eleven fine,
    # and that reading drives Monu's whole bearish half and Opy's bear-call
    # tiebreak. Refuse to interpret it instead; an absent sector publishes
    # nothing, which is the honest answer to "we could not tell".
    if not isinstance(raw.columns, pd.MultiIndex):
        logger.error(
            f"sector fetch returned a single unlabelled frame for {len(tags)} tickers "
            f"- cannot tell which ETF it is, so no sector gating this run"
        )
        return {}

    available = set(raw.columns.get_level_values(0))
    out = {}
    for tag in tags:
        try:
            if tag not in available:
                logger.warning(f"{tag}: missing from the sector response")
                continue
            close = raw[tag]["Close"].dropna()
            if len(close) < SMA:
                logger.warning(f"{tag}: only {len(close)} closes, need {SMA}")
                continue
            price = float(close.iloc[-1])
            sma = float(close.rolling(SMA).mean().iloc[-1])
            out[tag] = {
                "name": SECTOR_ETFS[tag],
                "price": round(price, 2),
                "sma": round(sma, 2),
                "pct": round((price / sma - 1) * 100, 2),
                "weak": price < sma,
            }
        except (KeyError, IndexError, ValueError, TypeError) as e:
            logger.warning(f"{tag}: could not read sector state ({e})")

    if out:
        weak = sorted(t for t, v in out.items() if v["weak"])
        logger.info(f"Sector regime: {len(weak)}/{len(out)} below their {SMA}-day SMA"
                    + (f" - {', '.join(weak)}" if weak else ""))
    return out


def weak_tags(state: Dict[str, Dict]) -> List[str]:
    """Just the sector tags that are below their average."""
    return sorted(t for t, v in state.items() if v["weak"])


def sector_of(tags: List[str]) -> str:
    """The sector ETF tag out of a symbol's universe.txt tags, or None.

    A symbol carries index tags and at most one sector tag in the same list
    (AAPL -> SPX,QQQ,XLK), so this is just the first one that names a sector.
    """
    for t in tags or ():
        if t in SECTOR_ETFS:
            return t
    return None


def summary_line(state: Dict[str, Dict]) -> str:
    """One line for an agent's dashboard context block."""
    if not state:
        return "unavailable"
    weak = weak_tags(state)
    return f"{len(weak)}/{len(state)} weak" + (f" ({', '.join(weak)})" if weak else "")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    st = read()
    for tag, v in sorted(st.items()):
        print(f"  {tag:<5} ${v['price']:>8.2f}  {SMA}d ${v['sma']:>8.2f}  "
              f"{v['pct']:+6.2f}%  {'WEAK' if v['weak'] else 'ok'}   {v['name']}")
    print(f"\n{summary_line(st)}")
