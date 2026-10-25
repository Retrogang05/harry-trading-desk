#!/usr/bin/env python3
"""Recent closes for the dashboard's price chart, shared by every agent.

Each published row gets `history`: the last ~3 months of daily closes, so the
detail panel can draw the chart without the page fetching prices itself (the
dashboard is static - GitHub Pages - and a browser call to Yahoo is blocked by
CORS anyway).

Agents pass the frames they already fetched, so the chart is drawn from the
same bars the signal was computed on. A symbol with no frame in hand - Opy's
screener keeps metrics, not bars - is fetched here in one small batch. Any
failure leaves the row without `history`; the panel then shows no chart
rather than the run failing over a picture.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

SESSIONS = 63          # ~3 months of trading days
FETCH_DAYS = 120       # calendar days to request when fetching ourselves

Frame = Union[pd.DataFrame, pd.Series]


def _closes(x: Optional[Frame]) -> Optional[pd.Series]:
    if x is None:
        return None
    c = x["Close"] if isinstance(x, pd.DataFrame) else x
    if isinstance(c, pd.DataFrame):            # yfinance MultiIndex leftovers
        c = c.iloc[:, 0]
    c = c.dropna()
    return c if len(c) >= 2 else None


def series(x: Optional[Frame], sessions: int = SESSIONS) -> Optional[Dict]:
    """{"dates": [...], "closes": [...]} for the last `sessions` closes."""
    c = _closes(x)
    if c is None:
        return None
    c = c.iloc[-sessions:]
    # Prices span $1 to $1,000+; 4 significant figures is finer than a
    # 300-pixel chart can show and keeps each agent's JSON small.
    return {"dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in c.index],
            "closes": [float(f"{v:.4g}") if abs(v) < 1000 else round(float(v), 1)
                       for v in c.values]}


def _fetch(symbols: List[str]) -> Dict[str, pd.Series]:
    if not symbols:
        return {}
    end = datetime.now()
    try:
        raw = yf.download(symbols, start=end - timedelta(days=FETCH_DAYS), end=end,
                          auto_adjust=True, progress=False, threads=False,
                          group_by="ticker")
    except Exception as e:
        logger.warning(f"history fetch failed ({e}) - no charts for {len(symbols)} rows")
        return {}
    out = {}
    for s in symbols:
        # group_by="ticker" nests columns under the symbol - for a single
        # symbol too, in current yfinance, but not in every version.
        try:
            out[s] = raw[s]["Close"]
        except Exception:
            try:
                if len(symbols) == 1:
                    out[s] = raw["Close"]
            except Exception:
                continue
    return out


def attach(rows: List[Dict], frames: Optional[Dict[str, Frame]] = None,
           sessions: int = SESSIONS) -> None:
    """Add `history` to every row in place. `frames` maps symbol -> bars or
    closes the agent already has; anything missing is fetched."""
    frames = frames or {}
    missing = sorted({o["symbol"] for o in rows if _closes(frames.get(o["symbol"])) is None})
    fetched = _fetch(missing)
    done = 0
    for o in rows:
        sym = o["symbol"]
        h = series(frames.get(sym), sessions) or series(fetched.get(sym), sessions)
        if h:
            o["history"] = h
            done += 1
    logger.info(f"price history attached to {done}/{len(rows)} rows"
                + (f" ({len(missing)} fetched)" if missing else ""))
