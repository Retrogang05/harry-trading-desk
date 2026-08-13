"""Opy: runs the options screener (iron condor, credit spreads, LEAPS, RSI
momentum) and publishes a combined shortlist to the Harry Trading Desk
dashboard as its own tab.

Mirrors ../agent.py (Monu)'s structure and the two bugs that were fixed there
the hard way:
  - reasoning is gated on ANTHROPIC_API_KEY up front, with a last-resort
    except so a narration failure can never take down a scan that otherwise
    succeeded (the Anthropic SDK raises a bare TypeError, not an anthropic.*
    exception, when no credential resolves - that killed Monu's first run)
  - Claude is called only on the published shortlist, never on every
    candidate that merely passed stage 2 (that was a 90-calls-to-publish-20
    bug on Monu; here it would be worse, since four pipelines each produce
    their own candidate pool before any of them get capped)

Unlike Monu, Opy's four strategies score on entirely different axes (an iron
condor's IV/HV richness has nothing in common with a LEAPS trade's delta and
leverage), so this publishes per-OPPORTUNITY `dimensions` and a declarative
`setup` block rather than the single per-agent `dimensions` + fixed
entry/stop/target shape Monu uses. The dashboard was extended to prefer a
per-opportunity override when present and fall back to Monu's original
per-agent contract otherwise - see docs/index.html for the reader side.
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from anthropic import Anthropic
import anthropic

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
import condor
import spreads
import leaps
import rsi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-haiku-4-5"

AGENT = {
    "id": "OPY",
    "name": "Opy",
    "strategy": "Options",
    "description": "Screens iron condors, credit spreads and LEAPS calls for liquidity and "
                    "IV pricing, plus an RSI momentum context list.",
    # Same low-saturation register as Monu's #c8974a, not neon - see agent.py's comment.
    "accent": "#8a7fa8",
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DATA_DIR = os.path.join(REPO_ROOT, "docs", "data")
UNIVERSE_FILE = os.path.join(REPO_ROOT, "universe.txt")

# ── Screener parameters -----------------------------------------------------
# Hardcoded rather than argparse: this runs unattended in CI. Values are
# screener.py's own CLI defaults - see that file for the research citations
# behind each one. Kept identical here rather than re-derived, so Opy's
# results match what running the original CLI locally would produce.
MIN_DOLLAR_VOLUME = 20_000_000
MIN_DAILY_RANGE, MAX_DAILY_RANGE = 0.5, 3.5
DTE_MIN, DTE_MAX, DTE_FLOOR = 25, 35, 21
MIN_EARNINGS_DAYS = 30
MAX_SPREAD_PCT = 15.0
MIN_OPEN_INTEREST = 50
MIN_IV_HV_RATIO = 1.0
STAGE1_LIMIT = 80
STAGE2_WORKERS = 8
RANGE_POSITION_LOW, RANGE_POSITION_HIGH = 0.25, 0.75
TARGET_DELTA = 0.30
MAX_WIDTH_PCT = 15.0
MIN_CREDIT_WIDTH_PCT = 20.0
LEAPS_TARGET_DELTA = 0.80
LEAPS_DTE_MIN, LEAPS_DTE_MAX = 270, 545
LEAPS_MAX_IV_HV_RATIO = 1.1
LEAPS_MAX_RSI = 30.0
LEAPS_MIN_EARNINGS_DAYS = 14
RSI_OVERSOLD_MAX = 30.0
RSI_APPROACHING_OVERSOLD_MAX = 40.0
RSI_APPROACHING_OVERBOUGHT_MIN = 60.0
RSI_OVERBOUGHT_MIN = 70.0
RSI_STAGE1_LIMIT = 25

# Published-row caps per strategy bucket. This is what Claude reasoning cost
# scales with (reasoning runs only on published rows), not stage-1/stage-2
# candidate counts - same cost-safety lesson as Monu's 90-calls-to-publish-20
# fix, applied from the start here instead of discovered the hard way.
TOP_N_CONDOR = 6
TOP_N_SPREADS = 8    # bull put + bear call combined
TOP_N_LEAPS = 6       # trend + reversal combined
TOP_N_RSI = 8          # 2 per band x 4 bands

# Dimension sets, one per strategy family. Each max is that dimension's
# weight in its strategy's score_and_rank() - see the breakdown_* functions
# below, which recompute those exact weighted-rank terms so the bars sum to
# the same score score_and_rank produced, not an approximation of it.
CONDOR_DIMENSIONS = [
    {"key": "iv_hv_richness", "label": "IV/HV Richness", "max": 40},
    {"key": "range_centrality", "label": "Range Centrality", "max": 20},
    {"key": "liquidity", "label": "Liquidity", "max": 15},
    {"key": "spread_tightness", "label": "Spread Tightness", "max": 15},
    {"key": "open_interest", "label": "Open Interest", "max": 10},
]
SPREADS_DIMENSIONS = [
    {"key": "credit_width", "label": "Credit / Width", "max": 30},
    {"key": "prob_profit", "label": "Prob. of Profit", "max": 25},
    {"key": "iv_hv_richness", "label": "IV/HV Richness", "max": 20},
    {"key": "liquidity", "label": "Liquidity", "max": 15},
    {"key": "open_interest", "label": "Open Interest", "max": 10},
]
LEAPS_DIMENSIONS = [
    {"key": "iv_cheapness", "label": "IV Cheapness", "max": 30},
    {"key": "quality", "label": "Trend / Reversal", "max": 20},
    {"key": "liquidity", "label": "Liquidity", "max": 20},
    {"key": "open_interest", "label": "Open Interest", "max": 15},
    {"key": "spread_tightness", "label": "Spread Tightness", "max": 15},
]
RSI_DIMENSIONS = [
    {"key": "extremity", "label": "RSI Extremity", "max": 60},
    {"key": "liquidity", "label": "Liquidity", "max": 40},
]


def load_universe() -> List[str]:
    """Reuses Monu's universe.txt rather than fetching S&P 500 constituents a
    second time (core.get_universe() would do its own SSGA/Wikipedia fetch).
    Keeps the two agents scanning the identical list, and halves the SSGA
    dependency surface. Falls back to core.get_universe() if the file is
    missing, so Opy still works if ever run standalone.
    """
    if not os.path.exists(UNIVERSE_FILE):
        logger.warning("universe.txt not found - falling back to core.get_universe()")
        return core.get_universe(cache_dir=os.path.join(REPO_ROOT, "opy", "data"))

    symbols, seen = [], set()
    with open(UNIVERSE_FILE) as f:
        for line in f:
            sym = line.split("#", 1)[0].strip().upper()
            if sym and sym not in seen:
                seen.add(sym)
                symbols.append(sym)
    logger.info(f"Loaded {len(symbols)} symbols from universe.txt (shared with Monu)")
    return symbols


# ── Reasoning -----------------------------------------------------------------

class Reasoner:
    def __init__(self):
        self.model = CLAUDE_MODEL
        self.enabled = bool(os.environ.get("ANTHROPIC_API_KEY"))
        self.client = Anthropic() if self.enabled else None
        if not self.enabled:
            logger.warning(
                "ANTHROPIC_API_KEY not set - screening will run, but signals "
                "will publish without written reasoning"
            )

    def explain(self, kind: str, row: Dict) -> str:
        if not self.enabled:
            return "[reasoning unavailable: ANTHROPIC_API_KEY not configured]"

        prompt = self._prompt(kind, row)
        try:
            msg = self.client.messages.create(
                model=self.model, max_tokens=220,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except anthropic.RateLimitError:
            logger.error(f"{row.get('ticker')}: rate limited")
            return "[reasoning unavailable: rate limited]"
        except anthropic.APIStatusError as e:
            logger.error(f"{row.get('ticker')}: API error {e.status_code}")
            return f"[reasoning unavailable: API error {e.status_code}]"
        except anthropic.APIConnectionError:
            logger.error(f"{row.get('ticker')}: could not reach the Claude API")
            return "[reasoning unavailable: connection error]"
        except Exception as e:
            # Same last-resort catch as Monu's fix: a bare TypeError from an
            # unresolved credential must never take the whole scan down.
            logger.error(f"{row.get('ticker')}: unexpected reasoning failure: {type(e).__name__}: {e}")
            return f"[reasoning unavailable: {type(e).__name__}]"

    def _prompt(self, kind: str, r: Dict) -> str:
        t = r["ticker"]
        if kind == "condor":
            return (
                f"Iron condor candidate: {t} at ${r['last_price']:.2f}, {r['dte']}d to "
                f"{r['expiry']}. ATM IV {r['atm_iv_pct']:.1f}% vs 30d realized vol "
                f"{r['hv30']:.1f}% (IV/HV {r['iv_hv_ratio']:.2f}x). Sitting at "
                f"{r['range_position_52w']*100:.0f}% of its 52-week range. Expected move "
                f"±{r.get('expected_move_pct') or 0:.1f}%. Open interest floor {r['min_open_interest']}, "
                f"avg bid/ask spread {r['avg_spread_pct']:.1f}%.\n\n"
                "Write 2-3 sentences: why this name fits a range-bound, premium-rich iron "
                "condor thesis, and what would invalidate it (a breakout, an earnings surprise, "
                "IV compressing). Professional, concise, no repeating the numbers verbatim."
            )
        if kind == "spread":
            return (
                f"{r['direction']} credit spread candidate: {t} at ${r['last_price']:.2f}, "
                f"short strike {r['short_strike']:g} (delta {r['short_delta']:.2f}), long strike "
                f"{r['long_strike']:g}, {r['dte']}d to {r['expiry']}. Credit ${r['net_credit']:.2f} "
                f"on ${r['width']:.2f} width ({r['credit_width_pct']:.1f}% of width), approx POP "
                f"{r['approx_pop']:.0f}%, ROC {r['roc_pct'] or 0:.1f}%. IV/HV {r['iv_hv_ratio']:.2f}x.\n\n"
                "Write 2-3 sentences: why the trend and premium support this spread, and the "
                "main risk (a reversal through the short strike, an IV crush after earnings). "
                "Professional, concise, no repeating the numbers verbatim."
            )
        if kind == "leaps":
            return (
                f"LEAPS call candidate ({r['style']}): {t} at ${r['last_price']:.2f}, strike "
                f"{r['strike']:g} (delta {r['delta']:.2f}, {r['itm_pct']:.1f}% ITM), {r['dte']}d to "
                f"{r['expiry']}. Premium ${r['premium']:.2f} ({r['premium_pct_of_stock']:.1f}% of "
                f"stock price), breakeven ${r['breakeven']:.2f}, leverage {r['leverage'] or 0:.1f}x. "
                f"IV/HV {r['iv_hv_ratio']:.2f}x (buying cheap vol).\n\n"
                "Write 2-3 sentences: why this is attractively priced as a leveraged stock "
                "substitute right now, and the main risk (paying too much time value, a trend "
                "reversal before the thesis plays out). Professional, concise, no repeating "
                "the numbers verbatim."
            )
        # rsi
        return (
            f"RSI momentum context: {t} at ${r['last_price']:.2f}, RSI14 {r['rsi14']:.1f} "
            f"({r['band']}). Price vs 50/200d SMA: {r['last_price']:.2f} vs "
            f"{r.get('sma50') or 0:.2f} / {r.get('sma200') or 0:.2f}.\n\n"
            "Write 1-2 sentences: what this RSI reading suggests about the stock's current "
            "momentum state. This is a signal for the trader's own judgment, not a specific "
            "options structure - do not suggest a trade. Professional, concise."
        )


# ── Per-strategy breakdown (reproduces each score_and_rank's own weights) ---
# These recompute the exact rank(pct=True)-weighted terms each vendored
# score_and_rank() sums into a single 'score' column, so the dashboard's
# dimension bars show real point contributions (bar sum == score, modulo
# rounding) rather than a placeholder split of the total.

def _breakdown_condor(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["_iv_hv_richness"] = df["iv_hv_ratio"].rank(pct=True) * 40
    df["_range_centrality"] = (1 - (df["range_position_52w"] - 0.5).abs() * 2).clip(lower=0) * 20
    df["_liquidity"] = df["avg_dollar_volume_m"].rank(pct=True) * 15
    df["_spread_tightness"] = (1 - df["avg_spread_pct"].rank(pct=True)) * 15
    df["_open_interest"] = df["min_open_interest"].rank(pct=True) * 10
    return df


def _breakdown_spreads(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["_credit_width"] = df["credit_width_pct"].rank(pct=True) * 30
    df["_prob_profit"] = df["approx_pop"].rank(pct=True) * 25
    df["_iv_hv_richness"] = df["iv_hv_ratio"].rank(pct=True) * 20
    df["_liquidity"] = df["avg_dollar_volume_m"].rank(pct=True) * 15
    df["_open_interest"] = df["min_open_interest"].rank(pct=True) * 10
    return df


def _breakdown_leaps(df: pd.DataFrame, style: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    quality = df["trend_strength"].rank(pct=True) if style == "trend" else (1 - df["rsi14"].rank(pct=True))
    df["_iv_cheapness"] = (1 - df["iv_hv_ratio"].rank(pct=True)) * 30
    df["_quality"] = quality * 20
    df["_liquidity"] = df["avg_dollar_volume_m"].rank(pct=True) * 20
    df["_open_interest"] = df["min_open_interest"].rank(pct=True) * 15
    df["_spread_tightness"] = (1 - df["avg_spread_pct"].rank(pct=True)) * 15
    return df


def _breakdown_rsi(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    extremity = (1 - df["rsi14"].rank(pct=True)) if direction == "bullish" else df["rsi14"].rank(pct=True)
    df["_extremity"] = extremity * 60
    df["_liquidity"] = df["avg_dollar_volume_m"].rank(pct=True) * 40
    return df


def _pts(row, key: str) -> float:
    v = row.get(f"_{key}")
    return round(float(v), 1) if v is not None and v == v else 0.0


# ── Per-strategy setup blocks -------------------------------------------------

def _fmt(v, spec="{:.2f}", default="—"):
    return default if v is None or (isinstance(v, float) and v != v) else spec.format(v)


def _setup_condor(r) -> Dict:
    bits = [f"{r['expiry']}", f"{r['dte']}d", f"IV/HV {_fmt(r['iv_hv_ratio'], '{:.2f}x')}"]
    return {
        "label": " · ".join(bits),
        "fields": [
            {"label": "Exp. Move", "value": f"±{_fmt(r.get('expected_move_pct'), '{:.1f}%')}"},
            {"label": "ATM IV", "value": _fmt(r["atm_iv_pct"], "{:.1f}%")},
            {"label": "HV30", "value": _fmt(r["hv30"], "{:.1f}%")},
            {"label": "Spread", "value": _fmt(r["avg_spread_pct"], "{:.1f}%")},
        ],
    }


def _setup_spread(r) -> Dict:
    max_loss = r["width"] - r["net_credit"] if r.get("width") is not None and r.get("net_credit") is not None else None
    return {
        "label": f"{r['direction']} {r['short_strike']:g}/{r['long_strike']:g} · {r['dte']}d",
        "fields": [
            {"label": "Credit", "value": f"${_fmt(r['net_credit'])}", "tone": "pos"},
            {"label": "Max Loss", "value": f"${_fmt(max_loss)}", "tone": "neg"},
            {"label": "ROC", "value": _fmt(r.get("roc_pct"), "{:.1f}%")},
            {"label": "Approx. POP", "value": _fmt(r["approx_pop"], "{:.0f}%")},
        ],
    }


def _setup_leaps(r) -> Dict:
    return {
        "label": f"{r['strike']:g}C · Δ{_fmt(r['delta'])} · {r['dte']}d",
        "fields": [
            {"label": "Premium", "value": f"${_fmt(r['premium'])}", "tone": "neg"},
            {"label": "Breakeven", "value": f"${_fmt(r['breakeven'])}"},
            {"label": "Leverage", "value": _fmt(r.get("leverage"), "{:.1f}x")},
            {"label": "% ITM", "value": _fmt(r["itm_pct"], "{:.1f}%")},
        ],
    }


# ── Per-strategy scan -----------------------------------------------------

def scan_condor(metrics: pd.DataFrame, reasoner: Reasoner) -> List[Dict]:
    stage1 = condor.filter_stage1(
        metrics, MIN_DOLLAR_VOLUME, MIN_DAILY_RANGE, MAX_DAILY_RANGE,
        (RANGE_POSITION_LOW, RANGE_POSITION_HIGH),
    ).sort_values("avg_dollar_volume_m", ascending=False).head(STAGE1_LIMIT)
    logger.info(f"[Condor] stage 1: {len(stage1)} candidates -> stage 2")

    stage2, _regime = condor.stage2_options_screen(
        stage1, DTE_MIN, DTE_MAX, DTE_FLOOR, MIN_EARNINGS_DAYS,
        MAX_SPREAD_PCT, MIN_OPEN_INTEREST, MIN_IV_HV_RATIO, max_workers=STAGE2_WORKERS,
    )
    ranked = condor.score_and_rank(stage2)
    ranked = _breakdown_condor(ranked)
    logger.info(f"[Condor] stage 2: {len(ranked)} passed")

    top = ranked.head(TOP_N_CONDOR)
    out = []
    for _, r in top.iterrows():
        r = r.to_dict()
        out.append({
            "symbol": r["ticker"], "price": round(float(r["last_price"]), 2),
            "score": round(float(r["score"])),
            "strategy": "Iron Condor",
            "dimensions": CONDOR_DIMENSIONS,
            "breakdown": {
                "iv_hv_richness": _pts(r, "iv_hv_richness"), "range_centrality": _pts(r, "range_centrality"),
                "liquidity": _pts(r, "liquidity"), "spread_tightness": _pts(r, "spread_tightness"),
                "open_interest": _pts(r, "open_interest"),
            },
            "setup": _setup_condor(r),
            "reasoning": reasoner.explain("condor", r),
        })
    return out


def scan_spreads(metrics: pd.DataFrame, reasoner: Reasoner) -> List[Dict]:
    bp1 = spreads.filter_bull_put_stage1(
        metrics, MIN_DOLLAR_VOLUME, MIN_DAILY_RANGE, MAX_DAILY_RANGE,
    ).sort_values("avg_dollar_volume_m", ascending=False).head(STAGE1_LIMIT)
    bc1 = spreads.filter_bear_call_stage1(
        metrics, MIN_DOLLAR_VOLUME, MIN_DAILY_RANGE, MAX_DAILY_RANGE,
    ).sort_values("avg_dollar_volume_m", ascending=False).head(STAGE1_LIMIT)
    logger.info(f"[Spreads] stage 1: {len(bp1)} bull put + {len(bc1)} bear call -> stage 2")

    kwargs = dict(dte_min=DTE_MIN, dte_max=DTE_MAX, dte_floor=DTE_FLOOR, min_earnings_days=MIN_EARNINGS_DAYS,
                  target_delta=TARGET_DELTA, max_width_pct=MAX_WIDTH_PCT, min_credit_width_pct=MIN_CREDIT_WIDTH_PCT,
                  max_spread_pct=MAX_SPREAD_PCT, min_open_interest=MIN_OPEN_INTEREST,
                  min_iv_hv_ratio=MIN_IV_HV_RATIO, max_workers=STAGE2_WORKERS)
    bp2 = spreads.stage2_spread_screen(bp1, "bull_put", **kwargs)
    bc2 = spreads.stage2_spread_screen(bc1, "bear_call", **kwargs)

    # score_and_rank is called once on the COMBINED pool in the original CLI
    # (screener.py), so ranks are computed across both directions together -
    # replicated here rather than ranking each direction in isolation.
    non_empty = [d for d in (bp2, bc2) if not d.empty]
    combined = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    ranked = spreads.score_and_rank(combined)
    ranked = _breakdown_spreads(ranked)
    logger.info(f"[Spreads] stage 2: {len(bp2)} bull put + {len(bc2)} bear call passed")

    top = ranked.head(TOP_N_SPREADS)
    out = []
    for _, r in top.iterrows():
        r = r.to_dict()
        out.append({
            "symbol": r["ticker"], "price": round(float(r["last_price"]), 2),
            "score": round(float(r["score"])),
            "strategy": f"{r['direction']} Spread",
            "dimensions": SPREADS_DIMENSIONS,
            "breakdown": {
                "credit_width": _pts(r, "credit_width"), "prob_profit": _pts(r, "prob_profit"),
                "iv_hv_richness": _pts(r, "iv_hv_richness"), "liquidity": _pts(r, "liquidity"),
                "open_interest": _pts(r, "open_interest"),
            },
            "setup": _setup_spread(r),
            "reasoning": reasoner.explain("spread", r),
        })
    return out


def scan_leaps(metrics: pd.DataFrame, reasoner: Reasoner) -> List[Dict]:
    trend1 = leaps.filter_leaps_stage1_trend(
        metrics, MIN_DOLLAR_VOLUME,
    ).sort_values("avg_dollar_volume_m", ascending=False).head(STAGE1_LIMIT)
    rev1 = leaps.filter_leaps_stage1_reversal(
        metrics, MIN_DOLLAR_VOLUME, max_rsi=LEAPS_MAX_RSI,
    ).sort_values("avg_dollar_volume_m", ascending=False).head(STAGE1_LIMIT)
    logger.info(f"[LEAPS] stage 1: {len(trend1)} trend + {len(rev1)} reversal -> stage 2")

    kwargs = dict(dte_min=LEAPS_DTE_MIN, dte_max=LEAPS_DTE_MAX, monthly_horizon=LEAPS_DTE_MAX + 45,
                  max_iv_hv_ratio=LEAPS_MAX_IV_HV_RATIO, max_spread_pct=MAX_SPREAD_PCT,
                  min_open_interest=MIN_OPEN_INTEREST, min_earnings_days=LEAPS_MIN_EARNINGS_DAYS,
                  max_workers=STAGE2_WORKERS)
    trend2 = leaps.stage2_leaps_screen(trend1, LEAPS_TARGET_DELTA, **kwargs)
    rev2 = leaps.stage2_leaps_screen(rev1, LEAPS_TARGET_DELTA, **kwargs)

    # Each style is scored (and thus rank-percentiled) within its OWN pool,
    # matching leaps.py's own score_and_rank(df, style) contract - a "top
    # quality" trend candidate and a "top quality" reversal candidate are not
    # compared against each other's population, only combined for display
    # after each is independently ranked.
    trend_r = _breakdown_leaps(leaps.score_and_rank(trend2, style="trend"), "trend")
    rev_r = _breakdown_leaps(leaps.score_and_rank(rev2, style="reversal"), "reversal")
    if not trend_r.empty:
        trend_r["style"] = "Trend Following"
    if not rev_r.empty:
        rev_r["style"] = "RSI Reversal"
    logger.info(f"[LEAPS] stage 2: {len(trend_r)} trend + {len(rev_r)} reversal passed")

    non_empty = [d for d in (trend_r, rev_r) if not d.empty]
    combined = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    if combined.empty:
        return []
    top = combined.sort_values("score", ascending=False).head(TOP_N_LEAPS)

    out = []
    for _, r in top.iterrows():
        r = r.to_dict()
        out.append({
            "symbol": r["ticker"], "price": round(float(r["last_price"]), 2),
            "score": round(float(r["score"])),
            "strategy": f"LEAPS ({r['style']})",
            "dimensions": LEAPS_DIMENSIONS,
            "breakdown": {
                "iv_cheapness": _pts(r, "iv_cheapness"), "quality": _pts(r, "quality"),
                "liquidity": _pts(r, "liquidity"), "open_interest": _pts(r, "open_interest"),
                "spread_tightness": _pts(r, "spread_tightness"),
            },
            "setup": _setup_leaps(r),
            "reasoning": reasoner.explain("leaps", r),
        })
    return out


def scan_rsi(metrics: pd.DataFrame, reasoner: Reasoner) -> List[Dict]:
    bands = {
        "Oversold": rsi.filter_oversold_stage1(metrics, MIN_DOLLAR_VOLUME, RSI_OVERSOLD_MAX),
        "Approaching Oversold": rsi.filter_approaching_oversold_stage1(
            metrics, MIN_DOLLAR_VOLUME, RSI_OVERSOLD_MAX, RSI_APPROACHING_OVERSOLD_MAX),
        "Approaching Overbought": rsi.filter_approaching_overbought_stage1(
            metrics, MIN_DOLLAR_VOLUME, RSI_APPROACHING_OVERBOUGHT_MIN, RSI_OVERBOUGHT_MIN),
        "Overbought": rsi.filter_overbought_stage1(metrics, MIN_DOLLAR_VOLUME, RSI_OVERBOUGHT_MIN),
    }
    bands = {n: d.sort_values("avg_dollar_volume_m", ascending=False).head(RSI_STAGE1_LIMIT) for n, d in bands.items()}
    logger.info("[RSI] stage 1: " + ", ".join(f"{len(d)} {n}" for n, d in bands.items()))

    directions = {"Oversold": "bullish", "Approaching Oversold": "bullish",
                  "Approaching Overbought": "bearish", "Overbought": "bearish"}
    tagged = []
    for name, stage1_df in bands.items():
        stage2_df = rsi.stage2_rsi_screen(stage1_df, DTE_MIN, DTE_MAX, max_workers=STAGE2_WORKERS)
        ranked = _breakdown_rsi(rsi.score_and_rank(stage2_df, directions[name]), directions[name])
        if not ranked.empty:
            ranked = ranked.copy()
            ranked["band"] = name
            tagged.append(ranked)
    logger.info("[RSI] stage 2 complete")

    if not tagged:
        return []
    combined = pd.concat(tagged, ignore_index=True)
    # Two per band rather than a flat top-N-overall: without this, one band
    # (e.g. Overbought in a strong uptrend) could crowd out the other three
    # entirely, defeating the point of showing all four zones.
    #
    # Sort globally first, then groupby().head() per band - NOT
    # groupby().apply(lambda g: g.sort_values(...).head(...)): pandas 3.x
    # defaults DataFrameGroupBy.apply to include_groups=False, which silently
    # drops the "band" column itself from what the lambda receives (and thus
    # from its output), so every row below crashed on a missing 'band' key.
    top = combined.sort_values("score", ascending=False).groupby("band", group_keys=False).head(TOP_N_RSI // 4)

    out = []
    for _, r in top.iterrows():
        r = r.to_dict()
        out.append({
            "symbol": r["ticker"], "price": round(float(r["last_price"]), 2),
            "score": round(float(r["score"])),
            "strategy": f"RSI {r['band']}",
            "dimensions": RSI_DIMENSIONS,
            "breakdown": {"extremity": _pts(r, "extremity"), "liquidity": _pts(r, "liquidity")},
            "setup": None,  # a signal, not a specific options structure - see rsi.py's docstring
            "reasoning": reasoner.explain("rsi", r),
        })
    return out


# ── Publish -------------------------------------------------------------------

def publish_to_dashboard(opportunities: List[Dict], context: List[Dict]) -> None:
    """Same additive-registration contract as Monu's publish_to_dashboard():
    writes only docs/data/OPY.json plus this agent's own entry in
    docs/data/agents.json, so Opy and Monu can run on independent schedules
    without either overwriting the other's data.
    """
    os.makedirs(DOCS_DATA_DIR, exist_ok=True)

    for i, opp in enumerate(sorted(opportunities, key=lambda o: o["score"], reverse=True), 1):
        opp["rank"] = i

    payload = {
        "agent": AGENT,
        "scan_date": datetime.now().isoformat(),
        "context": context,
        "opportunities": sorted(opportunities, key=lambda o: o["rank"]),
    }

    agent_file = os.path.join(DOCS_DATA_DIR, f"{AGENT['id']}.json")
    with open(agent_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)

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

    logger.info(f"Published {len(opportunities)} opportunities -> {agent_file}")


def main():
    logger.info("Starting Opy - options screener agent...")
    reasoner = Reasoner()

    universe = load_universe()
    logger.info(f"Computing price/trend metrics for {len(universe)} symbols...")
    metrics = core.compute_price_metrics(universe)
    logger.info(f"{len(metrics)} symbols have sufficient price history")
    if metrics.empty:
        logger.error("No usable price data - nothing to publish")
        sys.exit(1)

    opportunities: List[Dict] = []
    opportunities += scan_condor(metrics, reasoner)
    opportunities += scan_spreads(metrics, reasoner)
    opportunities += scan_leaps(metrics, reasoner)
    opportunities += scan_rsi(metrics, reasoner)

    by_strategy = {}
    for o in opportunities:
        by_strategy[o["strategy"]] = by_strategy.get(o["strategy"], 0) + 1

    context = [
        {"label": "Universe", "value": f"{len(universe)} symbols"},
        {"label": "Passed Filters", "value": f"{len(metrics)}"},
        {"label": "Published", "value": str(len(opportunities))},
        {"label": "Data", "value": "CBOE delayed (~15m)"},
        {"label": "Model", "value": CLAUDE_MODEL if reasoner.enabled else "reasoning off"},
    ]

    logger.info("Breakdown: " + ", ".join(f"{k}={v}" for k, v in by_strategy.items()) if by_strategy else "No candidates this run")

    publish_to_dashboard(opportunities, context)

    # Local artifact for debugging/history, mirroring Monu's monu_results.json.
    with open("opy_results.json", "w") as f:
        json.dump({"scan_date": datetime.now().isoformat(), "context": context,
                    "opportunities": opportunities}, f, indent=2, default=str)

    return opportunities


if __name__ == "__main__":
    main()
