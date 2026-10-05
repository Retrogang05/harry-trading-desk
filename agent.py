#!/usr/bin/env python3
"""
MONU - MNTM (Momentum Trader Agent)
An AI-powered momentum trading scanner using Claude for intelligent reasoning.

Author: Your Name
Date: 2026
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import logging

import anthropic
import numpy as np
import pandas as pd
import yfinance as yf
from anthropic import Anthropic
import ta  # Technical Analysis library

import sectors   # shared sector-ETF regime, also used by Opy
import universe  # shared universe.txt parser, also used by Opy and Goldy

# Which Claude model writes the reasoning. Haiku is the cheapest tier
# ($1/$5 per Mtok) and is what the cost estimate in the README assumes.
# Swap to "claude-opus-4-8" ($5/$25) for stronger analysis at ~5x the cost.
CLAUDE_MODEL = "claude-haiku-4-5"

# Identity this agent publishes under on the dashboard. A second strategy
# gets its own AGENT block and its own file in docs/data/ - the dashboard
# renders whatever agents the manifest lists, with no changes to the page.
AGENT = {
    "id": "MNTM",
    "name": "Monu",
    "strategy": "Momentum",
    "description": "Buys strength: breakouts confirmed by volume, riding the trend until momentum fades.",
    # Muted, low-saturation hue - the dashboard chrome is neutral, so each
    # agent's accent is the only identity colour it gets. Keep new agents in
    # the same register (e.g. #6f8fae blue, #8a7fa8 violet) rather than neon.
    "accent": "#c8974a",
}

# The score dimensions this agent reports, in display order. The dashboard
# reads this rather than hardcoding Monu's criteria, so an agent scoring on
# entirely different axes renders correctly without touching the page.
DIMENSIONS = [
    {"key": "trend_strength", "label": "Trend", "max": 25},
    {"key": "volume", "label": "Volume", "max": 20},
    {"key": "rsi", "label": "RSI", "max": 15},
    {"key": "macd", "label": "MACD", "max": 15},
    {"key": "relative", "label": "Relative", "max": 15},
    {"key": "breakout", "label": "Breakout", "max": 10},
]

# Published alongside DIMENSIONS so the dashboard renders the two filters that
# actually validated. Deliberately NOT folded into the 0-100 score: the score's
# own weights tested as noise, and mixing a measured signal into an unmeasured
# one would hide which part is working.
STRUCTURE_DIMENSIONS = [
    {"key": "room", "label": "Room to target", "max": 50},
    {"key": "own_trend", "label": "Own trend", "max": 50},
]

# Where the dashboard reads its data from.
DOCS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "data")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MomentumAnalyzer:
    """Analyzes stocks for momentum trading opportunities."""

    def __init__(self):
        self.model = CLAUDE_MODEL
        # Reasoning is optional. Everything else in this agent - the universe
        # fetch, all six score dimensions, the regime gate, entry/stop/target -
        # is pure price maths and must still run without a Claude key. The SDK
        # raises a bare TypeError at request time when no credential resolves,
        # which is not an anthropic.* exception, so it cannot be caught by the
        # handlers in generate_reasoning() - gate on the key up front instead.
        self.reasoning_enabled = bool(os.environ.get("ANTHROPIC_API_KEY"))
        self.client = Anthropic() if self.reasoning_enabled else None
        if not self.reasoning_enabled:
            logger.warning(
                "ANTHROPIC_API_KEY not set - scanning and scoring will run, "
                "but signals will publish without written reasoning"
            )

    # MA200 and the 52-week high need ~252 trading days. Calendar days are
    # ~30% weekends/holidays, so ask for 400 to land comfortably above that.
    CALENDAR_DAYS = 400
    MIN_ROWS = 200

    # Lookback for "recent" breakout and its volume confirmation. Both
    # dimensions must use the same window or they contradict each other.
    BREAKOUT_WINDOW = 5

    # How far above MA20 price can sit and still be treated as a pullback
    # setup. Beyond this the stock is extended and chasing it is the "late
    # entry" the strategy warns about. Tune with your paper-trading results.
    MAX_EXTENSION = 0.04

    # Tradability floor, matched to scripts/build_universe.py. Only bites once
    # the universe includes the Russell 2000: every S&P 500 name clears this by
    # two orders of magnitude, while a third of IWM does not clear it at all.
    MIN_PRICE = 5.0
    MIN_DOLLAR_VOLUME = 10_000_000
    LIQUIDITY_WINDOW = 60      # ~3 months, same basis as the build-time screen

    # A single session cannot legitimately move this far. Beyond it the series
    # has an unadjusted corporate action in it - a split, a spin-off, a reverse
    # split - and every average spanning the gap is arithmetic on two different
    # securities. Found via Goldy: CTVA came back with 2026-09-30 at $77.65 and
    # 2026-10-01 at $14.44, an unadjusted spin-off, which makes its MA50, MA200,
    # ATR and 52-week high all meaningless - and Monu scores on every one of
    # them. Real stocks do gap 40% on news, but nothing here can say anything
    # useful about one that has, so skipping is right either way.
    MAX_SESSION_MOVE = 0.40

    def continuous(self, data: pd.DataFrame) -> bool:
        """Reject a price series with an unadjusted corporate action in it.

        Checked across the whole MIN_ROWS window the indicators are computed
        over, not just recent bars: a gap 100 sessions back still poisons a
        200-day mean.
        """
        try:
            close = data['Close'].tail(self.MIN_ROWS + 1)
            if len(close) < 2:
                return False
            step = (close / close.shift(1)).dropna()
            return bool(((step - 1).abs() < self.MAX_SESSION_MOVE).all())
        except (KeyError, IndexError, ValueError, TypeError):
            return False

    def liquid_enough(self, data: pd.DataFrame) -> bool:
        """Can a position be entered and, more importantly, stopped out of?

        Median rather than mean dollar volume: one earnings-day print on an
        otherwise untraded small cap would carry a mean over the floor.
        """
        try:
            window = data.tail(self.LIQUIDITY_WINDOW)
            if float(data['Close'].iloc[-1]) < self.MIN_PRICE:
                return False
            dollar_volume = (window['Close'] * window['Volume']).median()
            return float(dollar_volume) >= self.MIN_DOLLAR_VOLUME
        except (KeyError, IndexError, ValueError, TypeError):
            return False  # malformed frame: treat as untradable, not as a pass

    def _usable(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalise a raw yfinance frame, or return None if it can't be scored."""
        if data is None or len(data) == 0:
            logger.warning(f"{symbol}: no data returned")
            return None

        # yfinance returns MultiIndex columns ('Close', 'AAPL'). The ta
        # library needs 1-D Series, so drop the ticker level.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # A batch slice for a delisted/bad ticker comes back as all-NaN rows
        # rather than an empty frame.
        data = data.dropna(how="all")

        if len(data) < self.MIN_ROWS:
            logger.warning(
                f"{symbol}: only {len(data)} rows, need {self.MIN_ROWS} for MA200 - skipping"
            )
            return None
        return data

    def fetch_many(self, symbols: List[str], days: int = CALENDAR_DAYS) -> Dict[str, pd.DataFrame]:
        """Fetch every symbol in one request.

        One download() call per symbol costs ~0.6s each; batching the whole
        universe into a single call costs ~0.08s each - about 7x faster, which
        is the difference between a 20-symbol toy list and scanning the S&P 500
        inside a GitHub Actions run.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        out = {}

        # Chunked so one bad ticker can't poison the whole universe, and to
        # stay within yfinance's per-request URL limits.
        #
        # threads=True is safe HERE only because scan_stocks() fetches SPY on
        # its own first, and that single unthreaded request creates yfinance's
        # SQLite timezone cache (~/.cache/py-yfinance, cold on every CI run)
        # before any thread pool touches it. Without that warm-up the first
        # batch has 100 workers racing to create one database and the losers
        # return OperationalError('database is locked') as empty frames -
        # silently, since the per-symbol loop below swallows it. Trey died of
        # exactly this race on runs #12 and #13. Keep the SPY fetch ahead of
        # this call, or warm the cache some other way before reordering.
        CHUNK = 100
        for i in range(0, len(symbols), CHUNK):
            chunk = symbols[i:i + CHUNK]
            logger.info(f"Fetching {len(chunk)} symbols ({i + 1}-{i + len(chunk)} of {len(symbols)})...")
            try:
                raw = yf.download(
                    chunk, start=start_date, end=end_date,
                    progress=False, group_by="ticker", auto_adjust=False,
                    threads=True,
                )
            except Exception as e:
                logger.error(f"Batch {i // CHUNK} failed: {e}")
                continue

            for sym in chunk:
                try:
                    # With several tickers yfinance nests per-ticker frames;
                    # with exactly one it returns the flat frame directly.
                    frame = raw[sym] if isinstance(raw.columns, pd.MultiIndex) and sym in raw.columns.levels[0] else raw
                    usable = self._usable(frame.copy(), sym)
                    if usable is not None:
                        out[sym] = usable
                except Exception as e:
                    logger.warning(f"{sym}: could not extract from batch ({e})")

        logger.info(f"{len(out)}/{len(symbols)} symbols have enough history to score")
        return out

    # Yahoo answers the odd request with an empty frame and recovers within
    # seconds. This matters more here than anywhere else in the scan: the one
    # symbol this function fetches is SPY, and SPY decides the regime, so a
    # blip closes the gate and costs the whole day's scan. Trey hit the same
    # transient on 2026-09-30 and 10-01 and ran clean on the 2nd.
    FETCH_ATTEMPTS = 3
    FETCH_BACKOFF = 5     # seconds, multiplied by the attempt number

    def fetch_stock_data(self, symbol: str, days: int = CALENDAR_DAYS) -> pd.DataFrame:
        """Fetch historical stock data for a single symbol (used for the index)."""
        logger.info(f"Fetching data for {symbol}...")
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            data = None
            for attempt in range(1, self.FETCH_ATTEMPTS + 1):
                data = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                if data is not None and len(data):
                    break
                if attempt < self.FETCH_ATTEMPTS:
                    wait = self.FETCH_BACKOFF * attempt
                    logger.warning(
                        f"{symbol}: empty response on attempt {attempt}/{self.FETCH_ATTEMPTS}"
                        f" - retrying in {wait}s"
                    )
                    time.sleep(wait)

            if data is None or len(data) == 0:
                logger.warning(
                    f"No data found for {symbol} after {self.FETCH_ATTEMPTS} attempts"
                )
                return None

            # yfinance returns MultiIndex columns ('Close', 'AAPL'). The ta
            # library needs 1-D Series, so drop the ticker level.
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            if len(data) < self.MIN_ROWS:
                logger.warning(
                    f"{symbol}: only {len(data)} rows, need {self.MIN_ROWS} for MA200 - skipping"
                )
                return None

            return data
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {str(e)}")
            return None

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators."""
        if data is None or len(data) < self.MIN_ROWS:
            return None

        try:
            # Moving Averages
            ma20 = ta.trend.sma_indicator(data['Close'], window=20)
            ma50 = ta.trend.sma_indicator(data['Close'], window=50)
            ma200 = ta.trend.sma_indicator(data['Close'], window=200)

            # RSI (Relative Strength Index)
            rsi = ta.momentum.rsi(data['Close'], window=14)

            # MACD (Moving Average Convergence Divergence)
            macd = ta.trend.macd_diff(data['Close'], window_slow=26, window_fast=12, window_sign=9)
            macd_line = ta.trend.macd(data['Close'], window_slow=26, window_fast=12)
            macd_signal = ta.trend.macd_signal(data['Close'], window_slow=26, window_fast=12, window_sign=9)

            # ATR (Average True Range) for volatility
            atr = ta.volatility.average_true_range(
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                window=14
            )

            # Volume analysis
            volume_avg_20 = data['Volume'].rolling(window=20).mean()

            return {
                'ma20': ma20,
                'ma50': ma50,
                'ma200': ma200,
                'rsi': rsi,
                'macd': macd,
                'macd_line': macd_line,
                'macd_signal': macd_signal,
                'atr': atr,
                'volume_avg': volume_avg_20
            }
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return None

    def score_breakdown_dimensions(self, data: pd.DataFrame, indicators: Dict) -> Dict:
        """Mirror image of the momentum score, for a stock breaking DOWN.

        Not a negated momentum score. Several axes do not simply invert: a
        falling stock's RSI band of interest is the one just under 50 rather
        than the mirror of 50-70, and "distance from the 52-week high" becomes
        "distance from the 52-week LOW", which is a different measurement on a
        different reference point.

        Scored on the same 0-100 scale as the long side so the two lists read
        against each other, and gated to weak sectors by the caller - this
        function describes a breakdown, it does not decide whether to look.
        """
        close = data['Close']
        price = float(close.iloc[-1])
        ma20 = float(indicators['ma20'].iloc[-1])
        ma50 = float(indicators['ma50'].iloc[-1])
        ma200 = float(indicators['ma200'].iloc[-1])
        rsi = float(indicators['rsi'].iloc[-1])
        macd = float(indicators['macd'].iloc[-1])
        avg_vol = float(indicators['volume_avg'].iloc[-1])

        if any(np.isnan(x) for x in (price, ma20, ma50, ma200, rsi, macd, avg_vol)):
            return None

        scores = {}

        # Trend, 25: stacked downward, price under each average in turn.
        t = 0
        if price < ma20: t += 8
        if price < ma50: t += 9
        if price < ma200: t += 8
        scores['trend_strength'] = t

        # Volume, 20: distribution. The heaviest day of the last week relative
        # to normal - selling on volume is the part that distinguishes a
        # breakdown from a drift.
        try:
            vol_ratio = float((data['Volume'].tail(self.BREAKOUT_WINDOW) / avg_vol).max())
        except (ZeroDivisionError, ValueError):
            vol_ratio = 0.0
        scores['volume'] = 20 if vol_ratio > 1.5 else 12 if vol_ratio > 1.2 else 6 if vol_ratio > 1.0 else 0

        # RSI, 15: the mirror of the long side's bands, and contiguous for the
        # same reason - weak but not yet washed out is the tradable zone, and
        # sub-30 is where short squeezes start.
        scores['rsi'] = 5 if rsi <= 30 else 15 if rsi <= 50 else 8 if rsi <= 60 else 0

        # MACD, 15: histogram below zero and still falling.
        hist = indicators['macd']
        falling = len(hist) > 3 and float(hist.iloc[-1]) < float(hist.iloc[-4])
        scores['macd'] = 15 if macd < 0 and falling else 9 if macd < 0 else 0

        # Relative, 15: nearness to the 52-week LOW. The long side measures
        # distance from the high; this is its own reference point, not a sign
        # flip of that one.
        try:
            low52 = float(close.tail(252).min())
            high52 = float(close.tail(252).max())
            span = high52 - low52
            frac = (price - low52) / span if span else 1.0   # 0 = at the low
        except (ValueError, ZeroDivisionError):
            frac = 1.0
        scores['relative'] = 15 if frac < 0.10 else 11 if frac < 0.25 else 6 if frac < 0.40 else 0

        # Breakout, 10: a fresh break of the 20-day low.
        try:
            low20 = float(data['Low'].iloc[-21:-1].min())
            scores['breakout'] = 10 if float(data['Low'].iloc[-1]) < low20 else 0
        except (ValueError, IndexError):
            scores['breakout'] = 0

        scores['total'] = sum(scores.values())
        scores['vol_ratio'] = vol_ratio
        scores['pct_off_low'] = round(frac * 100, 1)
        return scores

    # Floor for the bearish half. Higher than the long side's 60 on purpose:
    # the long list is the one with validated structure filters behind it,
    # while this half is new and unmeasured, so it publishes only its clearest
    # cases rather than filling the dashboard with marginal breakdowns.
    BEARISH_MIN_SCORE = 70

    def _bearish_row(self, symbol, data, indicators, bear, membership,
                     sector, sector_state, market_regime) -> Dict:
        """One published breakdown, with levels quoted for a SHORT.

        Entry/stop/target are the short-side mirror: the stop sits ABOVE
        entry, so risk is stop - entry, and the target sits below. They are
        labelled bearish on the row so the dashboard never colours or words
        them as a buy.
        """
        price = float(data['Close'].iloc[-1])
        atr = float(indicators['atr'].iloc[-1])
        if np.isnan(atr) or atr <= 0:
            atr = price * 0.02

        # Stop above the lower of the 20-day average and a 2-ATR band: a
        # reclaim of the average kills the premise, but on a name that has
        # already fallen far below it, the volatility stop is the usable one.
        structural = float(indicators['ma20'].iloc[-1])
        volatility = price + atr * 2
        stop = min(structural, volatility) if not np.isnan(structural) else volatility
        stop = min(stop, price * 1.15)          # cap risk at 15%, as the long side does
        risk = max(stop - price, atr * 0.5)

        sec = sector_state.get(sector, {})
        return {
            'rank': 0,
            'symbol': symbol,
            'price': price,
            'score': bear['total'],
            'strategy': 'Breakdown',
            'bias': 'bearish',
            'signal_label': 'Bearish',
            'signal_tone': 'neg',
            'dimensions': DIMENSIONS,
            'breakdown': {k: bear[k] for k in
                          ('trend_strength', 'volume', 'rsi', 'macd', 'relative', 'breakout')},
            'indexes': membership.get(symbol, [UNTAGGED]),
            'sector': sector,
            'entry': price,
            'stop_loss': stop,
            'take_profit': price - risk * 1.5,
            'risk_reward_ratio': 1.5,
            'pct_off_low': bear['pct_off_low'],
            'market_regime': market_regime,
            'setup': {
                'title': 'Breakdown detail',
                'label': (f"{sec.get('name', sector)} weak: {sector} "
                          f"{sec.get('pct', 0):+.2f}% vs its 20-day"),
                'fields': [
                    {'label': 'Direction', 'value': 'Short', 'tone': 'neg'},
                    {'label': 'Sector', 'value': f"{sector} {sec.get('pct', 0):+.2f}%",
                     'tone': 'neg'},
                    {'label': 'Off 52w low', 'value': f"{bear['pct_off_low']}%",
                     'tone': 'neg' if bear['pct_off_low'] < 15 else None},
                    {'label': 'Vol vs avg', 'value': f"{bear['vol_ratio']:.2f}x",
                     'tone': 'neg' if bear['vol_ratio'] > 1.5 else None},
                ],
            },
        }

    def score_momentum_dimensions(self, symbol: str, data: pd.DataFrame, indicators: Dict) -> Dict[str, float]:
        """Score stock across 6 momentum dimensions (0-100 scale)."""

        current_price = data['Close'].iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        current_rsi = indicators['rsi'].iloc[-1]
        current_macd = indicators['macd'].iloc[-1]
        current_macd_line = indicators['macd_line'].iloc[-1]
        current_macd_signal = indicators['macd_signal'].iloc[-1]
        current_atr = indicators['atr'].iloc[-1]
        avg_volume = indicators['volume_avg'].iloc[-1]
        current_ma20 = indicators['ma20'].iloc[-1]
        current_ma50 = indicators['ma50'].iloc[-1]
        current_ma200 = indicators['ma200'].iloc[-1]

        # A NaN indicator would silently score 0 and look like weak momentum
        # rather than missing data. Refuse to score instead.
        latest = {
            'rsi': current_rsi, 'macd': current_macd, 'macd_line': current_macd_line,
            'macd_signal': current_macd_signal, 'atr': current_atr,
            'volume_avg': avg_volume, 'ma20': current_ma20,
            'ma50': current_ma50, 'ma200': current_ma200,
        }
        missing = [name for name, value in latest.items() if pd.isna(value)]
        if missing:
            logger.warning(f"{symbol}: indicators are NaN {missing} - skipping")
            return None

        # 52-week high/low
        high_52w = data['High'].tail(252).max() if len(data) >= 252 else data['High'].max()

        scores = {}

        # 1. TREND STRENGTH (25 points max)
        trend_score = 0
        if current_price > current_ma20:
            trend_score += 8
        if current_price > current_ma50:
            trend_score += 8
        if current_price > current_ma200:
            trend_score += 9
        scores['trend_strength'] = min(trend_score, 25)

        # 2. VOLUME CONFIRMATION (20 points max)
        # Volume confirms the breakout, and the breakout may be a few days old.
        # Scoring only the latest day would zero out a stock that broke out on
        # 3x volume on Tuesday but drifted quietly on Friday, so take the best
        # ratio over the same 5-day window the breakout dimension uses.
        volume_ratio = (data['Volume'].tail(self.BREAKOUT_WINDOW) / avg_volume).max()
        if volume_ratio > 1.5:
            volume_score = 20  # Strong volume spike
        elif volume_ratio > 1.2:
            volume_score = 12  # Moderate volume increase
        elif volume_ratio > 1.0:
            volume_score = 6   # Slight volume increase
        else:
            volume_score = 0
        scores['volume'] = volume_score

        # 3. RSI STRENGTH (15 points max)
        # Bands are contiguous and closed on the low side so every value lands
        # somewhere. The previous version used `50 < r < 70` and `r > 70`,
        # which left exactly 70.0 (and exactly 40.0) scoring 0 - a value one
        # tick either side scored 15 or 5.
        if current_rsi >= 70:
            rsi_score = 5   # overbought - momentum present but late
        elif current_rsi >= 50:
            rsi_score = 15  # the momentum sweet spot
        elif current_rsi >= 40:
            rsi_score = 8   # weak but positive
        else:
            rsi_score = 0   # below 40: no momentum case to make (30-40 deliberately 0)
        scores['rsi'] = rsi_score

        # 4. MACD ALIGNMENT (15 points max)
        macd_score = 0
        if current_macd_line > current_macd_signal and current_macd > 0:
            macd_score = 15  # Strong bullish signal
        elif current_macd_line > current_macd_signal:
            macd_score = 10  # Bullish but histogram negative
        elif current_macd > 0:
            macd_score = 5   # Positive but line weak
        scores['macd'] = macd_score

        # 5. RELATIVE PERFORMANCE (15 points max)
        relative_score = 0
        if current_price > high_52w * 0.95:
            relative_score = 15  # Near 52-week high (strong momentum)
        elif current_price > high_52w * 0.90:
            relative_score = 12
        elif current_price > high_52w * 0.80:
            relative_score = 8
        scores['relative'] = relative_score

        # 6. BREAKOUT QUALITY (10 points max)
        # Check if recent breakout (last 5 days)
        breakout_score = 0
        window = self.BREAKOUT_WINDOW
        recent_close_above_ma50 = (indicators['ma50'].tail(window) < data['Close'].tail(window)).sum()
        recent_volume_spike = (data['Volume'].tail(window) > avg_volume * 1.5).sum()

        if recent_close_above_ma50 >= 3 and recent_volume_spike >= 2:
            breakout_score = 10
        elif recent_close_above_ma50 >= 2:
            breakout_score = 6
        elif current_price > current_ma50 and current_volume > avg_volume:
            breakout_score = 4
        scores['breakout'] = breakout_score

        # Calculate total
        total_score = sum(scores.values())
        scores['total'] = total_score

        return scores

    def check_market_regime(self, spy_data: pd.DataFrame) -> str:
        """Market regime from SPY's 50/200-day averages: UPTREND, DOWNTREND, or
        UNKNOWN when SPY couldn't be fetched.

        This is a GATE, not a label. Momentum breakouts are the strategy's
        thesis and the research behind it is explicit that they fail in a
        falling market - so in anything other than a confirmed UPTREND the
        scan is skipped and nothing is published. For its first month this
        function's result was attached to every row and displayed on the
        dashboard while gating nothing; Monu would have published the same
        twenty breakouts into a bear market.

        UNKNOWN closes the gate too: if the regime can't be verified, the
        safe default is to not put out momentum signals, not to assume an
        uptrend. (The old CAUTION branch fired only when the two averages
        were exactly equal to the tick - dead code - and is gone.)
        """
        if spy_data is None or len(spy_data) < 200:
            return "UNKNOWN"
        ma50 = ta.trend.sma_indicator(spy_data['Close'], window=50).iloc[-1]
        ma200 = ta.trend.sma_indicator(spy_data['Close'], window=200).iloc[-1]
        return "UPTREND" if ma50 > ma200 else "DOWNTREND"

    GATE_OPEN_REGIMES = {"UPTREND"}

    def generate_bearish_reasoning(self, o: Dict) -> str:
        """Narrate a breakdown. Its own prompt rather than the long one with
        the words swapped: the question that matters on a short is what makes
        it squeeze, which has no equivalent on the long side."""
        if not self.reasoning_enabled:
            return "[reasoning unavailable: ANTHROPIC_API_KEY not configured]"

        b = o['breakdown']
        prompt = f"""{o['symbol']} at ${o['price']:.2f} is breaking down, and its sector
ETF {o['sector']} is below its own 20-day average.

Breakdown score: {o['score']}/100
- Trend (below the averages): {b['trend_strength']}/25
- Volume (distribution): {b['volume']}/20
- RSI: {b['rsi']}/15
- MACD: {b['macd']}/15
- Position vs 52-week range: {b['relative']}/15  ({o['pct_off_low']}% off the low)
- Fresh 20-day low: {b['breakout']}/10

Short setup: entry ${o['entry']:.2f}, stop ${o['stop_loss']:.2f} (above),
target ${o['take_profit']:.2f}.

In 2-3 sentences: is this a genuine distribution or an oversold stock about to
bounce, and what would squeeze it? Be concrete and sceptical. Name the level
that would invalidate it. Do not recommend a trade."""
        try:
            message = self.client.messages.create(
                model=self.model, max_tokens=300,
                messages=[{"role": "user", "content": prompt}])
            return message.content[0].text.strip()
        except anthropic.RateLimitError:
            return "[reasoning unavailable: rate limited]"
        except anthropic.APIStatusError as e:
            return f"[reasoning unavailable: API error {e.status_code}]"
        except anthropic.APIConnectionError:
            return "[reasoning unavailable: connection error]"
        except Exception as e:
            return f"[reasoning unavailable: {type(e).__name__}]"

    def generate_reasoning(self, symbol: str, score_breakdown: Dict, price: float,
                          entry: float, stop_loss: float, take_profit: float) -> str:
        """Generate Claude-powered reasoning for the trade setup."""

        if not self.reasoning_enabled:
            return "[reasoning unavailable: ANTHROPIC_API_KEY not configured]"

        prompt = f"""You are a momentum trading expert. Given these indicators for {symbol},
provide a concise (2-3 sentence) explanation of the momentum trading setup.

Stock: {symbol}
Current Price: ${price:.2f}
Momentum Score: {score_breakdown['total']}/100

Score Breakdown:
- Trend Strength: {score_breakdown['trend_strength']}/25
- Volume: {score_breakdown['volume']}/20
- RSI: {score_breakdown['rsi']}/15
- MACD: {score_breakdown['macd']}/15
- Relative Performance: {score_breakdown['relative']}/15
- Breakout Quality: {score_breakdown['breakout']}/10

Trade Setup:
- Entry Level: ${entry:.2f}
- Stop Loss: ${stop_loss:.2f}
- Take Profit: ${take_profit:.2f}
- Risk/Reward Ratio: {(take_profit - entry) / (entry - stop_loss):.2f}:1

Please explain:
1. Why this stock shows momentum right now
2. Which indicators are most aligned
3. One key risk to watch

Format: Professional but conversational, suitable for a trader's quick decision."""


        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
        except anthropic.RateLimitError:
            logger.error(f"{symbol}: rate limited by the Claude API")
            return "[reasoning unavailable: rate limited]"
        except anthropic.APIStatusError as e:
            logger.error(f"{symbol}: Claude API error {e.status_code}: {e.message}")
            return f"[reasoning unavailable: API error {e.status_code}]"
        except anthropic.APIConnectionError:
            logger.error(f"{symbol}: could not reach the Claude API")
            return "[reasoning unavailable: connection error]"
        except Exception as e:
            # Last resort. A scan is ~30s of network work across 500 symbols;
            # losing all of it because one narration call raised something
            # unanticipated is never the right trade. This is what a bare
            # TypeError from an unresolved credential used to slip through.
            logger.error(f"{symbol}: unexpected reasoning failure: {type(e).__name__}: {e}")
            return f"[reasoning unavailable: {type(e).__name__}]"

        if message.stop_reason == "refusal":
            logger.warning(f"{symbol}: Claude declined to answer")
            return "[reasoning unavailable: request declined]"

        # content is a list of blocks, not a string - the first block is not
        # guaranteed to be text, so pick the text blocks out explicitly.
        text = "".join(b.text for b in message.content if b.type == "text").strip()
        return text or "[reasoning unavailable: empty response]"

    # ── Structure filters ────────────────────────────────────────────────
    # Two rules kept from the chart-setup-analysis skill, after validating its
    # 6-point checklist against all 759 signals Monu had published to date.
    # The checklist's own score did NOT predict outcome (corr -0.07; the
    # 5-of-6 signals were the worst group). Four of its six points were noise
    # or inverted as a screen. These two were not:
    #
    #   room to target   0 swing highs between entry and target -> +2.73% / 53% win
    #                    1-2 -> -3.83%,  3-4 -> -5.31%,  5+ -> -5.86%
    #   own trend up     stock's own 50d EMA rising -> -3.49% vs -8.08% when not
    #
    # Together they kept 9% of signals (~2/scan day) and were the only subset
    # that was positive in an 8-week window where the full list lost 4.2% and
    # SPY lost 0.6%. Small sample (32 signals, 17 days) - these are published
    # as fields and used to rank, not as a hard gate, so the effect stays
    # measurable in results/ rather than silently removing the counterfactual.

    # A swing high is a bar whose high is the max of the 3 bars each side -
    # the same pivot definition the skill uses.
    PIVOT_BARS = 3

    def swing_highs(self, data: pd.DataFrame) -> pd.Series:
        h = data['High']
        w = self.PIVOT_BARS * 2 + 1
        return h[h == h.rolling(w, center=True).max()].dropna()

    def structure_check(self, data: pd.DataFrame, entry: float, target: float) -> Dict:
        """Swing highs standing between entry and target, and whether the
        stock's own 50-day EMA is rising. Monu's regime gate only ever looked
        at SPY - a stock can be in its own downtrend inside a market uptrend,
        and those were the worst performers in the review."""
        # Only pivots confirmed before today: the centred window means the last
        # PIVOT_BARS bars cannot yet be known to be pivots, and using them
        # would be lookahead.
        confirmed = self.swing_highs(data.iloc[:-self.PIVOT_BARS]) if len(data) > self.PIVOT_BARS else pd.Series(dtype=float)
        blockers = int(((confirmed > entry) & (confirmed < target)).sum())

        ema50 = data['Close'].ewm(span=50, adjust=False).mean()
        slope = float(ema50.iloc[-1] / ema50.iloc[-21] - 1) if len(ema50) > 21 else 0.0
        return {
            "blockers": blockers,
            "room_clear": blockers == 0,
            "own_trend_pct": slope * 100,
            "own_trend_up": slope > 0.01,          # >1% over ~1 month
        }

    def calculate_entry_exit(self, data: pd.DataFrame, indicators: Dict,
                            score_breakdown: Dict) -> Dict[str, float]:
        """Calculate entry, stop-loss, and take-profit levels."""

        current_price = data['Close'].iloc[-1]
        current_atr = indicators['atr'].iloc[-1]
        current_ma20 = indicators['ma20'].iloc[-1]

        # Conservative entry waits for a pullback to MA20. But when price has
        # already run far above MA20 that pullback may never come, and a target
        # measured from MA20 can land below today's price - i.e. "sell lower
        # than it trades now", which is not a trade. Treat those as extended.
        extension = (current_price - current_ma20) / current_ma20

        if extension > self.MAX_EXTENSION:
            setup_type = 'EXTENDED'
            # Anchor to current price: this is a breakout/continuation entry,
            # not a pullback entry.
            entry = current_price
        else:
            setup_type = 'PULLBACK'
            entry = current_ma20

        stop_loss = entry - (current_atr * 1.5)
        take_profit = entry + (current_atr * 1.5 * 1.5)  # 1.5x risk-reward

        return {
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_per_trade': entry - stop_loss,
            'setup_type': setup_type,
            'extension_pct': extension * 100
        }

    def scan_stocks(self, symbols: List[str], membership: Dict[str, List[str]] = None,
                    sector_map: Dict[str, str] = None, per_index: int = 8):
        """Scan for momentum opportunities. Returns (opportunities, market_regime).

        `membership` maps symbol -> index tags (SPX/QQQ/IWM), `sector_map`
        symbol -> its sector SPDR, which is what the bearish half gates on.
        Publishing takes
        the best `per_index` from each index independently rather than the best
        N overall: the Russell 2000 contributes two thirds of the universe, and
        a single global ranking would hand it most of the list on volume of
        candidates alone. Quotas keep each index's own best setups visible and
        make the three groups comparable to each other over time.

        The union is deduplicated, so a name in both the S&P 500 and the
        Nasdaq-100 takes one row and carries both tags.

        The regime is returned explicitly rather than read back off the first
        opportunity - the old approach reported UNKNOWN whenever the list was
        empty, which is precisely when the regime matters most."""
        membership = membership or {}
        sector_map = sector_map or {}

        logger.info("Checking market regime...")
        spy_data = self.fetch_stock_data("SPY")
        market_regime = self.check_market_regime(spy_data)
        logger.info(f"Market Regime: {market_regime}")

        # Sector breadth, read before anything else because it decides whether
        # the bearish half of this scan runs at all. Independent of the SPY
        # gate above, and deliberately so: on 2026-10-05 ten of eleven sector
        # SPDRs were below their 20-day average while SPY's 50-day was still
        # over its 200-day, so the long gate read UPTREND through a market
        # where almost everything outside technology was rolling over. One
        # cap-weighted index trend is not breadth.
        sector_state = sectors.read()
        weak_sectors = set(sectors.weak_tags(sector_state))

        bullish_open = market_regime in self.GATE_OPEN_REGIMES
        if not bullish_open:
            logger.warning(
                f"Long gate CLOSED ({market_regime}): momentum breakouts are not traded "
                f"outside a confirmed uptrend."
            )
        if not weak_sectors:
            logger.info("No sector below its 20-day SMA - no bearish scan this run.")

        if not bullish_open and not weak_sectors:
            # Neither half has anything to do: nothing to fetch, nothing to
            # reason about, nothing to pay for.
            logger.warning(f"Both halves gated off; skipping the {len(symbols)}-symbol scan.")
            return [], market_regime, sector_state

        opportunities = []

        # One batched request for the whole universe, then score locally.
        frames = self.fetch_many(symbols)

        for symbol in symbols:
            data = frames.get(symbol)
            if data is None:
                continue

            # Liquidity backstop. build_universe.py already screens on this,
            # but it screens on the day it runs and universe.txt then sits for
            # weeks - a name can thin out or gap below $5 in between. Checked
            # here because the data is already in hand, and because a signal
            # that cannot be exited on its stop is worse than no signal.
            if not self.liquid_enough(data):
                continue

            # Unadjusted split or spin-off: every indicator below would be
            # computed across two different securities. See continuous().
            if not self.continuous(data):
                logger.warning(f"{symbol}: price discontinuity in the window - skipping")
                continue

            # Calculate indicators
            indicators = self.calculate_indicators(data)
            if indicators is None:
                continue

            # ── Bearish half ────────────────────────────────────────────
            # Only for names whose own sector ETF is below its 20-day
            # average. The sector gate is the whole premise: a breakdown in
            # a sector that is still working is far more likely to be noise
            # than one happening alongside its peers.
            sector = sector_map.get(symbol)
            if sector in weak_sectors:
                bear = self.score_breakdown_dimensions(data, indicators)
                if bear and bear['total'] >= self.BEARISH_MIN_SCORE:
                    opportunities.append(
                        self._bearish_row(symbol, data, indicators, bear,
                                          membership, sector, sector_state,
                                          market_regime))

            if not bullish_open:
                continue

            # Score momentum
            scores = self.score_momentum_dimensions(symbol, data, indicators)
            if scores is None:
                continue

            # Only include if score > 60
            if scores['total'] < 60:
                continue

            # Get current price
            current_price = data['Close'].iloc[-1]

            # Calculate entry/exit
            levels = self.calculate_entry_exit(data, indicators, scores)
            structure = self.structure_check(data, levels['entry'], levels['take_profit'])

            opportunity = {
                'rank': 0,  # Will be assigned after sorting
                'symbol': symbol,
                'price': current_price,
                'score': scores['total'],
                'breakdown': {
                    'trend_strength': scores['trend_strength'],
                    'volume': scores['volume'],
                    'rsi': scores['rsi'],
                    'macd': scores['macd'],
                    'relative': scores['relative'],
                    'breakout': scores['breakout'],
                    'room': 50 if structure['room_clear'] else 0,
                    'own_trend': 50 if structure['own_trend_up'] else 0,
                },
                'entry': levels['entry'],
                'stop_loss': levels['stop_loss'],
                'take_profit': levels['take_profit'],
                'risk_reward_ratio': (levels['take_profit'] - levels['entry']) / (levels['entry'] - levels['stop_loss']),
                'setup_type': levels['setup_type'],
                'extension_pct': levels['extension_pct'],
                'blockers': structure['blockers'],
                'room_clear': structure['room_clear'],
                'own_trend_pct': round(structure['own_trend_pct'], 1),
                'own_trend_up': structure['own_trend_up'],
                # Validated-structure tier, published so the dashboard can show
                # it and results/ can keep measuring it:
                #   A = clear room AND the stock's own trend is up
                #   B = one of the two
                #   C = neither
                'dimensions': DIMENSIONS + STRUCTURE_DIMENSIONS,
                'structure': ("A" if structure['room_clear'] and structure['own_trend_up']
                              else "B" if structure['room_clear'] or structure['own_trend_up']
                              else "C"),
                # Index membership, published so the dashboard can group and
                # filter by it. A list, not a string: names in both the S&P 500
                # and the Nasdaq-100 belong to both.
                'indexes': membership.get(symbol, [UNTAGGED]),
                'scores': scores,           # kept for the reasoning pass, stripped below
                'market_regime': market_regime
            }

            opportunities.append(opportunity)

        # Rank first, truncate, and only then pay for reasoning. Writing it
        # inside the scan loop bills a Claude call for every candidate over 60
        # even though most never get published - on a 165-symbol universe that
        # was 90 calls to publish 20. The bill is set by the published row
        # count, so widening the universe costs scan time, not API spend.
        #
        # Rank by the two VALIDATED structure filters first, momentum score only
        # as a tiebreak. The six-dimension score did not order outcomes in the
        # review (score 100 signals averaged -7.8% over 20 days), while clear
        # room to target did (+2.73% vs -4.2% for the rest). Tier A floats to
        # the top; nothing is dropped, so results/ still records what B and C
        # would have done.
        tier_rank = {"A": 0, "B": 1, "C": 2}

        def rank_key(o):
            # Bearish rows carry no structure tier - those filters were
            # validated on long setups and mean nothing on a short - so they
            # rank on score alone within their own quota.
            if o.get('bias') == 'bearish':
                return (0, -o['score'])
            return (tier_rank[o['structure']], -o['score'])

        bull = [o for o in opportunities if o.get('bias') != 'bearish']
        bear = [o for o in opportunities if o.get('bias') == 'bearish']
        logger.info(f"{len(bull)} long candidates, {len(bear)} breakdown candidates")

        # Quota per index, then dedupe. Insertion order of `picked` preserves
        # the INDEX_TAGS order for the groups themselves while each group is
        # internally ranked, so the final list reads SPX best-first, then the
        # QQQ names SPX did not already claim, then IWM.
        #
        # Long and short get SEPARATE quotas. Their scores are on the same
        # 0-100 scale but measure different things, and the whole point of the
        # bearish half is to be there when the long half is thin - letting one
        # ranking decide would hand every slot to whichever side the market
        # happens to favour, which is exactly the failure being fixed.
        buckets = {t: [] for t in INDEX_TAGS + [UNTAGGED]}
        bear_buckets = {t: [] for t in INDEX_TAGS + [UNTAGGED]}
        for o in bull:
            for tag in o['indexes']:
                if tag in buckets:
                    buckets[tag].append(o)
        for o in bear:
            for tag in o['indexes']:
                if tag in bear_buckets:
                    bear_buckets[tag].append(o)

        picked, selected = {}, []
        for tag in INDEX_TAGS + [UNTAGGED]:
            group = (sorted(buckets[tag], key=rank_key)[:per_index]
                     + sorted(bear_buckets[tag], key=rank_key)[:PER_INDEX_BEARISH])
            if buckets[tag] or bear_buckets[tag]:
                logger.info(f"  {tag}: {len(buckets[tag])} long / "
                            f"{len(bear_buckets[tag])} breakdown -> {len(group)} published")
            for o in group:
                if o['symbol'] not in picked:
                    picked[o['symbol']] = o
                    selected.append(o)

        # Longs first, then breakdowns, each by score. The dashboard sorts the
        # whole desk by score anyway, but this keeps the agent's own archive
        # in results/ grouped by direction.
        selected.sort(key=lambda o: (o.get('bias') == 'bearish', -o['score']))
        opportunities = selected
        logger.info(f"{len(opportunities)} opportunities published; generating reasoning for each")

        for i, opp in enumerate(opportunities, 1):
            opp['rank'] = i
            if opp.get('bias') == 'bearish':
                opp['reasoning'] = self.generate_bearish_reasoning(opp)
            else:
                opp['reasoning'] = self.generate_reasoning(
                    opp['symbol'], opp.pop('scores'), opp['price'],
                    opp['entry'], opp['stop_loss'], opp['take_profit']
                )

        return opportunities, market_regime, sector_state

    def format_results(self, opportunities: List[Dict], market_regime: str) -> str:
        """Format results for display/output."""

        output = f"""
╔══════════════════════════════════════════════════════════════╗
║           MONU - MNTM MOMENTUM TRADING SCAN                  ║
║                  Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}             ║
╚══════════════════════════════════════════════════════════════╝

Market Regime: {market_regime}  ({'gate open' if market_regime in MomentumAnalyzer.GATE_OPEN_REGIMES else 'GATE CLOSED - no momentum signals published'})
Stocks Scanned: {len(opportunities)}

"""

        if not opportunities:
            output += "No momentum opportunities found matching criteria.\n"
            return output

        for opp in opportunities:
            output += f"""
{"=" * 60}
#{opp['rank']} | {opp['symbol']} | Score: {opp['score']}/100
Price: ${opp['price']:.2f}

Dimensions:
  Trend:        {opp['breakdown']['trend_strength']}/25
  Volume:       {opp['breakdown']['volume']}/20
  RSI:          {opp['breakdown']['rsi']}/15
  MACD:         {opp['breakdown']['macd']}/15
  Relative:     {opp['breakdown']['relative']}/15
  Breakout:     {opp['breakdown']['breakout']}/10

Setup ({opp['setup_type']}, {opp['extension_pct']:+.1f}% vs MA20):
  Entry:        ${opp['entry']:.2f}
  Stop Loss:    ${opp['stop_loss']:.2f}
  Take Profit:  ${opp['take_profit']:.2f}
  Risk/Reward:  {opp['risk_reward_ratio']:.2f}:1

AI Reasoning:
{opp['reasoning']}

"""

        return output


# Fallback universe when no universe.txt is present. Deliberately small so a
# fresh clone runs in seconds; put real tickers in universe.txt to scan wide.
DEFAULT_UNIVERSE = [
    'NVDA', 'AAPL', 'MSFT', 'TSLA', 'AMZN', 'META', 'GOOGL', 'NFLX', 'AMD', 'AVGO',
    'ADBE', 'CRM', 'INTC', 'QCOM', 'CSCO', 'CRWD', 'NET', 'DDOG', 'MU', 'ORCL',
]


# Index buckets, in display order. A symbol can belong to more than one - the
# whole of QQQ except a handful of names is also in the S&P 500 - so these are
# tags, not a partition, and the per-bucket quota in scan_stocks() is applied
# to each tag independently.
# Re-exported from universe.py so this file reads naturally; that module is
# the single definition, shared with Opy and Goldy.
INDEX_TAGS = universe.INDEX_TAGS
UNTAGGED = universe.UNTAGGED

# Published rows per index. Three indexes at 8, minus the S&P 500 / Nasdaq-100
# overlap, lands around 20 - the same list length as before the universe grew,
# so the reasoning bill does not move. Raise it to see deeper into each group.
PER_INDEX = 8

# Breakdowns get their own smaller allocation on top of the long quota, so the
# bearish half is present whenever sectors are weak without taking the list
# over. See the quota comment in scan_stocks().
PER_INDEX_BEARISH = 3


def load_universe() -> Tuple[List[str], Dict[str, List[str]], Dict[str, str]]:
    """Symbols to scan, their index membership, and their sector ETF.

    Parsing lives in universe.py, shared with Opy and Goldy. Only the
    fallback is Monu's own: a missing or empty file means the built-in list,
    so a fresh clone runs in seconds without a universe.
    """
    try:
        symbols, membership, sector_map = universe.load()
    except FileNotFoundError:
        logger.info(f"No universe.txt - using built-in list of {len(DEFAULT_UNIVERSE)} symbols")
        return list(DEFAULT_UNIVERSE), {s: [UNTAGGED] for s in DEFAULT_UNIVERSE}, {}

    if not symbols:
        logger.warning("universe.txt is empty - falling back to built-in list")
        return list(DEFAULT_UNIVERSE), {s: [UNTAGGED] for s in DEFAULT_UNIVERSE}, {}

    logger.info(universe.describe(symbols, membership, sector_map))
    return symbols, membership, sector_map


def publish_to_dashboard(opportunities: List[Dict], market_regime: str,
                         membership: Dict[str, List[str]],
                         sector_state: Dict = None) -> None:
    """Write this agent's results where the dashboard can read them.

    Each agent owns exactly one file, docs/data/<AGENT_ID>.json, and registers
    itself in docs/data/agents.json. Registration is additive - publishing Monu
    never removes another agent's entry - so agents can run on independent
    schedules and in separate workflows without clobbering each other.
    """
    os.makedirs(DOCS_DATA_DIR, exist_ok=True)

    universe_counts = {t: sum(1 for tags in membership.values() if t in tags)
                       for t in INDEX_TAGS + [UNTAGGED]}
    universe_label = (
        f"{len(membership)} symbols  ("
        + " / ".join(f"{t} {n}" for t, n in universe_counts.items() if n) + ")"
    )
    sector_summary = sectors.summary_line(sector_state or {})
    n_short = sum(1 for o in opportunities if o.get("bias") == "bearish")
    n_long = len(opportunities) - n_short

    payload = {
        "agent": AGENT,
        "scan_date": datetime.now().isoformat(),
        "dimensions": DIMENSIONS,
        # Groups the dashboard offers as filters. Published rather than
        # hardcoded in the page, so changing the universe's index mix here is
        # the only edit needed.
        "groups": {
            "key": "indexes",
            "label": "Index",
            "values": [t for t in INDEX_TAGS + [UNTAGGED]
                       if any(t in o.get("indexes", []) for o in opportunities)],
        },
        "context": [
            {"label": "Market Regime", "value": market_regime},
            {"label": "Gate", "value": ("OPEN" if market_regime in MomentumAnalyzer.GATE_OPEN_REGIMES
                                        else "CLOSED - no signals in a downtrend")},
            {"label": "Universe", "value": universe_label},
            {"label": "Sector breadth", "value": sector_summary},
            {"label": "Passed Filter", "value": f"{n_long} long, {n_short} breakdown"},
            {"label": "Model", "value": CLAUDE_MODEL},
        ],
        "opportunities": opportunities,
    }
    if not opportunities:
        # The grid's default empty message says nothing cleared the filter.
        # When the gate is closed nothing was even scanned - say that instead.
        payload["empty_message"] = (
            f"Gate closed - market regime is {market_regime}. Momentum breakouts are not "
            f"traded outside a confirmed uptrend, so the scan was skipped."
            if market_regime not in MomentumAnalyzer.GATE_OPEN_REGIMES
            else "No setups cleared the filter. That is a normal scan result."
        )

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

    logger.info(f"Published to dashboard: {agent_file}")


def main():
    """Main entry point for the MONU agent."""

    logger.info("Starting MONU - MNTM Momentum Trading Agent...")

    # Initialize analyzer
    analyzer = MomentumAnalyzer()

    test_symbols, membership, sector_map = load_universe()

    # Run scan
    logger.info(f"Scanning {len(test_symbols)} stocks...")
    opportunities, market_regime, sector_state = analyzer.scan_stocks(
        test_symbols, membership, sector_map, per_index=PER_INDEX
    )

    # Format and print results
    results = analyzer.format_results(opportunities, market_regime)
    print(results)

    # Save results to JSON
    output_file = 'monu_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'scan_date': datetime.now().isoformat(),
            'market_regime': market_regime,
            'opportunities': opportunities
        }, f, indent=2, default=str)

    logger.info(f"Results saved to {output_file}")

    publish_to_dashboard(opportunities, market_regime, membership, sector_state)

    return opportunities


if __name__ == "__main__":
    main()
