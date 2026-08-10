#!/usr/bin/env python3
"""
MONU - MNTM (Momentum Trader Agent)
An AI-powered momentum trading scanner using Claude for intelligent reasoning.

Author: Your Name
Date: 2026
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

import anthropic
import numpy as np
import pandas as pd
import yfinance as yf
from anthropic import Anthropic
import ta  # Technical Analysis library

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
        self.client = Anthropic()
        self.model = CLAUDE_MODEL

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

    def fetch_stock_data(self, symbol: str, days: int = CALENDAR_DAYS) -> pd.DataFrame:
        """Fetch historical stock data for a single symbol (used for the index)."""
        logger.info(f"Fetching data for {symbol}...")
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False
            )

            if len(data) == 0:
                logger.warning(f"No data found for {symbol}")
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
        rsi_score = 0
        if 50 < current_rsi < 70:
            rsi_score = 15  # Perfect momentum range
        elif 40 < current_rsi <= 50:
            rsi_score = 8   # Weak but positive
        elif current_rsi > 70:
            rsi_score = 5   # Overbought, risky
        elif current_rsi < 30:
            rsi_score = 0   # Oversold
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
        """Detect market regime (UPTREND, DOWNTREND, CAUTION)."""
        if spy_data is None or len(spy_data) < 200:
            return "UNKNOWN"

        ma50 = ta.trend.sma_indicator(spy_data['Close'], window=50)
        ma200 = ta.trend.sma_indicator(spy_data['Close'], window=200)

        current_ma50 = ma50.iloc[-1]
        current_ma200 = ma200.iloc[-1]

        if current_ma50 > current_ma200:
            return "UPTREND"
        elif current_ma50 < current_ma200:
            return "DOWNTREND"
        else:
            return "CAUTION"

    def generate_reasoning(self, symbol: str, score_breakdown: Dict, price: float,
                          entry: float, stop_loss: float, take_profit: float) -> str:
        """Generate Claude-powered reasoning for the trade setup."""

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

        if message.stop_reason == "refusal":
            logger.warning(f"{symbol}: Claude declined to answer")
            return "[reasoning unavailable: request declined]"

        # content is a list of blocks, not a string - the first block is not
        # guaranteed to be text, so pick the text blocks out explicitly.
        text = "".join(b.text for b in message.content if b.type == "text").strip()
        return text or "[reasoning unavailable: empty response]"

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

    def scan_stocks(self, symbols: List[str], top_n: int = 20) -> List[Dict]:
        """Scan multiple stocks for momentum opportunities."""

        # Check market regime
        logger.info("Checking market regime...")
        spy_data = self.fetch_stock_data("SPY")
        market_regime = self.check_market_regime(spy_data)
        logger.info(f"Market Regime: {market_regime}")

        opportunities = []

        # One batched request for the whole universe, then score locally.
        frames = self.fetch_many(symbols)

        for symbol in symbols:
            data = frames.get(symbol)
            if data is None:
                continue

            # Calculate indicators
            indicators = self.calculate_indicators(data)
            if indicators is None:
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
                    'breakout': scores['breakout']
                },
                'entry': levels['entry'],
                'stop_loss': levels['stop_loss'],
                'take_profit': levels['take_profit'],
                'risk_reward_ratio': (levels['take_profit'] - levels['entry']) / (levels['entry'] - levels['stop_loss']),
                'setup_type': levels['setup_type'],
                'extension_pct': levels['extension_pct'],
                'scores': scores,           # kept for the reasoning pass, stripped below
                'market_regime': market_regime
            }

            opportunities.append(opportunity)

        # Rank first, truncate, and only then pay for reasoning. Writing it
        # inside the scan loop bills a Claude call for every candidate over 60
        # even though most never get published - on a 165-symbol universe that
        # was 90 calls to publish 20.
        opportunities = sorted(opportunities, key=lambda x: x['score'], reverse=True)[:top_n]
        logger.info(f"{len(opportunities)} opportunities ranked; generating reasoning for each")

        for i, opp in enumerate(opportunities, 1):
            opp['rank'] = i
            opp['reasoning'] = self.generate_reasoning(
                opp['symbol'], opp.pop('scores'), opp['price'],
                opp['entry'], opp['stop_loss'], opp['take_profit']
            )

        return opportunities

    def format_results(self, opportunities: List[Dict], market_regime: str) -> str:
        """Format results for display/output."""

        output = f"""
╔══════════════════════════════════════════════════════════════╗
║           MONU - MNTM MOMENTUM TRADING SCAN                  ║
║                  Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}             ║
╚══════════════════════════════════════════════════════════════╝

Market Regime: {opportunities[0]['market_regime'] if opportunities else 'UNKNOWN'}
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

UNIVERSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe.txt")


def load_universe() -> List[str]:
    """Symbols to scan: universe.txt if it exists, else the built-in list.

    universe.txt is one ticker per line; blank lines and #-comments ignored.
    Batched fetching means a few hundred symbols is a normal-sized scan.
    """
    if not os.path.exists(UNIVERSE_FILE):
        logger.info(f"No universe.txt - using built-in list of {len(DEFAULT_UNIVERSE)} symbols")
        return list(DEFAULT_UNIVERSE)

    symbols, seen = [], set()
    with open(UNIVERSE_FILE) as f:
        for line in f:
            sym = line.split("#", 1)[0].strip().upper()
            if sym and sym not in seen:
                seen.add(sym)
                symbols.append(sym)

    if not symbols:
        logger.warning("universe.txt is empty - falling back to built-in list")
        return list(DEFAULT_UNIVERSE)

    logger.info(f"Loaded {len(symbols)} symbols from universe.txt")
    return symbols


def publish_to_dashboard(opportunities: List[Dict], market_regime: str,
                         universe_size: int) -> None:
    """Write this agent's results where the dashboard can read them.

    Each agent owns exactly one file, docs/data/<AGENT_ID>.json, and registers
    itself in docs/data/agents.json. Registration is additive - publishing Monu
    never removes another agent's entry - so agents can run on independent
    schedules and in separate workflows without clobbering each other.
    """
    os.makedirs(DOCS_DATA_DIR, exist_ok=True)

    payload = {
        "agent": AGENT,
        "scan_date": datetime.now().isoformat(),
        "dimensions": DIMENSIONS,
        "context": [
            {"label": "Market Regime", "value": market_regime},
            {"label": "Universe", "value": f"{universe_size} symbols"},
            {"label": "Passed Filter", "value": str(len(opportunities))},
            {"label": "Model", "value": CLAUDE_MODEL},
        ],
        "opportunities": opportunities,
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

    logger.info(f"Published to dashboard: {agent_file}")


def main():
    """Main entry point for the MONU agent."""

    logger.info("Starting MONU - MNTM Momentum Trading Agent...")

    # Initialize analyzer
    analyzer = MomentumAnalyzer()

    test_symbols = load_universe()

    # Run scan
    logger.info(f"Scanning {len(test_symbols)} stocks...")
    opportunities = analyzer.scan_stocks(test_symbols, top_n=20)

    # Format and print results
    results = analyzer.format_results(opportunities,
                                     opportunities[0]['market_regime'] if opportunities else 'UNKNOWN')
    print(results)

    market_regime = opportunities[0]['market_regime'] if opportunities else 'UNKNOWN'

    # Save results to JSON
    output_file = 'monu_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'scan_date': datetime.now().isoformat(),
            'market_regime': market_regime,
            'opportunities': opportunities
        }, f, indent=2, default=str)

    logger.info(f"Results saved to {output_file}")

    publish_to_dashboard(opportunities, market_regime, len(test_symbols))

    return opportunities


if __name__ == "__main__":
    main()
