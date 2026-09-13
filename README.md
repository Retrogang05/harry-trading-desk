# Harry Trading Desk

A terminal-style research desk for AI trading agents. Each agent scans the market
on its own strategy, scores what it finds, and publishes a ranked shortlist to a
shared dashboard. **You execute every trade manually** — nothing here places orders.

> **Research signals, not investment advice.** Scores measure how well a setup
> matches a strategy's criteria. They are not forecasts, and past matches do not
> imply future returns. Nothing in this repository is a recommendation to buy or
> sell any security.

## Agents

| Agent | Code | Strategy | Status |
|-------|------|----------|--------|
| **Monu** | `MNTM` | Momentum — buys strength on volume-confirmed breakouts | Live |
| **Opy** | `OPY` | Options — iron condors, credit spreads, LEAPS calls, RSI momentum context | Live |
| **Trey** | `TREY` | TQQQ trend gate — OUT / HALF / FULL, signalled on QQQ's 200-day SMA | Live |

The dashboard renders whatever score dimensions an agent declares, so adding an
agent needs no changes to the page. See **Adding your second and third agents**
below for the contract.

---

## 📊 Monu (MNTM) — the momentum agent

**MONU** (Momentum Trader Agent) is a cloud-native momentum stock scanner that:

- ✅ Scans 500+ stocks daily for momentum setups
- ✅ Scores each stock across 6 momentum dimensions
- ✅ Detects market regime (uptrend/downtrend)
- ✅ Uses Claude AI to explain WHY each stock shows momentum
- ✅ Provides specific entry/stop/target levels
- ✅ Runs automatically on GitHub Actions (free tier)
- ✅ Costs ~$5-10/month to operate

## 🎯 The 6 Momentum Dimensions

Each stock is scored 0-100 across six critical dimensions:

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| **Trend Strength** | 25% | Price relative to 20/50/200-day moving averages |
| **Volume** | 20% | Best volume spike over the last 5 days (same window as the breakout check) |
| **RSI** | 15% | Momentum strength (50-70 = optimal) |
| **MACD** | 15% | Trend direction + momentum acceleration |
| **Relative Performance** | 15% | Position relative to 52-week high |
| **Breakout Quality** | 10% | Recent breakout confirmation |

### Score Interpretation

- **80-100** 🟢 **STRONG BUY** - All signals aligned, high conviction
- **70-79** 🟡 **BUY** - Most signals aligned, good setup
- **60-69** 🔵 **WATCH** - Mixed signals, monitor closely
- **<60** 🔴 **SKIP** - Insufficient momentum

## 🏗️ Architecture

```
GitHub Actions (Daily 9 AM EST)
    ↓
Python Agent (agent.py)
    ↓
├─ Fetch market data (yfinance)
├─ Calculate 6 momentum dimensions
├─ Detect market regime
├─ Score & rank top 20 stocks
    ↓
Claude API (claude-haiku-4-5)
    ↓
Generate natural language reasoning
    ↓
Save results to JSON
    ↓
Commit to repository / Display on dashboard
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- GitHub account
- Anthropic API key ([get one here](https://console.anthropic.com))

### 2. Setup

```bash
# Clone this repository
git clone https://github.com/Retrogang05/harry-trading-desk.git
cd harry-trading-desk

# Install dependencies
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY="your-key-here"
```

### 3. Run Locally

```bash
python agent.py
```

You'll see output like:

```
╔══════════════════════════════════════════════════════════════╗
║           MONU - MNTM MOMENTUM TRADING SCAN                  ║
║                  Scan Date: 2026-08-10 09:15                 ║
╚══════════════════════════════════════════════════════════════╝

Market Regime: UPTREND
Stocks Scanned: 20

============================================================
#1 | NVDA | Score: 87/100
Price: $891.50

Dimensions:
  Trend:        25/25
  Volume:       18/20
  RSI:          14/15
  MACD:         15/15
  Relative:     11/15
  Breakout:      4/10

Setup:
  Entry:        $890.20
  Stop Loss:    $868.75
  Take Profit:  $922.35
  Risk/Reward:  1.50:1

AI Reasoning:
NVDA is showing exceptional momentum with volume confirmation on a
clean breakout above $890 resistance. MACD histogram is expanding
and RSI remains in the 60-65 range (strong but not overbought).
Price just broke above all key moving averages after consolidating.
Key risk: If volume dries up, momentum could reverse quickly.
```

### 4. Deploy to GitHub Actions

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial MONU setup"
   git remote add origin https://github.com/Retrogang05/harry-trading-desk.git
   git push -u origin main
   ```

2. **Add Secrets**
   - Go to your repo → Settings → Secrets and variables → Actions
   - Add secret: `ANTHROPIC_API_KEY` = your Anthropic API key

3. **Enable Actions**
   - Go to Actions tab
   - Click "I understand my workflows, go ahead and enable them"

4. **Test the workflow**
   - Actions → MONU - MNTM Momentum Scan → Run workflow

## 📁 Project Structure

```
harry-trading-desk/
├── agent.py                 # Monu — momentum screener (S&P 500)
├── opy/                     # Opy — options screener (vendored from Options Screener/)
│   └── agent.py
├── trey/                    # Trey — TQQQ trend gate (one row, three states)
│   └── agent.py
├── universe.txt             # 503 S&P 500 tickers, shared by Monu and Opy
├── scripts/fetch_sp500.py   # regenerates universe.txt from SPY holdings
├── requirements.txt
├── .github/workflows/
│   ├── momentum-scan.yml    # Monu  — 14:00 UTC weekdays (pre-open)
│   ├── opy-scan.yml         # Opy   — 21:00 UTC weekdays (post-close)
│   └── trey-scan.yml        # Trey  — 21:00 UTC weekdays (post-close)
├── docs/                    # ← the dashboard (GitHub Pages serves this)
│   ├── index.html
│   └── data/
│       ├── agents.json      #   manifest: which agents to render
│       ├── MNTM.json  OPY.json  TREY.json
├── results/  opy/results/  trey/results/    # dated archives per agent
```

---

## 📐 Trey (TREY) — the TQQQ trend gate

Not a screener. Trey watches one instrument and publishes one row whose score
is really a **state**. Every rule is evaluated on **QQQ** (unlevered) and the
state applies to **TQQQ** — the 3× leverage would otherwise turn ordinary noise
into false crossovers.

| State | Condition on QQQ | Position | Score |
|-------|------------------|----------|-------|
| **OUT** | close below 200-day SMA | cash | 0 |
| **HALF** | above 200-day; fast trend not confirmed | half | 60 |
| **FULL** | above 200-day; 10d > 20d **and** close > 50d, held 3 days | full | 100 |

Read at the close, acted on the next session. The scheduled run fires after
the US close so the published state is tomorrow's position.

### Where the rules came from — and what was thrown out

Derived from Masonson's *QQQ and TQQQ Profit Machine*, then **backtested on
real data before any of it was coded.** The script is in this session's
history, not the repo; the numbers below are from it.

**Kept:**

- **The 200-day gate.** On real TQQQ over the last ten years it matched
  buy-and-hold's return (2,889% vs 2,877%) while cutting max drawdown from
  −82% to −56%. One 2022 save (−47% vs −79%) paid for every whipsaw around it.
  Recovering from −79% needs +376%; from −47%, +89%. That asymmetry is the
  whole strategy.
- **Fast trend as a *size* modifier, not a gate.** Best Sharpe of every
  variant tested (0.88 vs 0.61 buy-and-hold). It also caught the COVID crash
  a 200-day system is too slow for (−42% vs −55%).
- **3-day confirmation on the fast signal.** The raw 10>20 crossover flipped
  size ~every two weeks, a quarter of those reversing next day. Three days of
  confirmation improved *both* Sharpe and return and cut size trades by a
  third. Five days was already too slow to catch a fast crash.

**Thrown out:**

- **Seasonal "flat Jul–Oct".** Worse in every single variant. Tech summers
  were often strong in this decade; the 52-year stat behind it is a different
  index and era.
- **"All conditions must align"** as the entry rule. Worst configuration in
  every test — more filters just meant later entries.
- **225 vs 200 days.** Regime-dependent: 225 wins on 25 years of QQQ (one
  dot-com-era event), 200 wins on TQQQ's whole life and on the last decade.
  `GATE_SMA` is one constant in `trey/agent.py`; neither is a robust edge.

### The one alert: RSI > 80

Not a rule — a flag to go look at the chart. Tested before adding: after
QQQ's RSI(14) crosses **70**, TQQQ is still higher 65% of the time a month
later, and 70 fires on 13% of all days — useless as an alert, wrong as a sell
signal. At **80** the forward return turns negative at every horizon and it
fires ~1.5×/year. As a hard "step out" rule it added +6 pts CAGR and lifted
Sharpe 0.88 → 1.05 over the last decade — but on 23 episodes with the edge
concentrated in ~5 big drops. Too thin to move the position on; it's
surfaced in red on the row, the header, the setup panel and the rail when it
fires, and left quiet otherwise. `RSI_ALERT` in `trey/agent.py`.

### What it costs

~2 gate changes a year (full round trips) and ~8 size changes (half-position
trades) even after confirmation. Both are published separately in the agent's
context panel — a blended number hid the size churn the first time.

### The honest framing

This is **drawdown insurance, not a return machine.** Over TQQQ's full life
(2010–2026) every timing rule lost to buy-and-hold on total return; only in the
last decade did the gate break even. Whether it's worth running comes down to
one question: could you hold TQQQ through −82%? If yes, buy-and-hold won. If
not — and almost nobody can — the gate is the price of staying in the game,
and −46% to −56% is still brutal.

---

## 🖥 The Dashboard

`docs/index.html` is the homepage — a single static file, no build step and no
dependencies. It reads `docs/data/agents.json`, loads each listed agent's feed,
and renders them.

### Preview locally

`fetch()` is blocked on `file://`, so the page must be served over HTTP:

```bash
python agent.py            # generate docs/data/
python -m http.server 8000 --directory docs
# open http://localhost:8000
```

### Deploy free on GitHub Pages

Settings → Pages → Source: **Deploy from a branch** → branch `main`, folder
`/docs`. Every scan commit republishes it. No build, no Vercel, no cost.

### Adding your second and third agents

The dashboard has no per-agent code — it renders whatever the manifest lists.
A new agent needs to do exactly two things:

1. Write `docs/data/<ID>.json` in the shape below.
2. Append its ID to the `agents` array in `docs/data/agents.json`.

`publish_to_dashboard()` in `agent.py` does both and is ~30 lines — copy it
into the new agent and change the `AGENT` and `DIMENSIONS` constants at the top.

```jsonc
{
  "agent": {
    "id": "VALU",                    // short code, must match the filename
    "name": "Vera",
    "strategy": "Value",
    "description": "One line on what this agent buys and why.",
    "accent": "#60a5fa"              // colour used for its rule and chip
  },
  "scan_date": "2026-08-10T09:00:00",

  // The score axes THIS agent uses. The dashboard reads these rather than
  // assuming Monu's — a value or sentiment agent scoring on entirely
  // different criteria renders correctly with no changes to index.html.
  "dimensions": [
    { "key": "fcf_yield", "label": "FCF Yield", "max": 30 }
  ],

  // Free-form key/value pairs shown in the agent header.
  "context": [ { "label": "Universe", "value": "503 symbols" } ],

  "opportunities": [
    {
      "rank": 1, "symbol": "ABC", "price": 88.10, "score": 77,
      "breakdown": { "fcf_yield": 26 },   // keys match dimensions[].key
      "entry": 88.10, "stop_loss": 79.29, "take_profit": 101.32,
      "risk_reward_ratio": 1.5,
      "setup_type": "PULLBACK", "extension_pct": -1.2,
      "reasoning": "Claude's read on this setup."
    }
  ]
}
```

Only `agent.id`, `agent.name`, `score`, and `symbol` are strictly required —
omit `dimensions`, `context`, or the trade levels and those blocks are simply
skipped for that agent.

**Score bands are shared across all agents** so the numbers stay comparable:
80+ Strong · 70–79 Buy · 60–69 Watch. If a future agent uses a different
scale, normalise it to 0–100 before publishing.

#### When one agent runs several strategies at once

Monu's contract above assumes every row scores on the same axes and has the
same entry/stop/target shape. That's true for a single-strategy agent, but
**Opy** screens four option strategies in one run — an iron condor's IV/HV
richness has nothing in common with a LEAPS trade's delta and leverage, and a
credit spread's numbers are credit/max-loss/ROC, not entry/stop/target.

Two fields, both optional and both set **per opportunity** rather than once
per agent, cover this:

```jsonc
{
  "rank": 1, "symbol": "IWM", "price": 301.56, "score": 100,

  // Overrides the agent-level "dimensions" for THIS row only. Omit it and
  // the row falls back to the agent's declared dimensions (Monu's case).
  "strategy": "Bull Put Spread",
  "dimensions": [
    { "key": "credit_width", "label": "Credit / Width", "max": 30 }
  ],
  "breakdown": { "credit_width": 24 },

  // Replaces the fixed Entry/Stop/Target/R:R block with whatever fields
  // actually describe this trade. Omit it (and "entry") entirely for a row
  // that isn't a specific structure - Opy's RSI rows have neither, and the
  // dashboard simply skips the Setup panel for those.
  "setup": {
    "label": "Bull Put 294/292 · 38 DTE",
    "fields": [
      { "label": "Credit", "value": "$0.47", "tone": "pos" },
      { "label": "Max Loss", "value": "$1.53", "tone": "neg" }
    ]
  },
  "reasoning": "Claude's read on this setup."
}
```

`tone` on a setup field is `"pos"` (green), `"neg"` (red), or omitted
(neutral). The grid also gains a **Strategy** column showing `o.strategy`,
falling back to the agent's single `agent.strategy` when a row doesn't set
its own — so Monu's rows need no changes and just show "Momentum" throughout.

One consequence worth knowing if you build a multi-strategy agent: the
dashboard's row-selection key is `symbol + agent.id + strategy`, not just
`symbol + agent.id` — without the strategy segment, two rows for the same
symbol under different strategies (a stock that's both an iron condor
candidate and a LEAPS candidate) would be indistinguishable and only one
would ever be selectable.

**Resilience:** one agent's broken or missing file is logged to the console and
skipped — the rest of the desk still renders. The filter bar only appears once
there are two or more agents.

## ⚙️ Configuration

### Customize Stocks to Scan

Edit `agent.py`, line ~380:

```python
test_symbols = [
    'NVDA', 'AAPL', 'MSFT', 'TSLA', 'AMZN',
    'META', 'GOOGL', 'NFLX', 'AMD', 'AVGO',
    # Add your stocks here...
]
```

**Or load from a file:**

```python
with open('stocks.txt', 'r') as f:
    test_symbols = [line.strip() for line in f]
```

### Adjust Scoring Weights

Edit the `score_momentum_dimensions` method to change weights:

```python
# Currently: Trend=25, Volume=20, RSI=15, MACD=15, Relative=15, Breakout=10
# Adjust to your preferences
```

### Change Scan Schedule

Edit `.github/workflows/momentum-scan.yml`:

```yaml
schedule:
  # Every weekday at 9 AM EST (14:00 UTC)
  - cron: '0 14 * * 1-5'

  # Or every day at 6 PM (post-market)
  # - cron: '0 23 * * *'
```

## 💰 Cost Breakdown

| Component | Monthly Cost |
|-----------|--------------|
| Claude API (Haiku) | $5-10 |
| GitHub Actions | $0 (free tier) |
| yfinance data | $0 (free) |
| **Total** | **$5-10** |

## 📊 Understanding the Output

### JSON Structure

```json
{
  "scan_date": "2026-08-10T09:15:00",
  "market_regime": "UPTREND",
  "opportunities": [
    {
      "rank": 1,
      "symbol": "NVDA",
      "price": 891.50,
      "score": 87,
      "breakdown": {
        "trend_strength": 25,
        "volume": 18,
        "rsi": 14,
        "macd": 15,
        "relative": 11,
        "breakout": 4
      },
      "entry": 890.20,
      "stop_loss": 868.75,
      "take_profit": 922.35,
      "risk_reward_ratio": 1.50,
      "reasoning": "NVDA is showing exceptional momentum...",
      "market_regime": "UPTREND"
    }
  ]
}
```

## 🔍 How It Works

### Step 1: Market Regime Detection

```python
if S&P 500 50-day MA > 200-day MA:
    regime = "UPTREND"      # Momentum works well
elif S&P 500 50-day MA < 200-day MA:
    regime = "DOWNTREND"    # Consider short strategies
else:
    regime = "CAUTION"      # Skip momentum trades
```

### Step 2: Score Each Stock

For each stock, calculate:
- **Trend Strength**: Is price above key MAs?
- **Volume**: Is volume 1.5-2x average?
- **RSI**: Is RSI in 50-70 range?
- **MACD**: Is MACD line above signal? Is histogram expanding?
- **Relative Performance**: How close to 52-week high?
- **Breakout Quality**: Recent clean breakout?

### Step 3: Rank & Filter

- Sort by total score (descending)
- Filter: Only show scores > 60
- Take top 20 candidates

### Step 4: Generate AI Reasoning

Claude analyzes the score breakdown and generates:
- Why momentum is present
- Which indicators are aligned
- Key risks to watch

### Step 5: Calculate Trade Setup

```python
extension = (price - MA20) / MA20

if extension > 4%:          # stock has already run - chasing it is a late entry
    setup_type = EXTENDED
    entry = current price   # breakout/continuation entry
else:
    setup_type = PULLBACK
    entry = MA20            # wait for the pullback

stop_loss   = entry - (1.5 × ATR)
take_profit = entry + (1.5 × ATR × 1.5)  # 1.5:1 risk/reward
```

The extension check matters: without it, a stock trading 9% above its MA20
gets a take-profit *below the current price* — a setup that says "sell lower
than it trades right now." `MAX_EXTENSION` in `agent.py` tunes the threshold.

## ⚠️ Important Disclaimers

- **This is a research tool, not financial advice**
- **You execute all trades manually** (no automated execution)
- **Backtest before real money** (4-8 weeks paper trading recommended)
- **Past performance doesn't guarantee future results**
- **Momentum trading has 45-50% win rate** (position sizing critical)
- **Always use stop-losses** (1-2% risk per trade maximum)

## 🎯 Next Steps

### Week 1-2: Validate
- [ ] Run locally, verify data fetching works
- [ ] Check indicator calculations are correct
- [ ] Test with 5-10 stocks first
- [ ] Verify Claude reasoning is helpful

### Week 3-4: Deploy
- [ ] Push to GitHub
- [ ] Set up GitHub Actions
- [ ] Add API key to secrets
- [ ] Test automated runs

### Week 5-12: Paper Trade
- [ ] Track every signal (did it move as predicted?)
- [ ] Record win rate, average return, holding period
- [ ] Refine scoring weights based on results
- [ ] Build dashboard to visualize results

### Week 13+: Live Trading
- [ ] When win rate > 55%: Start with small positions
- [ ] Maintain detailed trade log
- [ ] Continue refining based on real results

## 🛠️ Troubleshooting

### "No data found for SYMBOL"
- Stock might be delisted or ticker changed
- Check if market is open (data may be delayed)
- Try a different ticker

### "only N rows, need 200 for MA200 - skipping"
The agent requests 400 *calendar* days to get ~270 *trading* days, because
MA200 and the 52-week high need at least 200 bars. Recently-listed stocks
don't have that history and are skipped rather than scored against a NaN
moving average. Raise `CALENDAR_DAYS` if you need deeper history.

### Every stock scores 0 on a dimension
Check whether the indicator is genuinely flat or the data is malformed.
`yfinance` returns MultiIndex columns (`('Close','AAPL')`); the agent flattens
them on fetch because the `ta` library needs 1-D Series and otherwise raises
`ValueError: Data must be 1-dimensional`.

### "Rate limit exceeded"
- yfinance has rate limits
- Add delays between requests: `time.sleep(0.5)`
- Or use a paid data API (Alpha Vantage, Finnhub)

### "API key error"
- Verify `ANTHROPIC_API_KEY` is set correctly
- Check key hasn't expired
- Ensure secret is added to GitHub

## 📚 Learn More

- [Momentum Trading Research](../momentum-trading-deep-research.html)
- [Strategy Comparison](../strategy-comparison-vs-research.html)
- [Implementation Plan](../momentum-agent-implementation-plan.md)

## 🤝 Contributing

This is your personal trading research tool. Customize it freely:

- Add more indicators (Bollinger Bands, Stochastic, etc.)
- Implement multi-timeframe analysis
- Add sector rotation logic
- Build a React dashboard
- Add backtesting capabilities

## 📝 License

MIT License - Use freely for personal or commercial purposes.

---

**Built with:** Python, yfinance, ta, Claude API, GitHub Actions

**Remember:** Momentum trading works best in trending markets. Always manage your risk. 🚀
