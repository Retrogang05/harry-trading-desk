# Harry Trading Desk

Seven scanning agents run on GitHub Actions after each US session and publish
ranked lists to one dashboard (GitHub Pages). Nothing places orders: every
trade is entered by hand.

> Research signals, not investment advice. A score measures how closely a
> setup matches an agent's rules. It is not a forecast.

## Agents

| Agent | Code | What it publishes |
|-------|------|-------------------|
| **Monu** | `MNTM` | Momentum breakouts on volume, plus breakdowns in weak sectors |
| **Opy** | `OPY` | Iron condors, credit spreads (bear calls get a floor), LEAPS calls, RSI context |
| **Trey** | `TREY` | TQQQ position state (OUT / HALF / FULL) from QQQ's 200-day SMA |
| **Goldy** | `GOLD` | 20/50 and 50/200 crosses in both directions, and crosses about to happen |
| **Pocket** | `PCKT` | The Finviz relative-volume screen, with major levels and pockets mapped |
| **Flag** | `FLAG` | Well-formed bull flags: forming, and breakouts on volume |
| **Leaders** | `LDRS` | Top 25 YTD in the S&P 500, Nasdaq-100 and Russell 2000 (the Stocktwits lists) |

The dashboard draws whatever score dimensions an agent declares, so a new
agent needs no page changes. See [Adding an agent](#adding-an-agent).

---

## Monu (MNTM): momentum

Scans the full universe, scores each stock on six dimensions, and publishes
the best 8 per index. The long half runs only when SPY's 50-day is above its
200-day; otherwise only the bearish half (below) can publish.

| Dimension | Points | Measures |
|-----------|-------:|----------|
| Trend | 25 | Price against the 20/50/200-day averages |
| Volume | 20 | Largest volume spike in the last 5 days |
| RSI | 15 | 50–70 scores highest |
| MACD | 15 | Line over signal, histogram expanding |
| Relative | 15 | Distance from the 52-week high |
| Breakout | 10 | A recent breakout |

Rows under 60 are dropped. Claude (Haiku) writes a short note on each
published row.

**Trade levels.** If price is more than 4% above the 20-day average
(`MAX_EXTENSION`), entry is the current price (`EXTENDED`); otherwise entry is
the 20-day average (`PULLBACK`). Stop is 1.5 ATR below entry and the target
2.25 ATR above, a 1.5:1 reward to risk. Without the extension check, a stock
9% above its average got a target below the current price.

### Bearish ideas and sector breadth

On 2026-10-05, ten of the eleven sector SPDRs were below their 20-day average
while SPY's 50-day was still over its 200-day, and Monu's gate read UPTREND.
One cap-weighted index is not breadth, so both Monu and Opy now gate bearish
ideas on sectors.

`sectors.py` reports which sector ETFs are under their 20-day SMA. Both agents
use it, so "weak" means the same on both lists.

- **Monu's breakdown half** scores a stock only when its own sector ETF is
  weak. The dimensions mirror the long side rather than flipping its sign: the
  RSI band of interest sits just under 50, and distance from the 52-week high
  becomes distance from the 52-week low. Levels are quoted for a short (stop
  above entry) and the row is labelled bearish. Quotas are 8 long + 3
  breakdown per index. The breakdown half needs only a weak sector, so a
  downtrend publishes breakdowns instead of an empty list. Its floor is 70,
  not 60, because it hasn't been validated yet.
- **Opy** used to rank bear calls against bull puts and take the top 8. On a
  10-of-11-weak day that still gave 6 bull puts and 2 bear calls. Bear calls
  now get at least 4 slots, with weak-sector names first.

Sector membership comes from the eleven sector SPDRs' holdings (SSGA serves
them at a stable URL). Russell names use BlackRock's sector column. The tag is
the ETF symbol, because that is what the agents price:

```
AAPL   # SPX,QQQ,XLK
JPM    # SPX,XLF
```

---

## The scan universe

`universe.txt` is built from three fund holdings exports:

| Tag | Fund | Source | Names |
|-----|------|--------|------:|
| `SPX` | SPDR S&P 500 ETF (SPY) | State Street | 501 |
| `QQQ` | Invesco QQQ Trust | Invesco | 100 |
| `IWM` | iShares Russell 2000 ETF | BlackRock | 1,000 |

That is 1,516 unique tickers. Tags overlap (85 names are in both the S&P 500
and the Nasdaq-100), so they are labels, not a partition. They live in a
trailing comment, so anything that wants a plain ticker list can read the file
as is.

`universe.py` is the one parser. It returns index membership and sector
separately: mixing them once let an early filter switch off Monu's whole
bearish half without any error.

**Who scans what**

- **Monu**: everything, best 8 per index. Quotas stop the Russell 2000, two
  thirds of the universe, from filling the list on numbers alone.
- **Opy**: `SPX` + `QQQ` only. Most Russell names have no options, or spreads
  wide enough to eat a condor's credit.
- **Pocket, Goldy, Flag, Leaders**: everything.
- **Trey**: reads QQQ, trades TQQQ.

**Liquidity.** The IWM export has ~1,950 rows, including cash, futures and
private positions, and ~960 names too thin to trade. The build keeps price
≥ $5 and median 3-month dollar volume ≥ $10M. Monu re-checks this at scan
time, since the file goes stale between rebuilds.

**ADRs.** QQQ rows count if they are common stock or an American Depositary
Receipt. Before October 2026 ADRs were excluded, which silently dropped ASML,
ARM and PDD.

**Rebuilding** (quarterly is enough):

```bash
python scripts/fetch_sp500.py      # fresh spy.xlsx
# download the QQQ and IWM holdings by hand into data/holdings/
python scripts/build_universe.py   # ~3 min, mostly pricing
```

Only SSGA serves holdings without a browser session, so QQQ and IWM are
manual. The build also writes `docs/data/names.json`, the company names shown
in the detail panel. `--names-only` refreshes names without re-pricing.

---

## Goldy (GOLD): moving-average crosses

Finds crosses in either direction within the last 10 sessions:

| Pair | Bullish | Bearish |
|---|---|---|
| 20/50 | 20/50 Cross | 20/50 Breakdown |
| 50/200 | Golden Cross | Death Cross |

**Approaching crosses.** By the time two averages touch, most of the move that
brought them together has happened. Goldy projects the gap forward (its
average daily change over 5 sessions) and publishes anything due to cross
within 10 sessions as `Nearing …`. The gap must be 0.25–2.0 ATR. Wider, and the
projection reaches too far. Narrower, and the averages are tangled together in
chop: the first run without the floor published 21 such rows, all scoring
93–100.

Confirmed and approaching crosses get separate quotas (6 + 3 per index), and
approach scores top out at 75, so a cross that might happen never outranks
one that did. Entry levels are published only for confirmed bullish crosses.

### Before trading the list

Backtest before building: 13,173 crosses across 580 names, 2019–2026, 60-day
return against SPY.

| | n | 60-day alpha | vs random |
|---|---|---|---|
| Random days, same stocks | 3,480 | +1.91% | — |
| 20/50 cross | 10,276 | +1.62% | t = −0.47 |
| 50/200 golden cross | 2,897 | +2.28% | t = +0.44 |

Neither cross beats a random day in the same stocks.

- Wide separation between the averages looks great on its own (+7.11% widest
  quartile vs −0.02% narrowest) but is a volatility proxy. Hold volatility
  constant and it disappears.
- Volatility orders outcomes: calmest ADR quartile −1.16%, wildest +9.03%. So
  ADR is shown on every row, as a risk measure.
- Requiring price above the 200-day (+1.23% vs +2.23%) or a rising 200-day
  (+0.88% vs +2.22%) made results worse. Goldy has no regime gate.

The score describes the cross (fresh, decisive, clean, confirmed) and does not
predict returns. Use Goldy to find charts worth a look.

---

## Pocket (PCKT): volume screen and level map

Automates a manual routine: run a Finviz screen, then look at each chart for a
**pocket**, a gap between major levels with nothing in it.

| Filter | Setting |
|---|---|
| Average volume | > 500K (63-session mean, excluding today) |
| Current volume | > 500K |
| Relative volume | > 1.5 |
| Market cap | Not applied (not in our data) |

**The map.** Major pivots are the extreme of ±10 sessions, except the latest
swing, which needs only 3 sessions after it (otherwise the most recent level
stays hidden for two weeks). Pivots within 0.75 ATR are merged. Each row shows
the nearest floor and ceiling, pocket width, room each way, volume inside the
zone, and today's state: broke up, broke down, testing (within 0.25 ATR of a
level) or inside.

Dropped: levels closer than 1.2 ATR (congestion), stale symbols, and pinned
ones (daily range under 0.5%, usually a takeover target).

**Trades.** Breakouts: entry at the close, stop 1 ATR under the broken level,
exit after 20 sessions or at the stop. Breakdowns get a short only within 10%
of the 52-week high. Other breakdowns, and inside or testing rows, have no
trade.

The 10-session pivot setting matched hand-drawn pockets on AAOI, GOOG, CRWV,
IREN and SE, and disagreed on DOCU, NOW and IBM, where a minor level splits
the zone.

### Before trading the list

2019–2026, up to 400 names, no lookahead:

- A clear path barely helps. A target 1–3 ATR away with no pivot in the way
  was reached only 0–3 points more often than one with a pivot in the way.
- Thin-volume pockets were crossed less often, not more. Zone volume is shown
  but not scored.
- On high relative volume days, breaking up into a pocket returned +1.12% over
  20 sessions vs +2.04% for any high-volume day. The pocket subtracted.
- Relative volume above 1.5 on its own: +2.04% vs +1.29% otherwise.

**Confluence and exit tests (Oct 2026)**, 32,799 point-in-time rows:

| | Trades | E[R] | 20-day |
|---|---:|---:|---:|
| Breakouts, original far-side target | 4,207 | +0.007 | +0.06% |
| **Breakouts, 1 ATR stop, 20-session exit (current)** | 4,207 | **+0.210** | +1.50% |
| Breakdowns, original | 4,157 | −0.124 | −0.66% |
| **Breakdowns within 10% of 52w high (current)** | 805 | **+0.031** | +0.03% |
| *Every stock passing the screen, held 20 days* | 48,030 | — | *+3.23%* |

- Of 36 confluence filters, each required to win in both 2020–22 and 2023–26,
  only one held: breakdowns near the 52-week high. Shorting already-weak
  stocks lost money, and stacking filters made it worse.
- The exit mattered more than any filter.
- Every layer of pocket logic returned less than the raw screen. Use the map
  for stops and context. Long-side figures are flattered by survivorship.

---

## Leaders (LDRS): top 25 per index

The Stocktwits "Top 25 Momentum" lists, rebuilt from closes the other agents
already fetch: S&P 500, Nasdaq-100 and Russell 2000 members ranked by YTD
return. Each row shows rank, last week's rank (▲/▼, New, Fell off), YTD,
week-to-date and sector. A stock on two lists appears twice.

Closes are unadjusted, as Stocktwits' are, so symbols with a split-shaped
one-day jump are dropped. Against Stocktwits for the week of 23 Oct 2026: S&P
figures matched exactly with 21 of 25 names in common, Nasdaq-100 23 of 25,
Russell 2000 18 of 25 (ours excludes sub-$5, thin, delisted and very new
stocks).

No Claude calls and no trade levels. It is a watchlist.

### Does the rank predict anything? (Oct 2026)

Every Friday 2019–2026, the top 25 against the equal-weight average of all the
list's members. t-stats use non-overlapping windows.

| List | 20 days | 50 days | 100 days | Verdict |
|------|--------:|--------:|---------:|---------|
| S&P 500 | +1.72% (t 3.5) | +4.16% (t 3.0) | +9.62% (t 2.3) | Real, but all of it from 2023–26 |
| Nasdaq-100 | +0.71% | +1.51% | +3.69% | Not significant |
| Russell 2000 | — | — | — | Reverses: top minus bottom 25 was −1.9% / −5.8% / −9.3% |

- About half the S&P excess in 2023–26 came from one stock per window.
- Deciles are U-shaped: the worst YTD names also beat the middle.
- Survivorship flatters the long side.

On the S&P list, leaders have kept leading recently. On the Russell list,
chasing the top 25 has hurt.

---

## Flag (FLAG): bull flags

The rules:

- **Uptrend**: close over the 21 EMA, 9 EMA over the 21.
- **Pole**: at least 4 ATR and 10%, into a high not exceeded since.
- **Flag**: 5–20 sessions of falling highs, pulling back at most half the pole,
  on volume below the pole's and falling.
- **Trigger**: the flag's upper trendline. A breakout is a close above it on at
  least 1.5× the 20-day average volume.

Only well-formed flags are published: flag volume at most 70% of the pole's,
pullback at most 40%, 5–8 sessions long.

- **Forming** rows show the trigger, the stop (flag low) and the distance to
  the trigger. No trade.
- **Breakout** rows: entry at the close, stop at the flag low, hold 20
  sessions or until stopped.

### Before trading the list

Point-in-time, 2019–2026. Built on 385 stocks, then re-run unchanged on the
1,098 others. 20-session hold or stop, in R:

| | 2019–22 | 2023–26 |
|---|---:|---:|
| Sample, well-formed breakouts | −0.02 | +0.70 |
| Unseen stocks, well-formed breakouts (n=499) | +0.00 | +0.54 |
| Unseen stocks, loose flags | −0.00 | +0.21 |
| Plain 20-day-high breakout on volume | +0.06 | +0.12 |

- A forming flag is not a signal. It did no better than any uptrend day, and
  more flags broke their low within 10 sessions (54%) than broke out (24%).
- Without the 1.5× breakout volume rule, the well-formed filter failed on the
  unseen stocks.
- 2019–22 broke even. Treat the edge as regime-dependent.
- The 20-session hold beat 2R and 3R targets. Exiting below the 9 EMA was the
  worst exit tested.
- No costs included. The average breakout stop was ~8% of price.
- Breakouts are rare, about two a week, so most days the tab is a short
  forming list.

---

## Trey (TREY): the TQQQ trend gate

One instrument, one row. The rules run on QQQ and the position applies to
TQQQ, because 3× leverage turns ordinary noise into false crossovers.

| State | Condition on QQQ | Position | Score |
|-------|------------------|----------|-------|
| OUT | close below 200-day SMA | cash | 0 |
| HALF | above 200-day, fast trend not confirmed | half | 60 |
| FULL | above 200-day; 10d > 20d and close > 50d, held 3 days | full | 100 |

Read at the close, acted on the next session.

### Where the rules came from

From Masonson's *QQQ and TQQQ Profit Machine*, backtested before coding.

**Kept:**

- **The 200-day gate.** Over ten years of TQQQ it matched buy-and-hold
  (2,889% vs 2,877%) and cut the worst drawdown from −82% to −56%. The 2022
  save alone (−47% vs −79%) paid for every whipsaw.
- **Fast trend as a size change, not a gate.** Best Sharpe of all variants
  (0.88 vs 0.61 buy-and-hold), and it caught the COVID crash that the 200-day
  was too slow for (−42% vs −55%).
- **3-day confirmation.** The raw 10/20 cross flipped size about every two
  weeks. Three days improved both Sharpe and return and cut size trades by a
  third. Five days was too slow for a fast crash.

**Dropped:**

- Seasonal "flat Jul–Oct": worse in every variant.
- Requiring all conditions to align: worst in every test.
- 225 vs 200 days: depends on the period. `GATE_SMA` in `trey/agent.py`.

### RSI > 80 alert

A prompt to look at the chart, not a rule. At RSI 70, TQQQ was still higher a
month later 65% of the time, and 70 fires on 13% of days. At 80, forward
returns turn negative and it fires about 1.5 times a year. As a hard exit it
added 6 points of CAGR, but on only 23 episodes, mostly ~5 big drops: too few
to trade on. It shows in red when it fires. `RSI_ALERT` in `trey/agent.py`.

### Cost and limits

About 2 gate changes and 8 size changes a year. Over TQQQ's full life
(2010–2026) every timing rule lost to buy-and-hold on total return. The gate
is drawdown insurance: worth it only if you couldn't sit through −82%, and
−46% to −56% is still severe.

---

## Dashboard

`docs/index.html` is one static file with no build step. It reads
`docs/data/agents.json` and loads each agent's feed.

- **Looks**: Day (light grey glass) and Night (dark glass). Auto, the
  default, follows the device setting. The switch in the top bar overrides it
  and is remembered per browser.
- **Tabs** sit in a dock at the bottom of the screen.
- **Detail panel**: the company name, score breakdown, trade levels and a
  3-month chart of daily closes with the 200-day SMA (amber; marked "off
  chart" when too far away to draw) and entry, stop and target as dashed
  lines. The chart uses the same bars the agent scanned (`history.py`); Opy
  keeps no bars, so it fetches its published symbols separately.
- **Results tab**: what happened to past signals (below).
- One agent's broken feed is logged and skipped; the rest still render.

### Results

`track/agent.py` replays every signal in the dated archives against the
prices that followed and writes `docs/data/results.json`. The archives were
written before the outcomes existed, so this is a forward test. It covers
Monu, Pocket, Flag, Goldy and Opy, and is rebuilt from scratch each night.

- **Entry**: the next session's open. A Monu `PULLBACK` is a limit at its
  entry, working for 5 sessions. A trade that opens through its stop is not
  taken.
- **Exit**: the published stop or target (stop wins a same-day tie; a gap
  through either exits at the open), or the time limit: Pocket and Flag 20
  sessions, Goldy 63. Monu has no time limit.
- **R**: the result as a multiple of the risk from fill to stop.
- **Repeats**: a signal for a stock the agent already holds is ignored.
- **Opy**: credit spreads are settled on the expiry close, at the published
  mid credit. Condors publish no strikes, so only whether the stock finished
  inside its expected move is counted. LEAPS and RSI rows are not tracked.

Each agent's headline figure includes open trades at the last close. Closed
trades alone lean to losers early on, because stops close in days and targets
or time exits take weeks.

**Preview locally** (`fetch()` doesn't work on `file://`):

```bash
python -m http.server 8000 --directory docs
```

**Hosting**: Settings → Pages → deploy from branch `main`, folder `/docs`.
Each scan commit republishes it.

### Adding an agent

1. Write `docs/data/<ID>.json` in the shape below.
2. Add its ID to `agents` in `docs/data/agents.json`.

`publish_to_dashboard()` in `agent.py` does both. Copy it and change the
`AGENT` and `DIMENSIONS` constants.

```jsonc
{
  "agent": {
    "id": "VALU",                    // must match the filename
    "name": "Vera",
    "strategy": "Value",
    "description": "One line on what this agent looks for.",
    "accent": "#60a5fa"
  },
  "scan_date": "2026-08-10T09:00:00",
  "dimensions": [ { "key": "fcf_yield", "label": "FCF Yield", "max": 30 } ],
  "context": [ { "label": "Universe", "value": "1516 symbols" } ],

  // Optional filter chips and column. The field may be a string or an array
  // (a stock in both the S&P 500 and Nasdaq-100 matches either chip).
  "groups": { "key": "indexes", "label": "Index", "values": ["SPX", "QQQ", "IWM"] },

  "opportunities": [
    {
      "rank": 1, "symbol": "ABC", "price": 88.10, "score": 77,
      "breakdown": { "fcf_yield": 26 },
      "indexes": ["SPX", "QQQ"],
      "entry": 88.10, "stop_loss": 79.29, "take_profit": 101.32,
      "risk_reward_ratio": 1.5,
      "reasoning": "Claude's note on this setup."
    }
  ]
}
```

Only `agent.id`, `agent.name`, `symbol` and `score` are required. Missing
blocks are skipped.

Score bands are shared so numbers compare across agents: 80+ Strong, 70–79
Buy, 60–69 Watch. Normalise other scales to 0–100.

**Rows with different strategies.** Opy mixes iron condors, spreads and LEAPS
in one feed, which share neither score axes nor entry/stop/target. Any row can
override them:

```jsonc
{
  "rank": 1, "symbol": "IWM", "price": 301.56, "score": 100,
  "strategy": "Bull Put Spread",
  "dimensions": [ { "key": "credit_width", "label": "Credit / Width", "max": 30 } ],
  "breakdown": { "credit_width": 24 },
  "setup": {
    "label": "Bull Put 294/292 · 38 DTE",
    "fields": [
      { "label": "Credit", "value": "$0.47", "tone": "pos" },
      { "label": "Max Loss", "value": "$1.53", "tone": "neg" }
    ]
  }
}
```

`tone` is `pos` (green), `neg` (red) or omitted. Rows without `strategy` use
the agent's. The selection key is symbol + agent + strategy, so one stock can
appear under two strategies. Other optional row fields: `history` (chart
data, from `history.py`) and `chart_levels` (`[{label, value, tone}]`, extra
lines on the chart).

---

## Running it

**Requirements**: Python 3.11+, and an Anthropic API key in the
`ANTHROPIC_API_KEY` GitHub secret. Without a key the agents still run; rows
just have no Claude note.

```bash
pip install -r requirements.txt
python agent.py                    # Monu; the others are <name>/agent.py
python tests_edge_cases.py
```

### Schedule

Every agent runs at 9pm New York, Monday–Friday, after the session's bar is
final. That is around midday in Sydney, but GitHub often starts scheduled runs
hours late.

| Workflow | Time (New York) |
|---|---|
| flag-scan.yml | 9:02pm |
| momentum-scan.yml | 9:07pm |
| opy-scan.yml | 9:17pm |
| trey-scan.yml | 9:27pm |
| goldy-scan.yml | 9:37pm |
| pocket-scan.yml | 9:47pm |
| leaders-scan.yml | 9:57pm |
| results.yml | 10:12pm (replays signals, no Claude calls) |

GitHub cron is UTC with no daylight saving, so each workflow has two crons
(01:xx and 02:xx UTC) and a `gate` job that lets through the one matching 9pm
New York that day. It checks which cron fired rather than the clock, because
runs start late.

Each run then waits up to 90 minutes for the day's SPY bar. In Sep–Oct 2026,
6 of 25 Trey runs had used the previous day's bar. If the bar never arrives
(holiday or Yahoo delay) the run publishes nothing. Archives are named after
the session scanned.

To change the time, edit both crons and the `want` hours in each workflow's
gate job.

### Cost

Claude Haiku for the notes, roughly $5–10 a month. GitHub Actions and yfinance
are free.

### Troubleshooting

- **"only N rows, need 200 for MA200"**: the stock is too new for a 200-day
  average and is skipped. `CALENDAR_DAYS` sets how much history is fetched.
- **A dimension scores 0 everywhere**: check for MultiIndex columns from
  yfinance. The `ta` library needs 1-D series.
- **"No data found"**: usually a delisted or renamed ticker.

## Project structure

```
harry-trading-desk/
├── agent.py            # Monu
├── opy/ trey/ goldy/ pocket/ leaders/ flag/   # one agent.py each
├── universe.py         # universe.txt parser
├── sectors.py          # sector-ETF breadth (Monu, Opy)
├── history.py          # 3-month closes + 200 SMA for the chart
├── track/agent.py      # Results: replays archived signals
├── universe.txt        # 1,516 tickers with index and sector tags
├── data/holdings/      # fund holdings exports universe.txt is built from
├── scripts/            # build_universe.py, fetch_sp500.py
├── tests_edge_cases.py
├── .github/workflows/  # one <agent>-scan.yml per agent
├── docs/               # the dashboard (GitHub Pages)
│   ├── index.html
│   └── data/           # agents.json, names.json, results.json, one <ID>.json per agent
└── results/ <agent>/results/   # dated archives
```

MIT License.
