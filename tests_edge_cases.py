"""Edge-case tests for Harry Trading Desk.

Written to find bugs, not to chase coverage: every case here is one where an
agent could plausibly publish something wrong rather than fail loudly. Run it
after changing any scanner.

    python tests_edge_cases.py

Needs the project dependencies (pandas, numpy, yfinance). Reads the live
universe.txt and the published docs/data/*.json, so it also acts as a contract
check on whatever the agents last wrote.
"""
import importlib.util, json, os, sys, tempfile
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

FAILS, PASSES = [], []

def check(name, cond, detail=""):
    (PASSES if cond else FAILS).append(name)
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"  — {detail}" if detail and not cond else ""))

def section(t): print(f"\n=== {t} ===")

def load_mod(name, rel, extra=None):
    if extra: sys.path.insert(0, os.path.join(ROOT, extra))
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def bars(prices, vol=1e6, start="2023-01-01"):
    idx = pd.date_range(start, periods=len(prices), freq="B")
    return pd.DataFrame({"Open": prices, "High": [p*1.02 for p in prices],
                         "Low": [p*0.98 for p in prices], "Close": prices,
                         "Volume": [vol]*len(prices)}, index=idx)

import universe, sectors
import agent as monu
goldy = load_mod("goldyagent", "goldy/agent.py", "goldy")

# ─────────────────────────────────────────────────────────────────────
section("universe.py")

u_syms, u_mem, u_sec = universe.load()
check("no symbol has a sector tag in membership",
      not any(t in sectors.SECTOR_ETFS for v in u_mem.values() for t in v))
check("every membership entry is non-empty", all(v for v in u_mem.values()))
check("sector_map keys are a subset of symbols", set(u_sec) <= set(u_syms))
check("no duplicate symbols", len(u_syms) == len(set(u_syms)))
check("keep= filters correctly",
      all(any(t in ("SPX","QQQ") for t in universe.load(keep=("SPX","QQQ"))[1][s])
          for s in universe.load(keep=("SPX","QQQ"))[0]))

with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write("# comment only\n\nAAPL # SPX,XLK\nAAPL # IWM\nbrk-b  # SPX\nNOTAG\n")
    tmp = f.name
s, m, sc = universe.load(path=tmp)
check("duplicate ticker keeps first occurrence", m.get("AAPL") == ["SPX"], str(m.get("AAPL")))
check("lowercase ticker upper-cased", "BRK-B" in m)
check("tagless line gets OTHER", m.get("NOTAG") == [universe.UNTAGGED], str(m.get("NOTAG")))
check("comment-only line ignored", len(s) == 3, f"{s}")
check("keep= does not drop tagless rows", "NOTAG" in universe.load(keep=("SPX",), path=tmp)[0])
os.unlink(tmp)

empty = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False); empty.close()
check("empty file returns empty, no crash", universe.load(path=empty.name)[0] == [])
os.unlink(empty.name)

# ─────────────────────────────────────────────────────────────────────
section("sectors.py")

check("sector_of finds the sector tag", sectors.sector_of(["SPX","QQQ","XLK"]) == "XLK")
check("sector_of returns None when absent", sectors.sector_of(["SPX","QQQ"]) is None)
check("sector_of tolerates None", sectors.sector_of(None) is None)
check("weak_tags on empty state", sectors.weak_tags({}) == [])
check("summary_line on empty state", sectors.summary_line({}) == "unavailable")

# The single-ticker collapse path: if yfinance returns a flat frame, does
# read() give every sector the SAME numbers?
import yfinance as _yf
_real = _yf.download
flat = bars([100.0 + i*0.1 for i in range(60)])
flat.columns = ["Open","High","Low","Close","Volume"]      # single-level
sectors.yf.download = lambda *a, **k: flat
st = sectors.read()
distinct = {(v["price"], v["sma"]) for v in st.values()}
check("flat frame does not clone one ETF across all 11",
      len(st) <= 1 or len(distinct) > 1,
      f"{len(st)} sectors returned, {len(distinct)} distinct price/sma pairs")
sectors.yf.download = _real

sectors.yf.download = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
check("fetch failure returns {} not a crash", sectors.read() == {})
sectors.yf.download = _real

# ─────────────────────────────────────────────────────────────────────
section("Monu — bearish scoring")

a = monu.MomentumAnalyzer.__new__(monu.MomentumAnalyzer)
falling = bars([200 - i*0.4 for i in range(300)])
ind = monu.MomentumAnalyzer.calculate_indicators(a, falling)
b = monu.MomentumAnalyzer.score_breakdown_dimensions(a, falling, ind)
check("breakdown score on a falling series is not None", b is not None)
if b:
    parts = ("trend_strength","volume","rsi","macd","relative","breakout")
    check("breakdown total == sum of its six parts", b["total"] == sum(b[k] for k in parts),
          f"total={b['total']} sum={sum(b[k] for k in parts)}")
    check("breakdown total within 0-100", 0 <= b["total"] <= 100, str(b["total"]))
    maxes = dict(zip(parts, (25,20,15,15,15,10)))
    check("no axis exceeds its max", all(b[k] <= maxes[k] for k in parts),
          str({k: b[k] for k in parts}))

flatp = bars([100.0]*300)
bf = monu.MomentumAnalyzer.score_breakdown_dimensions(a, flatp, monu.MomentumAnalyzer.calculate_indicators(a, flatp))
check("flat series does not divide by zero", bf is None or 0 <= bf["total"] <= 100)

# Short levels must put the stop ABOVE entry and target below.
if b:
    row = monu.MomentumAnalyzer._bearish_row(
        a, "TEST", falling, ind, b, {"TEST": ["SPX"]}, "XLV",
        {"XLV": {"name": "Health Care", "pct": -1.1}}, "UPTREND")
    check("short stop is above entry", row["stop_loss"] > row["entry"],
          f"entry={row['entry']:.2f} stop={row['stop_loss']:.2f}")
    check("short target is below entry", row["take_profit"] < row["entry"])
    check("short risk capped at 15%", (row["stop_loss"]/row["entry"] - 1) <= 0.1501,
          f"{(row['stop_loss']/row['entry']-1)*100:.1f}%")
    check("bearish row carries bias/label/tone",
          row["bias"]=="bearish" and row["signal_tone"]=="neg")
    check("bearish row indexes has no sector tag",
          not any(t in sectors.SECTOR_ETFS for t in row["indexes"]), str(row["indexes"]))
    check("bearish row is JSON-serialisable",
          isinstance(json.dumps(row, default=str), str))

# ─────────────────────────────────────────────────────────────────────
section("Monu — guards")

check("continuous() rejects a 50% gap",
      not monu.MomentumAnalyzer.continuous(a, bars([100.0]*150 + [50.0]*150)))
check("continuous() accepts a clean series",
      monu.MomentumAnalyzer.continuous(a, bars(list(np.linspace(100,140,300)))))
check("continuous() on a 1-row frame is False",
      not monu.MomentumAnalyzer.continuous(a, bars([100.0])))
check("liquid_enough rejects a $2 stock",
      not monu.MomentumAnalyzer.liquid_enough(a, bars([2.0]*300)))
check("liquid_enough rejects thin volume",
      not monu.MomentumAnalyzer.liquid_enough(a, bars([50.0]*300, vol=100)))
check("liquid_enough accepts a normal name",
      monu.MomentumAnalyzer.liquid_enough(a, bars([50.0]*300, vol=5e6)))
check("check_market_regime(None) is UNKNOWN",
      monu.MomentumAnalyzer.check_market_regime(a, None) == "UNKNOWN")
check("UNKNOWN closes the long gate",
      "UNKNOWN" not in monu.MomentumAnalyzer.GATE_OPEN_REGIMES)

# ─────────────────────────────────────────────────────────────────────
section("Goldy")

check("find_cross on a flat series is None", goldy.find_cross(bars([100.0]*300), 20, 50) is None)
check("find_approach on a flat series is None", goldy.find_approach(bars([100.0]*300), 20, 50) is None)
check("find_approach on a widening gap is None",
      goldy.find_approach(bars([100.0]*230 + [100+i*1.5 for i in range(1,30)]), 20, 50) is None)
check("find_cross on a short frame is None", goldy.find_cross(bars([100.0]*30), 20, 50) is None)
check("find_approach on a short frame is None", goldy.find_approach(bars([100.0]*30), 20, 50) is None)

for lab, dims in (("cross", goldy.CROSS_DIMENSIONS), ("approach", goldy.APPROACH_DIMENSIONS)):
    tot = sum(d["max"] for d in dims)
    check(f"{lab} dimensions sum to {100 if lab=='cross' else 75}",
          tot == (100 if lab == "cross" else 75), str(tot))

check("approach ceiling is below the cross ceiling",
      goldy.APPROACH_MAX_SCORE < sum(d["max"] for d in goldy.CROSS_DIMENSIONS))
check("approach band excludes entanglement", goldy.APPROACH_MIN_GAP_ATR > 0)
check("continuous() rejects a gap", not goldy.continuous(bars([100.0]*150+[40.0]*150), 200))

# ─────────────────────────────────────────────────────────────────────
section("Published JSON contracts")

for fn in ("MNTM","OPY","TREY","GOLD","PCKT","LDRS"):
    p = os.path.join(ROOT, "docs", "data", f"{fn}.json")
    d = json.load(open(p))
    rows = d.get("opportunities", [])
    check(f"{fn}: has agent id/name", bool(d.get("agent",{}).get("id") and d["agent"].get("name")))
    check(f"{fn}: every row has symbol+score",
          all("symbol" in o and isinstance(o.get("score"), (int,float)) for o in rows))
    check(f"{fn}: breakdown keys match declared dimensions",
          all(set(o.get("breakdown",{})) == {x["key"] for x in (o.get("dimensions") or d.get("dimensions") or [])}
              for o in rows if o.get("breakdown")))
    check(f"{fn}: no breakdown value exceeds its max",
          all(o["breakdown"].get(x["key"], 0) <= x["max"]
              for o in rows if o.get("breakdown")
              for x in (o.get("dimensions") or d.get("dimensions") or [])))
    bad = [o["symbol"] for o in rows
           if o.get("bias") == "bearish" and isinstance(o.get("entry"), (int,float))
           and isinstance(o.get("stop_loss"), (int,float)) and o["stop_loss"] <= o["entry"]]
    check(f"{fn}: no bearish row has a stop below entry", not bad, str(bad))
    bull_bad = [o["symbol"] for o in rows
                if o.get("bias") != "bearish" and isinstance(o.get("entry"), (int,float))
                and isinstance(o.get("stop_loss"), (int,float)) and o["stop_loss"] >= o["entry"]]
    check(f"{fn}: no bullish row has a stop above entry", not bull_bad, str(bull_bad))
    check(f"{fn}: groups.values all appear on rows",
          all(any(v in (o.get(d["groups"]["key"]) or []) for o in rows)
              for v in d.get("groups",{}).get("values",[])) if d.get("groups") else True)

man = json.load(open(os.path.join(ROOT,"docs","data","agents.json")))["agents"]
check("manifest lists every data file", set(man) == {"MNTM","OPY","TREY","GOLD","PCKT","LDRS"}, str(man))

# ─────────────────────────────────────────────────────────────────────
section("Every consumer of an opportunity row, in BOTH directions")

# The gap that let Monu's CI run #44 fail: the scan and the publish were both
# tested with bearish rows, but format_results - which main() calls between
# them - was not, and it read setup_type straight off the row. Anything that
# takes an opportunity dict gets exercised here with a mixed list.
mixed = json.load(open(os.path.join(ROOT, "docs", "data", "MNTM.json")))["opportunities"]
has_bear = any(o.get("bias") == "bearish" for o in mixed)
has_long = any(o.get("bias") != "bearish" for o in mixed)
check("fixture has both directions", has_bear and has_long,
      "published MNTM.json has no bearish rows - this section proves little")

_a = monu.MomentumAnalyzer.__new__(monu.MomentumAnalyzer)
try:
    txt = monu.MomentumAnalyzer.format_results(_a, mixed, "UPTREND")
    check("format_results survives bearish rows", True)
    check("format_results labels direction", "BEARISH" in txt and "LONG" in txt)
except Exception as e:
    check("format_results survives bearish rows", False, f"{type(e).__name__}: {e}")

try:
    tmpdir = tempfile.mkdtemp()
    _real_dir = monu.DOCS_DATA_DIR
    monu.DOCS_DATA_DIR = tmpdir
    monu.publish_to_dashboard(mixed, "UPTREND", {o["symbol"]: ["SPX"] for o in mixed}, {})
    written = json.load(open(os.path.join(tmpdir, "MNTM.json")))
    check("publish_to_dashboard survives bearish rows", len(written["opportunities"]) == len(mixed))
    monu.DOCS_DATA_DIR = _real_dir
except Exception as e:
    check("publish_to_dashboard survives bearish rows", False, f"{type(e).__name__}: {e}")

# Reasoning prompts: both branches must format against a real row without a
# live API call. A stub client returns canned text so the f-strings run.
class _StubMsg:
    # Faithful to what the SDK returns and to what both reasoning paths read:
    # content blocks carrying .type and .text, plus stop_reason, which the
    # long path checks for a refusal.
    def __init__(self):
        self.content = [type("C", (), {"text": "ok", "type": "text"})()]
        self.stop_reason = "end_turn"
class _StubClient:
    class messages:
        @staticmethod
        def create(**k):
            assert k["messages"][0]["content"], "empty prompt"
            return _StubMsg()
_a.reasoning_enabled, _a.client, _a.model = True, _StubClient(), "stub"
for o in mixed:
    try:
        if o.get("bias") == "bearish":
            monu.MomentumAnalyzer.generate_bearish_reasoning(_a, o)
        else:
            monu.MomentumAnalyzer.generate_reasoning(
                _a, o["symbol"], {**o["breakdown"], "total": o["score"]}, o["price"],
                o["entry"], o["stop_loss"], o["take_profit"])
    except Exception as e:
        check(f"reasoning prompt formats for {o['symbol']}", False, f"{type(e).__name__}: {e}")
        break
else:
    check("reasoning prompts format for every row, both directions", True)

# Goldy's printer takes the same shape of row.
gold = json.load(open(os.path.join(ROOT, "docs", "data", "GOLD.json")))["opportunities"]
try:
    for o in gold:
        f"{o['rank']:>2} {o['symbol']} {o['strategy']} {o['score']} {o.get('cross_date')}"
    check("Goldy rows format without KeyError", True)
except Exception as e:
    check("Goldy rows format without KeyError", False, f"{type(e).__name__}: {e}")

# ─────────────────────────────────────────────────────────────────────
section("Pocket")

pocket = load_mod("pocketagent", "pocket/agent.py", "pocket")

def vol_bars(n, base_vol, last_vol, price=50.0):
    df = bars([price + (i % 7) * 0.5 for i in range(n)], vol=base_vol)
    df.iloc[-1, df.columns.get_loc("Volume")] = last_vol
    return df

# The screen, exactly as the Finviz settings read.
check("screen passes 3x volume on a liquid name",
      pocket.screen(vol_bars(300, 1e6, 3e6)) is not None)
check("screen rejects rel vol under 1.5",
      pocket.screen(vol_bars(300, 1e6, 1.4e6)) is None)
check("screen rejects avg volume under 500K",
      pocket.screen(vol_bars(300, 4e5, 2e6)) is None)
check("screen rejects current volume under 500K",
      pocket.screen(vol_bars(300, 3e5, 4.9e5)) is None)
# Average excludes today, as Finviz's does - otherwise a spike inflates its
# own denominator. 1M baseline, 3M today: 3.0x excluding, ~2.97x including.
s3 = pocket.screen(vol_bars(300, 1e6, 3e6))
check("relative volume excludes today from the average",
      s3 is not None and abs(s3["rel_volume"] - 3.0) < 1e-6, str(s3 and s3["rel_volume"]))

# No lookahead: a pivot needs PIVOT_BARS sessions after it, so a spike in the
# last PIVOT_BARS bars must NOT appear as a level yet.
spike = bars([100.0 + (i % 5) for i in range(300)])
spike.iloc[-3, spike.columns.get_loc("High")] = 200.0
lv = pocket.major_levels(spike, pocket.atr(spike))
check("a swing with fewer than RECENT_RIGHT_BARS sessions after it is not yet a level",
      all(x["level"] < 150 for x in lv), str([round(x["level"], 1) for x in lv if x["level"] > 150]))
# ...but the most recent swing, once RECENT_RIGHT_BARS sessions have printed
# after it, IS a level - the DECK 88.40 case the full 10-bar rule missed.
spike5 = bars([100.0 + (i % 5) for i in range(300)])
spike5.iloc[-5, spike5.columns.get_loc("High")] = 200.0
lv5 = pocket.major_levels(spike5, pocket.atr(spike5))
check("a recent swing with RECENT_RIGHT_BARS sessions after it is a level",
      any(abs(x["level"] - 200.0) < 1 for x in lv5), str([round(x["level"], 1) for x in lv5]))

# Published rows: breaks carry levels on the right side, inside rows carry
# none and never read as a buy.
pk = json.load(open(os.path.join(ROOT, "docs", "data", "PCKT.json")))["opportunities"]
check("every Pocket row has a signal_label (no default 'Buy')",
      all(o.get("signal_label") for o in pk))
check("inside and testing rows carry no entry/stop/target",
      all("entry" not in o for o in pk if o["signal_state"] in ("inside", "testing")))
check("only breaks earn the fresh-break points",
      all((o["breakdown"]["fresh_break"] > 0) == (o["signal_state"] in ("broke_up", "broke_down")) for o in pk))
# The exit rules that tested best - see BREAKOUT_* and SHORT_MAX_OFF_HIGH.
check("breakouts carry a stop and a time exit, no fixed target",
      all("entry" in o and "take_profit" not in o and "sessions" in o.get("exit_rule", "")
          for o in pk if o["signal_state"] == "broke_up"))
check("breakout stop sits BREAKOUT_STOP_ATR under the crossed level",
      all(abs(o["stop_loss"] - (o["map"]["floor"] - pocket.BREAKOUT_STOP_ATR * o["map"]["atr"])) < 1e-6
          for o in pk if o["signal_state"] == "broke_up"))
_off = lambda o: next(float(f["value"].rstrip("%")) for f in o["setup"]["fields"] if f["label"] == "Off 52w high")
check("only breakdowns within 10% of the 52-week high carry a short",
      all(("entry" in o) == (_off(o) > -pocket.SHORT_MAX_OFF_HIGH * 100)
          for o in pk if o["signal_state"] == "broke_down"))
check("breakdowns are labelled bearish",
      all(o.get("bias") == "bearish" for o in pk if o["signal_state"] == "broke_down"))
check("no published pocket narrower than MIN_POCKET_ATR",
      all(o["map"]["width_atr"] is None or o["map"]["width_atr"] >= pocket.MIN_POCKET_ATR for o in pk))

# Stale and pinned symbols - both found on BZH. scan() is run against a
# stubbed fetch so the guards are tested on the path the agent actually uses.
def _screen_hit(prices, end="2026-10-23"):
    idx = pd.date_range(end=end, periods=len(prices), freq="B")
    df = pd.DataFrame({"Open": prices, "High": [x*1.02 for x in prices],
                       "Low": [x*0.98 for x in prices], "Close": prices,
                       "Volume": [1e6]*len(prices)}, index=idx)
    df.iloc[-1, df.columns.get_loc("Volume")] = 3e6
    return df
wave = [100 + 15*np.sin(i/9) for i in range(400)]
fresh = _screen_hit(wave)
stale = _screen_hit(wave, end="2026-10-19")
pinned = _screen_hit([33.45 + 0.01*np.sin(i) for i in range(400)])
pinned["High"] = pinned["Close"] + 0.02; pinned["Low"] = pinned["Close"] - 0.02
_real_fetch = pocket.fetch_many
pocket.fetch_many = lambda syms: {"FRESH": fresh, "STALE": stale, "PIN": pinned}
_rows, _st = pocket.scan(["FRESH", "STALE", "PIN"], {"FRESH": ["SPX"], "STALE": ["SPX"], "PIN": ["SPX"]})
pocket.fetch_many = _real_fetch
_syms = {o["symbol"] for o in _rows}
check("stale symbol (last bar older than the run) is dropped",
      "STALE" not in _syms and _st["stale"] == 1, f"stale={_st['stale']} rows={_syms}")
check("pinned symbol (ATR < 0.5% of price) is dropped",
      "PIN" not in _syms, f"rows={_syms} pinned={_st['pinned']}")
check("every Pocket row reports today's move",
      all("day_move_pct" in o for o in pk + _rows))

# ─────────────────────────────────────────────────────────────────────
section("Leaders")

leaders = load_mod("leadersagent", "leaders/agent.py", "leaders")

# The universe builder once kept only QQQ rows labelled "Common Stock", which
# silently dropped the Nasdaq-100's ADRs from every agent.
_qqq = {s for s in universe.load()[0] if "QQQ" in universe.load()[1].get(s, [])}
check("Nasdaq-100 ADRs (ASML, ARM, PDD) are in the universe",
      {"ASML", "ARM", "PDD"} <= _qqq, str({"ASML", "ARM", "PDD"} - _qqq))

# A synthetic panel: 30 SPX names whose YTD return is their number, so the
# expected ranking is known. Prior year closes at 100; this week ends Fri
# 2026-10-23, last week Fri 2026-10-16.
_days = pd.bdate_range("2025-12-01", "2026-10-23")
_C = pd.DataFrame(100.0, index=_days, columns=[f"S{i:02d}" for i in range(30)])
for i in range(30):
    _C.loc[_C.index >= "2026-01-02", f"S{i:02d}"] = 100.0 + i          # YTD = i%
# Last week S00 led (YTD +80%); this week it falls back to the bottom. 1.8x
# is deliberately not split-shaped, or the split filter would drop it.
_C.loc["2026-10-16", "S00"] = 180.0
_mem = {s: ["SPX"] for s in _C.columns}
_etf = pd.DataFrame({"SPY": 100.0, "QQQ": 100.0, "IWM": 100.0}, index=_days)
_rows, _ctx = leaders.build(_C, _mem, {}, _etf)
_top = [o for o in _rows if o["signal_label"] != "Fell off"]
check("ytd_at measures from the prior year's last close",
      abs(leaders.ytd_at(_C, _days[-1])["S29"] - 0.29) < 1e-9)
check("a list holds at most TOP_N names", len(_top) == leaders.TOP_N, str(len(_top)))
check("rank 1 is the best YTD", _top[0]["symbol"] == "S29" and _top[0]["rank"] == 1)
check("each row belongs to exactly one list", all(o["indexes"] == ["SPX"] for o in _rows))
_s00 = next((o for o in _rows if o["symbol"] == "S00"), None)
check("last week's #1 that dropped out shows as Fell off",
      _s00 is not None and _s00["signal_label"] == "Fell off" and _s00["prev_rank"] == 1, str(_s00 and _s00["signal_label"]))
check("fell-off rows score below every listed row",
      all(o["score"] < min(t["score"] for t in _top) for o in _rows if o["signal_label"] == "Fell off"))
check("a name that rose one place reads ▲1",
      next(o for o in _rows if o["symbol"] == "S29")["signal_label"] == "▲1")
# Split artefacts and stale symbols never reach a list.
_C2 = _C.copy(); _C2["SPLIT"] = 100.0; _C2.loc[_C2.index >= "2026-06-01", "SPLIT"] = 1000.0
_C2["STALE"] = 200.0; _C2.loc[_C2.index[-1], "STALE"] = np.nan
_r2, _c2 = leaders.build(_C2, {**_mem, "SPLIT": ["SPX"], "STALE": ["SPX"]}, {}, _etf)
check("split-shaped and stale symbols are dropped",
      not {"SPLIT", "STALE"} & {o["symbol"] for o in _r2} and _c2["splits"] == 1 and _c2["stale"] == 1,
      f"splits={_c2['splits']} stale={_c2['stale']}")

ld = json.load(open(os.path.join(ROOT, "docs", "data", "LDRS.json")))["opportunities"]
for g in ("SPX", "QQQ", "IWM"):
    on = [o for o in ld if o["indexes"] == [g] and o["signal_label"] != "Fell off"]
    check(f"published {g} list has {leaders.TOP_N} names ranked 1..{leaders.TOP_N}",
          sorted(o["rank"] for o in on) == list(range(1, leaders.TOP_N + 1)))
check("Leaders rows carry no trade levels (a watchlist, not a signal)",
      not any(k in o for o in ld for k in ("entry", "stop_loss", "take_profit")))

# ─────────────────────────────────────────────────────────────────────
section("Price history (detail-panel chart)")

import history
_h = history.series(bars([float(i) for i in range(1, 101)]))
check("history keeps the last SESSIONS closes, oldest first",
      _h is not None and len(_h["closes"]) == history.SESSIONS and _h["closes"][-1] == 100.0
      and _h["dates"] == sorted(_h["dates"]))
check("history of a one-bar series is None (nothing to draw)", history.series(bars([5.0])) is None)
_rows = [{"symbol": "AAA"}, {"symbol": "BBB"}]
_real = history._fetch
history._fetch = lambda syms: {}                 # no network: BBB stays chartless
history.attach(_rows, {"AAA": bars([10.0 + i for i in range(80)])})
history._fetch = _real
check("attach uses the agent's own frames and skips what it cannot get",
      "history" in _rows[0] and "history" not in _rows[1])
for fn in ("MNTM", "OPY", "TREY", "GOLD", "PCKT", "LDRS"):
    _rs = json.load(open(os.path.join(ROOT, "docs", "data", f"{fn}.json")))["opportunities"]
    check(f"{fn}: every history has matching dates/closes",
          all(len(o["history"]["dates"]) == len(o["history"]["closes"]) >= 2
              for o in _rs if "history" in o))

print(f"\n{'='*60}\n{len(PASSES)} passed, {len(FAILS)} failed")
if FAILS:
    print("FAILURES:")
    for f in FAILS:
        print("  -", f)
sys.exit(1 if FAILS else 0)
