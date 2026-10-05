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

for fn in ("MNTM","OPY","TREY","GOLD"):
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
check("manifest lists every data file", set(man) == {"MNTM","OPY","TREY","GOLD"}, str(man))

print(f"\n{'='*60}\n{len(PASSES)} passed, {len(FAILS)} failed")
if FAILS:
    print("FAILURES:")
    for f in FAILS:
        print("  -", f)
sys.exit(1 if FAILS else 0)
