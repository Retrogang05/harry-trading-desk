#!/usr/bin/env python3
"""What actually happened to the signals an agent published?

Reads the dated archive in results/ (Monu), scores every published signal
against real subsequent prices, and writes a per-signal CSV plus a summary.
This is the accountability loop none of the agents had: the dashboard shows
what was predicted; this shows what happened.

    python scripts/review_signals.py            # Monu, all history
    python scripts/review_signals.py --csv out.csv

Signals are taken at the scan-day close (Monu's own `price` field). Outcome
is measured against Monu's own published stop and target over the following
20 sessions: which was touched first.
"""
import argparse, glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_signals(results_dir):
    sig = []
    for f in sorted(glob.glob(os.path.join(results_dir, "scan_*.json"))):
        day = pd.Timestamp(os.path.basename(f)[5:13])
        for o in json.load(open(f))["opportunities"]:
            sig.append({"date": day, "sym": o["symbol"], "rank": o["rank"], "score": o["score"], "px": o["price"],
                        "stop": o["stop_loss"], "target": o["take_profit"], "setup": o["setup_type"], "ext_pct": o["extension_pct"]})
    return pd.DataFrame(sig)

def score(S, horizon=20):
    syms = sorted(S.sym.unique()) + ["SPY"]
    raw = yf.download(syms, start=(S.date.min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                      auto_adjust=False, progress=False, group_by="ticker", threads=True)
    px = lambda s: raw[s].dropna(how="all")
    spy = px("SPY")
    rows = []
    for r in S.itertuples():
        p = px(r.sym); fut = p[p.index > r.date]
        if fut.empty: continue
        base = r.px
        out = r._asdict(); out.pop("Index", None)
        out["sessions_since"] = len(fut)
        out["open_gap"] = fut.Open.iloc[0] / base - 1
        s_fut = spy[spy.index > r.date]; s_base = spy.loc[:r.date].Close.iloc[-1]
        for h in (5, 10, 20):
            out[f"ret{h}"] = fut.Close.iloc[h-1] / base - 1 if len(fut) >= h else np.nan
            out[f"spy{h}"] = s_fut.Close.iloc[h-1] / s_base - 1 if len(s_fut) >= h else np.nan
        w = fut.iloc[:horizon]
        out["max_dd"] = w.Low.min() / base - 1
        out["max_up"] = w.High.max() / base - 1
        hs = w.index[w.Low <= r.stop]; ht = w.index[w.High >= r.target]
        fs = hs.min() if len(hs) else None; ft = ht.min() if len(ht) else None
        out["outcome"] = "target" if ft is not None and (fs is None or ft <= fs) else ("stop" if fs is not None else "open")
        out["days_to_outcome"] = (len(w.loc[:ft]) if out["outcome"] == "target" else len(w.loc[:fs]) if out["outcome"] == "stop" else None)
        rows.append(out)
    R = pd.DataFrame(rows)
    for h in (5, 10, 20): R[f"alpha{h}"] = R[f"ret{h}"] - R[f"spy{h}"]
    return R

def summarize(R):
    pct = lambda x: f"{x*100:+.2f}%"
    print(f"{len(R)} signals, {R.sym.nunique()} stocks, {R.date.min().date()} -> {R.date.max().date()}\n")
    print(f"{'horizon':>8} {'n':>4} {'avg':>8} {'median':>8} {'win':>5} {'SPY':>8} {'alpha':>8}")
    for h in (5, 10, 20):
        g = R.dropna(subset=[f"ret{h}"])
        print(f"{h:>7}d {len(g):>4} {pct(g[f'ret{h}'].mean()):>8} {pct(g[f'ret{h}'].median()):>8} {(g[f'ret{h}']>0).mean()*100:4.0f}% {pct(g[f'spy{h}'].mean()):>8} {pct(g[f'alpha{h}'].mean()):>8}")
    oc = R.outcome.value_counts()
    print(f"\nvs own levels (20d): stop {oc.get('stop',0)}  target {oc.get('target',0)}  open {oc.get('open',0)}"
          f"   |  avg max drawdown {pct(R.max_dd.mean())}  avg max gain {pct(R.max_up.mean())}")
    print("\nby score:")
    R["band"] = pd.cut(R.score, [59, 79, 89, 99, 100], labels=["60-79", "80-89", "90-99", "100"])
    for b, g in R.groupby("band", observed=True):
        g20 = g.dropna(subset=["ret20"])
        print(f"  {b:6} n={len(g):>4}  ret10 {pct(g.ret10.mean()):>8}  ret20 {pct(g20.ret20.mean()):>8}  win20 {(g20.ret20>0).mean()*100:3.0f}%"
              f"  stopped {(g.outcome=='stop').mean()*100:3.0f}%  target {(g.outcome=='target').mean()*100:3.0f}%")
    print("\nby setup:")
    for st, g in R.groupby("setup"):
        g20 = g.dropna(subset=["ret20"])
        print(f"  {st:9} n={len(g):>4}  ret20 {pct(g20.ret20.mean()):>8}  win20 {(g20.ret20>0).mean()*100:3.0f}%  stopped {(g.outcome=='stop').mean()*100:3.0f}%  target {(g.outcome=='target').mean()*100:3.0f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--csv", help="write per-signal rows here")
    a = ap.parse_args()
    S = load_signals(a.results)
    if S.empty: sys.exit(f"no scan_*.json under {a.results}")
    R = score(S)
    summarize(R)
    if a.csv:
        cols = ["date","sym","rank","score","setup","ext_pct","px","open_gap","ret5","ret10","ret20","alpha20","max_dd","max_up","stop","target","outcome","days_to_outcome"]
        R[cols].sort_values(["date","rank"]).to_csv(a.csv, index=False, float_format="%.4f")
        print(f"\nwrote {a.csv}")
