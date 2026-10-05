#!/usr/bin/env python3
"""The shared scan universe: one parser for universe.txt, used by every agent.

Monu, Opy and Goldy each had their own hand-copied copy of this, and they had
already drifted - Monu kept sector tags inside `membership` and narrowed them
again at publish time, Opy kept them in a separate map, Goldy dropped them
entirely. That divergence is not cosmetic: filtering tags too early in Monu's
copy silently disabled its entire bearish half on the first run, because the
sector lookup read a list the loader had already stripped.

So this module settles the shape once:

    symbols      the tickers to scan, file order
    membership   symbol -> INDEX tags only (SPX / QQQ / IWM, or OTHER)
    sector_map   symbol -> its sector SPDR tag (XLV, XLF, ...), where known

Index and sector are separate returns rather than one mixed list, because they
are different axes: the dashboard groups by index, the bearish gating keys off
sector, and the bug above came from carrying them in the same list.

File format - one ticker per line, tags in a trailing comment:

    AAPL   # SPX,QQQ,XLK
    JPM    # SPX,XLF
    TWST   # IWM

Anything that only wants tickers can still read the file by splitting on "#",
which is why the tags live in a comment at all. A file written before the tags
existed loads fine: those symbols come back tagged OTHER with no sector.
"""

import logging
import os
from typing import Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.abspath(__file__))
UNIVERSE_FILE = os.path.join(ROOT, "universe.txt")

# Index tags, in display order. These are the groups the dashboard offers as
# filter chips.
INDEX_TAGS = ["SPX", "QQQ", "IWM"]

# A symbol with no index tag at all: a universe.txt predating the tags, or a
# hand-added ticker.
UNTAGGED = "OTHER"

# Sector tags are the eleven sector SPDRs. Imported from sectors.py rather than
# restated, so "which ETFs are sectors" has exactly one definition - that
# module prices them, this one only recognises them.
try:
    from sectors import SECTOR_ETFS
except ImportError:                      # pragma: no cover - standalone use
    SECTOR_ETFS = {}


def parse_tags(comment: str) -> Tuple[List[str], str]:
    """Split one line's trailing comment into (index tags, sector tag).

    Unknown tags are dropped rather than guessed at. The sector is a single
    value because the sector SPDRs do not overlap - a ticker is in exactly one
    of them, or in none.
    """
    tags = [t.strip().upper() for t in comment.split(",") if t.strip()]
    idx = [t for t in tags if t in INDEX_TAGS]
    sector = next((t for t in tags if t in SECTOR_ETFS), None)
    return idx, sector


def load(keep: Sequence[str] = None,
         path: str = None) -> Tuple[List[str], Dict[str, List[str]], Dict[str, str]]:
    """Read universe.txt. Returns (symbols, membership, sector_map).

    `keep` restricts to symbols carrying at least one of those index tags -
    Opy passes ("SPX", "QQQ") because most Russell names have no tradable
    options. None keeps everything.

    Raises FileNotFoundError if the file is missing. Each agent wants a
    different answer to that (a built-in list, a live fetch, or exit), so the
    fallback stays with the caller rather than being decided here.
    """
    path = path or UNIVERSE_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    symbols, membership, sector_map, skipped = [], {}, {}, 0
    with open(path) as f:
        for line in f:
            ticker, _, comment = line.partition("#")
            sym = ticker.strip().upper()
            if not sym or sym in membership:
                continue

            idx, sector = parse_tags(comment)
            # An untagged file predates the tags; it was an S&P 500 list, so
            # `keep` must not filter every symbol out of it.
            if keep and idx and not any(t in keep for t in idx):
                skipped += 1
                continue

            symbols.append(sym)
            membership[sym] = idx or [UNTAGGED]
            if sector:
                sector_map[sym] = sector

    return symbols, membership, sector_map


def describe(symbols: List[str], membership: Dict[str, List[str]],
             sector_map: Dict[str, str], extra: str = "") -> str:
    """The one-line summary each agent logs after loading."""
    counts = {t: sum(1 for v in membership.values() if t in v)
              for t in INDEX_TAGS + [UNTAGGED]}
    body = ", ".join(f"{t} {n}" for t, n in counts.items() if n)
    return (f"Loaded {len(symbols)} symbols from universe.txt  ({body}; "
            f"{len(sector_map)} with a sector tag){extra}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    syms, mem, sec = load()
    print(describe(syms, mem, sec))
    kept, mem2, sec2 = load(keep=("SPX", "QQQ"))
    print(describe(kept, mem2, sec2, extra="   [keep=SPX,QQQ]"))
