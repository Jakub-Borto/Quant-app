"""Passive wall entry: a cluster of big resting passive orders on the defended side.

No absorption candle and no delta confirmation — the wall of stacked liquidity is the signal.
Mirrors consecutive_absorption's price-cluster logic, but counts big passive orders (by raw
resting size) instead of absorption candles. The wall builds across bars: a running `seen`
list of qualifying levels is never cleared.
"""

import numpy as np

_NO_ENTRY = (None, None, None, None, None)


def find_entry(win, params: dict) -> tuple:
    """
    Accumulates big passive orders on the defended side (long: resting bids below candle
    open; short: resting asks above candle open) whose raw size >= passive_baseline *
    passive_wall_mult. When passive_wall_n of them cluster within passive_wall_ticks of one
    level, a wall is confirmed and we enter on the next bar's open.

    The candidate prefilter (per-bar max defended-side size from DayData) only skips bars
    that could not contribute any level — identical accumulation to the original loop.

    Returns (entry_rel, entry_price, invalidation_rel, entry_notes, trade_type).
    """
    n = win.n
    if n == 0:
        return _NO_ENTRY

    first_inv = win.first_inv
    if first_inv == 0:                   # breakout bar: invalidation only
        return None, None, 0, None, None

    o          = win.o
    long_      = win.direction == "long"
    tick_size  = params["tick_size"]
    level_tol  = params["passive_wall_ticks"] * tick_size
    required_n = params["passive_wall_n"]
    wall_mult  = params["passive_wall_mult"]

    with np.errstate(invalid="ignore"):
        cand = (win.p_base > 0) & (win.best_passive >= win.p_base * wall_mult)
    cand[0] = False

    # (price_level, window bar index) of every big passive order seen — never cleared
    seen: list[tuple] = []

    for i in map(int, np.flatnonzero(cand)):
        if i >= first_inv:
            break

        prices, sizes, counts = win.po[i]
        required = float(win.p_base[i]) * wall_mult
        if long_:
            m = (counts > 0) & (prices < o[i]) & (sizes >= required)
        else:
            m = (counts > 0) & (prices > o[i]) & (sizes >= required)
        if not m.any():
            continue

        # append each qualifying level (document order); trigger at the earliest level
        # that completes a wall
        for lvl in prices[m].tolist():
            seen.append((lvl, i))

            nearby = [(p, t) for p, t in seen if abs(p - lvl) <= level_tol]
            if len(nearby) < required_n:
                continue

            entry_rel = i + 1
            if entry_rel >= n:
                return _NO_ENTRY
            if entry_rel >= first_inv:
                return None, None, entry_rel, None, None

            entry_notes = {
                "wall_levels": [round(p, 2) for p, _ in nearby],
                "wall_times":  [win.ts(t).strftime("%H:%M") for _, t in nearby],
                "wall_count":  len(nearby),
            }
            return entry_rel, float(o[entry_rel]), None, entry_notes, "passive_wall"

    if first_inv < n:
        return None, None, first_inv, None, None
    return _NO_ENTRY
