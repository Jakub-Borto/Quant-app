"""Consecutive absorption entry: N absorption candles at the same level + body midpoint."""

import numpy as np

from ..absorption import absorption_scan, wick_bounds

_NO_ENTRY = (None, None, None, None, None)


def find_entry(win, params: dict) -> tuple:
    """
    Scans the window's candidate absorption bars. For each absorption candle found, checks
    whether n-1 prior absorption candles exist within ±consec_abs_ticks of its absorption
    level and body midpoint. If so, enters on the open of the next bar.
    No confirmation candle required.

    An absorption level that price has since CLOSED through is dead (long: any later close
    strictly below the level; closing exactly AT it keeps it). The original pruned the `seen`
    list bar-by-bar with each close; applying the min (long) / max (short) close over the
    interval since the last visited candidate kills exactly the same levels.

    Returns (entry_rel, entry_price, invalidation_rel, entry_notes, trade_type).
    """
    n = win.n
    if n == 0:
        return _NO_ENTRY

    first_inv = win.first_inv
    if first_inv == 0:                   # breakout bar: invalidation only
        return None, None, 0, None, None

    o, h, l, c = win.o, win.h, win.l, win.c
    long_      = win.direction == "long"
    tick_size  = params["tick_size"]
    level_tol  = params["consec_abs_ticks"] * tick_size
    required_n = params["consec_abs_n"]
    mult       = params["consec_abs_mult"]

    cand = (win.wick_frac >= params["consec_wick_threshold"]) & (win.abs_base > 0)
    cand[0] = False

    seen: list[tuple] = []               # (abs_level, body_mid, window bar index)
    last = 0                             # closes up to and including `last` already applied

    for i in map(int, np.flatnonzero(cand)):
        if i >= first_inv:
            break

        if seen:
            if long_:
                worst = c[last + 1 : i + 1].min()
                seen = [e for e in seen if worst >= e[0]]
            else:
                worst = c[last + 1 : i + 1].max()
                seen = [e for e in seen if worst <= e[0]]
        last = i

        baseline = float(win.abs_base[i])
        wl, wh   = wick_bounds(o[i], h[i], l[i], c[i], win.direction)
        found, trigger_price, trigger_volume = absorption_scan(
            win.tv[i], wl, wh, baseline * mult, win.direction
        )
        if not found:
            continue

        abs_level = float(l[i]) if long_ else float(h[i])
        body_mid  = (
            min(float(o[i]), float(c[i])) +
            max(float(o[i]), float(c[i]))
        ) / 2

        seen.append((abs_level, body_mid, i))

        nearby = [
            (lvl, t) for lvl, bm, t in seen
            if abs(lvl - abs_level) <= level_tol
            and abs(bm - body_mid) <= level_tol
        ]

        if len(nearby) < required_n:
            continue

        entry_rel = i + 1
        if entry_rel >= n:
            return _NO_ENTRY
        if entry_rel >= first_inv:
            return None, None, entry_rel, None, None

        # trigger reflects the nth (triggering) candle
        entry_notes = {
            "absorption_time": [win.ts(t).strftime("%H:%M") for _, t in nearby],
            "abs_baseline":    round(baseline, 2),
            "trigger_price":   trigger_price,
            "trigger_volume":  trigger_volume,
        }
        return entry_rel, float(o[entry_rel]), None, entry_notes, "consecutive_absorption"

    if first_inv < n:
        return None, None, first_inv, None, None
    return _NO_ENTRY
