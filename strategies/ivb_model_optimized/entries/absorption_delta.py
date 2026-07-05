"""Absorption + delta entry: absorption candle followed by a confirming entry candle."""

import numpy as np

from ..absorption import absorption_scan, wick_bounds

_NO_ENTRY = (None, None, None, None, None)


def find_entry(win, params: dict) -> tuple:
    """Returns (entry_rel, entry_price, invalidation_rel, entry_notes, trade_type).

    Vectorized prechecks (wick fraction, baseline validity, confirm-candle mask) come from
    the shared EntryWindow; the loop only visits candidate absorption bars. Invalidation
    semantics are position-based: the first invalid bar (win.first_inv) wins over any chain
    that has not produced an entry strictly before it — exactly the original bar-by-bar order.
    """
    n = win.n
    if n == 0:
        return _NO_ENTRY

    first_inv = win.first_inv
    if first_inv == 0:                   # breakout bar (index 0): invalidation only
        return None, None, 0, None, None

    o, h, l, c = win.o, win.h, win.l, win.c
    long_   = win.direction == "long"
    confirm = win.confirm_any            # this finder's confirm has NO candle-direction test
    K       = params["entry_after_absorption"]
    mult    = params["absorption_mult"]

    cand = (win.wick_frac >= params["wick_threshold"]) & (win.abs_base > 0)
    cand[0] = False                      # breakout bar is never an absorption candidate

    for i in map(int, np.flatnonzero(cand)):
        if i >= first_inv:
            break

        baseline = float(win.abs_base[i])
        wl, wh   = wick_bounds(o[i], h[i], l[i], c[i], win.direction)
        found, trigger_price, trigger_volume = absorption_scan(
            win.tv[i], wl, wh, baseline * mult, win.direction
        )
        if not found:
            continue

        absorption_level = float(l[i]) if long_ else float(h[i])

        scan_end = min(i + 1 + K, n)
        for j in range(i + 1, scan_end):
            if j >= first_inv:
                return None, None, first_inv, None, None

            # close back through the absorption level kills this candidate
            if long_:
                if c[j] < absorption_level:
                    break
            else:
                if c[j] > absorption_level:
                    break

            if not confirm[j]:
                continue

            entry_rel = j + 1
            if entry_rel >= n:
                return _NO_ENTRY
            if entry_rel >= first_inv:
                return None, None, entry_rel, None, None

            entry_notes = {
                "absorption_time": win.ts(i).strftime("%H:%M"),
                "abs_baseline":    round(baseline, 2),
                "trigger_price":   trigger_price,
                "trigger_volume":  trigger_volume,
            }
            return entry_rel, float(o[entry_rel]), None, entry_notes, "absorption_delta"

    if first_inv < n:
        return None, None, first_inv, None, None
    return _NO_ENTRY
