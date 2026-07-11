"""Passive absorption (size only): big resting order by raw size + absorption + confirmation."""

import numpy as np

from ..absorption import absorption_scan, wick_bounds

_NO_ENTRY = (None, None, None, None, None)


def find_entry(win, params: dict) -> tuple:
    """
    Looks for a candle with both:
      - A big passive order on the defended side (raw size only, order count ignored):
          Long: resting bid below candle open, size >= passive_baseline * passive_size_order_mult
          Short: resting ask above candle open, size >= passive_baseline * passive_size_order_mult
      - Absorption in the defended wick (passive_size_absorption_mult / passive_size_wick_threshold)

    Both conditions must be on the same candle. Then scans up to entry_after_absorption bars
    for a confirmation candle and enters on the open of the bar after it.

    The candidate prefilter uses the per-bar max defended-side size precomputed in DayData
    (a qualifying order exists iff the max qualifies); the per-bar scan then picks the FIRST
    qualifying level in document order for the notes, as the original dict loop did.

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
    order_mult = params["passive_size_order_mult"]
    abs_mult   = params["passive_size_absorption_mult"]
    confirm    = win.confirm_dir
    K          = params["entry_after_absorption"]

    with np.errstate(invalid="ignore"):
        cand = (
            (win.p_base > 0)
            & (win.best_passive >= win.p_base * order_mult)
            & (win.wick_frac >= params["passive_size_wick_threshold"])
            & (win.abs_base > 0)
        )
    cand[0] = False

    for i in map(int, np.flatnonzero(cand)):
        if i >= first_inv:
            break

        # --- passive order check (raw size only): first qualifying defended-side level ---
        prices, sizes, counts = win.po[i]
        required_po = float(win.p_base[i]) * order_mult
        if long_:
            m = (counts > 0) & (prices < o[i]) & (sizes >= required_po)
        else:
            m = (counts > 0) & (prices > o[i]) & (sizes >= required_po)
        if not m.any():
            continue
        k = int(m.argmax())
        passive_trigger_price = float(prices[k])
        passive_trigger_size  = sizes[k].item()
        passive_trigger_count = counts[k].item()

        # --- absorption check on the same candle ---
        baseline = float(win.abs_base[i])
        wl, wh   = wick_bounds(o[i], h[i], l[i], c[i], win.direction)
        found, _, _ = absorption_scan(win.tv[i], wl, wh, baseline * abs_mult, win.direction)
        if not found:
            continue

        # --- confirmation candle scan ---
        conf_scan_end = min(i + 1 + K, n)
        for j in range(i + 1, conf_scan_end):
            if j >= first_inv:
                return None, None, first_inv, None, None
            if not confirm[j]:
                continue

            entry_rel = j + 1
            if entry_rel >= n:
                return _NO_ENTRY
            if entry_rel >= first_inv:
                return None, None, entry_rel, None, None

            entry_notes = {
                "absorption_time": win.ts(i).strftime("%H:%M"),
                "passive_price":   passive_trigger_price,
                "passive_size":    passive_trigger_size,
                "passive_count":   passive_trigger_count,
            }
            return (entry_rel, float(o[entry_rel]), None, entry_notes,
                    "passive_absorption_size_only")

    if first_inv < n:
        return None, None, first_inv, None, None
    return _NO_ENTRY
