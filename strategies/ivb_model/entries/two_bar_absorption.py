"""Two bar absorption entry: reversal pair merged into a synthetic candle + confirmation.

The 2-bar-specific helpers (merge_tv, the strided pair-density baseline) live here because
this is their only entry-side consumer (vwap_trailing_risk keeps its own copies per the
risk-script convention).
"""

import numpy as np

_NO_ENTRY = (None, None, None, None, None)


def merge_tv(tv1, tv2) -> dict:
    """Merge two pre-parsed tick_volume tuples into {price: [buy, sell]}, summing volumes.

    Insertion order (bar1's levels, then bar2's unseen levels) matches the original
    JSON-string merge, so a first-qualifying-level scan picks the same level.
    """
    merged: dict = {}
    for tv in (tv1, tv2):
        if tv is None:
            continue
        prices, buys, sells = tv
        pl, bl, sl_ = prices.tolist(), buys.tolist(), sells.tolist()
        for k in range(len(pl)):
            p = pl[k]
            e = merged.get(p)
            if e is None:
                merged[p] = [bl[k], sl_[k]]
            else:
                e[0] += bl[k]
                e[1] += sl_[k]
    return merged


def pair_densities(h, l, vol, tick_size: float) -> np.ndarray:
    """density[j] of the merged pair (bars j, j+1): summed counter-volume / merged range in
    ticks; NaN where the merged range is 0 (the original skipped those pairs)."""
    vol_pair = vol[:-1] + vol[1:]
    rng = (np.maximum(h[:-1], h[1:]) - np.minimum(l[:-1], l[1:])) / tick_size
    with np.errstate(divide="ignore", invalid="ignore"):
        d = vol_pair / rng
    d[rng <= 0] = np.nan
    return d


def baseline_from_densities(dens: np.ndarray, i: int, n_pairs: int, lowest_j: int) -> float:
    """Mean pair density anchored backwards from bar i: pairs (i-1,i-2), (i-3,i-4), ...
    — i.e. dens[i-2], dens[i-4], ... down to j >= lowest_j, first n_pairs of them, NaN pairs
    skipped. Summation order matches the original loop exactly."""
    js   = np.arange(i - 2, lowest_j - 1, -2)[:n_pairs]
    if js.size == 0:
        return float("nan")
    vals = dens[js]
    vals = vals[~np.isnan(vals)].tolist()
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def find_entry(win, params: dict) -> tuple:
    """
    Pattern (long bias):
      - Bar i:   bearish, small bottom wick (<= two_bar_wick_ticks)
      - Bar i+1: bullish, small bottom wick (<= two_bar_wick_ticks), just positive delta

    Pattern (short bias): mirrored.

    Merges both bars, checks absorption in the defended wick against the 2-bar paired
    baseline (pairs anchored backwards from the pattern, breakout bar excluded), then scans
    from bar i+1 (inclusive) for a confirmation candle. Enters on the open of the bar after
    the confirmation candle.

    Returns (entry_rel, entry_price, invalidation_rel, entry_notes, trade_type).
    """
    n = win.n
    if n == 0:
        return _NO_ENTRY

    inv       = win.invalid
    first_inv = win.first_inv
    if first_inv == 0:                   # breakout bar: invalidation only
        return None, None, 0, None, None

    o, h, l, c = win.o, win.h, win.l, win.c
    vdp        = win.vdp
    long_      = win.direction == "long"
    tick_size  = params["tick_size"]
    max_wick   = params["two_bar_wick_ticks"] * tick_size
    req_mult   = params["two_bar_abs_mult"]
    confirm    = win.confirm_dir
    K          = params["entry_after_absorption"]
    n_pairs    = params["absorption_baseline_window"] // 2

    # vectorized bar1/bar2 pattern masks; pair candidate at i = bars (i, i+1)
    if long_:
        wick = np.minimum(o, c) - l
        bar1_ok = (c < o) & (wick <= max_wick)
        bar2_ok = (c > o) & (wick <= max_wick) & (vdp > 0)
    else:
        wick = h - np.maximum(o, c)
        bar1_ok = (c > o) & (wick <= max_wick)
        bar2_ok = (c < o) & (wick <= max_wick) & (vdp < 0)
    pair = bar1_ok[:-1] & bar2_ok[1:]

    # the original returns invalidation at pair index i when EITHER bar of the pair is
    # invalid — note the reported bar is i even when the invalid bar is i+1
    if n >= 3:
        pinv = inv[1 : n - 1] | inv[2:n]
        fi2  = 1 + int(pinv.argmax()) if pinv.any() else n
    else:
        fi2 = n

    dens = None                          # lazy: only built when a pattern candidate exists

    for i in map(int, np.flatnonzero(pair)):
        if i == 0:                       # breakout bar can never start the pair
            continue
        if i >= fi2:
            return None, None, fi2, None, None

        # --- 2-bar baseline from bars before bar i (excludes breakout bar at index 0) ---
        if dens is None:
            vol  = win.day.sell_vol[win.pos] if long_ else win.day.buy_vol[win.pos]
            dens = pair_densities(h, l, vol, tick_size)
        baseline = baseline_from_densities(dens, i, n_pairs, lowest_j=1)

        if np.isnan(baseline) or baseline <= 0:
            continue

        required = baseline * req_mult

        # --- build merged synthetic candle ---
        merged_open  = float(o[i])
        merged_high  = float(max(h[i], h[i + 1]))
        merged_low   = float(min(l[i], l[i + 1]))
        merged_close = float(c[i + 1])

        # --- absorption in defended wick of merged candle ---
        if long_:
            wick_low  = merged_low
            wick_high = min(merged_open, merged_close)
        else:
            wick_low  = max(merged_open, merged_close)
            wick_high = merged_high

        if wick_high <= wick_low:
            continue

        # absorption must register in the defended half of the full merged candle
        mid_price = (merged_low + merged_high) / 2.0

        merged = merge_tv(win.tv[i], win.tv[i + 1])

        absorbed       = False
        trigger_price  = None
        trigger_volume = None

        for price, (buy_qty, sell_qty) in merged.items():
            if not (wick_low <= price <= wick_high):
                continue
            if long_ and price > mid_price:
                continue
            if (not long_) and price < mid_price:
                continue
            volume_at_level = sell_qty if long_ else buy_qty
            if volume_at_level >= required:
                absorbed       = True
                trigger_price  = price
                trigger_volume = volume_at_level
                break

        if not absorbed:
            continue

        # --- confirmation candle scan starting from bar2 (inclusive) ---
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
                "absorption_time":  [win.ts(i).strftime("%H:%M"),
                                     win.ts(i + 1).strftime("%H:%M")],
                "two_bar_baseline": round(baseline, 2),
                "trigger_price":    trigger_price,
                "trigger_volume":   trigger_volume,
            }
            return entry_rel, float(o[entry_rel]), None, entry_notes, "two_bar_absorption"

    if fi2 < n:
        return None, None, fi2, None, None
    return _NO_ENTRY
