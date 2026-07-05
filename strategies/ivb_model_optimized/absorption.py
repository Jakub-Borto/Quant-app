"""Shared absorption grading used by all absorption-based entry finders.

Vectorized split of the original is_absorption_candle / find_absorption_trigger pair:

  - The scalar prechecks (valid baseline, bar range > 0, defended wick fraction >=
    wick_threshold) are done VECTORIZED by the callers over the whole window
    (see EntryWindow.wick_frac / abs_base in _daydata).
  - absorption_scan() does the remaining per-bar work on the pre-parsed tick_volume
    arrays: find the FIRST price level inside the defended wick whose aggressive volume
    against the trade direction crosses baseline * absorption_mult. "First" is JSON
    document order — exactly the dict-iteration order the original used — so it doubles
    as find_absorption_trigger (same scan, same first hit).
"""

import numpy as np


def absorption_scan(tv, wick_low: float, wick_high: float, required: float, direction: str):
    """Returns (found, trigger_price, trigger_volume) for the first level in
    [wick_low, wick_high] with counter-volume >= required. tv is the pre-parsed
    (prices, buys, sells) tuple or None."""
    if tv is None:
        return False, None, None
    prices, buys, sells = tv
    vols = sells if direction == "long" else buys
    hit  = (prices >= wick_low) & (prices <= wick_high) & (vols >= required)
    if not hit.any():
        return False, None, None
    k = int(hit.argmax())
    return True, float(prices[k]), vols[k].item()


def wick_bounds(o: float, h: float, l: float, c: float, direction: str) -> tuple:
    """Defended-wick price bounds of a bar: (low, body_bottom) long / (body_top, high) short."""
    if direction == "long":
        return l, min(o, c)
    return max(o, c), h
