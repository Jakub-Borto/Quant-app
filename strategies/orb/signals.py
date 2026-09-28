"""Opening range and the first breakout after it."""

import numpy as np
import pandas as pd


def first_true(mask: np.ndarray) -> int:
    """Index of the first True, or -1."""
    return int(mask.argmax()) if mask.any() else -1


def opening_range(core, range_minutes: int):
    """(split, orb_high, orb_low), where `split` is the first bar AFTER the
    range — or None when the range is empty, covers the whole day, or has no
    height."""
    n = len(core.i8)
    range_end_ns = core.rth_start.value + pd.Timedelta(minutes=range_minutes).value
    split = int(core.i8.searchsorted(range_end_ns, side="left"))
    if split == 0 or split == n:
        return None
    orb_high = np.nanmax(core.high[:split])
    orb_low  = np.nanmin(core.low[:split])
    if orb_high - orb_low <= 0:
        return None
    return split, orb_high, orb_low


def first_breakout(core, split: int, orb_high: float, orb_low: float):
    """(direction, rel) for the first bar after the range that CLOSES beyond it
    with a body in the breakout direction (`rel` counts from `split`), or None.
    The day's last bar is excluded (the entry needs the next bar's open); long
    is checked first on each bar, and the earliest signal wins."""
    n = len(core.i8)
    cs, os_ = core.close[split:n - 1], core.open[split:n - 1]
    long_m  = (cs > orb_high) & (cs > os_)
    short_m = (cs < orb_low) & (cs < os_)
    rel = first_true(long_m | short_m)
    if rel == -1:
        return None
    return ("long" if long_m[rel] else "short"), rel
