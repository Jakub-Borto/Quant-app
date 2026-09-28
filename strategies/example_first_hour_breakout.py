"""
example_first_hour_breakout — the minimal single-file strategy from
STRATEGY_GUIDE.md (worked example 1). Copy it as a starting template.

Rules (one trade per day at most):
  1. The "first hour" is the RTH bars from 09:30 up to (not including)
     09:30 + range_minutes (NY time). Its high/low form the range.
  2. The first later bar that CLOSES above the range high is a long signal,
     below the range low a short signal. We enter at the NEXT bar's open
     (never at the signal bar's own close — that price is already history
     when the bar is known to have closed).
  3. Stop = the opposite side of the range, pushed stop_buffer_ticks further
     away. Target = entry +/- (entry-to-stop distance) x target_rr.
  4. If neither is hit, exit at the close of the last bar at or before
     exit_time. Within one bar the stop is checked first (pessimistic).
"""

import numpy as np
import pandas as pd

from modules.engine import Trade

PARAMS = {
    "tick_size":         0.25,     # auto-filled from ASSET_INFO (read-only in the UI)
    "range_minutes":     60,       # length of the opening range in minutes
    "stop_buffer_ticks": 2,        # extra ticks beyond the range for the stop
    "target_rr":         1.5,      # target distance = stop distance x this
    "exit_time":         "15:55",  # flat at the close of the last bar at/before this (NY)
}

PARAM_SECTIONS = {
    "Range": ["tick_size", "range_minutes"],
    "Risk":  ["stop_buffer_ticks", "target_rr", "exit_time"],
}

DATA = {"main": ["open", "high", "low", "close"]}


def process_day(day, params):
    candles = day.data["main"]                  # this day's bars, declared columns only
    tz = candles.index.tz                       # America/New_York
    d = day.date                                # datetime.date of the RTH session

    rth_open = pd.Timestamp(f"{d} 09:30", tz=tz)
    range_end = rth_open + pd.Timedelta(minutes=params["range_minutes"])
    exit_at = pd.Timestamp(f"{d} {params['exit_time']}", tz=tz)

    # Compare the index with Timestamps (works for ns- and us-unit indexes).
    idx = candles.index
    in_range = (idx >= rth_open) & (idx < range_end)
    after = (idx >= range_end) & (idx <= exit_at)
    if not in_range.any() or after.sum() < 2:
        return None                             # half day / missing data: no trade

    rng = candles[in_range]
    hi, lo = float(rng["high"].max()), float(rng["low"].min())

    bars = candles[after]
    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    l = bars["low"].to_numpy()
    c = bars["close"].to_numpy()

    # signal on a CLOSE beyond the range; the last bar can't signal (no next open)
    long_sig = c[:-1] > hi
    short_sig = c[:-1] < lo
    either = long_sig | short_sig
    if not either.any():
        return None
    s = int(np.argmax(either))                  # first signal bar
    direction = "long" if long_sig[s] else "short"

    e = s + 1                                   # entry bar = the bar after the signal
    entry = float(o[e])
    buffer = params["stop_buffer_ticks"] * params["tick_size"]
    if direction == "long":
        sl = lo - buffer
        tp = entry + (entry - sl) * params["target_rr"]
    else:
        sl = hi + buffer
        tp = entry - (sl - entry) * params["target_rr"]
    if (direction == "long" and sl >= entry) or (direction == "short" and sl <= entry):
        return None                             # entry already beyond the stop

    # walk the bars from the entry bar on; stop first within a bar
    for k in range(e, len(c)):
        if direction == "long":
            if l[k] <= sl:
                return _trade(direction, bars, e, k, entry, sl, "sl", sl, tp, hi, lo)
            if h[k] >= tp:
                return _trade(direction, bars, e, k, entry, tp, "tp", sl, tp, hi, lo)
        else:
            if h[k] >= sl:
                return _trade(direction, bars, e, k, entry, sl, "sl", sl, tp, hi, lo)
            if l[k] <= tp:
                return _trade(direction, bars, e, k, entry, tp, "tp", sl, tp, hi, lo)
    last = len(c) - 1
    return _trade(direction, bars, e, last, entry, float(c[last]), "time_exit", sl, tp, hi, lo)


def _trade(direction, bars, e, k, entry, exit_price, reason, sl, tp, hi, lo):
    return Trade(
        direction=direction,
        entry_time=bars.index[e],
        exit_time=bars.index[k],
        entry_price=entry,
        exit_price=float(exit_price),
        exit_reason=reason,
        sl=float(sl),
        tp=float(tp),
        trade_type="first_hour_breakout",
        notes={"range_high": hi, "range_low": lo, "bars_held": int(k - e + 1)},
    )
