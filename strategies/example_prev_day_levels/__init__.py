"""
example_prev_day_levels — the package strategy from STRATEGY_GUIDE.md
(worked example 2). Shows an additional data slot, prepare_day with
PREPARE_PARAMS, looking back over previous days, and rich notes.

Rules (one trade per day at most):
  1. Yesterday's RTH high/low are the levels (from day.previous).
  2. From `signal_after` on, the first bar that CLOSES above yesterday's high
     with CVD rising (and above RTH VWAP when require_vwap) is a long signal;
     below yesterday's low with CVD falling (and below VWAP) a short signal.
  3. Entry = the NEXT bar's open. Stop distance = the average RTH range of the
     last `lookback_days` days x stop_range_frac; target = stop distance x
     target_rr. Time exit at `exit_time`.
  4. Days without enough history (fewer than lookback_days usable previous
     days) are not traded.

Files:
  params.py   PARAMS / PARAM_SECTIONS / PARAMS_OPTIONS
  day.py      prepare_day's RthDay (cached by the engine)
  logic.py    signal detection + exit simulation -> Trade
"""

import math

from .day import build
from .logic import find_signal, simulate
from .params import PARAM_SECTIONS, PARAMS, PARAMS_OPTIONS

DATA = {
    "main":       ["open", "high", "low", "close"],
    "indicators": ["vwap_bar_rth", "cumulative_delta"],
}

PREPARE_PARAMS = ["rth_start"]


def prepare_day(date, data, params):
    return build(date, data["main"], data["indicators"], params["rth_start"])


def process_day(day, params):
    core = day.prepared
    # earlier days, oldest first; skip days missing an additional-data file
    # (reading them would raise) and days prepare_day rejected (None)
    history = [d.prepared for d in day.previous_days(params["lookback_days"])
               if not d.missing]
    history = [h for h in history if h is not None]
    if len(history) < params["lookback_days"]:
        return None                                      # not enough history yet
    prev = history[-1]                                   # the most recent usable day
    avg_range = sum(h.rth_range for h in history) / len(history)

    sig = find_signal(core, prev, params)
    if sig is None:
        return None
    s, direction = sig
    e = s + 1                                            # entry at the next bar's open
    entry = float(core.open[e])
    stop_dist = avg_range * params["stop_range_frac"]
    if stop_dist <= 0:
        return None
    # prices must sit on the instrument's tick grid: round the stop AWAY from
    # the entry (never tighter than intended) and the target TOWARD it
    tick = params["tick_size"]
    if direction == "long":
        sl = math.floor((entry - stop_dist) / tick) * tick
        tp = math.floor((entry + stop_dist * params["target_rr"]) / tick) * tick
    else:
        sl = math.ceil((entry + stop_dist) / tick) * tick
        tp = math.ceil((entry - stop_dist * params["target_rr"]) / tick) * tick

    notes = {
        "prev_date":     str(prev.index[0].date()),   # the day the levels come from
        "prev_high":     prev.rth_high,
        "prev_low":      prev.rth_low,
        "avg_range":     round(avg_range, 2),
        "signal_time":   core.index[s].strftime("%H:%M"),
        "vwap_at_entry": None if math.isnan(core.vwap[e]) else float(core.vwap[e]),
    }
    return simulate(core, e, direction, sl, tp, params, notes)


__all__ = ["DATA", "PREPARE_PARAMS", "PARAMS", "PARAM_SECTIONS", "PARAMS_OPTIONS",
           "prepare_day", "process_day"]
