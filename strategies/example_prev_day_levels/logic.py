"""The per-day trading logic: previous-day high/low breakout, VWAP- and
CVD-filtered, with a stop sized by the average RTH range of the last days."""

import numpy as np
import pandas as pd

from modules.engine import Trade


def find_signal(core, prev, params):
    """(bar_index, direction) of the first bar at/after `signal_after` that
    closes beyond yesterday's RTH high (long) / low (short) and passes the
    filters — or None. The last bar can't signal (the entry is the NEXT open)."""
    tz = core.index.tz
    date = core.index[0].date()
    t0 = pd.Timestamp(f"{date} {params['signal_after']}", tz=tz)
    t_exit = pd.Timestamp(f"{date} {params['exit_time']}", tz=tz)
    allow_long = params["direction_filter"] in ("both", "long_only")
    allow_short = params["direction_filter"] in ("both", "short_only")

    for i in range(1, len(core.close) - 1):
        t = core.index[i]
        if t < t0:
            continue
        if t >= t_exit:
            return None
        c, v = core.close[i], core.vwap[i]
        cvd_up = core.cvd[i] > core.cvd[i - 1]
        if allow_long and c > prev.rth_high and cvd_up \
                and (not params["require_vwap"] or (not np.isnan(v) and c > v)):
            return i, "long"
        if allow_short and c < prev.rth_low and not cvd_up \
                and (not params["require_vwap"] or (not np.isnan(v) and c < v)):
            return i, "short"
    return None


def simulate(core, e, direction, sl, tp, params, notes):
    """Walk from the entry bar `e`: stop first within a bar, then target, then
    the time exit at the close of the last bar at/before `exit_time`."""
    tz = core.index.tz
    date = core.index[0].date()
    t_exit = pd.Timestamp(f"{date} {params['exit_time']}", tz=tz)
    entry = float(core.open[e])
    last = e
    for k in range(e, len(core.close)):
        if core.index[k] > t_exit:
            break
        last = k
        if direction == "long":
            if core.low[k] <= sl:
                return _trade(core, e, k, direction, entry, sl, "sl", sl, tp, notes)
            if core.high[k] >= tp:
                return _trade(core, e, k, direction, entry, tp, "tp", sl, tp, notes)
        else:
            if core.high[k] >= sl:
                return _trade(core, e, k, direction, entry, sl, "sl", sl, tp, notes)
            if core.low[k] <= tp:
                return _trade(core, e, k, direction, entry, tp, "tp", sl, tp, notes)
    return _trade(core, e, last, direction, entry, float(core.close[last]),
                  "time_exit", sl, tp, notes)


def _trade(core, e, k, direction, entry, exit_price, reason, sl, tp, notes):
    return Trade(
        direction=direction,
        entry_time=core.index[e],
        exit_time=core.index[k],
        entry_price=entry,
        exit_price=float(exit_price),
        exit_reason=reason,
        sl=float(sl),
        tp=float(tp),
        trade_type=f"prev_day_{'high' if direction == 'long' else 'low'}_break",
        notes={**notes, "bars_held": int(k - e + 1)},
    )
