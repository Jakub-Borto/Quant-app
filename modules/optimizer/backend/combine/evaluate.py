"""
Per-set metrics on one slice (IS or OOS): total ticks of the merged stream —
the only optimized number — plus daily Sharpe and max drawdown, displayed
alongside so a high-ticks set that is secretly a rollercoaster is visible.

Conventions match optimization/metrics.py: Sharpe = mean/std(ddof=1)×sqrt(252)
of per-day merged pnl over TRADED days, NaN when < 2 days or std == 0.
Max drawdown is peak-to-trough on the DAILY merged equity curve (consistent
with the Sharpe's daily aggregation), reported in ticks and as % of the
running peak (NaN when the curve never rises above 0).
"""

import math

import pandas as pd

from ..metrics import ANNUALIZATION_DAYS
from .merge import merge_streams, no_overlap_walk


def evaluate_set(member_tuple_lists: list) -> dict:
    """
    Merge the members' (sorted) tuple lists under the no-overlap rule and
    return {total_ticks, sharpe_daily, max_dd_ticks, max_dd_pct, n_trades,
    n_days, empty}.
    """
    streams = [s for s in member_tuple_lists if s]
    if not streams:
        return {"total_ticks": 0.0, "sharpe_daily": float("nan"),
                "max_dd_ticks": float("nan"), "max_dd_pct": float("nan"),
                "n_trades": 0, "n_days": 0, "empty": True}

    kept = no_overlap_walk(merge_streams(*streams))

    daily = {}
    total = 0.0
    for t in kept:
        total += t[4]
        daily[t[5]] = daily.get(t[5], 0.0) + t[4]
    series = pd.Series(daily).sort_index()

    if len(series) < 2:
        sharpe = float("nan")
    else:
        std = float(series.std(ddof=1))
        sharpe = float("nan") if std == 0 else \
            float(series.mean()) / std * math.sqrt(ANNUALIZATION_DAYS)

    equity  = series.cumsum()
    peak    = equity.cummax()
    dd      = equity - peak
    dd_ticks = float(dd.min()) if len(dd) else float("nan")
    # % relative to the running peak at the time of each trough
    valid  = peak > 0
    dd_pct = float((dd[valid] / peak[valid]).min() * 100.0) \
        if valid.any() else float("nan")

    return {"total_ticks": total, "sharpe_daily": sharpe,
            "max_dd_ticks": dd_ticks, "max_dd_pct": dd_pct,
            "n_trades": len(kept), "n_days": len(series), "empty": False}
