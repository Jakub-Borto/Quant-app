"""
Pure metric functions over a trades DataFrame.

Both entry points implement the SAME eight metrics and are used at build time
and at every heatmap filter recompute — no hidden state, no Streamlit:

  compute_metrics(trades)                 reference implementation, one dict
  compute_metrics_by_cell(trades, cols)   vectorized groupby over the swept-
                                          param columns, one row per grid cell

Invariant (tested): for every cell, compute_metrics_by_cell(...) row ==
compute_metrics(trades of that cell).

Inputs need `pnl_ticks` (float) and `date` columns. `be_band_ticks` defines a
break-even band around zero: win = pnl > be, breakeven = |pnl| <= be,
loss = pnl < -be. Default 0.0 = exact break-even.

NaN / inf conventions (callers exclude both from the color scale):
  profit_factor  inf when there are no gross losses (denominator 0)
  sharpe_trade   NaN when n < 2 or the per-trade std is 0
  sharpe_daily   NaN when < 2 trading days or the daily std is 0; the sqrt(252)
                 annualization is approximate once day types are filtered out
                 (fewer days) — comparative use only
"""

import math

import numpy as np
import pandas as pd

ANNUALIZATION_DAYS = 252

# The eight heatmap metrics, in UI order.
METRIC_ORDER = [
    "total_ticks",
    "total_trades",
    "avg_trade",
    "profit_factor",
    "win_rate",
    "win_rate_be",
    "sharpe_trade",
    "sharpe_daily",
]

METRIC_LABELS = {
    "total_ticks":   "Total Ticks",
    "total_trades":  "Total Trades",
    "avg_trade":     "Avg Trade (ticks)",
    "profit_factor": "Profit Factor",
    "win_rate":      "Win Rate %",
    "win_rate_be":   "Win Rate % (incl. BE)",
    "sharpe_trade":  "Sharpe (trade)",
    "sharpe_daily":  "Sharpe (daily, ann.)",
}


def compute_metrics(trades: pd.DataFrame, be_band_ticks: float = 0.0) -> dict:
    """The eight metrics for one set of trades (reference implementation)."""
    n = len(trades)
    if n == 0:
        return {
            "total_ticks":   0.0,
            "total_trades":  0,
            "avg_trade":     float("nan"),
            "profit_factor": float("nan"),
            "win_rate":      float("nan"),
            "win_rate_be":   float("nan"),
            "sharpe_trade":  float("nan"),
            "sharpe_daily":  float("nan"),
        }

    be  = float(be_band_ticks)
    pnl = trades["pnl_ticks"].astype(float)

    total      = float(pnl.sum())
    gross_win  = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    profit_factor = float("inf") if gross_loss == 0 else gross_win / gross_loss

    std = float(pnl.std(ddof=1)) if n >= 2 else float("nan")
    if n < 2 or not math.isfinite(std) or std == 0:
        sharpe_trade = float("nan")
    else:
        sharpe_trade = float(pnl.mean()) / std

    daily = pnl.groupby(pd.to_datetime(trades["date"]).dt.normalize()).sum()
    if len(daily) < 2:
        sharpe_daily = float("nan")
    else:
        daily_std = float(daily.std(ddof=1))
        if daily_std == 0:
            sharpe_daily = float("nan")
        else:
            sharpe_daily = float(daily.mean()) / daily_std * math.sqrt(ANNUALIZATION_DAYS)

    return {
        "total_ticks":   total,
        "total_trades":  n,
        "avg_trade":     total / n,
        "profit_factor": profit_factor,
        "win_rate":      100.0 * int((pnl > be).sum()) / n,
        "win_rate_be":   100.0 * int((pnl >= -be).sum()) / n,
        "sharpe_trade":  sharpe_trade,
        "sharpe_daily":  sharpe_daily,
    }


def compute_metrics_by_cell(trades: pd.DataFrame, cell_cols: list,
                            be_band_ticks: float = 0.0) -> pd.DataFrame:
    """
    The eight metrics per grid cell in ONE vectorized pass (no Python loop
    over cells). Returns a DataFrame indexed by `cell_cols` with METRIC_ORDER
    columns; cells with zero rows in `trades` simply don't appear — the
    heatmap reindexes onto the full grid and masks them.
    """
    if trades.empty:
        idx = pd.MultiIndex.from_arrays([[] for _ in cell_cols], names=cell_cols) \
            if len(cell_cols) > 1 else pd.Index([], name=cell_cols[0])
        return pd.DataFrame(columns=METRIC_ORDER, index=idx)

    be  = float(be_band_ticks)
    pnl = trades["pnl_ticks"].astype(float)

    work = trades[cell_cols].copy()
    work["_pnl"]        = pnl
    work["_gross_win"]  = pnl.clip(lower=0.0)
    work["_gross_loss"] = (-pnl).clip(lower=0.0)
    work["_is_win"]     = pnl > be
    work["_is_win_be"]  = pnl >= -be
    work["_date"]       = pd.to_datetime(trades["date"]).dt.normalize()

    agg = work.groupby(cell_cols, sort=True).agg(
        total_ticks=("_pnl", "sum"),
        total_trades=("_pnl", "size"),
        _std=("_pnl", "std"),            # pandas GroupBy.std is ddof=1
        _gross_win=("_gross_win", "sum"),
        _gross_loss=("_gross_loss", "sum"),
        _wins=("_is_win", "sum"),
        _wins_be=("_is_win_be", "sum"),
    )

    n = agg["total_trades"]
    out = pd.DataFrame(index=agg.index)
    out["total_ticks"]  = agg["total_ticks"]
    out["total_trades"] = n
    out["avg_trade"]    = agg["total_ticks"] / n

    no_loss = agg["_gross_loss"] == 0
    out["profit_factor"] = (agg["_gross_win"] / agg["_gross_loss"].where(~no_loss)) \
        .mask(no_loss, np.inf)

    out["win_rate"]    = 100.0 * agg["_wins"] / n
    out["win_rate_be"] = 100.0 * agg["_wins_be"] / n

    std = agg["_std"]
    out["sharpe_trade"] = (out["avg_trade"] / std).where((n >= 2) & (std > 0))

    daily = work.groupby(cell_cols + ["_date"], sort=True)["_pnl"].sum()
    by_cell    = daily.groupby(level=list(range(len(cell_cols))))
    daily_mean = by_cell.mean()
    daily_std  = by_cell.std(ddof=1)
    daily_n    = by_cell.size()
    sharpe_daily = (daily_mean / daily_std * math.sqrt(ANNUALIZATION_DAYS)) \
        .where((daily_n >= 2) & (daily_std > 0))
    out["sharpe_daily"] = sharpe_daily.reindex(out.index)

    return out[METRIC_ORDER]
