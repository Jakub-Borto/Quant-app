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

Sharpe definitions (both daily-aggregated, both annualized ×sqrt(252)):
  sharpe_trade   daily P&L over TRADED days only — days without a trade
                 simply don't exist for it
  sharpe_daily   daily P&L over EVERY business day between the first and last
                 (filtered) trade — days without trades count as 0

NaN / inf conventions (callers exclude both from the color scale):
  profit_factor  inf when there are no gross losses (denominator 0)
  sharpe_trade   NaN when < 2 traded days or the daily std is 0
  sharpe_daily   NaN when the business-day span is < 2 days or the
                 (zero-filled) variance is 0
The sqrt(252) annualization is approximate once day types are filtered out
(fewer days) — comparative use only.
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
    "sharpe_trade":  "Sharpe (traded days)",
    "sharpe_daily":  "Sharpe (daily, zero-filled)",
}


def _business_day_span(first, last, n_traded: int) -> int:
    """
    Business days from `first` to `last` inclusive — the zero-fill spine
    length for sharpe_daily. Never smaller than the number of traded days
    (weekend/holiday trades still count as days).
    """
    n = int(np.busday_count(np.datetime64(first, "D"),
                            np.datetime64(last, "D") + 1))
    return max(n, n_traded)


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

    daily = pnl.groupby(pd.to_datetime(trades["date"]).dt.normalize()).sum()
    k = len(daily)

    # traded days only
    if k < 2:
        sharpe_trade = float("nan")
    else:
        daily_std = float(daily.std(ddof=1))
        sharpe_trade = float("nan") if daily_std == 0 else \
            float(daily.mean()) / daily_std * math.sqrt(ANNUALIZATION_DAYS)

    # zero-filled over the business-day span (first -> last trade). Computed
    # from the traded-day sums alone: with N spine days, mean = S/N and
    # var(ddof=1) = (sum(d^2) - S^2/N) / (N-1) — the zeros contribute nothing.
    n_days = _business_day_span(daily.index.min(), daily.index.max(), k)
    if n_days < 2:
        sharpe_daily = float("nan")
    else:
        s     = float(daily.sum())
        var_z = (float((daily ** 2).sum()) - s * s / n_days) / (n_days - 1)
        sharpe_daily = float("nan") if var_z <= 0 else \
            (s / n_days) / math.sqrt(var_z) * math.sqrt(ANNUALIZATION_DAYS)

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

    ann = math.sqrt(ANNUALIZATION_DAYS)
    daily   = work.groupby(cell_cols + ["_date"], sort=True)["_pnl"].sum()
    levels  = list(range(len(cell_cols)))
    by_cell = daily.groupby(level=levels)

    # sharpe_trade: daily P&L over traded days only
    k         = by_cell.size()
    daily_std = by_cell.std(ddof=1)
    out["sharpe_trade"] = (by_cell.mean() / daily_std * ann) \
        .where((k >= 2) & (daily_std > 0)).reindex(out.index)

    # sharpe_daily: zero-filled over each cell's business-day span (same
    # closed-form as compute_metrics — the zero days contribute nothing)
    s      = by_cell.sum()
    sumsq  = (daily ** 2).groupby(level=levels).sum()
    d_min  = work.groupby(cell_cols, sort=True)["_date"].min()
    d_max  = work.groupby(cell_cols, sort=True)["_date"].max()
    n_days = np.busday_count(d_min.to_numpy().astype("datetime64[D]"),
                             d_max.to_numpy().astype("datetime64[D]")
                             + np.timedelta64(1, "D"))
    n_days = pd.Series(np.maximum(n_days, k.reindex(d_min.index).to_numpy()),
                       index=d_min.index)
    var_z  = (sumsq - s ** 2 / n_days) / (n_days - 1)
    out["sharpe_daily"] = ((s / n_days) / np.sqrt(var_z.where(var_z > 0)) * ann) \
        .where(n_days >= 2).reindex(out.index)

    return out[METRIC_ORDER]
