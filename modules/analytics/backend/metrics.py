"""
Dollar-space metrics for the Analytics module.

Extracted verbatim from legacy_streamlit/views/analytics.py. All metric
helpers work on the dollar-denominated columns produced by the sizer
(trade_pnl, equity). Do NOT use the `ticks` column here — that's the raw
backtester output before sizing.

METRIC_REGISTRY is the single source of truth for the per-instance tile grid
and the comparison table: (key, label, formatter) triples. Add a metric here
and it appears everywhere automatically.
"""

import numpy as np
import pandas as pd

# Trading-day count used to annualize the Sharpe ratio. Standard convention
# for US equity/futures backtests.
TRADING_DAYS_PER_YEAR = 252

# Human labels for the day-type keys written into trades-file metadata by the
# backtester (DAY_TYPE_ORDER keys).
DAY_TYPE_LABELS = {
    "holiday":     "Holidays",
    "fomc":        "FOMC",
    "cpi":         "CPI",
    "nfp":         "Non-Farm Employment",
    "ppi":         "PPI",
    "high_impact": "Other High Impact News",
    "normal":      "Normal Trading Days",
}


def _taken_subset(trades: pd.DataFrame) -> pd.DataFrame:
    """
    The "taken" trades — rows the sizer actually allocated to (contracts > 0).

    Skipped trades are contracts-0 rows the sizer leaves in the frame; counting
    them as non-wins distorts win rate. Fallback: if there's no `contracts`
    column, treat every row as taken.
    """
    if "contracts" in trades.columns:
        return trades[trades["contracts"] > 0]
    return trades


def _sharpe_trade(trades: pd.DataFrame) -> float:
    """
    Per-trade Sharpe annualized by actual trading days — mirrors the
    backtester's trade-Sharpe: mean/std(ddof=1) * sqrt(n_trading_days).
    """
    pnl = trades["trade_pnl"]
    std = pnl.std(ddof=1)
    if len(pnl) <= 1 or not np.isfinite(std) or std == 0:
        return 0.0

    n_trading_days = trades["date"].nunique() if "date" in trades.columns else len(pnl)
    return float(pnl.mean() / std * np.sqrt(n_trading_days))


def _calmar(total_pnl: float, max_dd_dollars: float) -> float:
    """
    Cheap Calmar (not time-annualized): total P&L / |max drawdown $|.
    ∞ when there's no drawdown (max DD >= 0). Consistent with the backtester.
    """
    if max_dd_dollars < 0:
        return total_pnl / abs(max_dd_dollars)
    return float("inf")


def _max_drawdown(equity: pd.Series) -> tuple[float, float]:
    """
    Return (max_drawdown_dollars, max_drawdown_percent).

    Both are expressed as negative numbers (or zero). The percentage is
    computed against the running peak at each point, not the global peak —
    this is the standard definition and is what people usually expect when
    they see "max drawdown %".
    """
    if equity.empty:
        return 0.0, 0.0

    running_peak = equity.cummax()
    drawdown_abs = equity - running_peak
    max_dd_dollars = float(drawdown_abs.min())

    # Guard against div-by-zero if starting equity somehow was non-positive.
    dd_pct_series = np.where(
        running_peak > 0,
        drawdown_abs / running_peak * 100.0,
        0.0,
    )
    max_dd_pct = float(np.min(dd_pct_series))

    return max_dd_dollars, max_dd_pct


def _annualized_sharpe(trades: pd.DataFrame) -> float:
    """
    Sharpe ratio computed on daily PnL, annualized by sqrt(252).

    Preferred input: a 'date' column (day-grain, no timezone issues).
    Fallback: resample 'entry_time' to daily. This matters because a strategy
    might enter and exit within a single day, and we want one PnL sample per
    trading day for Sharpe to make sense.
    """
    if "date" in trades.columns:
        daily_pnl = trades.groupby("date")["trade_pnl"].sum()
    else:
        daily_pnl = trades.set_index("entry_time")["trade_pnl"].resample("1D").sum()

    # Need at least 2 observations and non-zero std for Sharpe to be defined.
    if len(daily_pnl) <= 1 or daily_pnl.std(ddof=1) == 0:
        return 0.0

    return float(
        daily_pnl.mean() / daily_pnl.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    )


def _win_rate_and_profit_factor(pnl: pd.Series) -> tuple[float, float]:
    """
    Return (win_rate, profit_factor).

    Profit factor edge cases:
    - Zero losses and at least one win  -> inf (well-defined edge case)
    - No trades at all                   -> 0.0 (avoids NaN in the table)
    """
    if pnl.empty:
        return 0.0, 0.0

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    win_rate = float(len(wins) / len(pnl))

    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())  # make loss positive for the ratio

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    return win_rate, profit_factor


# ── Metric registry — single source of truth for selector, blocks, table ──────
# Each entry: (key, label, formatter). The formatter turns a raw value into the
# string shown in per-instance metric tiles; the comparison table stores the
# rounded raw value (so it stays sortable). Add a metric here and it appears in
# the per-instance grid and the table automatically.

def _fmt_money(x):  return f"${x:,.2f}"
def _fmt_ratio(x):  return "∞" if x == float("inf") else f"{x:.2f}"
def _fmt_pct(x):    return f"{x:.1%}"          # win rate is a 0–1 fraction
def _fmt_pct_pts(x):return f"{x:.2f}%"         # max-dd % is already in percent
def _fmt_int(x):    return f"{int(x)}"

METRIC_REGISTRY = [
    ("final_equity",  "Final Equity ($)", _fmt_money),
    ("total",         "Total ($)",        _fmt_money),
    ("avg_trade",     "Avg Trade ($)",    _fmt_money),
    ("avg_win",       "Avg Win ($)",      _fmt_money),
    ("avg_loss",      "Avg Loss ($)",     _fmt_money),
    ("largest_win",   "Largest Win ($)",  _fmt_money),
    ("largest_loss",  "Largest Loss ($)", _fmt_money),
    ("sharpe_daily",  "Sharpe (daily)",   _fmt_ratio),
    ("sharpe_trade",  "Sharpe (trade)",   _fmt_ratio),
    ("profit_factor", "Profit Factor",    _fmt_ratio),
    ("calmar",        "Calmar",           _fmt_ratio),
    ("max_dd_dollars","Max Drawdown ($)", _fmt_money),
    ("max_dd_pct",    "Max Drawdown (%)", _fmt_pct_pts),
    ("max_peak",      "Max Peak ($)",     _fmt_money),
    ("total_trades",  "Total Trades",     _fmt_int),
    ("win_rate",      "Win Rate",         _fmt_pct),
    ("skipped",       "Skipped Trades",   _fmt_int),
    ("cost_drag",     "Cost Drag ($)",    _fmt_money),
]

# Convenience lookups.
_KEY_TO_LABEL     = {key: label for key, label, _f in METRIC_REGISTRY}
_KEY_TO_FORMATTER = {key: fmt for key, _label, fmt in METRIC_REGISTRY}

# The four equity curves the statistics can be computed on. The user picks one;
# the chart always shows all four. DEFAULT_CURVE = net (the cost-adjusted truth).
CURVE_LABELS  = ["Gross", "+ Commissions", "+ Slippage", "+ Both (net)"]
DEFAULT_CURVE = "+ Both (net)"


def _empty_metric_values(skipped: int = 0) -> dict:
    """Zero-filled values for every registry key — empty/all-skipped instances."""
    vals = {key: 0.0 for key, _l, _f in METRIC_REGISTRY}
    vals["total_trades"] = 0
    vals["skipped"]      = skipped
    return vals


def compute_metric_values(view: pd.DataFrame, gross_total: float) -> dict:
    """
    Raw value for every registry key, computed on a single curve's view frame
    (`trade_pnl`/`equity` set to the chosen curve — gross / +comm / +slip / net).
    `cost_drag = gross_total − curve_total` (0 for the gross curve, full Σcosts
    for net). Taken set (contracts > 0) drives total_trades / win_rate /
    profit_factor. Empty/zero-safe.
    """
    skipped = int(view.attrs.get("skipped_trades", 0))

    if view.empty or "trade_pnl" not in view.columns or "equity" not in view.columns:
        vals = _empty_metric_values(skipped)
        vals["cost_drag"] = round(float(gross_total), 2)   # no curve total → full drag
        return vals

    pnl    = view["trade_pnl"]
    equity = view["equity"]
    taken  = _taken_subset(view)

    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    max_dd_dollars, max_dd_pct = _max_drawdown(equity)
    win_rate, profit_factor    = _win_rate_and_profit_factor(taken["trade_pnl"])
    curve_total = float(pnl.sum())

    return {
        "final_equity":   round(float(equity.iloc[-1]), 2),
        "total":          round(curve_total, 2),
        "avg_trade":      round(float(pnl.mean()), 2),
        "avg_win":        round(float(wins.mean()), 2)   if len(wins)   else 0.0,
        "avg_loss":       round(float(losses.mean()), 2) if len(losses) else 0.0,
        "largest_win":    round(float(pnl.max()), 2),
        "largest_loss":   round(float(pnl.min()), 2),
        "sharpe_daily":   round(_annualized_sharpe(view), 2),
        "sharpe_trade":   round(_sharpe_trade(view), 2),
        "profit_factor":  (round(profit_factor, 2) if np.isfinite(profit_factor) else float("inf")),
        "calmar":         (round(_calmar(curve_total, max_dd_dollars), 2)
                           if np.isfinite(_calmar(curve_total, max_dd_dollars)) else float("inf")),
        "max_dd_dollars": round(max_dd_dollars, 2),
        "max_dd_pct":     round(max_dd_pct, 2),
        "max_peak":       round(float(equity.cummax().max()), 2),
        "total_trades":   int(len(taken)),
        "win_rate":       round(win_rate, 4),
        "skipped":        skipped,
        "cost_drag":      round(float(gross_total) - curve_total, 2),
    }


def _curve_frame(run: dict, curve_label: str) -> pd.DataFrame:
    """View frame for one curve: a copy of the sized trades with trade_pnl /
    equity replaced by the selected curve's series. Stats are computed on this."""
    trades = run["trades"]
    frame  = trades.copy()
    if not trades.empty and "trade_pnl" in trades.columns:
        pnl, equity = run["curves"][curve_label]
        frame["trade_pnl"] = pnl
        frame["equity"]    = equity
    return frame
