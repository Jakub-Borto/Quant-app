"""
Trade statistics shared by the Backtester and the Optimizer's cell drill-down.

Extracted verbatim from legacy_streamlit/views/trade_report.py — the pure
computation behind the report panel: the full metric suite, the day-type
catalogue, the news/holiday breakdown rows, the exit-breakdown table and the
RR-distribution series/bins. Rendering lives in modules/common/ui/trade_report/.

Every function expects backtester-shaped trades: a `ticks` column (the
optimizer aliases pnl_ticks), `cumulative_ticks` for equity/drawdown,
normalized-able `date`, lowercase `direction`, and the standard strategy
columns (entry/exit prices & times, sl/tp, exit_reason, pnl_points).
"""

import math

import pandas as pd

# Day-type categories in precedence order — drives the filter UIs and the
# news/holiday breakdown table in both modules.
DAY_TYPE_ORDER = [
    ("holiday",     "Holidays"),
    ("fomc",        "FOMC"),
    ("cpi",         "CPI"),
    ("nfp",         "Non-Farm Employment"),
    ("ppi",         "PPI"),
    ("high_impact", "Other High Impact News"),
    ("normal",      "Normal Trading Days"),
]


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(trades: pd.DataFrame) -> dict:
    winning   = trades[trades["ticks"] > 0]
    losing    = trades[trades["ticks"] < 0]
    breakeven = trades[trades["ticks"] == 0]

    avg_win   = winning["ticks"].mean() if len(winning) > 0 else 0.0
    avg_loss  = losing["ticks"].mean()  if len(losing)  > 0 else 0.0
    win_rate  = len(winning) / len(trades)
    loss_rate = len(losing)  / len(trades)

    gross_wins   = winning["ticks"].sum()
    gross_losses = abs(losing["ticks"].sum())
    if gross_losses > 0:
        profit_factor = gross_wins / gross_losses
    elif gross_wins > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    # Both Sharpes are daily-aggregated and annualized ×sqrt(252). Dates are
    # normalized to Timestamps first: strategies return strings (orb) or date
    # objects (ivb), and reindexing a string-keyed daily series against a
    # DatetimeIndex silently matches nothing — all-zero series, Sharpe 0.
    dates        = pd.to_datetime(trades["date"]).dt.normalize()
    traded_daily = trades.groupby(dates)["ticks"].sum()

    # Daily Sharpe — zero-fill every business day between first & last trade
    # (union keeps any weekend/holiday traded day in the spine too)
    spine     = pd.bdate_range(dates.min(), dates.max()).union(traded_daily.index)
    daily_pnl = traded_daily.reindex(spine, fill_value=0)
    daily_std = daily_pnl.std(ddof=1)
    sharpe_daily = (daily_pnl.mean() / daily_std) * (252 ** 0.5) if daily_std > 0 else 0.0

    # Trade Sharpe — daily P&L over TRADED days only (no zero-fill)
    traded_std   = traded_daily.std(ddof=1)
    sharpe_trade = (traded_daily.mean() / traded_std) * (252 ** 0.5) \
        if len(traded_daily) >= 2 and traded_std > 0 else 0.0

    # Equity curve / drawdown
    cumulative   = trades["cumulative_ticks"]
    rolling_max  = cumulative.cummax()
    drawdown     = cumulative - rolling_max
    max_drawdown = drawdown.min()

    total       = trades["ticks"].sum()
    global_peak = rolling_max.max()
    calmar      = total / abs(max_drawdown) if max_drawdown < 0 else float("inf")

    # ── Risk/reward ───────────────────────────────────────────────────────────
    # Planned RR (target geometry): abs(tp - entry) / abs(sl - entry).
    # Realised RR (signed R-multiple): pnl_points / abs(entry - sl) — a full
    # target reads +planned_RR, a full stop reads -1.0, scratches scale between.
    # Both use the same valid rows: stop distance > 0.
    if all(c in trades.columns for c in ["entry_price", "sl"]):
        sl_dist = (trades["entry_price"] - trades["sl"]).abs()
        valid   = sl_dist > 0
    else:
        sl_dist = None
        valid   = None

    if sl_dist is not None and "tp" in trades.columns:
        tp_dist           = (trades["tp"] - trades["entry_price"]).abs()
        rr_planned        = (tp_dist / sl_dist)[valid]
        avg_rr_planned    = rr_planned.mean()    if len(rr_planned) > 0 else None
        median_rr_planned = rr_planned.median()  if len(rr_planned) > 0 else None
    else:
        avg_rr_planned = median_rr_planned = None

    if sl_dist is not None and "pnl_points" in trades.columns:
        rr_realised        = (trades["pnl_points"] / sl_dist)[valid]
        avg_rr_realised    = rr_realised.mean()   if len(rr_realised) > 0 else None
        median_rr_realised = rr_realised.median() if len(rr_realised) > 0 else None
    else:
        avg_rr_realised = median_rr_realised = None

    # Consecutive wins / losses
    results = (trades["ticks"] > 0).astype(int).tolist()
    max_consec_wins = max_consec_losses = cur_wins = cur_losses = 0
    for r in results:
        if r == 1:
            cur_wins  += 1
            cur_losses = 0
        else:
            cur_losses += 1
            cur_wins   = 0
        max_consec_wins   = max(max_consec_wins,   cur_wins)
        max_consec_losses = max(max_consec_losses, cur_losses)

    # Trade duration
    if "entry_time" in trades.columns and "exit_time" in trades.columns:
        durations           = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 60
        avg_duration_min    = durations.mean()
        median_duration_min = durations.median()
    else:
        avg_duration_min = median_duration_min = None

    long_trades  = trades[trades["direction"] == "long"]
    short_trades = trades[trades["direction"] == "short"]

    return {
        "total_ticks":          total,
        "avg_trade":            trades["ticks"].mean(),
        "avg_win":              avg_win,
        "avg_loss":             avg_loss,
        "largest_win":          winning["ticks"].max() if len(winning) > 0 else 0,
        "largest_loss":         losing["ticks"].min()  if len(losing)  > 0 else 0,
        "avg_rr_planned":       avg_rr_planned,
        "median_rr_planned":    median_rr_planned,
        "avg_rr_realised":      avg_rr_realised,
        "median_rr_realised":   median_rr_realised,
        "total_trades":         len(trades),
        "win_rate":             win_rate,
        "loss_rate":            loss_rate,
        "breakeven_rate":       len(breakeven) / len(trades),
        "sharpe_daily":         sharpe_daily,
        "sharpe_trade":         sharpe_trade,
        "profit_factor":        profit_factor,
        "calmar":               calmar,
        "max_drawdown":         max_drawdown,
        "max_peak":             global_peak,
        "max_consec_wins":      max_consec_wins,
        "max_consec_losses":    max_consec_losses,
        "avg_duration_min":     avg_duration_min,
        "median_duration_min":  median_duration_min,
        "long_trades":          len(long_trades),
        "short_trades":         len(short_trades),
        "long_winrate":         (long_trades["ticks"] > 0).mean()  if len(long_trades)  > 0 else 0.0,
        "short_winrate":        (short_trades["ticks"] > 0).mean() if len(short_trades) > 0 else 0.0,
    }


# ── Exit breakdown ────────────────────────────────────────────────────────────

def exit_breakdown_table(trades: pd.DataFrame) -> pd.DataFrame:
    """Per-exit-reason count / avg / total ticks table (from render_metrics)."""
    exit_stats = trades.groupby("exit_reason")["ticks"].agg(
        count="count", avg="mean", total="sum"
    ).reset_index()
    exit_stats.columns        = ["Exit Reason", "Count", "Avg Ticks", "Total Ticks"]
    exit_stats["Avg Ticks"]   = exit_stats["Avg Ticks"].round(1)
    exit_stats["Total Ticks"] = exit_stats["Total Ticks"].round(0).astype(int)
    return exit_stats


# ── News & holiday breakdown ─────────────────────────────────────────────────

def news_holiday_rows(trades: pd.DataFrame) -> list[dict] | None:
    """
    Rows for the News & Holiday Exposure table — computed from the
    trade_type-filtered trades but unaffected by the day_type filter. Rows are
    driven by DAY_TYPE_ORDER so every day-type category (incl. the carved-out
    event categories) appears. None when trades carry no day_type column.
    """
    if "day_type" not in trades.columns:
        return None

    rows = []
    for tag, label in DAY_TYPE_ORDER:
        subset = trades[trades["day_type"] == tag]
        n      = len(subset)
        if n == 0:
            rows.append({
                "Day Type":    label,
                "Trades":      0,
                "Wins":        0,
                "Losses":      0,
                "Win Rate":    "N/A",
                "Total Ticks": 0,
                "Avg Ticks":   "N/A",
            })
        else:
            wins   = (subset["ticks"] > 0).sum()
            losses = (subset["ticks"] < 0).sum()
            rows.append({
                "Day Type":    label,
                "Trades":      n,
                "Wins":        wins,
                "Losses":      losses,
                "Win Rate":    f"{wins / n:.1%}",
                "Total Ticks": int(round(subset["ticks"].sum())),
                "Avg Ticks":   f"{subset['ticks'].mean():.1f}",
            })
    return rows


# ── RR distribution ───────────────────────────────────────────────────────────

def rr_distribution_series(trades: pd.DataFrame) -> dict | None:
    """
    The four RR histogram series. Same formulas as compute_metrics:
      planned  = |tp - entry| / |entry - sl|
      realised = pnl_points   / |entry - sl|
    over rows with stop distance > 0. Returns None when the required columns
    are missing or both main series are empty.
    """
    if not all(c in trades.columns for c in ["entry_price", "sl"]):
        return None

    sl_dist = (trades["entry_price"] - trades["sl"]).abs()
    valid   = sl_dist > 0

    planned  = ((trades["tp"] - trades["entry_price"]).abs() / sl_dist)[valid] \
        if "tp" in trades.columns else None
    realised = (trades["pnl_points"] / sl_dist)[valid] \
        if "pnl_points" in trades.columns else None

    # Realised RR of WINNERS only — trades that went to profit (realised > 0);
    # a planned-3 trade that only banked +1 R shows here as 1; SL hits excluded.
    won = realised[realised > 0]  if realised is not None else None
    # Break-even trades — realised exactly 0 R (their own column at x=0).
    be  = realised[realised == 0] if realised is not None else None

    have_planned  = planned  is not None and len(planned)  > 0
    have_realised = realised is not None and len(realised) > 0
    if not have_planned and not have_realised:
        return None

    return {"planned": planned, "realised": realised, "won": won, "be": be}


def rr_bin_edges(series: dict, w: float) -> tuple[float, float]:
    """
    Shared histogram bin range so all series line up (from
    render_rr_distribution). Widened slightly past the data:
    one extra bin of headroom on the right edge.
    """
    values = pd.concat([s for s in (series["planned"], series["realised"])
                        if s is not None])
    lo, hi = float(values.min()), float(values.max())
    start  = math.floor(lo / w) * w
    end    = math.ceil(hi / w) * w + w   # +1 bin of headroom on the right edge
    return start, end
