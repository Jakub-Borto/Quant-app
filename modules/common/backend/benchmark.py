"""
Market-exposure (alpha/beta) benchmark regression, shared by the Backtester
and Optimizer report panels.

Extracted verbatim from legacy_streamlit/views/trade_report.py. Regresses the
strategy's daily tick P&L against the traded asset's own daily move in ticks —
real alpha vs disguised beta. Sizing-agnostic, tick space. Four cells:
{days traded, all days} x {settlement move, RTH move}.

Change vs the old view: load_asset_statistics takes the parquet root
explicitly (`<data_root>/parquet`) instead of the hardcoded data/parquet.
"""

from pathlib import Path

import pandas as pd

_ALPHA_BETA_TOOLTIP = """\
**Alpha (α):** the strategy's return that is NOT explained by the benchmark — the part
of the edge independent of just being long/short the asset. Positive α = the strategy
generates tick-edge beyond market exposure. Reported in ticks/day (and annualized ×252).

**Beta (β):** how much the strategy's daily P&L moves with the asset's daily move.
β≈0 = no linear market exposure (market-neutral in direction); β≈1 = moves like
buy-and-hold; β<0 = moves opposite.

**Limitations:** in-sample — if the strategy was tuned on this data, α is optimistic by
construction and does not predict out-of-sample performance. Single-factor — tests
market-*direction* exposure only, not volatility, tail, or liquidity risk. A strategy
can show β≈0 here and still blow up on a vol spike.
"""

_TSTAT_TOOLTIP = """\
**t-stat = estimate ÷ its standard error** (signal-to-noise); |t| > ~2 ≈ "unlikely to
be noise."

**Days-traded (A) vs all-days (B):** going to B usually *shrinks* α and β (flat
zero-return days pull the line toward flat) but *balloons* the day count. More days
shrink the standard error, which can inflate the t-stat — and plain OLS treats the
zero-variance flat days as if each were a real independent observation, badly
overstating B's significance. That's why these use **HAC (Newey-West) standard
errors**, which down-weight the empty/autocorrelated days. Trust only the HAC t-stat
shown here; a naive-OLS t-stat (especially for B) is flattering you with empty days.

**Settlement vs RTH:** settlement spans the overnight gap you didn't hold; RTH is only
your in-market hours. Comparing the two shows whether overnight movement is distorting
the exposure estimate.
"""


def load_asset_statistics(parquet_root: Path, asset: str) -> pd.DataFrame | None:
    """Per-asset daily statistics file, or None when unavailable.
    `parquet_root` is `<data_root>/parquet` (path shape below unchanged)."""
    path = Path(parquet_root) / "Futures" / asset / f"{asset}_statistics" / "statistics.parquet"
    if not path.exists():
        return None
    stats = pd.read_parquet(path)
    stats["date"] = pd.to_datetime(stats["date"]).dt.normalize()
    return stats.sort_values("date").reset_index(drop=True)


def build_benchmark_series(stats: pd.DataFrame, tick_size: float) -> dict:
    """
    Build the two benchmark daily-move series (in ticks) with exclusions applied.

    - settle: settle_px.diff() over the previous ROW (previous available trading
      day, not a calendar shift), where settle_px = settlement when
      settle_is_actual else globex_close (never rth_close — different mark).
    - rth: rth_close − rth_open, same-day only.
    - Roll days are excluded from BOTH series: prices jump old→new contract,
      a discontinuity, not a market move. One uncaught roll can dominate the fit.
      (Symbol changes always fall on flagged rows; a symbol-change guard is
      included as insurance.)

    Returns {"settle": Series, "rth": Series (both date-indexed),
             "n_roll", "n_missing_settle", "n_missing_rth"}.
    """
    s = stats
    is_roll = s["is_roll_day"].fillna(False).astype(bool)
    # Insurance: any cross-contract diff is invalid even if somehow unflagged.
    symbol_switch = (s["symbol"] != s["symbol"].shift(1)).fillna(False)
    symbol_switch.iloc[0] = False

    actual    = s["settle_is_actual"].fillna(False).astype(bool) & s["settlement"].notna()
    settle_px = s["settlement"].where(actual, s["globex_close"])
    settle_move = settle_px.diff() / tick_size
    settle_move[symbol_switch] = float("nan")

    rth_move = (s["rth_close"] - s["rth_open"]) / tick_size

    settle_valid = settle_move.notna() & ~is_roll
    rth_valid    = rth_move.notna()    & ~is_roll

    return {
        "settle": pd.Series(settle_move[settle_valid].values,
                            index=s.loc[settle_valid, "date"]),
        "rth":    pd.Series(rth_move[rth_valid].values,
                            index=s.loc[rth_valid, "date"]),
        "n_roll":           int(is_roll.sum()),
        "n_missing_settle": int((settle_move.isna() & ~is_roll).sum()),
        "n_missing_rth":    int((rth_move.isna() & ~is_roll).sum()),
    }


def _fit_alpha_beta(y: pd.Series, x: pd.Series) -> dict | None:
    """
    OLS of strategy_ticks ~ const + benchmark_ticks with HAC (Newey-West)
    standard errors, maxlags=5 (~one trading week). Returns None when the fit
    is undefined (< 3 rows or zero benchmark variance).
    """
    if len(y) < 3 or len(x) < 3 or float(x.std(ddof=0)) == 0.0:
        return None

    import statsmodels.api as sm
    X   = sm.add_constant(x.to_numpy(dtype=float))
    res = sm.OLS(y.to_numpy(dtype=float), X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    return {
        "alpha":     float(res.params[0]),
        "alpha_ann": float(res.params[0]) * 252,
        "beta":      float(res.params[1]),
        "t_alpha":   float(res.tvalues[0]),
        "t_beta":    float(res.tvalues[1]),
        "r2":        float(res.rsquared),
        "n":         int(res.nobs),
    }


def _regression_cell_md(res: dict | None) -> str:
    """Compact per-cell markdown table (or an insufficient-data note)."""
    if res is None:
        return "*insufficient data*"
    return (
        "| | |\n|---|---|\n"
        f"| α (ticks/day) | {res['alpha']:.2f} |\n"
        f"| α annualized | {res['alpha_ann']:.0f} |\n"
        f"| β | {res['beta']:.3f} |\n"
        f"| t(α) HAC | {res['t_alpha']:.2f} |\n"
        f"| t(β) HAC | {res['t_beta']:.2f} |\n"
        f"| R² | {res['r2']:.3f} |\n"
        f"| n | {res['n']} |"
    )


def market_exposure_data(trades: pd.DataFrame, stats: pd.DataFrame,
                         tick_size: float) -> dict | None:
    """
    The full 2x2 regression grid on the FILTERED trades (the computation
    previously inline in render_market_exposure). Returns None when the stats
    file has no overlap with the backtest window; else
    {"bench": ..., "n_absent": int, "cells": {(sample_label, bench_label): fit}}.
    """
    # Strategy daily tick P&L — same aggregation as the daily Sharpe.
    daily = trades.groupby("date")["ticks"].sum()
    daily.index = pd.to_datetime(daily.index).normalize()
    daily = daily.sort_index()

    # Restrict the stats spine to the backtest window (Sharpe convention).
    lo, hi = daily.index.min(), daily.index.max()
    stats  = stats[(stats["date"] >= lo) & (stats["date"] <= hi)]
    if stats.empty:
        return None

    bench = build_benchmark_series(stats, tick_size)

    # Traded dates entirely absent from the stats file (never regressed
    # against a phantom benchmark).
    stats_dates = set(stats["date"])
    n_absent    = int(sum(d not in stats_dates for d in daily.index))

    cells = {}
    for sample_label, all_days in [("Days traded", False), ("All days", True)]:
        for bench_label, series in [("Settlement", bench["settle"]),
                                    ("RTH", bench["rth"])]:
            if all_days:
                # Stats-file business-day spine, strategy zero-filled —
                # matches the daily-Sharpe zero-fill convention.
                x = series
                y = daily.reindex(series.index, fill_value=0.0)
            else:
                common = daily.index.intersection(series.index)
                y = daily.loc[common]
                x = series.loc[common]
            cells[(sample_label, bench_label)] = _fit_alpha_beta(y, x)

    return {"bench": bench, "n_absent": n_absent, "cells": cells}
