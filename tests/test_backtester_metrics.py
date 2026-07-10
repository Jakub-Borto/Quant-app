"""
Backtester Sharpe regressions. Strategies return dates as strings (orb) or
date objects (ivb); the old code reindexed the raw-typed daily series against
a DatetimeIndex, which silently matched nothing — an all-zero series and a
0.00 daily Sharpe.
"""

import datetime as dt
import math

import pandas as pd
import pytest

# TEMPORARY (PyQt rebuild phase 1): views/ moved to legacy_streamlit/ and is
# frozen. compute_metrics gets a proper backend home in phase 3, when this
# import is pointed at modules.common.backend.trade_stats.
compute_metrics = pytest.importorskip(
    "views.backtester", reason="views frozen during PyQt rebuild (until phase 3)"
).compute_metrics

SQRT252 = math.sqrt(252)


def trades_df(pnls, dates) -> pd.DataFrame:
    df = pd.DataFrame({
        "ticks":     [float(p) for p in pnls],
        "date":      dates,
        "direction": ["long"] * len(pnls),
    })
    df["cumulative_ticks"] = df["ticks"].cumsum()
    return df


@pytest.mark.parametrize("dates", [
    ["2026-01-05", "2026-01-06", "2026-01-07"],                    # orb-style
    [dt.date(2026, 1, 5), dt.date(2026, 1, 6), dt.date(2026, 1, 7)],  # ivb-style
    pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
])
def test_sharpe_daily_not_zeroed_by_date_dtype(dates):
    m = compute_metrics(trades_df([10, -5, 20], dates))
    daily = pd.Series([10.0, -5.0, 20.0])
    expected = daily.mean() / daily.std(ddof=1) * SQRT252
    assert m["sharpe_daily"] == pytest.approx(expected)
    assert m["sharpe_trade"] == pytest.approx(expected)   # traded every day


def test_sharpe_definitions_differ_on_gap_days():
    # Mon, Tue, Fri: daily zero-fills Wed+Thu; traded-days does not
    m = compute_metrics(trades_df([10, 20, 30],
                                  ["2026-01-05", "2026-01-06", "2026-01-09"]))
    zero_filled = pd.Series([10.0, 20.0, 0.0, 0.0, 30.0])
    traded      = pd.Series([10.0, 20.0, 30.0])
    assert m["sharpe_daily"] == pytest.approx(
        zero_filled.mean() / zero_filled.std(ddof=1) * SQRT252)
    assert m["sharpe_trade"] == pytest.approx(
        traded.mean() / traded.std(ddof=1) * SQRT252)


def test_sharpe_single_day_is_zero():
    # backtester convention: undefined Sharpe displays as 0.0, not NaN
    m = compute_metrics(trades_df([10, -5], ["2026-01-05", "2026-01-05"]))
    assert m["sharpe_daily"] == 0.0
    assert m["sharpe_trade"] == 0.0
