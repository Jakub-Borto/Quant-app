"""Metric functions vs hand-computed tiny trade sets (spec §6.3 / §14)."""

import math

import numpy as np
import pandas as pd
import pytest

from optimization.metrics import (
    ANNUALIZATION_DAYS, METRIC_ORDER, compute_metrics, compute_metrics_by_cell,
)


def trades_df(pnls, dates=None, **cells) -> pd.DataFrame:
    n = len(pnls)
    df = pd.DataFrame({"pnl_ticks": [float(p) for p in pnls]})
    df["date"] = pd.to_datetime(dates if dates is not None else ["2026-01-05"] * n)
    for col, values in cells.items():
        df[col] = values
    return df


def test_all_wins_multi_day():
    m = compute_metrics(trades_df([10, 20, 30],
                                  ["2026-01-05", "2026-01-06", "2026-01-07"]))
    assert m["total_ticks"] == 60
    assert m["total_trades"] == 3
    assert m["avg_trade"] == 20
    assert m["profit_factor"] == float("inf")      # zero losing trades
    assert m["win_rate"] == 100.0
    assert m["win_rate_be"] == 100.0
    # std([10,20,30], ddof=1) = 10
    assert m["sharpe_trade"] == pytest.approx(2.0)
    assert m["sharpe_daily"] == pytest.approx(2.0 * math.sqrt(ANNUALIZATION_DAYS))


def test_all_losses():
    m = compute_metrics(trades_df([-5, -5]))
    assert m["total_ticks"] == -10
    assert m["profit_factor"] == 0.0               # 0 gross win / 10 gross loss
    assert m["win_rate"] == 0.0
    assert m["win_rate_be"] == 0.0
    assert math.isnan(m["sharpe_trade"])           # std == 0
    assert math.isnan(m["sharpe_daily"])           # single day


def test_single_trade():
    m = compute_metrics(trades_df([7]))
    assert m["total_trades"] == 1
    assert m["avg_trade"] == 7
    assert m["profit_factor"] == float("inf")
    assert math.isnan(m["sharpe_trade"])           # n < 2
    assert math.isnan(m["sharpe_daily"])           # < 2 days


def test_breakeven_band():
    # be = 2: win = pnl > 2, breakeven = |pnl| <= 2, loss = pnl < -2
    m = compute_metrics(trades_df([3, 2, 0, -2, -3]), be_band_ticks=2.0)
    assert m["win_rate"] == pytest.approx(100 * 1 / 5)      # only the 3
    assert m["win_rate_be"] == pytest.approx(100 * 4 / 5)   # all but the -3
    # profit factor stays sign-based, not band-based
    assert m["profit_factor"] == pytest.approx((3 + 2) / (2 + 3))


def test_exact_breakeven_default_band():
    m = compute_metrics(trades_df([0, 5, -5]))
    assert m["win_rate"] == pytest.approx(100 / 3)          # 0 is not a win
    assert m["win_rate_be"] == pytest.approx(200 / 3)       # 0 counts with BE


def test_std_zero_sharpe_nan():
    m = compute_metrics(trades_df([5, 5, 5]))
    assert math.isnan(m["sharpe_trade"])
    assert m["win_rate"] == 100.0


def test_multi_day_daily_sharpe():
    df = trades_df([10, -5, 20, 5],
                   ["2026-01-05", "2026-01-05", "2026-01-06", "2026-01-07"])
    daily = pd.Series([5.0, 20.0, 5.0])
    expected = daily.mean() / daily.std(ddof=1) * math.sqrt(ANNUALIZATION_DAYS)
    assert compute_metrics(df)["sharpe_daily"] == pytest.approx(expected)


def test_daily_sharpe_equal_days_nan():
    # 2 days, identical daily pnl -> daily std 0 -> NaN (excluded from scale)
    df = trades_df([5, 5], ["2026-01-05", "2026-01-06"])
    assert math.isnan(compute_metrics(df)["sharpe_daily"])


def test_empty_trades():
    m = compute_metrics(trades_df([]))
    assert m["total_trades"] == 0
    assert m["total_ticks"] == 0.0
    for key in ("avg_trade", "profit_factor", "win_rate", "win_rate_be",
                "sharpe_trade", "sharpe_daily"):
        assert math.isnan(m[key])


# ── vectorized groupby == reference implementation, cell by cell ──────────────

def _assert_same(a: float, b: float):
    if isinstance(a, float) and math.isnan(a):
        assert isinstance(b, float) and math.isnan(b)
    else:
        assert a == pytest.approx(b)


@pytest.mark.parametrize("cell_cols", [["p1"], ["p1", "p2"]])
def test_by_cell_matches_reference(cell_cols):
    rng = np.random.default_rng(7)
    n = 200
    df = trades_df(
        rng.normal(0, 10, n).round(1),
        list(pd.to_datetime("2026-01-05") + pd.to_timedelta(rng.integers(0, 15, n), "D")),
        p1=rng.choice([1, 2, 3], n),
        p2=rng.choice([0.5, 1.0], n),
    )
    # inject degenerate cells: single trade / all-zero / zero-loss
    df.loc[df.index[:1], ["p1", "p2"]] = [9, 9.0]
    df.loc[df.index[:1], "pnl_ticks"] = 0.0

    by_cell = compute_metrics_by_cell(df, cell_cols, be_band_ticks=1.0)
    for cell, row in by_cell.iterrows():
        key = cell if isinstance(cell, tuple) else (cell,)
        mask = np.logical_and.reduce([df[c] == v for c, v in zip(cell_cols, key)])
        ref = compute_metrics(df[mask], be_band_ticks=1.0)
        for metric in METRIC_ORDER:
            _assert_same(float(row[metric]), float(ref[metric]))


def test_by_cell_empty():
    out = compute_metrics_by_cell(trades_df([], p1=[]), ["p1"])
    assert out.empty
    assert list(out.columns) == METRIC_ORDER
