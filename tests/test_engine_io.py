"""
Grid engine + run persistence: combo enumeration/injection, enrichment,
golden heatmap on a toy deterministic strategy, round-trip / partition
invariants, save/load (spec §14).
"""

import math

import numpy as np
import pandas as pd
import pytest

from optimization.engine import (
    check_param_columns, median_split_date, run_grid,
)
from optimization.io import list_runs, load_run, save_run
from optimization.metrics import METRIC_ORDER, compute_metrics, compute_metrics_by_cell

TICKS_PER_POINT = 4
DAYS = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]


class ToyStrategy:
    """
    Deterministic, analytically known: `b` trades on the first `b` DAYS, each
    with pnl_points == a. a == 99 -> zero trades (masked-cell path).
    Records every params dict it is called with.
    """
    PARAMS = {"a": 1, "b": 2, "hold": "x"}

    def __init__(self):
        self.calls = []

    def run(self, folder_path, start_date, end_date, params):
        self.calls.append(dict(params))
        a, b = params["a"], params["b"]
        if a == 99:
            return pd.DataFrame()
        rows = []
        for day in DAYS[:b]:
            rows.append({
                "date":        day,
                "direction":   "long",
                "entry_time":  pd.Timestamp(f"{day} 10:00", tz="America/New_York"),
                "exit_time":   pd.Timestamp(f"{day} 11:00", tz="America/New_York"),
                "entry_price": 100.0,
                "exit_price":  100.0 + a,
                "sl":          99.0,
                "tp":          100.0 + a,
                "exit_reason": "tp",
                "pnl_points":  float(a),
            })
        return pd.DataFrame(rows)


AXES = [
    {"param": "a", "values": [1, 2, 99], "role": "x"},
    {"param": "b", "values": [2, 3],     "role": "y"},
]
BUCKET_MAP = {"2026-01-05": "cpi", "2026-01-08": "holiday"}


def grid_run(strategy=None, axes=AXES, bucket_map=BUCKET_MAP, on_progress=None):
    strategy = strategy or ToyStrategy()
    trades = run_grid(
        strategy, "unused_folder", "2026-01-01", "2026-12-31",
        base_params=dict(ToyStrategy.PARAMS), axes=axes,
        tick_size=0.25, ticks_per_point=TICKS_PER_POINT,
        bucket_map=bucket_map, on_progress=on_progress,
    )
    return strategy, trades


def test_combo_count_and_injection():
    strategy, trades = grid_run()
    assert len(strategy.calls) == 3 * 2                       # product of sizes
    assert all(p["tick_size"] == 0.25 for p in strategy.calls)
    assert all(p["hold"] == "x" for p in strategy.calls)      # held param passed
    swept = [(p["a"], p["b"]) for p in strategy.calls]
    assert swept == [(1, 2), (1, 3), (2, 2), (2, 3), (99, 2), (99, 3)]


def test_enrichment_and_zero_trade_cells():
    _, trades = grid_run()
    # 4 non-empty cells: b=2 -> 2 trades, b=3 -> 3 trades; a=99 -> zero rows
    assert len(trades) == 2 + 3 + 2 + 3
    assert not ((trades["a"] == 99).any())
    assert (trades["pnl_ticks"] == trades["pnl_points"] * TICKS_PER_POINT).all()
    by_day = dict(zip(trades["date"].dt.strftime("%Y-%m-%d"), trades["day_bucket"]))
    assert by_day["2026-01-05"] == "cpi"
    assert by_day["2026-01-06"] == "normal"                   # unlisted date
    # swept params are the leading columns (cell identity)
    assert list(trades.columns[:2]) == ["a", "b"]


def test_progress_stream():
    seen = []
    grid_run(on_progress=lambda cur, total, msg: seen.append((cur, total)))
    assert seen == [(i, 6) for i in range(1, 7)]


def test_golden_heatmap():
    _, trades = grid_run()
    grid = compute_metrics_by_cell(trades, ["a", "b"])
    for (a, b) in [(1, 2), (1, 3), (2, 2), (2, 3)]:
        row = grid.loc[(a, b)]
        assert row["total_trades"] == b
        assert row["total_ticks"] == pytest.approx(a * b * TICKS_PER_POINT)
        assert row["avg_trade"] == pytest.approx(a * TICKS_PER_POINT)
        assert row["win_rate"] == 100.0
        assert row["profit_factor"] == float("inf")           # no losses
        assert math.isnan(row["sharpe_trade"])                # identical pnls
    assert (99, 2) not in grid.index                          # zero rows


def test_round_trip_invariant():
    # recompute with ALL buckets and BOTH halves selected == build-time metrics
    _, trades = grid_run()
    build_time = compute_metrics_by_cell(trades, ["a", "b"])

    filtered = trades[trades["day_bucket"].isin(
        ["holiday", "fomc", "cpi", "nfp", "ppi", "other_high_impact", "normal"])]
    split = median_split_date(trades)
    dates = pd.to_datetime(filtered["date"])
    both = pd.concat([filtered[dates <= split], filtered[dates > split]])
    recomputed = compute_metrics_by_cell(both.sort_index(), ["a", "b"])

    pd.testing.assert_frame_equal(build_time, recomputed)


def test_partition_invariant():
    # 1st ∪ 2nd == both, disjoint, no gaps
    _, trades = grid_run()
    split = median_split_date(trades)
    dates = pd.to_datetime(trades["date"])
    first, second = trades[dates <= split], trades[dates > split]
    assert len(first) + len(second) == len(trades)
    assert set(first.index).isdisjoint(set(second.index))
    assert set(first["date"]).isdisjoint(set(second["date"]))
    assert not first.empty and not second.empty


def test_median_split_date():
    def days_df(days):
        return pd.DataFrame({"date": pd.to_datetime(days)})
    assert median_split_date(days_df(DAYS)) == pd.Timestamp("2026-01-06")   # 4 days -> 2|2
    assert median_split_date(days_df(DAYS[:3])) == pd.Timestamp("2026-01-06")  # 3 -> 2|1
    assert median_split_date(pd.DataFrame({"date": []})) is None


def test_determinism():
    _, t1 = grid_run()
    _, t2 = grid_run()
    pd.testing.assert_frame_equal(t1, t2)


def test_param_column_collision():
    with pytest.raises(ValueError):
        check_param_columns([{"param": "date", "values": [1], "role": "x"}])


def test_all_empty_grid():
    _, trades = grid_run(axes=[{"param": "a", "values": [99], "role": "x"},
                               {"param": "b", "values": [2], "role": "y"}])
    assert trades.empty
    assert "pnl_ticks" in trades.columns and "day_bucket" in trades.columns
    assert median_split_date(trades) is None


# ── persistence ───────────────────────────────────────────────────────────────

def test_save_load_round_trip(tmp_path):
    _, trades = grid_run()
    meta = {
        "strategy": "toy",
        "axes": {"x": {"param": "a", "values": [1, 2, 99]},
                 "y": {"param": "b", "values": [2, 3]},
                 "slider": None},
        "n_trades": np.int64(len(trades)),          # numpy type must serialize
        "be_band_ticks": np.float64(0.0),
        "split_date": str(median_split_date(trades).date()),
    }
    run_dir = save_run("toy run: v1", trades, meta, root=tmp_path)
    assert run_dir.parent == tmp_path
    assert list_runs(tmp_path) == [run_dir.name]

    loaded_trades, loaded_meta = load_run(run_dir.name, root=tmp_path)
    pd.testing.assert_frame_equal(trades, loaded_trades)
    assert loaded_meta["axes"]["x"]["values"] == [1, 2, 99]
    assert loaded_meta["axes"]["slider"] is None
    assert loaded_meta["n_trades"] == len(trades)
    assert loaded_meta["run_name"] == run_dir.name

    # exploring a reloaded run == exploring the in-memory run (no re-run needed)
    pd.testing.assert_frame_equal(
        compute_metrics_by_cell(trades, ["a", "b"]),
        compute_metrics_by_cell(loaded_trades, ["a", "b"]),
    )


def test_save_name_collision_suffix(tmp_path):
    _, trades = grid_run()
    d1 = save_run("same", trades, {}, root=tmp_path)
    d2 = save_run("same", trades, {}, root=tmp_path)
    assert d1.name == "same" and d2.name == "same_2"
    assert set(list_runs(tmp_path)) == {"same", "same_2"}


def test_by_cell_vs_reference_on_engine_output():
    # spot-check the vectorized grid against the pure per-cell function
    _, trades = grid_run()
    grid = compute_metrics_by_cell(trades, ["a", "b"])
    for cell, row in grid.iterrows():
        subset = trades[(trades["a"] == cell[0]) & (trades["b"] == cell[1])]
        ref = compute_metrics(subset)
        for metric in METRIC_ORDER:
            got = float(row[metric])
            want = float(ref[metric])
            assert (math.isnan(got) and math.isnan(want)) or got == pytest.approx(want)
