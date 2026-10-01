"""
Grid engine + run persistence: combo enumeration/injection, enrichment,
golden heatmap on a toy deterministic strategy, round-trip / partition
invariants, save/load (spec §14).

The toy strategies follow the engine contract (DATA + process_day) and run on
a tiny on-disk dataset (TOY_FOLDER: one parquet per DAYS entry), because every
optimizer combo is a real modules.engine.run_strategy() call.
"""

import datetime
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules.engine import Trade
from modules.optimizer.backend.engine import (
    WORKER_BASELINE_MB, WORKER_MB_PER_DISK_MB, check_param_columns,
    estimate_worker_memory, median_split_date, run_grid,
)
from modules.optimizer.backend.io import list_runs, load_run, save_run
from modules.optimizer.backend.loader import load_strategy
from modules.optimizer.backend.metrics import METRIC_ORDER, compute_metrics, compute_metrics_by_cell

TICKS_PER_POINT = 4
DAYS = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]


def _make_toy_folder() -> Path:
    """A 4-day dataset: DAYS as YYYY-MM-DD.parquet with a tz-aware 1m index."""
    folder = Path(tempfile.mkdtemp(prefix="toy_engine_io_")) / "TOY_1m"
    folder.mkdir()
    for day in DAYS:
        idx = pd.date_range(f"{day} 09:30", periods=3, freq="1min",
                            tz="America/New_York")
        pd.DataFrame({"close": [100.0, 100.5, 101.0]}, index=idx) \
            .to_parquet(folder / f"{day}.parquet")
    return folder


TOY_FOLDER = _make_toy_folder()


def toy_trade(day: datetime.date, a) -> Trade:
    t0 = pd.Timestamp(f"{day} 10:00", tz="America/New_York")
    return Trade("long", t0, t0 + pd.Timedelta(hours=1), 100.0, 100.0 + a, "tp",
                 sl=99.0, tp=100.0 + a)


class ToyStrategy:
    """
    Deterministic, analytically known: `b` trades on the first `b` DAYS, each
    with pnl_points == a. a == 99 -> zero trades (masked-cell path).
    Records the params dict of every RUN (process_day on the first day).
    """
    PARAMS = {"a": 1, "b": 2, "hold": "x"}
    DATA = {"main": ["close"]}

    def __init__(self):
        self.calls = []

    def process_day(self, day, params):
        if day.date.isoformat() == DAYS[0]:
            self.calls.append(dict(params))
        a, b = params["a"], params["b"]
        if a == 99 or day.date.isoformat() not in DAYS[:b]:
            return None
        return toy_trade(day.date, a)


AXES = [
    {"param": "a", "values": [1, 2, 99], "role": "x"},
    {"param": "b", "values": [2, 3],     "role": "y"},
]
BUCKET_MAP = {"2026-01-05": "cpi", "2026-01-08": "holiday"}


def grid_run(strategy=None, axes=AXES, bucket_map=BUCKET_MAP, on_progress=None):
    strategy = strategy or ToyStrategy()
    trades = run_grid(
        strategy, TOY_FOLDER, "2026-01-01", "2026-12-31",
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
    grid_run(on_progress=lambda cur, total, msg: seen.append((cur, total, msg)))
    combo_lines = [(cur, total) for cur, total, msg in seen if msg.startswith("[")]
    assert combo_lines == [(i, 6) for i in range(1, 7)]
    # between combo lines the engine's per-day calls pass through (cancellation
    # points) with the count of combos finished so far
    assert all(cur <= 6 for cur, _, _ in seen) and len(seen) > 6
    # each combo line says where its data came from (the shared cache may be warm)
    assert all(("files read from disk" in m) or ("all data from RAM" in m)
               for _, _, m in seen if m.startswith("["))
    assert "all data from RAM" in next(m for _, _, m in seen if m.startswith("[2/6]"))


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


# ── parallel execution ────────────────────────────────────────────────────────

# File twin of ToyStrategy — written to tmp_path so pool workers (separate
# processes) can load it by name via modules.optimizer.backend.loader without polluting
# the real strategies/ folder (whose contents feed the UI dropdown).
TOY_STRATEGY_SOURCE = '''
import pandas as pd

from modules.engine import Trade

PARAMS = {"a": 1, "b": 2, "hold": "x"}
DATA = {"main": ["close"]}
DAYS = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]


def process_day(day, params):
    a, b = params["a"], params["b"]
    if a == 99 or day.date.isoformat() not in DAYS[:b]:
        return None
    t0 = pd.Timestamp(f"{day.date} 10:00", tz="America/New_York")
    return Trade("long", t0, t0 + pd.Timedelta(hours=1), 100.0, 100.0 + a, "tp",
                 sl=99.0, tp=100.0 + a)
'''


@pytest.fixture()
def toy_strategy_dir(tmp_path):
    (tmp_path / "toy_grid.py").write_text(TOY_STRATEGY_SOURCE)
    return tmp_path


def test_parallel_matches_serial(toy_strategy_dir):
    """The determinism contract: a pool run (out-of-order completion) must
    produce the exact same trades table as the serial loop — including the
    zero-trade a=99 cells. Also pins the progress stream: done-counts are
    monotone and reach the total."""
    _, serial = grid_run()

    progress = []
    parallel = run_grid(
        None, TOY_FOLDER, "2026-01-01", "2026-12-31",
        base_params=dict(ToyStrategy.PARAMS), axes=AXES,
        tick_size=0.25, ticks_per_point=TICKS_PER_POINT,
        bucket_map=BUCKET_MAP,
        on_progress=lambda cur, total, msg: progress.append((cur, total, msg)),
        n_workers=3, strategy_name="toy_grid", strategies_dir=toy_strategy_dir,
    )

    pd.testing.assert_frame_equal(serial, parallel)

    counts = [cur for cur, _, msg in progress if msg.startswith("[")]   # combo lines
    assert counts == sorted(counts)                     # monotone
    assert counts == [1, 2, 3, 4, 5, 6]                 # every combo reported once
    assert all(total == 6 for _, total, _ in progress)
    msgs = [m for _, _, m in progress if m]
    assert msgs[0].startswith("Starting 3 worker processes: the 4 days are split into 3 chunks")
    loaded = [m for m in msgs if " loaded its data in " in m]
    assert len(loaded) == 3                             # once per worker
    assert "worker 1/3 (2026-01-05 … 2026-01-06, 2 days)" in "".join(loaded)


LOOKBACK_STRATEGY_SOURCE = '''
import pandas as pd

from modules.engine import Trade

PARAMS = {"k": 1}
DATA = {"main": ["close"], "indicators": ["vwap"]}


def process_day(day, params):
    prev = day.previous_days(2)          # reaches across a worker's chunk edge
    t0 = pd.Timestamp(f"{day.date} 10:00", tz="America/New_York")
    notes = {"prev": [d.date.isoformat() for d in prev], "k": params["k"]}
    return Trade("long", t0, t0 + pd.Timedelta(hours=1), 100.0, 100.0 + len(prev), "tp",
                 trade_type=None if day.date.day % 2 else "even", notes=notes)
'''


def test_parallel_date_chunks_match_serial_with_lookback_and_skips(tmp_path):
    """Chunks of days per worker: lookback across a chunk edge, a mixed
    None/str column and days skipped for a missing slot file all come out
    exactly as in one serial run."""
    main, ind = tmp_path / "ES_1m", tmp_path / "ES_ind"
    days = pd.bdate_range("2026-02-02", periods=11)
    for folder, col in ((main, "close"), (ind, "vwap")):
        folder.mkdir()
        for n, d in enumerate(days):
            if folder is ind and n in (3, 8):             # two days without indicators
                continue
            idx = pd.date_range(f"{d.date()} 09:30", periods=3, freq="1min", tz="America/New_York")
            pd.DataFrame({col: [1.0, 2.0, 3.0]}, index=idx).to_parquet(folder / f"{d.date()}.parquet")
    (tmp_path / "lookback_toy.py").write_text(LOOKBACK_STRATEGY_SOURCE)
    axes = [{"param": "k", "values": [1, 2], "role": "x"}]
    kw = dict(base_params={"k": 1}, axes=axes, tick_size=0.25, ticks_per_point=4,
              bucket_map={}, extra_folders={"indicators": ind})
    serial_warn, parallel_warn = [], []
    serial = run_grid(load_strategy("lookback_toy", tmp_path), main, "2026-01-01", "2026-12-31",
                      warnings_out=serial_warn, **kw)
    parallel = run_grid(None, main, "2026-01-01", "2026-12-31", n_workers=4,
                        strategy_name="lookback_toy", strategies_dir=tmp_path,
                        warnings_out=parallel_warn, **kw)
    pd.testing.assert_frame_equal(serial, parallel)
    assert len(serial) == 2 * 9
    assert serial_warn == parallel_warn and "2 day(s), so those days were SKIPPED" in serial_warn[0]


def test_day_chunks_are_contiguous_and_balanced(tmp_path):
    from modules.optimizer.backend.engine import day_chunks
    folder = tmp_path / "ES_1m"
    folder.mkdir()
    for d in pd.bdate_range("2026-03-02", periods=10):
        (folder / f"{d.date()}.parquet").write_bytes(b"")
    chunks = day_chunks(folder, "2026-01-01", "2026-12-31", 3)
    assert [n for _, _, n in chunks] == [4, 3, 3]
    assert chunks[0][0] == datetime.date(2026, 3, 2) and chunks[-1][1] == datetime.date(2026, 3, 13)
    assert [c[1] < n[0] for c, n in zip(chunks, chunks[1:])] == [True, True]
    assert len(day_chunks(folder, "2026-01-01", "2026-12-31", 50)) == 10   # <= one day each
    assert day_chunks(folder, "2027-01-01", "2027-12-31", 4) == []


def test_parallel_requires_strategy_name():
    with pytest.raises(ValueError, match="strategy_name"):
        run_grid(
            None, "unused_folder", "2026-01-01", "2026-12-31",
            base_params={}, axes=AXES,
            tick_size=0.25, ticks_per_point=TICKS_PER_POINT,
            bucket_map={}, n_workers=3,
        )


def test_estimate_worker_memory(tmp_path):
    data = tmp_path / "ES_1m"
    data.mkdir()
    (data / "2026-01-05.parquet").write_bytes(b"x" * 1_000_000)
    (data / "2026-01-06.parquet").write_bytes(b"y" * 500_000)
    (data / "2026-02-01.parquet").write_bytes(b"z" * 700_000)   # out of range
    (data / "meta.parquet").write_bytes(b"m" * 900_000)         # not a day file
    est = estimate_worker_memory(data, "2026-01-01", "2026-01-31")
    assert est["n_days"] == 2
    assert est["disk_mb"] == pytest.approx(1.5)
    assert est["est_mb"] == pytest.approx(
        WORKER_BASELINE_MB + 1.5 * WORKER_MB_PER_DISK_MB)

    # additional-data folders the strategy also reads count toward the estimate
    indicators = tmp_path / "ES_indicators"
    indicators.mkdir()
    (indicators / "2026-01-05.parquet").write_bytes(b"i" * 400_000)
    (indicators / "2026-03-01.parquet").write_bytes(b"i" * 900_000)  # out of range
    est = estimate_worker_memory(data, "2026-01-01", "2026-01-31",
                                 extra_folders=[indicators])
    assert est["n_days"] == 2                                   # primary only
    assert est["disk_mb"] == pytest.approx(1.9)
    assert est["est_mb"] == pytest.approx(
        WORKER_BASELINE_MB + 1.9 * WORKER_MB_PER_DISK_MB)


def test_loader_custom_dir(toy_strategy_dir):
    mod = load_strategy("toy_grid", toy_strategy_dir)
    assert callable(mod.process_day)
    assert mod.PARAMS["b"] == 2
    with pytest.raises(FileNotFoundError):
        load_strategy("does_not_exist", toy_strategy_dir)


def test_loader_default_dir_finds_real_strategies():
    mod = load_strategy("orb")          # repo strategies/ (a package), cwd-independent
    assert callable(mod.process_day) and callable(mod.prepare_day)


# ── persistence ───────────────────────────────────────────────────────────────

def test_save_load_round_trip(tmp_path):
    _, trades = grid_run()
    meta = {
        "strategy": "toy",
        "axes": {"x": {"param": "a", "values": [1, 2, 99]},
                 "y": {"param": "b", "values": [2, 3]},
                 "slider": None, "slider2": None},
        "n_trades": np.int64(len(trades)),          # numpy type must serialize
        "be_band_ticks": np.float64(0.0),
        "split_date": str(median_split_date(trades).date()),
    }
    run_dir = save_run(trades, meta, run_name="toy run: v1", root=tmp_path)
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
    d1 = save_run(trades, {}, run_name="same", root=tmp_path)
    d2 = save_run(trades, {}, run_name="same", root=tmp_path)
    assert d1.name == "same" and d2.name == "same_2"
    assert set(list_runs(tmp_path)) == {"same", "same_2"}


def test_save_into_folders(tmp_path):
    from modules.optimizer.backend.io import list_folders

    _, trades = grid_run()
    # new folder is created on demand; a second save reuses it
    d1 = save_run(trades, {}, run_name="run_a", folder="my sweeps", root=tmp_path)
    d2 = save_run(trades, {}, run_name="run_b", folder="my_sweeps", root=tmp_path)
    d3 = save_run(trades, {}, run_name="run_root", root=tmp_path)

    assert d1.parent == d2.parent == tmp_path / "my_sweeps"   # name sanitized
    assert list_folders(tmp_path) == ["my_sweeps"]
    assert set(list_runs(tmp_path)) == {"my_sweeps/run_a", "my_sweeps/run_b",
                                        "run_root"}

    # nested runs load by their relative path; meta records it
    loaded_trades, loaded_meta = load_run("my_sweeps/run_a", root=tmp_path)
    pd.testing.assert_frame_equal(trades, loaded_trades)
    assert loaded_meta["run_name"] == "my_sweeps/run_a"

    # collision suffix works inside a folder too
    d4 = save_run(trades, {}, run_name="run_a", folder="my_sweeps", root=tmp_path)
    assert d4.name == "run_a_2"


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


def test_bool_axis_end_to_end(tmp_path):
    """A bool sweep axis ([False, True]) must survive the whole optimizer
    pipeline: enrichment, parquet round trip, meta.json, per-cell metrics and
    the explore-style equality filter."""
    class BoolToyStrategy:
        PARAMS = {"a": 1, "flag": True}
        DATA = {"main": ["close"]}

        def process_day(self, day, params):
            if day.date.isoformat() != DAYS[0]:
                return None
            return toy_trade(day.date, 2.0 if params["flag"] else 1.0)

    axes = [{"param": "flag", "values": [False, True], "role": "x"}]
    trades = run_grid(
        BoolToyStrategy(), TOY_FOLDER, "2026-01-01", "2026-12-31",
        base_params=dict(BoolToyStrategy.PARAMS), axes=axes,
        tick_size=0.25, ticks_per_point=TICKS_PER_POINT,
        bucket_map={},
    )
    meta = {"axes": {"x": {"param": "flag", "values": [False, True]},
                     "y": None, "slider": None, "slider2": None}}
    run_dir = save_run(trades, meta, run_name="bool axis", root=tmp_path)
    loaded_trades, loaded_meta = load_run(run_dir.name, root=tmp_path)

    assert loaded_meta["axes"]["x"]["values"] == [False, True]
    assert loaded_trades["flag"].dtype == bool
    grid = compute_metrics_by_cell(loaded_trades, ["flag"])
    assert grid.loc[False]["total_ticks"] == 1.0 * TICKS_PER_POINT
    assert grid.loc[True]["total_ticks"] == 2.0 * TICKS_PER_POINT
    # the explore/cell-detail filter pattern: meta value == trades column
    for v in loaded_meta["axes"]["x"]["values"]:
        assert len(loaded_trades[loaded_trades["flag"] == v]) == 1


def test_parallel_cancel_stops_promptly(toy_strategy_dir):
    """Raising inside on_progress (the app's Cancel) ends a parallel run: the
    queued parts are cancelled and every pool is shut down."""
    import time

    class Stop(Exception):
        pass

    def on_progress(cur, total, msg):
        if msg.startswith("["):
            raise Stop
    axes = [{"param": "a", "values": list(range(1, 41)), "role": "x"},
            {"param": "b", "values": [2, 3], "role": "y"}]
    t0 = time.perf_counter()
    with pytest.raises(Stop):
        run_grid(None, TOY_FOLDER, "2026-01-01", "2026-12-31",
                 base_params=dict(ToyStrategy.PARAMS), axes=axes, tick_size=0.25,
                 ticks_per_point=TICKS_PER_POINT, bucket_map={}, on_progress=on_progress,
                 n_workers=2, strategy_name="toy_grid", strategies_dir=toy_strategy_dir)
    assert time.perf_counter() - t0 < 60
