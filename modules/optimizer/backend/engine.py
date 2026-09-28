"""
Grid engine: run one strategy across the cartesian product of the swept
parameter values and return every trade of every cell in one long-format
DataFrame (the run's source of truth).

Every combo is one modules.engine.run_strategy() call. Serial by default
(n_workers=1): the strategy module is loaded once by the caller and the runs
share the app's in-process engine cache (modules.engine.cache — data and
prepared days are param-independent, so every combo after the first runs
warm, and so do later optimizer and backtester runs).

n_workers > 1 runs combos on a ProcessPoolExecutor. Workers are long-lived
for the whole grid and each loads the strategy ONCE by name (via
.loader — never through a UI module, so workers stay Qt-free) and fills its
OWN engine cache, capped at `cache_bytes` (the optimizer's memory budget
divided by the worker count, so the budget is the total across all
workers): every worker pays one cold start, then runs warm. Results are reassembled in combo order, so a
parallel run produces the exact same trades table as a serial one. Cleanup is
a hard shutdown(cancel_futures=True) in a finally — a Streamlit Stop (raised
inside the caller's on_progress st.* call) cancels everything queued and
waits only for the one in-flight combo per worker.
"""

import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pandas as pd

from modules.engine import OUTPUT_COLUMNS, get_cache, run_strategy

from .buckets import tag_day_bucket
from .loader import load_strategy
from .param_space import enumerate_combos

# The engine's trade columns (every strategy returns exactly these); used to
# build the empty frame when a whole grid produces zero trades.
TRADE_COLUMNS = list(OUTPUT_COLUMNS)
ENRICHED_COLUMNS = ["pnl_ticks", "day_bucket"]

# Per-worker memory heuristic (see estimate_worker_memory): fixed
# interpreter + numpy/pandas/pyarrow baseline, plus roughly this much
# resident memory per MB of (compressed) parquet the strategy will read and
# cache. Calibrated on ES_1m_advanced (~0.25 MB/day on disk, ~0.3 MB/day
# parsed ivb day core). Tunable.
WORKER_BASELINE_MB    = 200.0
WORKER_MB_PER_DISK_MB = 0.6

# win32 ProcessPoolExecutor raises ValueError above 61 workers
_MAX_POOL_WORKERS = 61


def check_param_columns(axes: list) -> None:
    """Swept param names become trades-table columns — reject collisions."""
    reserved = set(TRADE_COLUMNS) | set(ENRICHED_COLUMNS) | {"notes", "trade_type"}
    clashes = [a["param"] for a in axes if a["param"] in reserved]
    if clashes:
        raise ValueError(
            f"swept param name(s) collide with trade columns: {clashes}"
        )


def _range_files(folder_path, start_date, end_date) -> list:
    """Dated day files in range — the same filter every strategy applies."""
    start = pd.Timestamp(start_date).date()
    end   = pd.Timestamp(end_date).date()
    return [
        f for f in sorted(Path(folder_path).glob("*.parquet"))
        if f.stem[0].isdigit() and start <= pd.Timestamp(f.stem).date() <= end
    ]


def estimate_worker_memory(folder_path, start_date, end_date,
                           extra_folders=()) -> dict:
    """
    Rough per-worker memory need: fixed process baseline + a multiple of the
    date-filtered parquet bytes the strategy will read/cache — summed over
    the selected dataset AND `extra_folders` (the folders chosen for the
    strategy's additional DATA slots). Still a heuristic (in-memory size per
    byte varies by strategy; the engine cache is capped separately), but good
    enough to budget a worker count.
    Returns {"n_days", "disk_mb", "est_mb"} — n_days counts the primary
    dataset only.
    """
    files   = _range_files(folder_path, start_date, end_date)
    disk_mb = sum(f.stat().st_size for f in files) / 1e6
    for folder in extra_folders:
        disk_mb += sum(f.stat().st_size
                       for f in _range_files(folder, start_date, end_date)) / 1e6
    return {
        "n_days":  len(files),
        "disk_mb": disk_mb,
        "est_mb":  WORKER_BASELINE_MB + disk_mb * WORKER_MB_PER_DISK_MB,
    }


def _enrich(trades, combo: dict, axis_names: list, ticks_per_point: float,
            bucket_map: dict):
    """Per-combo enrichment shared by both paths; None for empty results."""
    if trades is None or len(trades) == 0:
        return None
    trades = trades.copy()
    # ns so the dtype survives the parquet round-trip exactly
    trades["date"] = (pd.to_datetime(trades["date"]).dt.normalize()
                      .astype("datetime64[ns]"))
    trades["pnl_ticks"] = trades["pnl_points"].astype(float) * ticks_per_point
    trades = tag_day_bucket(trades, bucket_map)
    for name in axis_names:
        trades[name] = combo[name]
    return trades


def _combo_desc(combo: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in combo.items())


# ── process-pool worker plumbing ──────────────────────────────────────────────
# Module-level so Windows spawn can pickle them by qualified name. This module
# imports no Streamlit, so worker processes stay lean.

_WORKER_STRATEGY = None


def _init_worker(strategy_name: str, strategies_dir, cache_bytes) -> None:
    """Runs once per worker process: load the strategy, size this process's
    engine cache to its share of the optimizer budget, keep both warm."""
    global _WORKER_STRATEGY
    _WORKER_STRATEGY = load_strategy(strategy_name, strategies_dir)
    if cache_bytes:
        get_cache().set_budget(int(cache_bytes))


def _run_combo(index: int, folder_path: str, start_iso: str, end_iso: str,
               params: dict, tick_size: float, extra_folders: dict):
    """One backtest in a worker. Returns (index, trades|None, warnings, elapsed_s)."""
    t0 = time.perf_counter()
    result = run_strategy(_WORKER_STRATEGY, folder_path, start_iso, end_iso, params,
                          tick_size=tick_size, extra_folders=extra_folders,
                          verbose=False)
    trades = result.trades if len(result.trades) else None   # don't ship empty frames
    return index, trades, result.warnings, time.perf_counter() - t0


# ── grid runners ──────────────────────────────────────────────────────────────

def run_grid(strategy, folder_path, start_date, end_date, base_params: dict,
             axes: list, *, tick_size: float, ticks_per_point: float,
             bucket_map: dict, on_progress=None, n_workers: int = 1,
             strategy_name: str = None, strategies_dir=None,
             extra_folders: dict | None = None, cache_bytes: int | None = None,
             warnings_out: list | None = None) -> pd.DataFrame:
    """
    Long-format trades table: one row per trade, carrying the swept-param
    values as extra columns. Combos with zero trades contribute zero rows
    (their cells render masked). `axes` order defines enumeration order, and
    the output is identical for ANY n_workers — parallel results are
    reassembled in combo order.

    n_workers <= 1: serial, runs `strategy` in-process through the engine
    (reusing the app's warm engine cache). n_workers > 1: process pool;
    requires `strategy_name` (each worker loads the strategy itself);
    `strategy` may be None; each worker's engine cache is capped at
    `cache_bytes`. `extra_folders` = {slot: folder} for the strategy's
    additional DATA slots. The engine's data warnings (e.g. days skipped for
    a missing additional-data file — identical for every combo) are appended
    once to `warnings_out`. NOTE for headless scripts: Windows spawn re-imports __main__
    when Python is launched as `python script.py`, so such callers must
    guard their entry point with `if __name__ == "__main__":` (Streamlit and
    pytest launches are unaffected).
    """
    check_param_columns(axes)
    combos = enumerate_combos(axes)
    axis_names = [a["param"] for a in axes]

    if n_workers > 1:
        if not strategy_name:
            raise ValueError("n_workers > 1 requires strategy_name — "
                             "workers load the strategy themselves")
        frames, warnings = _run_grid_parallel(
            strategy_name, strategies_dir, folder_path, start_date, end_date,
            base_params, combos, axis_names, tick_size=tick_size,
            ticks_per_point=ticks_per_point, bucket_map=bucket_map,
            on_progress=on_progress, n_workers=n_workers,
            extra_folders=extra_folders or {}, cache_bytes=cache_bytes,
        )
    else:
        frames, warnings = _run_grid_serial(
            strategy, folder_path, start_date, end_date,
            base_params, combos, axis_names, tick_size=tick_size,
            ticks_per_point=ticks_per_point, bucket_map=bucket_map,
            on_progress=on_progress, extra_folders=extra_folders or {},
        )
    if warnings_out is not None:
        warnings_out.extend(warnings)

    frames = [f for f in frames if f is not None]     # index order preserved
    if not frames:
        return pd.DataFrame(columns=axis_names + TRADE_COLUMNS + ENRICHED_COLUMNS)

    long = pd.concat(frames, ignore_index=True)
    # swept params first — the cell identity of every row
    ordered = axis_names + [c for c in long.columns if c not in axis_names]
    return long[ordered]


def _run_grid_serial(strategy, folder_path, start_date, end_date,
                     base_params, combos, axis_names, *, tick_size,
                     ticks_per_point, bucket_map, on_progress, extra_folders):
    total = len(combos)
    frames = []
    warnings: list = []
    for i, combo in enumerate(combos, start=1):
        t0 = time.perf_counter()
        params = {**base_params, **combo, "tick_size": tick_size}
        result = run_strategy(strategy, folder_path, start_date, end_date, params,
                              tick_size=tick_size, extra_folders=extra_folders,
                              verbose=False)
        trades = result.trades if len(result.trades) else None
        if not warnings:
            warnings = list(result.warnings)
        n = 0 if trades is None else len(trades)
        frames.append(_enrich(trades, combo, axis_names, ticks_per_point,
                              bucket_map))
        if on_progress is not None:
            on_progress(i, total,
                        f"[{i}/{total}] {_combo_desc(combo)} -> {n} trades "
                        f"({time.perf_counter() - t0:.2f}s)")
    return frames, warnings


def _run_grid_parallel(strategy_name, strategies_dir, folder_path, start_date,
                       end_date, base_params, combos, axis_names, *,
                       tick_size, ticks_per_point, bucket_map, on_progress,
                       n_workers, extra_folders, cache_bytes):
    total   = len(combos)
    workers = max(1, min(n_workers, total, _MAX_POOL_WORKERS,
                         os.process_cpu_count() or 1))
    # absolute path: workers inherit the parent cwd at spawn time, but don't
    # depend on it
    folder    = str(Path(folder_path).resolve())
    start_iso = str(pd.Timestamp(start_date))
    end_iso   = str(pd.Timestamp(end_date))
    dir_arg   = str(strategies_dir) if strategies_dir is not None else None
    extras    = {slot: str(Path(f).resolve()) for slot, f in extra_folders.items()}
    per_worker = int(cache_bytes // workers) if cache_bytes else None

    results = [None] * total
    warnings: list = []
    done = 0

    # NOT a `with` block: Executor.__exit__ is shutdown(wait=True) WITHOUT
    # cancel_futures, which would block a Streamlit Stop until every queued
    # combo ran. The explicit finally cancels the queue and waits only for
    # the in-flight combo per worker.
    executor = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(strategy_name, dir_arg, per_worker),
    )
    try:
        pending = set()
        for i, combo in enumerate(combos):
            params = {**base_params, **combo, "tick_size": tick_size}
            pending.add(executor.submit(_run_combo, i, folder, start_iso,
                                        end_iso, params, tick_size, extras))

        while pending:
            finished, pending = wait(pending, timeout=0.5,
                                     return_when=FIRST_COMPLETED)
            if not finished:
                # idle tick — the caller's st.* call is Streamlit's chance
                # to raise Stop/rerun while workers grind
                if on_progress is not None:
                    on_progress(done, total, "")
                continue
            for fut in finished:
                try:
                    index, trades, run_warnings, elapsed = fut.result()
                except BrokenProcessPool as e:
                    raise RuntimeError(
                        f"worker pool died while running '{strategy_name}' — "
                        f"usually the strategy failed to load in a worker or "
                        f"a worker ran out of memory ({e})"
                    ) from e
                if not warnings:
                    warnings = list(run_warnings)
                combo = combos[index]
                results[index] = _enrich(trades, combo, axis_names,
                                         ticks_per_point, bucket_map)
                done += 1
                if on_progress is not None:
                    n = 0 if trades is None else len(trades)
                    on_progress(done, total,
                                f"[{done}/{total}] #{index + 1} "
                                f"{_combo_desc(combo)} -> {n} trades "
                                f"({elapsed:.2f}s)")
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    return results, warnings


def median_split_date(trades: pd.DataFrame):
    """
    The run's train/test split date: the median unique trading day of the FULL
    (unfiltered) trades table. 1st half = date <= split, 2nd half = date >
    split — a disjoint, gap-free partition. None when there are no trades.
    """
    if trades.empty:
        return None
    days = pd.to_datetime(trades["date"]).dt.normalize().drop_duplicates().sort_values()
    return days.iloc[(len(days) - 1) // 2]
