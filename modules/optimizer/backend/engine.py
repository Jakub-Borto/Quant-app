"""
Grid engine: run one strategy across the cartesian product of the swept
parameter values and return every trade of every cell in one long-format
DataFrame (the run's source of truth).

Serial (n_workers=1): every combo is one modules.engine.run_strategy() call
over the whole date range, in-process, sharing the app's engine cache
(modules.engine.cache — data and prepared days are param-independent, so
every combo after the first runs warm, and so do later optimizer and
backtester runs). The engine's own per-day progress is forwarded (at most one
log line a second) so a slow first combo shows what it is loading.

Parallel (n_workers > 1) SPLITS THE DATES, not the combos: the date range is
cut into one contiguous chunk of days per worker, and every worker runs EVERY
combo on its own chunk. So each day file is read and prepared by exactly one
worker, once (plus a few lookback days at a chunk edge). The old layout gave
each worker a share of the combos over ALL the days, so every worker read and
prepared the whole dataset at the same time and the first round of combos
took as long as a cold backtest. Each worker is its own single-process pool,
so its chunk always lands on the same process and stays warm in that
process's cache (capped at `cache_bytes` / workers — the budget is the total,
and each worker only holds its 1/workers share of the data). Workers load the
strategy ONCE by name (via .loader — never through a UI module, so workers
stay Qt-free).

Determinism: workers return their (date, Trade) rows; the parent joins a
combo's chunks in date order and builds ONE frame with the engine's
trades_to_frame — exactly what a serial run builds. This relies on the engine
contract that process_day carries no state from one day to the next (the
days of one combo run in different processes). Cleanup is a hard
shutdown(cancel_futures=True) of every pool in a finally — a Stop (raised
inside the caller's on_progress) cancels everything queued and waits only for
the in-flight part per worker.
"""

import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pandas as pd

from modules.engine import (GB, OUTPUT_COLUMNS, day_files, estimate_run_memory,
                            get_cache, run_strategy, skipped_warnings,
                            trades_to_frame)

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
# parsed ivb day core). Tunable. estimate_grid_memory is the strategy-aware
# estimate the New Run tab uses.
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
    strategy's additional DATA slots). A heuristic without the strategy;
    estimate_grid_memory (from the strategy's declared columns) is the
    accurate one.
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


def estimate_grid_memory(strategy, folder_path, start_date, end_date,
                         extra_folders: dict | None = None) -> dict:
    """RAM a grid run needs, from the strategy's DATA declaration and the
    parquet footers (modules.engine.estimate_run_memory): the day data +
    prepared days held in the engine caches (split across parallel workers,
    never duplicated) and the per-process baseline.

    Returns {"n_days", "data_mb", "baseline_mb"}; a run with W parallel
    workers needs about data_mb + W * baseline_mb."""
    est = estimate_run_memory(strategy, folder_path, start_date, end_date,
                              extra_folders=extra_folders)
    return {"n_days": est["n_days"], "data_mb": est["total_bytes"] / 1024 ** 2,
            "baseline_mb": WORKER_BASELINE_MB}


def day_chunks(folder_path, start_date, end_date, n: int) -> list[tuple]:
    """The run's day files split into <= n contiguous, near-equal chunks:
    [(first_date, last_date, n_days), ...] in date order."""
    lo, hi = pd.Timestamp(start_date).date(), pd.Timestamp(end_date).date()
    days = [d for d, _ in day_files(folder_path) if lo <= d <= hi]
    if not days:
        return []
    n = max(1, min(int(n), len(days)))
    return [(c[0], c[-1], len(c))
            for c in np.array_split(np.array(days, dtype=object), n) if len(c)]


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


def _data_desc(files_read: int, days_prepared: int) -> str:
    if not files_read and not days_prepared:
        return "all data from RAM"
    return f"{files_read} files read from disk, {days_prepared} days prepared"


# ── process-pool worker plumbing ──────────────────────────────────────────────
# Module-level so Windows spawn can pickle them by qualified name. This module
# imports no Qt, so worker processes stay lean.

_WORKER_STRATEGY = None


def _init_worker(strategy_name: str, strategies_dir, cache_bytes) -> None:
    """Runs once per worker process: load the strategy, size this process's
    engine cache to its share of the optimizer budget, keep both warm."""
    global _WORKER_STRATEGY
    _WORKER_STRATEGY = load_strategy(strategy_name, strategies_dir)
    if cache_bytes:
        get_cache().set_budget(int(cache_bytes))


def _run_part(index: int, chunk: int, folder_path: str, start_iso: str, end_iso: str,
              params: dict, tick_size: float, extra_folders: dict) -> dict:
    """One combo on one worker's date chunk: the (date, Trade) rows plus what
    the engine did (for the progress log and the merge)."""
    t0 = time.perf_counter()
    result = run_strategy(_WORKER_STRATEGY, folder_path, start_iso, end_iso, params,
                          tick_size=tick_size, extra_folders=extra_folders,
                          verbose=False)
    return {"index": index, "chunk": chunk, "rows": result.rows,
            "skipped": result.skipped, "cache_note": result.cache_note,
            "elapsed": time.perf_counter() - t0, "files_read": result.files_read,
            "days_prepared": result.days_prepared}


# ── grid runners ──────────────────────────────────────────────────────────────

def run_grid(strategy, folder_path, start_date, end_date, base_params: dict,
             axes: list, *, tick_size: float, ticks_per_point: float,
             bucket_map: dict, on_progress=None, n_workers: int = 1,
             strategy_name: str = None, strategies_dir=None,
             extra_folders: dict | None = None, cache_bytes: int | None = None,
             warnings_out: list | None = None,
             notes_out: list | None = None) -> pd.DataFrame:
    """
    Long-format trades table: one row per trade, carrying the swept-param
    values as extra columns. Combos with zero trades contribute zero rows
    (their cells render masked). `axes` order defines enumeration order, and
    the output is identical for ANY n_workers — parallel parts are
    reassembled in combo and date order.

    n_workers <= 1: serial, runs `strategy` in-process through the engine
    (reusing the app's warm engine cache). n_workers > 1: one worker process
    per date chunk (see the module docstring); requires `strategy_name` (each
    worker loads the strategy itself); `strategy` may be None; the workers'
    engine caches share `cache_bytes`. `extra_folders` = {slot: folder} for
    the strategy's additional DATA slots. The engine's data warnings (e.g.
    days skipped for a missing additional-data file — identical for every
    combo) are appended once to `warnings_out`; a cache-too-small note is
    appended once to `notes_out`. NOTE for headless scripts: Windows spawn
    re-imports __main__ when Python is launched as `python script.py`, so such
    callers must guard their entry point with `if __name__ == "__main__":`
    (pytest launches are unaffected).
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
    warnings, cache_note = warnings
    if warnings_out is not None:
        warnings_out.extend(warnings)
    if notes_out is not None and cache_note:
        notes_out.append(cache_note)

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
    cache_note = None
    for i, combo in enumerate(combos, start=1):
        t0 = time.perf_counter()
        params = {**base_params, **combo, "tick_size": tick_size}
        engine_progress = None
        if on_progress is not None:
            last = [t0]

            def engine_progress(_cur, _days, text, _i=i, _last=last):
                # the engine reports every step of every day: the log gets at
                # most one line a second (a slow combo shows what it is doing),
                # and every call stays a cancellation point
                now = time.perf_counter()
                if text and now - _last[0] >= 1.0:
                    _last[0] = now
                    on_progress(_i - 1, total, f"   ↳ combo {_i}: {text.splitlines()[0]}")
                else:
                    on_progress(_i - 1, total, "")
        result = run_strategy(strategy, folder_path, start_date, end_date, params,
                              tick_size=tick_size, extra_folders=extra_folders,
                              on_progress=engine_progress, verbose=False)
        trades = result.trades if len(result.trades) else None
        if not warnings:
            warnings = list(result.warnings)
        cache_note = cache_note or result.cache_note
        n = 0 if trades is None else len(trades)
        frames.append(_enrich(trades, combo, axis_names, ticks_per_point,
                              bucket_map))
        if on_progress is not None:
            on_progress(i, total,
                        f"[{i}/{total}] {_combo_desc(combo)} -> {n} trades "
                        f"({time.perf_counter() - t0:.2f}s; "
                        f"{_data_desc(result.files_read, result.days_prepared)})")
    return frames, (warnings, cache_note)


def _run_grid_parallel(strategy_name, strategies_dir, folder_path, start_date,
                       end_date, base_params, combos, axis_names, *,
                       tick_size, ticks_per_point, bucket_map, on_progress,
                       n_workers, extra_folders, cache_bytes):
    total  = len(combos)
    chunks = day_chunks(folder_path, start_date, end_date,
                        min(n_workers, _MAX_POOL_WORKERS, os.process_cpu_count() or 1))
    if not chunks:                     # no days in range: nothing to split
        return [None] * total, ([], None)
    workers = len(chunks)
    # absolute path: workers inherit the parent cwd at spawn time, but don't
    # depend on it
    folder     = str(Path(folder_path).resolve())
    dir_arg    = str(strategies_dir) if strategies_dir is not None else None
    extras     = {slot: str(Path(f).resolve()) for slot, f in extra_folders.items()}
    per_worker = int(cache_bytes // workers) if cache_bytes else None
    names = {"main": Path(folder_path).name,
             **{slot: Path(f).name for slot, f in extra_folders.items()}}

    parts: list[dict] = [{} for _ in range(total)]     # combo -> {chunk: part}
    results = [None] * total
    skipped: dict = {}
    cache_note = None
    done = 0
    t_start = time.perf_counter()

    def log(msg: str) -> None:
        if on_progress is not None:
            on_progress(done, total, msg)

    log(f"Starting {workers} worker processes: the {sum(c[2] for c in chunks)} days are "
        f"split into {workers} chunks of ~{chunks[0][2]} days — each worker reads and "
        f"prepares only its own days, once, then runs every combo on them from RAM"
        + (f" (day cache {per_worker / GB:.2f} GB per worker)" if per_worker else ""))

    # NOT `with` blocks: Executor.__exit__ is shutdown(wait=True) WITHOUT
    # cancel_futures, which would block a Stop until every queued part ran.
    # The finally cancels the queues and waits only for the in-flight part per
    # worker. One single-process pool per chunk = the chunk's days always hit
    # the same warm cache.
    pools = [ProcessPoolExecutor(max_workers=1, initializer=_init_worker,
                                 initargs=(strategy_name, dir_arg, per_worker))
             for _ in chunks]
    pending: set = set()
    try:
        # combo-major submission: every worker starts on combo 0, so combos
        # complete (all chunks in) roughly in order
        for i, combo in enumerate(combos):
            params = {**base_params, **combo, "tick_size": tick_size}
            for k, (lo, hi, _n) in enumerate(chunks):
                pending.add(pools[k].submit(_run_part, i, k, folder, str(pd.Timestamp(lo)),
                                            str(pd.Timestamp(hi)), params, tick_size, extras))

        loaded = set()
        while pending:
            finished, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            if not finished:
                # idle tick — the caller's chance to raise Stop while workers grind
                log("")
                continue
            for fut in finished:
                try:
                    part = fut.result()
                except BrokenProcessPool as e:
                    raise RuntimeError(
                        f"worker pool died while running '{strategy_name}' — "
                        f"usually the strategy failed to load in a worker or "
                        f"a worker ran out of memory ({e})"
                    ) from e
                i, k = part["index"], part["chunk"]
                parts[i][k] = part
                cache_note = cache_note or part["cache_note"]
                if k not in loaded:
                    loaded.add(k)
                    lo, hi, n = chunks[k]
                    log(f"worker {k + 1}/{workers} ({lo} … {hi}, {n} days) loaded its data "
                        f"in {part['elapsed']:.2f}s: "
                        f"{_data_desc(part['files_read'], part['days_prepared'])}")
                if len(parts[i]) < workers:
                    continue
                # every chunk of combo i is in: join them in date order and
                # build ONE frame, exactly like a serial run
                ordered = [parts[i][c] for c in range(workers)]
                rows = [r for p in ordered for r in p["rows"]]
                if not skipped:
                    for p in ordered:
                        for slot, dates in p["skipped"].items():
                            skipped.setdefault(slot, []).extend(dates)
                combo = combos[i]
                trades = trades_to_frame(rows) if rows else None
                results[i] = _enrich(trades, combo, axis_names, ticks_per_point,
                                     bucket_map)
                parts[i] = {}                          # free the rows
                done += 1
                slowest = max(p["elapsed"] for p in ordered)
                log(f"[{done}/{total}] #{i + 1} {_combo_desc(combo)} -> {len(rows)} trades "
                    f"(slowest worker {slowest:.2f}s; {time.perf_counter() - t_start:.1f}s "
                    f"since start)")
    finally:
        # cancel every queued part first (all pools at once), then wait for
        # each worker's in-flight part — they finish concurrently
        for fut in pending:
            fut.cancel()
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)

    return results, (skipped_warnings(skipped, names), cache_note)


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
