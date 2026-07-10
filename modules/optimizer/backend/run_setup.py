"""
Grid-run job for the Optimizer's New Run tab — the worker-side body of the
old execute_grid_run (legacy_streamlit/views/optimizer.py), minus the
Streamlit progress/session plumbing.

Runs the grid and builds the run's meta dict (verbatim field set — this dict
is persisted as meta.json by io.save_run and its exact content is part of the
on-disk contract with the Explore tab and the Combiner).

Change vs the old view: the FF events parquet path arrives explicitly (from
the settings' data roots) instead of the module-level FF_EVENTS_PATH default.
"""

from pathlib import Path

import pandas as pd

from .buckets import EVENT_KEYWORDS, load_bucket_map
from .engine import median_split_date, run_grid
from .param_space import ROLES, combo_count


def run_grid_job(strategy, folder_path, start_date, end_date, *,
                 fixed_params: dict, axes: list,
                 asset_type: str, asset: str, dataset: str, strategy_name: str,
                 tick_size: float, ticks_per_point: float,
                 be_band_ticks: float, min_trades_default: int,
                 n_workers: int, ff_events_path,
                 strategies_dir=None, on_progress=None) -> tuple[pd.DataFrame, dict]:
    """
    Run the whole grid and return (trades, meta). Raises RuntimeError from the
    engine on a broken pool (the window shows it as an error banner, exactly
    like the old st.error path).
    """
    ff_found   = ff_events_path is not None and Path(ff_events_path).exists()
    bucket_map = load_bucket_map(ff_events_path) if ff_events_path else {}

    trades = run_grid(
        strategy, folder_path, start_date, end_date,
        base_params=fixed_params, axes=axes,
        tick_size=tick_size,
        ticks_per_point=ticks_per_point,
        bucket_map=bucket_map,
        on_progress=on_progress,
        n_workers=n_workers,
        strategy_name=strategy_name,
        strategies_dir=strategies_dir,
    )

    split = median_split_date(trades)
    axes_by_role = {a["role"]: {"param": a["param"], "values": a["values"]}
                    for a in axes}
    meta = {
        "strategy":           strategy_name,
        "dataset":            f"{asset_type}/{asset}/{dataset}",
        "ticker":             asset,
        "tick_size":          tick_size,
        "ticks_per_point":    ticks_per_point,
        "start_date":         str(start_date),
        "end_date":           str(end_date),
        "axes": {role: axes_by_role.get(role) for role in ROLES},
        "fixed_params":       fixed_params,
        "min_trades_default": min_trades_default,
        "be_band_ticks":      be_band_ticks,
        "event_keywords":     EVENT_KEYWORDS,
        "ff_events_found":    ff_found,
        "split_date":         None if split is None else str(split.date()),
        "n_combos":           combo_count(axes),
        "n_trades":           len(trades),
        "created_at":         pd.Timestamp.now().isoformat(),
    }
    return trades, meta
