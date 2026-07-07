"""
Grid engine: run one strategy across the cartesian product of the swept
parameter values and return every trade of every cell in one long-format
DataFrame (the run's source of truth).

The strategy module is loaded ONCE by the caller; run_grid() only calls its
run() per combo. Speed therefore rides on the strategy's internal day cache
being independent of params (true for ivb_model_optimized — its day cores are
keyed on files + session_start only). A strategy without such a cache re-reads
its data every combo, and grid runtime scales accordingly — the optimizer view
warns about this.
"""

import time
from pathlib import Path

import pandas as pd

from .buckets import tag_day_bucket
from .param_space import enumerate_combos

# Standard trade columns every strategy returns (see strategies/base.py);
# used to build the empty frame when a whole grid produces zero trades.
TRADE_COLUMNS = [
    "date", "direction", "entry_time", "exit_time", "entry_price",
    "exit_price", "sl", "tp", "exit_reason", "pnl_points",
]
ENRICHED_COLUMNS = ["pnl_ticks", "day_bucket"]


def check_param_columns(axes: list) -> None:
    """Swept param names become trades-table columns — reject collisions."""
    reserved = set(TRADE_COLUMNS) | set(ENRICHED_COLUMNS) | {"notes", "trade_type"}
    clashes = [a["param"] for a in axes if a["param"] in reserved]
    if clashes:
        raise ValueError(
            f"swept param name(s) collide with trade columns: {clashes}"
        )


def run_grid(strategy, folder_path, start_date, end_date, base_params: dict,
             axes: list, *, tick_size: float, ticks_per_point: float,
             bucket_map: dict, on_progress=None) -> pd.DataFrame:
    """
    Long-format trades table: one row per trade, carrying the swept-param
    values as extra columns. Combos with zero trades contribute zero rows
    (their cells render masked). `axes` order defines enumeration order.
    """
    check_param_columns(axes)
    combos = enumerate_combos(axes)
    total = len(combos)
    axis_names = [a["param"] for a in axes]

    frames = []
    for i, combo in enumerate(combos, start=1):
        t0 = time.perf_counter()
        params = {**base_params, **combo, "tick_size": tick_size}
        trades = strategy.run(
            folder_path=Path(folder_path),
            start_date=pd.Timestamp(start_date),
            end_date=pd.Timestamp(end_date),
            params=params,
        )

        n = 0 if trades is None else len(trades)
        if n:
            trades = trades.copy()
            # ns so the dtype survives the parquet round-trip exactly
            trades["date"] = (pd.to_datetime(trades["date"]).dt.normalize()
                              .astype("datetime64[ns]"))
            trades["pnl_ticks"] = trades["pnl_points"].astype(float) * ticks_per_point
            trades = tag_day_bucket(trades, bucket_map)
            for name in axis_names:
                trades[name] = combo[name]
            frames.append(trades)

        if on_progress is not None:
            combo_desc = ", ".join(f"{k}={v}" for k, v in combo.items())
            on_progress(i, total,
                        f"[{i}/{total}] {combo_desc} -> {n} trades "
                        f"({time.perf_counter() - t0:.2f}s)")

    if not frames:
        return pd.DataFrame(columns=axis_names + TRADE_COLUMNS + ENRICHED_COLUMNS)

    long = pd.concat(frames, ignore_index=True)
    # swept params first — the cell identity of every row
    ordered = axis_names + [c for c in long.columns if c not in axis_names]
    return long[ordered]


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
