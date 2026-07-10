"""
Candidate pool assembly: saved entry runs -> flat pool of variants.

A variant is one parameter cell of one entry run: identity =
(run, trade_type, param-tuple) — a single run holding several trade_types
splits correctly. Pipeline order is strict (spec §6.1): pool -> day filter ->
chronological IS/OOS split -> per-entry min-trades on the IN-SAMPLE slice
only. pnl_ticks is read as-is and never recomputed from pnl_points.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..io import RUNS_ROOT
from .merge import trades_to_tuples

COMBINED_DIR = "_combined"

# the only trade columns the combiner needs
_POOL_COLUMNS = ["date", "entry_time", "exit_time", "day_bucket", "pnl_ticks"]


@dataclass
class Variant:
    vid: str                      # unique: "run · trade_type · k=v, ..."
    run: str
    trade_type: str
    params: dict
    is_tuples: list = field(default_factory=list)     # sorted merge tuples
    oos_tuples: list = field(default_factory=list)
    n_is: int = 0
    n_oos: int = 0
    is_daily: pd.Series = None    # in-sample per-day pnl (traded days only)


def list_containers(root: Path = RUNS_ROOT) -> list:
    """Folders under data/optimizations that hold at least one entry run."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and discover_entry_runs(d.name, root))


def discover_entry_runs(container: str, root: Path = RUNS_ROOT) -> list:
    """
    Child folders holding BOTH meta.json and trades.parquet. `_combined/`
    (this module's own output) and incomplete folders are excluded.
    """
    base = Path(root) / container
    if not base.is_dir():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and d.name != COMBINED_DIR
        and (d / "meta.json").exists() and (d / "trades.parquet").exists()
    )


def load_entry_runs(container: str, run_names: list,
                    root: Path = RUNS_ROOT) -> dict:
    """{run_name: (meta, trades)} for the ticked runs."""
    out = {}
    for name in run_names:
        run_dir = Path(root) / container / name
        with open(run_dir / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        out[name] = (meta, pd.read_parquet(run_dir / "trades.parquet"))
    return out


def _param_columns(meta: dict) -> list:
    axes = meta.get("axes", {}) or {}
    return [ax["param"] for ax in axes.values() if ax]


def _format_params(params: dict) -> str:
    return ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in params.items())


def assert_shared_timezone(runs: dict) -> str:
    """All runs' entry_time must share one tz — merged ordering depends on it."""
    tzs = {str(trades["entry_time"].dtype) for _, trades in runs.values()}
    if len(tzs) > 1:
        raise ValueError(f"entry_time timezones differ across runs: {sorted(tzs)}")
    return next(iter(tzs), "")


def build_pool(runs: dict, enabled_buckets: set,
               shared_start=None, shared_end=None) -> list:
    """
    Group every run's trades by (trade_type, swept-param columns) -> variants.
    Rows outside the enabled day_buckets or the shared date window are dropped
    HERE, before the split — freed slots re-merge later by construction.
    Variants are returned with raw per-variant trades attached (split happens
    in split_pool).
    """
    assert_shared_timezone(runs)
    variants = []
    for run_name, (meta, trades) in sorted(runs.items()):
        param_cols = [c for c in _param_columns(meta) if c in trades.columns]

        df = trades
        if shared_start is not None:
            dates = pd.to_datetime(df["date"])
            df = df[(dates >= pd.Timestamp(shared_start))
                    & (dates <= pd.Timestamp(shared_end))]
        if enabled_buckets is not None:
            df = df[df["day_bucket"].isin(enabled_buckets)]
        if df.empty:
            continue

        group_cols = (["trade_type"] if "trade_type" in df.columns else []) \
            + param_cols
        if not group_cols:
            groups = [((), df)]
        else:
            groups = df.groupby(group_cols, sort=True, dropna=False)

        for key, cell in groups:
            key = key if isinstance(key, tuple) else (key,)
            named = dict(zip(group_cols, key))
            trade_type = str(named.pop("trade_type", "unknown"))
            params = named
            vid = f"{run_name} · {trade_type}"
            if params:
                vid += f" · {_format_params(params)}"
            variants.append(Variant(
                vid=vid, run=run_name, trade_type=trade_type, params=params,
                is_tuples=trades_to_tuples(
                    cell[[c for c in _POOL_COLUMNS if c in cell.columns]], vid),
            ))
    return variants


def split_date_boundary(variants: list, is_fraction: float):
    """
    The last IN-SAMPLE calendar date: chronological cut over the pool's
    unique trade dates so no day straddles the boundary. None if the pool
    is empty. The cut is clamped so both slices hold at least one date.
    """
    dates = sorted({t[5] for v in variants for t in v.is_tuples})
    if len(dates) < 2:
        return None
    cut = int(len(dates) * is_fraction)
    cut = min(max(cut, 1), len(dates) - 1)
    return dates[cut - 1]           # last IS date (as int64 ns)


def split_pool(variants: list, boundary_ns: int) -> None:
    """
    In place: split every variant's tuples into IS (date <= boundary) and
    OOS (date > boundary), attach counts and the in-sample daily-pnl series
    used by the redundancy penalty.
    """
    for v in variants:
        all_tuples   = v.is_tuples
        v.is_tuples  = [t for t in all_tuples if t[5] <= boundary_ns]
        v.oos_tuples = [t for t in all_tuples if t[5] > boundary_ns]
        v.n_is, v.n_oos = len(v.is_tuples), len(v.oos_tuples)
        if v.is_tuples:
            daily = {}
            for t in v.is_tuples:
                daily[t[5]] = daily.get(t[5], 0.0) + t[4]
            v.is_daily = pd.Series(daily).sort_index()
        else:
            v.is_daily = pd.Series(dtype=float)


def apply_min_trades(variants: list, floors: dict, default_floor: int = 0) -> list:
    """
    Drop variants whose IN-SAMPLE trade count is below their trade_type's
    floor (entries fire at different rates, hence per-type). The OOS slice
    is untouched by this filter.
    """
    return [v for v in variants
            if v.n_is >= floors.get(v.trade_type, default_floor)]
