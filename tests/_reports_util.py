"""Shared helpers for the Reports tests: synthetic Optimizer runs written
with the Optimizer's own io.save_run, and the Explore tab's grid computation
reproduced verbatim (filter by slider values -> compute_metrics_by_cell over
(x, y) -> _build_grid_arrays) as the reference every report cell must equal."""

import hashlib
import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from modules.optimizer.backend import io as opt_io
from modules.optimizer.backend.heatmap_model import _build_grid_arrays
from modules.optimizer.backend.metrics import compute_metrics_by_cell
from modules.optimizer.backend.param_space import ROLES

AXES_BY_N = {
    1: [{"param": "stop", "values": [4, 6, 8, 10]}],
    2: [{"param": "stop", "values": [4, 6, 8]},
        {"param": "target", "values": [0.5, 1.0, 1.5, 2.0]}],
    3: [{"param": "stop", "values": [4, 6, 8]},
        {"param": "target", "values": [0.5, 1.0, 1.5]},
        {"param": "side", "values": ["long", "short"]}],
    4: [{"param": "stop", "values": [4, 6]},
        {"param": "target", "values": [0.5, 1.0, 1.5]},
        {"param": "side", "values": ["both", "long", "short"]},
        {"param": "filter_on", "values": [False, True]}],
}


def make_trades(axes, seed=0, empty_combo=True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=60)
    names = [a["param"] for a in axes]
    rows = []
    combos = list(product(*[a["values"] for a in axes]))
    for n, combo in enumerate(combos):
        if empty_combo and n == 1:
            continue                                   # a zero-trade cell
        k = 3 if n == 2 else int(rng.integers(8, 40))  # n == 2: below min trades
        days = rng.choice(dates, size=k)
        pnl = np.round(rng.normal(0.4, 6.0, size=k), 2)
        if n == 3:
            pnl = np.abs(pnl) + 0.25                   # no losses -> PF inf
        for d, p in zip(days, pnl):
            rows.append({**dict(zip(names, combo)), "pnl_ticks": float(p),
                         "date": pd.Timestamp(d).date(), "day_bucket": "normal",
                         "direction": "long", "entry_price": 100.0})
    df = pd.DataFrame(rows)
    return df.sort_values(names + ["date"], kind="stable").reset_index(drop=True)


def make_meta(axes, trades) -> dict:
    by_role = dict(zip(ROLES, axes))
    return {
        "strategy": "toy", "dataset": "futures/ES/ES_1m", "ticker": "ES",
        "tick_size": 0.25, "ticks_per_point": 4,
        "start_date": "2025-01-02", "end_date": "2025-03-26",
        "axes": {r: by_role.get(r) for r in ROLES},
        "fixed_params": {"session": "rth", "risk": 1.0},
        "min_trades_default": 5, "be_band_ticks": 0.5,
        "split_date": "2025-02-12", "n_combos": int(np.prod([len(a["values"]) for a in axes])),
        "n_trades": len(trades), "created_at": "2025-04-01T10:00:00",
    }


def write_run(parent: Path, n_params: int, name: str = None, seed: int = 0) -> Path:
    axes = AXES_BY_N[n_params]
    trades = make_trades(axes, seed=seed)
    meta = make_meta(axes, trades)
    return opt_io.save_run(trades, meta, run_name=name or f"run_{n_params}p",
                           root=parent)


def explore_arrays(trades, meta, x_axis, y_axis, slider_values) -> dict:
    """The Explore tab's _refresh_heatmap grid (no bucket filter, both halves)."""
    df = trades
    for p, v in slider_values.items():
        df = df[df[p] == v]
    cell_cols = [x_axis["param"]] + ([y_axis["param"]] if y_axis else [])
    grid = compute_metrics_by_cell(df, cell_cols, meta.get("be_band_ticks", 0.0))
    return _build_grid_arrays(grid, x_axis["param"], x_axis["values"],
                              y_axis["param"] if y_axis else None,
                              y_axis["values"] if y_axis else None)


def every_layout(axes):
    """(x_axis, y_axis | None, slider_values) for every axis pair and every
    slice of the remaining params."""
    if len(axes) == 1:
        yield axes[0], None, {}
        return
    for x in axes:
        for y in axes:
            if x is y:
                continue
            rest = [a for a in axes if a is not x and a is not y]
            for vals in product(*[a["values"] for a in rest]):
                yield x, y, {a["param"]: v for a, v in zip(rest, vals)}


def tree_hashes(folder) -> dict:
    out = {}
    for root, _dirs, files in os.walk(folder):
        for f in files:
            p = Path(root) / f
            out[p.relative_to(folder).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out
