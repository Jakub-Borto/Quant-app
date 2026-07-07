"""
Persistence of an optimization run under data/optimizations/{run_name}/:

  trades.parquet   long-format trades table (source of truth)
  meta.json        everything needed to reproduce + explore the run without
                   re-running any backtest: strategy, dataset, ticker,
                   ticks_per_point, date range, axis definitions, held params,
                   be_band_ticks, EVENT_KEYWORDS used, split date, created_at

Nothing outside data/optimizations/ is ever written.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RUNS_ROOT = Path("data/optimizations")


def _jsonable(obj):
    """Recursively convert numpy/pandas scalars and dates for json.dump."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def save_run(run_name: str, trades: pd.DataFrame, meta: dict,
             root: Path = RUNS_ROOT) -> Path:
    """
    Write trades.parquet + meta.json under root/run_name (suffix _2, _3, ...
    when the folder already exists). Returns the run directory.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in run_name.strip())
    if not safe:
        raise ValueError("empty run name")

    run_dir = root / safe
    n = 2
    while run_dir.exists():
        run_dir = root / f"{safe}_{n}"
        n += 1
    run_dir.mkdir()

    trades.to_parquet(run_dir / "trades.parquet")
    meta = dict(meta)
    meta["run_name"] = run_dir.name
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(meta), f, indent=2)
    return run_dir


def list_runs(root: Path = RUNS_ROOT) -> list:
    """Saved run names (folders holding a meta.json), newest first."""
    root = Path(root)
    if not root.exists():
        return []
    runs = [d for d in root.iterdir() if d.is_dir() and (d / "meta.json").exists()]
    runs.sort(key=lambda d: (d / "meta.json").stat().st_mtime, reverse=True)
    return [d.name for d in runs]


def load_run(run_name: str, root: Path = RUNS_ROOT):
    """(trades, meta) of a saved run — no backtests re-run."""
    run_dir = Path(root) / run_name
    trades = pd.read_parquet(run_dir / "trades.parquet")
    with open(run_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return trades, meta
