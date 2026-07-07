"""
Persistence of an optimization run under data/optimizations/, either at the
root or grouped one level deep in a user-named folder (Data Formatter logic):

  {run_name}/  or  {folder}/{run_name}/
      trades.parquet   long-format trades table (source of truth)
      meta.json        everything needed to reproduce + explore the run
                       without re-running any backtest

A directory holding a meta.json IS a run; a directory without one is a
grouping folder. Runs are addressed by their path relative to the root
("my_run" or "my_folder/my_run"). Nothing outside data/optimizations/ is
ever written.
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


def _safe_name(name: str, what: str) -> str:
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name.strip())
    if not safe:
        raise ValueError(f"empty {what}")
    return safe


def save_run(trades: pd.DataFrame, meta: dict, run_name: str,
             folder: str = "", root: Path = RUNS_ROOT) -> Path:
    """
    Write trades.parquet + meta.json under root[/folder]/run_name (suffix _2,
    _3, ... when the run directory already exists; the folder — new or
    existing — is created as needed). Returns the run directory; the saved
    meta's run_name is the root-relative path ("folder/run" or "run").
    """
    root = Path(root)
    parent = root / _safe_name(folder, "folder name") if folder.strip() else root
    parent.mkdir(parents=True, exist_ok=True)

    safe = _safe_name(run_name, "run name")
    run_dir = parent / safe
    n = 2
    while run_dir.exists():
        run_dir = parent / f"{safe}_{n}"
        n += 1
    run_dir.mkdir()

    trades.to_parquet(run_dir / "trades.parquet")
    meta = dict(meta)
    meta["run_name"] = run_dir.relative_to(root).as_posix()
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(meta), f, indent=2)
    return run_dir


def list_runs(root: Path = RUNS_ROOT) -> list:
    """
    Root-relative paths of all saved runs ("run" or "folder/run"),
    newest first.
    """
    root = Path(root)
    if not root.exists():
        return []
    run_dirs = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if (d / "meta.json").exists():
            run_dirs.append(d)
        else:                                   # grouping folder — one level
            run_dirs.extend(sub for sub in d.iterdir()
                            if sub.is_dir() and (sub / "meta.json").exists())
    run_dirs.sort(key=lambda d: (d / "meta.json").stat().st_mtime, reverse=True)
    return [d.relative_to(root).as_posix() for d in run_dirs]


def list_folders(root: Path = RUNS_ROOT) -> list:
    """Existing grouping folders (dirs that are not themselves runs)."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and not (d / "meta.json").exists())


def load_run(run_name: str, root: Path = RUNS_ROOT):
    """(trades, meta) of a saved run ("run" or "folder/run") — no re-runs."""
    run_dir = Path(root) / run_name
    trades = pd.read_parquet(run_dir / "trades.parquet")
    with open(run_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return trades, meta
