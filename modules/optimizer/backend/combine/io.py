"""
Persistence of a combine run under
data/optimizations/{container}/_combined/{combine_run_name}/:

  path.parquet     one row per path point: k, stage, member vids, IS/OOS
                   ticks + Sharpe + max DD, is_oos_peak flag
  members.parquet  long: (k, stage) -> member identity (run, trade_type,
                   params as JSON) — a chosen set is fully reproducible
  meta.json        container, ticked runs, filters, split, λ, seeds, shared
                   window, created_at

The `_combined/` prefix keeps combine outputs out of the entry-run discovery
(pool.discover_entry_runs skips it). Writes are confined to this folder.
"""

import json
from pathlib import Path

import pandas as pd

from ..io import RUNS_ROOT, _jsonable
from .pool import COMBINED_DIR


def _combined_root(container: str, root: Path = RUNS_ROOT) -> Path:
    return Path(root) / container / COMBINED_DIR


def save_combine_run(container: str, name: str, path_df: pd.DataFrame,
                     members_df: pd.DataFrame, meta: dict,
                     root: Path = RUNS_ROOT) -> Path:
    """Write the three artifacts (name suffixed _2, _3, ... on collision)."""
    base = _combined_root(container, root)
    base.mkdir(parents=True, exist_ok=True)

    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name.strip())
    if not safe:
        raise ValueError("empty combine run name")
    run_dir = base / safe
    n = 2
    while run_dir.exists():
        run_dir = base / f"{safe}_{n}"
        n += 1
    run_dir.mkdir()

    path_df.to_parquet(run_dir / "path.parquet")
    members_df.to_parquet(run_dir / "members.parquet")
    meta = dict(meta)
    meta["combine_run_name"] = run_dir.name
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(meta), f, indent=2)
    return run_dir


def list_combine_runs(container: str, root: Path = RUNS_ROOT) -> list:
    base = _combined_root(container, root)
    if not base.exists():
        return []
    runs = [d for d in base.iterdir()
            if d.is_dir() and (d / "path.parquet").exists()]
    runs.sort(key=lambda d: (d / "path.parquet").stat().st_mtime, reverse=True)
    return [d.name for d in runs]


def load_combine_run(container: str, name: str, root: Path = RUNS_ROOT):
    """(path_df, members_df, meta) — no backtests re-run, ever."""
    run_dir = _combined_root(container, root) / name
    path_df    = pd.read_parquet(run_dir / "path.parquet")
    members_df = pd.read_parquet(run_dir / "members.parquet")
    with open(run_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return path_df, members_df, meta
