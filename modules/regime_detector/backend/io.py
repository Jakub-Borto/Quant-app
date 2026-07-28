"""
Regime-run persistence & exploration I/O (pure, Qt-free).

A regime run is a folder {data_root}/regimes/{ASSET}/{run_name}/ holding one
YYYY-MM-DD.parquet per trading day plus a meta.json (the optimizations/
precedent: a folder with a meta.json IS a run). Daily files are written by
runner.RegimeContext; this module owns naming, meta round-trip, and the
Explore tab's readers.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

META_NAME = "meta.json"
NY = "America/New_York"


# ── small shared helpers (same shape as optimizer/backend/io.py) ─────────────

def jsonable(obj):
    """Recursively convert numpy/pandas scalars and dates for json.dump."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def safe_name(name: str, what: str = "run name") -> str:
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name.strip())
    if not safe:
        raise ValueError(f"empty {what}")
    return safe


def propose_run_name(asset: str, script_stem: str) -> str:
    """The prefilled run name: {SYMBOL}_{script_stem} (leading underscores of
    scaffold-style stems stripped so 'ES__scaffold_example' doesn't happen)."""
    return f"{asset}_{script_stem.lstrip('_')}"


# ── meta.json ────────────────────────────────────────────────────────────────

def write_meta(run_dir: Path, meta: dict) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / META_NAME, "w", encoding="utf-8") as f:
        json.dump(jsonable(meta), f, indent=2)


def read_meta(run_dir: Path) -> dict:
    with open(Path(run_dir) / META_NAME, encoding="utf-8") as f:
        return json.load(f)


def tiers(meta: dict) -> dict:
    """The run's {regime/score/diagnostic: [columns]} split.

    meta_version 2 nests it under schema.tiers; version-1 metas held the tier
    dict directly under schema. Every consumer goes through here."""
    block = meta.get("schema", {}) or {}
    return block.get("tiers", block)


# ── daily-file scanning / loading ────────────────────────────────────────────

def day_files(folder: Path) -> dict[str, Path]:
    """{YYYY-MM-DD: path} for a folder's daily parquets, sorted by date (the
    digit-stem filter from data_roots.available_dates — meta.json and any
    sidecars are naturally excluded)."""
    files = [f for f in Path(folder).glob("*.parquet") if f.stem[0:1].isdigit()]
    return {f.stem: f for f in sorted(files, key=lambda f: f.stem)}


def files_in_range(folder: Path, start: str | None = None,
                   end: str | None = None) -> dict[str, Path]:
    """day_files() narrowed to [start, end] (inclusive date strings)."""
    return {d: p for d, p in day_files(folder).items()
            if (start is None or d >= start) and (end is None or d <= end)}


def estimate_load_size(run_dir: Path, start: str | None = None,
                       end: str | None = None) -> tuple[int, float]:
    """
    (number of day files, estimated GB once loaded into memory) for a range.

    Parquet is compressed and columnar, so on-disk size understates the loaded
    footprint badly — instead ONE representative file is read and measured,
    then scaled by the file count. Costs a single file read, which is what
    makes it safe to call before deciding whether a load is affordable.
    """
    files = files_in_range(run_dir, start, end)
    if not files:
        return 0, 0.0
    paths = list(files.values())
    sample = paths[len(paths) // 2]
    try:
        frame = pd.read_parquet(sample)
    except (OSError, ValueError):
        return len(files), 0.0
    per_file = float(frame.memory_usage(deep=True).sum())
    return len(files), per_file * len(files) / (1024 ** 3)


def load_day_frames(run_dir: Path, start: str | None = None,
                    end: str | None = None,
                    on_progress=None) -> dict[str, pd.DataFrame]:
    """Read the run's daily frames for [start, end] (date strings, inclusive,
    None = unbounded). Progress per file via the universal callback."""
    files = files_in_range(run_dir, start, end)
    frames: dict[str, pd.DataFrame] = {}
    total = len(files)
    for i, (date, path) in enumerate(files.items()):
        if on_progress is not None:
            on_progress(i, total, date)
        frames[date] = pd.read_parquet(path)
    if on_progress is not None:
        on_progress(total, total, "loaded")
    return frames


# ── as-of row selection (the Explore tab's second selector) ──────────────────

FINAL = "final"


def pick_rows(frames: dict[str, pd.DataFrame], at: str,
              session_start: str) -> pd.DataFrame:
    """
    One row per day: the day's last snapshot at/before wall time `at`
    (HH:MM), or the file's final row for at=FINAL. Days with no snapshot yet
    at that time are dropped (nothing was knowable). Result is indexed by the
    RTH date (as Timestamps) and keeps every column plus 'snapshot_ts'.

    Cross-midnight rule: a wall time at/after `session_start` (e.g. 18:30 for
    an 18:00 globex open) belongs to the EVENING BEFORE the RTH date.
    """
    rows, dates = [], []
    for date, df in frames.items():
        if df.empty:
            continue
        if at == FINAL:
            row = df.iloc[-1]
        else:
            day = pd.Timestamp(date)
            if at >= session_start:               # HH:MM strings compare fine
                day = day - pd.Timedelta(days=1)
            target = pd.Timestamp(f"{day.date()} {at}", tz=NY)
            pos = df.index.searchsorted(target, side="right")
            if pos == 0:
                continue
            row = df.iloc[pos - 1]
        rows.append({**row.to_dict(), "snapshot_ts": row.name})
        dates.append(pd.Timestamp(date))
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"))
    return out.sort_index()


def snapshot_grid_labels(session_start: str, session_end: str,
                         minutes: int) -> list[str]:
    """The as-of dropdown options: clock-anchored HH:MM labels stepping
    `minutes` through the (possibly midnight-crossing) session, from the
    first boundary strictly after the open through the boundary at/after the
    close — session order (evening first), e.g. 18:30, 19:00, ..., 17:00."""
    start = pd.Timestamp(f"2000-01-01 {session_start}")
    end = pd.Timestamp(f"2000-01-01 {session_end}")
    if end <= start:                              # crosses midnight
        end += pd.Timedelta(days=1)
    step = pd.Timedelta(minutes=minutes)
    t = start.ceil(step)
    if t <= start:
        t += step
    last = end.ceil(step)                         # aligned close stays itself
    labels = []
    while t <= last:
        labels.append(t.strftime("%H:%M"))
        t += step
    return labels
