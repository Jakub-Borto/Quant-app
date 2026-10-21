"""
Optimizer runs inside reports: loading (full OR compact), the per-combination
summary, 2-D slicing, and the Compress tool — pure, Qt-free.

A FULL run is a copy of an Optimizer run folder (trades.parquet + meta.json,
the on-disk contract of modules/optimizer/backend/io.py). A COMPACT run is
what Compress writes: summary.parquet (one row per parameter combination —
the swept values + every heatmap metric, NO trades) + the source meta.json
with an added `compact` block.

One implementation of the numbers: the summary is
`compute_metrics_by_cell(trades, <all swept params>, be_band)` — the very
function the Optimizer's Explore heatmap calls — reindexed onto the full
cartesian product, so zero-trade combinations exist as all-NaN rows exactly
like the Explore grid's reindexed cells. Explore (no day-type filter, both
halves) groups the slider-filtered trades by (x, y); grouping by every swept
param and then slicing yields the same groups over the same rows in the same
order, so every cell is identical (tests/test_reports_compact.py pins it for
2-, 3- and 4-parameter runs, full vs compact vs Explore).

All values are pre-cost — exactly what the Optimizer shows.
"""

import json
import os
import secrets
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from modules.optimizer.backend.metrics import METRIC_ORDER, compute_metrics_by_cell
from modules.optimizer.backend.param_space import ROLES, enumerate_combos

from .scan import (folder_size, is_hidden_name, is_in_archive,
                   owning_reports_folder)

TRADES_FILE = "trades.parquet"
META_FILE = "meta.json"
SUMMARY_FILE = "summary.parquet"
COMPACT_FORMAT_VERSION = 1
TRADE_COLUMNS_NEEDED = ["pnl_ticks", "date"]


class RunError(Exception):
    """A missing / malformed run (message names what is wrong)."""


class CompressError(Exception):
    """A user-facing Compress refusal or failure."""


class TargetExists(CompressError):
    """`<destination>/<run_name>/` exists — the caller must ask before
    calling again with replace=True."""


@dataclass
class LoadedRun:
    path: Path
    meta: dict
    summary: pd.DataFrame       # one row per combination: params + METRIC_ORDER
    compact: bool

    @property
    def axes(self) -> list[dict]:
        return swept_axes(self.meta)


# ── meta helpers ─────────────────────────────────────────────────────────────

def swept_axes(meta: dict) -> list[dict]:
    """The run's swept params ({"param", "values"}) in their saved role
    order (x, y, slider, slider2) — the Explore tab's default layout."""
    axes = (meta or {}).get("axes") or {}
    return [axes[r] for r in ROLES if axes.get(r)]


def run_kind(path) -> str | None:
    """'compact', 'full' or None for a folder."""
    path = Path(path)
    if not (path / META_FILE).is_file():
        return None
    if (path / SUMMARY_FILE).is_file():
        return "compact"
    if (path / TRADES_FILE).is_file():
        return "full"
    return None


def _read_meta(path: Path) -> dict:
    try:
        with open(path / META_FILE, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise RunError(f"meta.json unreadable: {e}") from e
    if not isinstance(meta, dict):
        raise RunError("meta.json is not a JSON object")
    if not swept_axes(meta):
        raise RunError("meta.json has no swept axes (meta.axes)")
    return meta


def _parquet_columns(path: Path) -> list[str]:
    try:
        return list(pq.read_schema(path).names)
    except Exception as e:  # noqa: BLE001 — any unreadable parquet
        raise RunError(f"{path.name} unreadable: {e}") from e


# ── the summary (one row per combination) ────────────────────────────────────

def _full_index(axes: list[dict]):
    names = [a["param"] for a in axes]
    if len(axes) == 1:
        return pd.Index(axes[0]["values"], name=names[0])
    return pd.MultiIndex.from_product([a["values"] for a in axes], names=names)


def summarize_trades(trades: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Per-combination metrics with the Optimizer's own compute_metrics_by_cell,
    reindexed onto every combination of the swept axes (zero-trade
    combinations = all-NaN rows, as in the Explore grid)."""
    axes = swept_axes(meta)
    cols = [a["param"] for a in axes]
    missing = [c for c in cols + TRADE_COLUMNS_NEEDED if c not in trades.columns]
    if missing:
        raise RunError(f"trades.parquet is missing columns: {', '.join(missing)}")
    be_band = float(meta.get("be_band_ticks", 0.0))
    grid = compute_metrics_by_cell(trades, cols, be_band)
    out = grid.reindex(_full_index(axes))
    return out[METRIC_ORDER].reset_index()


def slice_grid(summary: pd.DataFrame, x_param: str, y_param,
               slider_values: dict) -> pd.DataFrame:
    """The 2-D (or 1-D) slice the heatmap shows: rows matching every slider
    value, indexed by the cell columns — the shape compute_metrics_by_cell
    returns, ready for heatmap_model._build_grid_arrays."""
    df = summary
    for param, value in slider_values.items():
        df = df[df[param] == value]
    cell_cols = [x_param] + ([y_param] if y_param is not None else [])
    return df.set_index(cell_cols)[METRIC_ORDER]


# ── loading ──────────────────────────────────────────────────────────────────

def load_run(path) -> LoadedRun:
    """Load a full or compact run folder. Raises RunError naming the problem
    (missing files, malformed meta, missing columns)."""
    path = Path(path)
    if not path.is_dir():
        raise RunError("folder not found")
    kind = run_kind(path)
    if kind is None:
        raise RunError("not an Optimizer run — needs trades.parquet + meta.json "
                       "(full run) or summary.parquet + meta.json (compact run)")
    meta = _read_meta(path)
    cols = [a["param"] for a in swept_axes(meta)]

    if kind == "compact":
        have = _parquet_columns(path / SUMMARY_FILE)
        missing = [c for c in cols + METRIC_ORDER if c not in have]
        if missing:
            raise RunError(f"summary.parquet is missing columns: {', '.join(missing)}")
        summary = pd.read_parquet(path / SUMMARY_FILE, columns=cols + METRIC_ORDER)
        return LoadedRun(path, meta, summary, compact=True)

    have = _parquet_columns(path / TRADES_FILE)
    missing = [c for c in cols + TRADE_COLUMNS_NEEDED if c not in have]
    if missing:
        raise RunError(f"trades.parquet is missing columns: {', '.join(missing)}")
    trades = pd.read_parquet(path / TRADES_FILE, columns=cols + TRADE_COLUMNS_NEEDED)
    return LoadedRun(path, meta, summarize_trades(trades, meta), compact=False)


def run_signature(path) -> tuple:
    """(path, mtime/size of every run file) — the cache key that changes
    whenever the run's files change."""
    path = Path(path)
    sig = [str(path.resolve())]
    for name in (META_FILE, TRADES_FILE, SUMMARY_FILE):
        p = path / name
        try:
            st = p.stat()
            sig.append((name, st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append((name, None, None))
    return tuple(sig)


# ── the Compress tool ────────────────────────────────────────────────────────

def find_runs(folders) -> list[tuple[Path, str]]:
    """(run folder, kind) for every Optimizer run inside the reports folders
    (hidden / `_archive` folders skipped; a run folder is not descended)."""
    found = []
    for folder in folders:
        folder = Path(folder)
        if not folder.is_dir():
            continue
        for root, dirs, _files in os.walk(folder):
            dirs[:] = sorted(d for d in dirs if not is_hidden_name(d))
            kind = run_kind(root)
            if kind is not None:
                found.append((Path(root), kind))
                dirs[:] = []
    return found


def _check_inside(path: Path, folders, what: str) -> Path:
    owner = owning_reports_folder(path, folders)
    if owner is None:
        raise CompressError(f"The {what} must be inside a reports folder "
                            f"(Settings → Report folders): {path}")
    if is_in_archive(path, owner):
        raise CompressError(f"The {what} must not be inside _archive/: {path}")
    return owner


def compress_target(source, destination) -> Path:
    return Path(destination) / Path(source).name


def compress_run(source, destination, folders, replace: bool = False,
                 on_progress=None) -> dict:
    """Write `<destination>/<run_name>/` = the compact form of the full run
    `source`. The source is never modified. Output goes to a hidden
    temporary folder, is verified (one row per combination, every
    combination present) and only then renamed into place; cancel (an
    exception inside on_progress) or failure leaves no partial output.

    Returns {"target", "n_combinations", "source_bytes", "result_bytes"}.
    Raises TargetExists when the target exists and replace is False."""
    on_progress = on_progress or (lambda *_a: None)
    source = Path(source)
    destination = Path(destination)
    owner = _check_inside(source, folders, "source")
    _check_inside(destination, folders, "destination")

    kind = run_kind(source)
    if kind == "compact":
        raise CompressError(f"{source.name} is already a compact run — nothing "
                            "to compress.")
    if kind != "full":
        raise CompressError(f"{source} is not an Optimizer run (needs "
                            "trades.parquet + meta.json).")

    src_res, dst_res = source.resolve(), destination.resolve()
    if dst_res == src_res or src_res in dst_res.parents:
        raise CompressError("The destination must not be the source run or a "
                            "folder inside it — the source is never modified.")
    target = destination / source.name
    if target.exists() and not replace:
        raise TargetExists(f"{target} already exists.")

    steps = 5
    on_progress(0, steps, f"Reading {source.name}…")
    try:
        run = load_run(source)
    except RunError as e:
        raise CompressError(f"{source.name}: {e}") from e
    on_progress(1, steps, "Computed the per-combination metrics")

    axes = run.axes
    combos = enumerate_combos([{"param": a["param"], "values": a["values"]}
                               for a in axes])
    meta = dict(run.meta)
    meta["compact"] = {
        "format_version": COMPACT_FORMAT_VERSION,
        "source": source.resolve().relative_to(Path(owner).resolve()).as_posix(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_combinations": len(combos),
    }

    destination.mkdir(parents=True, exist_ok=True)
    tmp = destination / f".{source.name}.compact_tmp_{secrets.token_hex(3)}"
    tmp.mkdir()
    aside = None
    try:
        run.summary.to_parquet(tmp / SUMMARY_FILE, index=False)
        with open(tmp / META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        on_progress(2, steps, "Wrote summary.parquet + meta.json")

        _verify_summary(tmp, axes, combos)
        on_progress(3, steps, "Verified: one row per combination, all present")

        if target.exists():
            aside = destination / f".{source.name}.replaced_{secrets.token_hex(3)}"
            os.rename(target, aside)
        try:
            os.rename(tmp, target)
        except OSError:
            if aside is not None:
                os.rename(aside, target)        # put the old one back
                aside = None
            raise
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if aside is not None:
        shutil.rmtree(aside, ignore_errors=True)

    result = {
        "target": target,
        "n_combinations": len(combos),
        "source_bytes": folder_size(source),
        "result_bytes": folder_size(target),
    }
    on_progress(steps, steps, f"Compressed {source.name} → {target}")
    return result


def _verify_summary(folder: Path, axes: list[dict], combos: list[dict]) -> None:
    summary = pd.read_parquet(folder / SUMMARY_FILE)
    if len(summary) != len(combos):
        raise CompressError(f"Verification failed: {len(summary)} rows, "
                            f"{len(combos)} combinations in the source")
    names = [a["param"] for a in axes]
    got = set(zip(*(summary[n].tolist() for n in names)))
    expected = {tuple(c[n] for n in names) for c in combos}
    if got != expected:
        raise CompressError("Verification failed: the summary does not contain "
                            "every combination of the source run")
