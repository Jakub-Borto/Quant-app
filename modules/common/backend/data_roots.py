"""
Multi-root data-folder scanning.

Every configured data root is a full tree following the project blueprint:

    <root>/raw_dbn/{type}/{ASSET}/{dataset}/    *.dbn.zst inputs
    <root>/parquet/{type}/{ASSET}/{dataset}/    YYYY-MM-DD.parquet candles
    <root>/trades/{name}.parquet                saved backtests
    <root>/optimizations/{run}/                 optimizer runs
    <root>/news_and_holidays/ff_usd_events.parquet

Pickers show the UNION across roots; a label gets a "[rootname]" suffix only
when the same entry exists in several roots. Outputs are always written back
to the root the input came from (each window remembers its ref's root).

The per-root directory walk is the verbatim three-level scan from the old
views (get_structure / get_parquet_structure); this module adds the
multi-root merging and the output-path helpers.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from modules.data_formatter.backend.scan import get_structure


@dataclass(frozen=True)
class DatasetRef:
    """One dataset folder inside one data root."""
    root: Path         # the data root
    source: str        # "raw_dbn" | "parquet"
    asset_type: str    # e.g. "Futures"
    asset: str         # UPPERCASE ticker, e.g. "ES"
    dataset: str       # dataset folder name
    label: str         # dataset, or "dataset  [rootname]" on collision

    @property
    def path(self) -> Path:
        return self.root / self.source / self.asset_type / self.asset / self.dataset


@dataclass(frozen=True)
class TradesRef:
    """One saved trades parquet inside one data root."""
    root: Path
    path: Path
    label: str         # filename, or "filename  [rootname]" on collision

    @property
    def filename(self) -> str:
        return self.path.name


# ── structure scanning ────────────────────────────────────────────────────────

def scan_structure(roots: list[Path], source: str = "parquet") -> dict:
    """
    Union of every root's {type: {asset: [DatasetRef, ...]}} for one source
    layer. Types/assets merge by name; dataset entries that exist in several
    roots each keep their own ref, disambiguated via the label.
    """
    merged: dict[str, dict[str, list[DatasetRef]]] = {}
    for root in roots:
        root = Path(root)
        structure = get_structure(root / source)
        for asset_type, assets in structure.items():
            for asset, datasets in assets.items():
                bucket = merged.setdefault(asset_type, {}).setdefault(asset, [])
                for dataset in datasets:
                    bucket.append(DatasetRef(root, source, asset_type, asset,
                                             dataset, dataset))

    # collision labels: same dataset name under the same type/asset in ≥2 roots
    for assets in merged.values():
        for asset, refs in assets.items():
            counts = {}
            for r in refs:
                counts[r.dataset] = counts.get(r.dataset, 0) + 1
            assets[asset] = [
                DatasetRef(r.root, r.source, r.asset_type, r.asset, r.dataset,
                           f"{r.dataset}  [{r.root.name}]"
                           if counts[r.dataset] > 1 else r.dataset)
                for r in refs
            ]
    return merged


def available_dates(folder_path: Path) -> list[pd.Timestamp]:
    """Sorted trading dates from a dataset folder's YYYY-MM-DD.parquet files
    (verbatim digit-stem filter from the old render_controls)."""
    return sorted([
        pd.Timestamp(f.stem) for f in Path(folder_path).glob("*.parquet")
        if f.stem[0].isdigit()
    ])


# ── trades files ──────────────────────────────────────────────────────────────

def list_trades_files(roots: list[Path]) -> list[TradesRef]:
    """Union of every root's trades/*.parquet, sorted by filename per root;
    filename collisions across roots get the root-name label suffix."""
    refs = []
    for root in roots:
        root = Path(root)
        tdir = root / "trades"
        if not tdir.exists():
            continue
        for p in sorted(tdir.glob("*.parquet")):
            refs.append(TradesRef(root, p, p.name))

    counts = {}
    for r in refs:
        counts[r.path.name] = counts.get(r.path.name, 0) + 1
    return [
        TradesRef(r.root, r.path,
                  f"{r.path.name}  [{r.root.name}]"
                  if counts[r.path.name] > 1 else r.path.name)
        for r in refs
    ]


# ── output locations (always inside a specific root) ─────────────────────────

def trades_dir(root: Path) -> Path:
    return Path(root) / "trades"


def temp_dir(root: Path) -> Path:
    return Path(root) / "temp"


def temp_trades_ref(path: Path) -> TradesRef:
    """Ref for a temp handoff file at <root>/temp/<name> — never produced by
    list_trades_files (temp files don't appear in regular pickers)."""
    p = Path(path)
    return TradesRef(root=p.parent.parent, path=p, label=p.name)


def clear_temp_files(roots: list[Path]) -> int:
    """
    Delete every {ASSET}_temp_file_{N}.parquet in every root's temp/ folder
    (app-startup cleanup — temp handoff files live for one app session).
    Only the app's own naming pattern is touched; anything else a user drops
    into temp/ is left alone. Best effort: locked/undeletable files are
    skipped. Returns the number of files deleted.
    """
    deleted = 0
    for root in roots:
        tdir = Path(root) / "temp"
        if not tdir.exists():
            continue
        for p in tdir.glob("*_temp_file_*.parquet"):
            try:
                p.unlink()
                deleted += 1
            except OSError:
                pass
    return deleted


def optimizations_root(root: Path) -> Path:
    return Path(root) / "optimizations"


def ff_events_path(root: Path) -> Path:
    return Path(root) / "news_and_holidays" / "ff_usd_events.parquet"


def resolve_ff_events(preferred_root: Path | None,
                      all_roots: list[Path]) -> Path | None:
    """
    The FF calendar parquet to use: the preferred root's copy when it exists
    (the root the run's dataset lives in), else the first configured root
    that has one, else None (-> empty bucket map, every day 'normal').
    """
    if preferred_root is not None:
        p = ff_events_path(preferred_root)
        if p.exists():
            return p
    for root in all_roots:
        p = ff_events_path(root)
        if p.exists():
            return p
    return None
