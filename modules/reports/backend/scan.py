"""
Report + archive discovery across the configured reports folders.

A report is a DIRECT subfolder of a reports folder that contains `report.py`
(the entry point); its display name is the folder name. Ignored: folders
whose name starts with "_" or "." (this covers the `_archive/` folder and the
hidden temporary folders the archive/restore/compress tools write) and
folders without `report.py`.

Archives live in `<reports folder>/_archive/<report>__<YYYY-MM-DD_HHMMSS>.zip`.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ENTRY_POINT = "report.py"
ARCHIVE_DIR = "_archive"
ARCHIVE_SEP = "__"
ARCHIVE_TS_FORMAT = "%Y-%m-%d_%H%M%S"


@dataclass(frozen=True)
class ReportRef:
    name: str
    path: Path          # the report folder
    folder: Path        # the reports folder it was found in
    modified: datetime  # newest file anywhere inside the report folder

    @property
    def entry(self) -> Path:
        return self.path / ENTRY_POINT


@dataclass(frozen=True)
class ArchiveRef:
    name: str              # report name
    archived_at: datetime | None
    size: int              # bytes
    folder: Path           # the reports folder (parent of _archive/)
    path: Path             # the zip


def is_hidden_name(name: str) -> bool:
    return name.startswith(("_", "."))


def newest_mtime(folder) -> datetime:
    """The newest file mtime anywhere under `folder` (the folder's own mtime
    when it holds no files)."""
    folder = Path(folder)
    try:
        newest = folder.stat().st_mtime
    except OSError:
        return datetime.fromtimestamp(0)
    stack = [folder]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            newest = max(newest, entry.stat().st_mtime)
                    except OSError:
                        continue
        except OSError:
            continue
    return datetime.fromtimestamp(newest)


def list_reports(folders) -> list[ReportRef]:
    """Every report in every reports folder, name-sorted within each folder
    (folders in the given order — two folders may hold a report of the same
    name; both are listed)."""
    refs = []
    for folder in folders:
        folder = Path(folder)
        try:
            entries = sorted(folder.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for d in entries:
            if d.is_dir() and not is_hidden_name(d.name) \
                    and (d / ENTRY_POINT).is_file():
                refs.append(ReportRef(d.name, d, folder, newest_mtime(d)))
    return refs


def archive_name(report_name: str, when: datetime) -> str:
    return f"{report_name}{ARCHIVE_SEP}{when.strftime(ARCHIVE_TS_FORMAT)}.zip"


def parse_archive_name(filename: str) -> tuple[str, datetime | None]:
    """'<report>__<timestamp>.zip' -> (report, timestamp); a zip that does
    not follow the pattern keeps its stem as the name and no timestamp."""
    stem = filename[:-4] if filename.lower().endswith(".zip") else filename
    name, sep, ts = stem.rpartition(ARCHIVE_SEP)
    if sep:
        base_ts = ts.split("_")
        try:
            # a collision suffix ("..._HHMMSS_2") is not part of the time
            return name, datetime.strptime("_".join(base_ts[:2]), ARCHIVE_TS_FORMAT)
        except ValueError:
            pass
    return stem, None


def list_archives(folders) -> list[ArchiveRef]:
    """Every archived zip under `<folder>/_archive/`, newest first."""
    refs = []
    for folder in folders:
        folder = Path(folder)
        arch = folder / ARCHIVE_DIR
        try:
            zips = [p for p in arch.iterdir()
                    if p.is_file() and p.suffix.lower() == ".zip"
                    and not p.name.startswith(".")]
        except OSError:
            continue
        for z in zips:
            name, when = parse_archive_name(z.name)
            try:
                size = z.stat().st_size
            except OSError:
                size = 0
            refs.append(ArchiveRef(name, when, size, folder, z))
    refs.sort(key=lambda r: (r.archived_at or datetime.fromtimestamp(0)),
              reverse=True)
    return refs


def owning_reports_folder(path, folders) -> Path | None:
    """The reports folder `path` lies inside (strictly or equal), else None.
    Paths are compared resolved, so `..` tricks and different spellings of
    the same folder behave."""
    try:
        p = Path(path).resolve()
    except OSError:
        return None
    for folder in folders:
        try:
            root = Path(folder).resolve()
        except OSError:
            continue
        if p == root or root in p.parents:
            return Path(folder)
    return None


def is_in_archive(path, folder) -> bool:
    """True when `path` lies inside `<folder>/_archive/`."""
    try:
        p = Path(path).resolve()
        arch = (Path(folder) / ARCHIVE_DIR).resolve()
    except OSError:
        return False
    return p == arch or arch in p.parents


def human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def folder_size(folder) -> int:
    total = 0
    for root, _dirs, files in os.walk(folder):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total
