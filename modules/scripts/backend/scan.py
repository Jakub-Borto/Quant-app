"""Script classification + file metadata for the Scripts module.

A script is a *Streamlit* script only if it declares so inside the file —
never by filename. Either form, in the first 30 lines, marks it:

    # app: streamlit
    STREAMLIT = True

No marker means a plain Python script (run with `python -u`).
"""

import re
from datetime import datetime
from pathlib import Path

_MARKER_SCAN_LINES = 30
_COMMENT_MARKER = re.compile(r"^\s*#\s*app\s*:\s*streamlit\s*$", re.IGNORECASE)
_CONSTANT_MARKER = re.compile(r"^STREAMLIT\s*=\s*True\b")


def is_streamlit_script(path) -> bool:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= _MARKER_SCAN_LINES:
                    break
                if _COMMENT_MARKER.match(line) or _CONSTANT_MARKER.match(line):
                    return True
    except OSError:
        return False
    return False


def script_mtime(path) -> datetime:
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime)
    except OSError:
        return datetime.fromtimestamp(0)


_EXCLUDED_STEMS = {"__init__", "base"}


def list_folder(path) -> tuple[list[Path], list[Path]]:
    """(subfolders, scripts) directly inside `path`, each name-sorted.

    Folders starting with "." or "_" (venvs, __pycache__, .git) are hidden;
    scripts follow the plugin exclusion rules (__init__/base skipped).
    Nesting is arbitrary — a subfolder may hold more subfolders.
    """
    try:
        entries = sorted(Path(path).iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return [], []
    folders = [p for p in entries
               if p.is_dir() and not p.name.startswith((".", "_"))]
    scripts = [p for p in entries
               if p.is_file() and p.suffix == ".py"
               and p.stem not in _EXCLUDED_STEMS]
    return folders, scripts


def folder_summary(path) -> tuple[int, int]:
    """(n_subfolders, n_scripts) directly inside `path` — for folder rows."""
    folders, scripts = list_folder(path)
    return len(folders), len(scripts)
