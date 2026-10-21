"""
Saved views for kit components — the file layer (pure, no Streamlit).

One file per report: `<report folder>/.report_state.json`
    {"version": 1, "components": {"<key>": {<view fields>, "locked": bool}}}

- Writes are atomic (temporary file + os.replace) and touch only the one
  component's entry; keys no component uses any more are left alone.
- A missing file = no saved views. A corrupt/unreadable file is set aside as
  `.report_state.json.bad` (never silently overwritten) and reported once —
  check_state_file returns the warning text.
- Two browser tabs on one report: last write wins.
"""

import json
import os
import time
from pathlib import Path

STATE_FILE = ".report_state.json"
STATE_VERSION = 1


def state_path(report_dir) -> Path:
    return Path(report_dir) / STATE_FILE


def _parse(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    comps = data.get("components", {})
    if not isinstance(comps, dict):
        raise ValueError("'components' is not an object")
    return data


def check_state_file(report_dir) -> str | None:
    """Set a corrupt state file aside (`.bad`) and return a warning; None
    when the file is fine or absent."""
    path = state_path(report_dir)
    if not path.exists():
        return None
    try:
        _parse(path)
        return None
    except (OSError, ValueError) as e:      # JSONDecodeError is a ValueError
        bad = path.with_name(STATE_FILE + ".bad")
        n = 2
        while bad.exists():
            bad = path.with_name(f"{STATE_FILE}.bad{n}")
            n += 1
        try:
            os.replace(path, bad)
            where = f"it was kept as {bad.name}"
        except OSError:
            where = "it could not be renamed and is left in place"
        return (f"Saved views could not be read ({STATE_FILE}: {e}) — every "
                f"component starts from its defaults; {where}.")


def read_components(report_dir) -> dict:
    """{key: entry} of the saved views ({} when absent or corrupt)."""
    path = state_path(report_dir)
    if not path.exists():
        return {}
    try:
        return dict(_parse(path).get("components", {}))
    except (OSError, ValueError):
        return {}


def write_entry(report_dir, key: str, entry: dict) -> None:
    """Atomically replace one component's saved view."""
    path = state_path(report_dir)
    data = {"version": STATE_VERSION, "components": {}}
    if path.exists():
        try:
            data = _parse(path)
        except (OSError, ValueError):
            check_state_file(report_dir)    # never overwrite a corrupt file
            data = {"version": STATE_VERSION, "components": {}}
    data.setdefault("components", {})[key] = entry
    data["version"] = STATE_VERSION
    tmp = path.with_name(f"{STATE_FILE}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str),
                   encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:             # Windows: briefly held by a reader
            if attempt == 4:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.05)
