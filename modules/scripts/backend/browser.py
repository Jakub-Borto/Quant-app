"""Dedicated browser window for Streamlit scripts.

Launching Chrome/Edge with a private `--user-data-dir` creates a browser
instance fully separate from the user's own browser (Opera GX tabs are
never touched). The first launch opens the dedicated window; every later
URL passed with the same profile dir is routed into that instance and opens
as a NEW TAB in it. Closing the window ends the instance, so the next run
opens a fresh window again.

Opera/Opera GX can't be a candidate: it ignores these Chromium flags and
would open a tab in the user's current window — the exact thing this
exists to avoid. Chrome and Edge both behave; Edge ships with Windows, so
in practice there is always a hit. Callers still handle None by falling
back to the default browser.
"""

import os
from pathlib import Path


def _candidates() -> list[Path]:
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    return [
        Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
        Path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
        local / "Google/Chrome/Application/chrome.exe",
        Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
        Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
    ]


def find_app_browser() -> Path | None:
    for p in _candidates():
        if p.exists():
            return p
    return None


def profile_dir() -> Path:
    """The dedicated profile folder (created on demand) that makes the
    scripts browser its own instance."""
    base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    d = base / "QuantResearchPlatform" / "scripts_browser_profile"
    d.mkdir(parents=True, exist_ok=True)
    return d


def launch_args(url: str) -> list[str]:
    """Arguments that open `url` in the dedicated instance — first call
    opens its window, subsequent calls add tabs to it."""
    return [f"--user-data-dir={profile_dir()}",
            "--no-first-run", "--no-default-browser-check", url]
