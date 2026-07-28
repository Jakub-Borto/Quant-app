"""
User settings for the desktop app: configurable plugin folders + data roots.

Persisted as human-readable JSON at <repo>/settings.json (gitignored;
auto-created with defaults on first load). Schema:

    {
      "version": 1,
      "extra_plugin_dirs": {
        "strategies":      ["D:/somewhere/my_strategies", ...],
        "data_transforms": [...],
        "position_sizing": [...]
      },
      "data_roots": ["data", "E:/market_data"],
      "ui_prefs": {"trade_report_sections": {...}}
    }

Rules:
- For each plugin category the FIRST folder is always the in-repo default
  (strategies/, data_transforms/, position_sizing/) — it is re-prepended on
  every load and is not stored, so it can never be removed in the dialog.
- Data roots are fully user-editable (add/remove/reorder), minimum one.
  Each data root is a full tree: raw_dbn/, parquet/, trades/, optimizations/,
  news_and_holidays/. Relative entries resolve against the repo root (the
  shipped default is the in-repo "data" folder).
- ui_prefs is a free-form namespace for per-module UI preferences (section
  order/visibility, ...), keyed by a UI_PREF_* constant. Read/write it through
  ui_pref()/set_ui_pref() rather than touching the dict.

ADDING A KEY: to_json() rebuilds the document from scratch, so a new top-level
key must be added in FOUR places together — _DEFAULTS, __init__, to_json and
load_settings — or it is silently dropped the next time anything saves.
"""

import copy
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]   # modules/common/backend -> repo root
SETTINGS_PATH = REPO_ROOT / "settings.json"

# category key -> in-repo default folder name (the locked first entry)
PLUGIN_CATEGORIES = {
    "strategies":       "strategies",
    "data_transforms":  "data_transforms",
    "position_sizing":  "position_sizing",
    "scripts":          "scripts",
    "regime_detectors": "regime_detectors",
}

# Human labels for the settings dialog.
CATEGORY_LABELS = {
    "strategies":       "Strategy folders",
    "data_transforms":  "Data-transform folders",
    "position_sizing":  "Position-sizing folders",
    "scripts":          "Script folders",
    "regime_detectors": "Regime-detector folders",
}

# Default data root — the machine's dedicated data drive location (the data
# tree moved out of the in-repo data/ folder in July 2026).
DEFAULT_DATA_ROOT = "D:/market_data"

# ui_prefs keys (one per preference blob; see the module docstring).
UI_PREF_TRADE_REPORT = "trade_report_sections"

_DEFAULTS = {
    "version": 1,
    "extra_plugin_dirs": {key: [] for key in PLUGIN_CATEGORIES},
    "data_roots": [DEFAULT_DATA_ROOT],
    "ui_prefs": {},
}


def _resolve(entry: str) -> Path:
    """Resolve a stored path entry — relative entries anchor at the repo root."""
    p = Path(entry)
    return p if p.is_absolute() else (REPO_ROOT / p)


class Settings:
    """In-memory settings. Mutate extra_plugin_dirs / data_roots_raw, then save()."""

    def __init__(self, extra_plugin_dirs: dict[str, list[str]],
                 data_roots_raw: list[str], ui_prefs: dict | None = None):
        self.extra_plugin_dirs = {
            key: list(extra_plugin_dirs.get(key, [])) for key in PLUGIN_CATEGORIES
        }
        self.data_roots_raw = list(data_roots_raw) or [DEFAULT_DATA_ROOT]
        # keyword-with-default so every existing Settings(dirs, roots) call
        # site (including the tests) keeps working untouched
        self.ui_prefs = copy.deepcopy(dict(ui_prefs or {}))

    # ── plugin folders ────────────────────────────────────────────────────────
    def default_plugin_dir(self, category: str) -> Path:
        return REPO_ROOT / PLUGIN_CATEGORIES[category]

    def plugin_dirs(self, category: str) -> list[Path]:
        """[locked in-repo default] + the user's extra folders, in order."""
        return [self.default_plugin_dir(category)] + [
            _resolve(p) for p in self.extra_plugin_dirs[category]
        ]

    # ── data roots ────────────────────────────────────────────────────────────
    @property
    def data_roots(self) -> list[Path]:
        return [_resolve(p) for p in self.data_roots_raw]

    # ── per-module UI preferences ─────────────────────────────────────────────
    def ui_pref(self, key: str, default=None):
        """One UI preference blob (deep-copied — callers may mutate freely)."""
        value = self.ui_prefs.get(key)
        return copy.deepcopy(value) if value is not None else default

    def set_ui_pref(self, key: str, value) -> None:
        """Store a UI preference blob. Call save() to persist it."""
        self.ui_prefs[key] = copy.deepcopy(value)

    # ── persistence ───────────────────────────────────────────────────────────
    def to_json(self) -> dict:
        return {
            "version": 1,
            "extra_plugin_dirs": {k: list(v) for k, v in self.extra_plugin_dirs.items()},
            "data_roots": list(self.data_roots_raw),
            "ui_prefs": copy.deepcopy(self.ui_prefs),
        }

    def save(self, path: Path = SETTINGS_PATH) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2) + "\n",
                              encoding="utf-8")


def load_settings(path: Path = SETTINGS_PATH) -> Settings:
    """Load settings, creating the file with defaults on first run. A corrupt
    file falls back to defaults (and is rewritten on the next save)."""
    path = Path(path)
    if not path.exists():
        settings = Settings(_DEFAULTS["extra_plugin_dirs"], _DEFAULTS["data_roots"],
                            _DEFAULTS["ui_prefs"])
        settings.save(path)
        return settings

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Settings(raw.get("extra_plugin_dirs", {}),
                        raw.get("data_roots", [DEFAULT_DATA_ROOT]),
                        raw.get("ui_prefs", {}))
    except (json.JSONDecodeError, OSError):
        return Settings(_DEFAULTS["extra_plugin_dirs"], _DEFAULTS["data_roots"],
                        _DEFAULTS["ui_prefs"])
