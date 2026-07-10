"""
Multi-folder plugin discovery + loading.

Unifies the per-view copies of the old plugin scanning/loading code onto one
API. The loading mechanics are unchanged:

- flat plugins (transforms, sizers, MC methods) — spec_from_file_location +
  exec_module, NO sys.modules registration (exactly how the old views loaded
  them; sibling imports inside a plugin dir keep working via the plugins' own
  sys.path hacks);
- strategies — flat .py OR package dir with __init__.py, loaded through
  modules.optimizer.backend.loader.load_strategy (registers in sys.modules so
  package-relative imports resolve — same loader the optimizer's process-pool
  workers use).

Discovery unions the configured folders in order; the in-repo default folder
comes first. A name collision across folders keeps every entry, disambiguated
in the UI label with the folder name.
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from modules.optimizer.backend.loader import load_strategy as _load_strategy_from_dir

# Filenames excluded from every plugin scan (helper modules, not plugins).
DEFAULT_EXCLUDE = ("__init__", "base")


@dataclass(frozen=True)
class PluginRef:
    """One discovered plugin: its stem name, file/dir path, source folder and
    the (possibly folder-disambiguated) UI label."""
    name: str          # module stem (the old display name)
    path: Path         # .py file, or the package dir for package strategies
    dir: Path          # the plugin folder it was found in
    label: str         # name, or "name  [folder]" when the name collides
    is_package: bool = False


def _disambiguate(refs: list[PluginRef]) -> list[PluginRef]:
    """Suffix labels with the folder name where a stem occurs in >1 folder."""
    counts = {}
    for r in refs:
        counts[r.name] = counts.get(r.name, 0) + 1
    return [
        PluginRef(r.name, r.path, r.dir,
                  f"{r.name}  [{r.dir.name}]" if counts[r.name] > 1 else r.name,
                  r.is_package)
        for r in refs
    ]


def list_plugins(dirs: list[Path], exclude=DEFAULT_EXCLUDE) -> list[PluginRef]:
    """Flat .py plugins from every folder, per-folder sorted (transforms,
    sizers, MC methods). Same scan rule as the old views: *.py minus
    __init__/base."""
    refs = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for f in sorted(d.glob("*.py")):
            if f.stem not in exclude:
                refs.append(PluginRef(f.stem, f, d, f.stem))
    return _disambiguate(refs)


def list_strategies(dirs: list[Path]) -> list[PluginRef]:
    """Strategies: flat .py files + package folders with __init__.py — the
    old views/backtester.get_strategies logic, per folder, sorted within
    each folder."""
    refs = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        found = []
        for f in d.glob("*.py"):                     # flat .py files
            if f.stem not in DEFAULT_EXCLUDE:
                found.append(PluginRef(f.stem, f, d, f.stem))
        for f in d.iterdir():                        # folders with __init__.py
            if f.is_dir() and (f / "__init__.py").exists():
                found.append(PluginRef(f.name, f, d, f.name, is_package=True))
        refs.extend(sorted(found, key=lambda r: r.name))
    return _disambiguate(refs)


def load_module(ref: PluginRef):
    """Load a flat plugin module by file path (verbatim mechanics of the old
    views' _load_module/load_transform/load_sizer)."""
    path = Path(ref.path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_strategy(ref: PluginRef):
    """Load a strategy (flat or package) through the pure optimizer loader,
    anchored at the ref's own folder — this is also what gets passed to
    run_grid(strategies_dir=...) so pool workers load the same file."""
    return _load_strategy_from_dir(ref.name, strategies_dir=ref.dir)
