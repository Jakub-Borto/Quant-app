"""
Pure strategy loader — no Streamlit import, usable from views AND from
process-pool workers (a worker importing a view would drag Streamlit into
every worker process).

Paths are anchored to the repo root (not the cwd) so workers and tests load
strategies regardless of where the process was started. `strategies_dir` can
be overridden, which lets tests exercise the real pool path against a toy
strategy written to tmp_path instead of polluting strategies/ (whose contents
feed the UI dropdown).
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STRATEGIES_DIR = REPO_ROOT / "strategies"


def load_strategy(name: str, strategies_dir=None):
    """Import a strategy by name — flat file or package — and validate run()."""
    strategies_dir = STRATEGIES_DIR if strategies_dir is None \
        else Path(strategies_dir)

    flat_path   = strategies_dir / f"{name}.py"
    folder_path = strategies_dir / name / "__init__.py"

    if flat_path.exists():
        path   = flat_path
        is_pkg = False
    elif folder_path.exists():
        path   = folder_path
        is_pkg = True
    else:
        raise FileNotFoundError(f"Strategy '{name}' not found in {strategies_dir}")

    if is_pkg:
        spec = importlib.util.spec_from_file_location(
            name, path,
            submodule_search_locations=[str(strategies_dir / name)],
        )
    else:
        spec = importlib.util.spec_from_file_location(name, path)

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # register so relative imports resolve
    spec.loader.exec_module(module)

    if not hasattr(module, "run") or not callable(module.run):
        raise ValueError(f"Strategy '{name}' has no callable run()")
    return module
