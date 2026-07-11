"""
Qt smoke tests for the PySide6 desktop app (replaces the old Streamlit
AppTest smoke test).

- Every module window + the main menu + the settings dialog must construct
  offscreen against the repo data/ root without raising.
- The optimizer window must expose its three tabs.
- Spawn safety: importing the optimizer engine (what pool workers do) must
  never drag PySide6 in.

Skips window construction when data/parquet is absent (same policy as the
old smoke test).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

REPO = Path(__file__).resolve().parents[1]
HAS_DATA = (REPO / "data" / "parquet").exists()

needs_data = pytest.mark.skipif(not HAS_DATA, reason="data/parquet not present")


@pytest.fixture()
def settings(tmp_path):
    from modules.common.backend.settings import load_settings
    return load_settings(tmp_path / "settings.json")


@pytest.fixture(autouse=True)
def _theme(qapp):
    from modules.common.ui.theme import apply_theme
    apply_theme(qapp)


def test_main_menu_constructs(qtbot, settings):
    from modules.main_menu.window import MainMenuWindow
    menu = MainMenuWindow(settings)
    qtbot.addWidget(menu)
    menu.show()
    assert "Data roots" in menu._footer.text()


def test_settings_dialog_round_trip(qtbot, tmp_path):
    from modules.common.backend.settings import load_settings
    from modules.common.ui.settings_dialog import SettingsDialog
    s = load_settings(tmp_path / "settings.json")
    dlg = SettingsDialog(s)
    qtbot.addWidget(dlg)
    dlg._on_ok()
    reloaded = load_settings(tmp_path / "settings.json")
    assert reloaded.data_roots_raw == ["data"]
    assert [p.name for p in reloaded.plugin_dirs("strategies")] == ["strategies"]
    assert json.loads((tmp_path / "settings.json").read_text())["version"] == 1


@needs_data
def test_data_formatter_window(qtbot, settings):
    from modules.data_formatter.window import DataFormatterWindow
    win = DataFormatterWindow(settings)
    qtbot.addWidget(win)
    win.show()
    assert win._transform.count() > 0


@needs_data
def test_backtester_window(qtbot, settings):
    from modules.backtester.window import BacktesterWindow
    win = BacktesterWindow(settings)
    qtbot.addWidget(win)
    win.show()
    assert win._strategy.count() > 0
    assert win._params_form is not None or win._strategy.count() == 0


@needs_data
def test_analytics_window(qtbot, settings):
    from modules.analytics.window import AnalyticsWindow
    win = AnalyticsWindow(settings)
    qtbot.addWidget(win)
    win.show()
    assert len(win._editors) == 1


@needs_data
def test_monte_carlo_window(qtbot, settings):
    from modules.monte_carlo.window import MonteCarloWindow
    win = MonteCarloWindow(settings)
    qtbot.addWidget(win)
    win.show()
    methods = [win._method.itemText(i) for i in range(win._method.count())]
    assert "bootstrap" in methods


def _make_temp_trades(tmp_path):
    """A hermetic data root holding only a temp handoff file (no trades/)."""
    import pandas as pd
    root = tmp_path / "root"
    (root / "temp").mkdir(parents=True)
    p = root / "temp" / "ES_temp_file_1.parquet"
    pd.DataFrame({"date": ["2026-01-05"], "direction": ["long"],
                  "pnl_points": [1.0], "ticks": [4.0]}).to_parquet(p)
    return root, p


def test_analytics_window_initial_trades(qtbot, tmp_path):
    from modules.analytics.window import AnalyticsWindow
    from modules.common.backend.settings import Settings
    root, p = _make_temp_trades(tmp_path)
    win = AnalyticsWindow(Settings({}, [str(root)]), initial_trades=p)
    qtbot.addWidget(win)
    win.show()
    assert win._default_file.currentData().path == p
    assert win._editors[0]._file.findText(p.name) >= 0
    win._rescan()  # "Refresh files" must not lose or deselect the temp file
    assert win._default_file.currentData().path == p


def test_monte_carlo_window_initial_trades(qtbot, tmp_path):
    from modules.common.backend.settings import Settings
    from modules.monte_carlo.window import MonteCarloWindow
    root, p = _make_temp_trades(tmp_path)
    win = MonteCarloWindow(Settings({}, [str(root)]), initial_trades=p)
    qtbot.addWidget(win)
    win.show()
    assert win._file.currentData().path == p
    win._rescan()
    assert win._file.currentData().path == p


@needs_data
def test_optimizer_window_three_tabs(qtbot, settings):
    from modules.optimizer.window import OptimizerWindow
    win = OptimizerWindow(settings)
    qtbot.addWidget(win)
    win.show()
    labels = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert labels == ["New Run", "Explore", "Combine"]


def test_worker_import_chain_is_qt_free():
    """Pool workers import the engine module by name — Qt must never come
    along (each worker would load ~100 MB of GUI)."""
    code = (
        "import sys; "
        "import modules.optimizer.backend.engine; "
        "import modules.optimizer.backend.run_setup; "
        "assert 'PySide6' not in sys.modules, 'engine import pulled in Qt'; "
        "assert 'pyqtgraph' not in sys.modules, 'engine import pulled in pyqtgraph'; "
        "print('CLEAN')"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout
