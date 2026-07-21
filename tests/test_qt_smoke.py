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


def _configured_data_roots() -> list[str]:
    """Data roots from the repo's real settings.json (falls back to the
    in-repo default data/) so data-dependent smoke tests follow the machine's
    actual data location."""
    cfg = REPO / "settings.json"
    if cfg.exists():
        try:
            return json.loads(cfg.read_text(encoding="utf-8")).get(
                "data_roots", ["data"])
        except (json.JSONDecodeError, OSError):
            pass
    return ["data"]


DATA_ROOTS = _configured_data_roots()
HAS_DATA = any(
    ((Path(r) if Path(r).is_absolute() else REPO / r) / "parquet").exists()
    for r in DATA_ROOTS
)

needs_data = pytest.mark.skipif(
    not HAS_DATA, reason="no configured data root has a parquet/ folder")


@pytest.fixture()
def settings():
    from modules.common.backend.settings import Settings
    return Settings({}, DATA_ROOTS)


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
    from modules.common.backend.settings import DEFAULT_DATA_ROOT, load_settings
    from modules.common.ui.settings_dialog import SettingsDialog
    s = load_settings(tmp_path / "settings.json")
    dlg = SettingsDialog(s)
    qtbot.addWidget(dlg)
    dlg._on_ok()
    reloaded = load_settings(tmp_path / "settings.json")
    assert reloaded.data_roots_raw == [DEFAULT_DATA_ROOT]
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


def test_scripts_window_constructs(qtbot, tmp_path):
    from modules.common.backend.settings import Settings
    from modules.scripts.window import ScriptsWindow
    extra = tmp_path / "extra_scripts"
    extra.mkdir()
    (extra / "quick_check.py").write_text("print('hi')\n", encoding="utf-8")
    win = ScriptsWindow(Settings({"scripts": [str(extra)]}, DATA_ROOTS))
    qtbot.addWidget(win)
    win.show()
    assert "quick_check" in [r.name for r in win._refs]
    assert not win._instances    # constructing must not spawn processes


@needs_data
def test_optimizer_window_three_tabs(qtbot, settings):
    from modules.optimizer.window import OptimizerWindow
    win = OptimizerWindow(settings)
    qtbot.addWidget(win)
    win.show()
    labels = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert labels == ["New Run", "Explore", "Combine"]


# ── params_form: PARAMS_OPTIONS widgets (dropdowns + bit-flag groups) ────────

def test_param_widget_dropdown_str(qtbot):
    from PySide6.QtWidgets import QComboBox
    from modules.common.ui.params_form import make_param_widget
    w, get = make_param_widget("globex", ["globex", "rth"])
    qtbot.addWidget(w)
    assert isinstance(w, QComboBox)
    assert get() == "globex"
    w.setCurrentIndex(1)
    assert get() == "rth"


def test_param_widget_dropdown_keeps_option_type(qtbot):
    from modules.common.ui.params_form import make_param_widget
    w, get = make_param_widget(2, [2, 3])
    qtbot.addWidget(w)
    assert get() == 2 and type(get()) is int      # typed, not "2"


def test_param_widget_flags_round_trip(qtbot):
    from modules.common.ui.params_form import FlagsGroup, make_param_widget
    names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
    w, get = make_param_widget("1010100", names)
    qtbot.addWidget(w)
    assert isinstance(w, FlagsGroup)
    assert get() == "1010100"
    w._boxes[1].setChecked(True)
    assert get() == "1110100"


def test_param_widget_bool_ignores_options(qtbot):
    from PySide6.QtWidgets import QCheckBox
    from modules.common.ui.params_form import make_param_widget
    w, get = make_param_widget(True, [False, True])
    qtbot.addWidget(w)
    assert isinstance(w, QCheckBox) and get() is True


def test_param_widget_options_fit_neither_rule(qtbot):
    from PySide6.QtWidgets import QLineEdit
    from modules.common.ui.params_form import make_param_widget
    w, get = make_param_widget("zzz", ["a", "b"])   # not in options, not a bitstring
    qtbot.addWidget(w)
    assert isinstance(w, QLineEdit) and get() == "zzz"


def test_params_form_with_options(qtbot):
    from modules.common.ui.params_form import ParamsForm
    params = {"mode": "fast", "entries": "110", "on": True, "n": 3}
    options = {"mode": ["fast", "slow"], "entries": ["a", "b", "c"]}
    form = ParamsForm(params, options=options)
    qtbot.addWidget(form)
    assert form.values() == {"mode": "fast", "entries": "110", "on": True, "n": 3}


def test_sweep_panel_new_kinds(qtbot):
    import types
    from modules.optimizer.sweep_panel import SweepPanel
    stub = types.SimpleNamespace(
        PARAMS={"flag": True, "mode": "fast", "entries": "110", "n": 3},
        PARAM_SECTIONS={"All": ["flag", "mode", "entries", "n"]},
        PARAMS_OPTIONS={"mode": ["fast", "slow"], "entries": ["a", "b", "c"]},
    )
    panel = SweepPanel(stub)
    qtbot.addWidget(panel)

    # fixed values keep their new shapes (bool / typed choice / bitstring)
    assert panel.fixed_params() == {"flag": True, "mode": "fast",
                                    "entries": "110", "n": 3}

    # bool param: sweepable, axis is always [False, True]
    panel._cells["flag"].check.setChecked(True)
    assert panel.swept_values()["flag"] == [False, True]

    # choice param: all options checked initially; unchecking prunes the axis
    panel._cells["mode"].check.setChecked(True)
    editor = panel._cells["mode"].sweep_editor
    assert panel.swept_values()["mode"] == ["fast", "slow"]
    editor._choice_boxes[0].setChecked(False)
    assert panel.swept_values()["mode"] == ["slow"]
    editor._choice_boxes[1].setChecked(False)           # zero checked -> invalid
    assert panel.swept_values()["mode"] is None

    # flags param: comma-separated bitstrings, validated against option count
    panel._cells["entries"].check.setChecked(True)
    panel._cells["entries"].sweep_editor._text.setText("110, 011")
    assert panel.swept_values()["entries"] == ["110", "011"]
    panel._cells["entries"].sweep_editor._text.setText("11")   # wrong length
    assert panel.swept_values()["entries"] is None


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
