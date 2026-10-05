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
    # never the default path: that is the user's real repo settings.json
    dlg = SettingsDialog(s, settings_path=tmp_path / "settings.json")
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
    from modules.common.trade_report import DEFAULT_ORDER
    from modules.backtester.window import BacktesterWindow
    win = BacktesterWindow(settings)
    qtbot.addWidget(win)
    win.show()
    assert win._strategy.count() > 0
    assert win._params_form is not None or win._strategy.count() == 0
    # every registry section is composed, and the layout gear is pinned
    assert sorted(win._report.panel.sections.keys()) == sorted(DEFAULT_ORDER)
    assert win._header_actions is not None



def test_engine_progress_panel_paints_latest_and_finishes(qtbot):
    from modules.common.ui.engine_progress import EngineProgressPanel
    panel = EngineProgressPanel()
    qtbot.addWidget(panel)
    panel.start("Starting toy on ES_1m…")
    panel.on_progress(3, 10, "Day 4 of 10 · 2026-01-07 — preparing the day (prepare_day)\n"
                             "Reading ahead: …\nSo far: 2 trades")
    panel.on_progress(4, 10, "Day 5 of 10 · 2026-01-08 — running the strategy (process_day)\n"
                             "Reading ahead: …\nSo far: 3 trades")
    panel._paint()                                  # only the LATEST call is painted
    bar, now, detail = panel.current_text()
    assert bar.startswith("day 4 / 10") and now.endswith("running the strategy (process_day)")
    assert detail.splitlines()[-1] == "So far: 3 trades"
    panel.finish("toy: 10 days in range …", "[toy timing] wall 1.000s")
    assert panel.current_text()[1:] == ("Finished", "toy: 10 days in range …")
    assert not panel._timing.isHidden()


def test_backtester_run_shows_progress_summary_and_timing(qtbot, tmp_path):
    """A real run through the window: the progress panel ends on the engine's
    summary and fills the timing table; Cancel is only visible while running."""
    import numpy as np
    import pandas as pd
    from modules.backtester.window import BacktesterWindow
    from modules.common.backend.settings import Settings
    folder = tmp_path / "parquet" / "Futures" / "ES" / "ES_1m_test"
    folder.mkdir(parents=True)
    for day in ("2026-03-02", "2026-03-03", "2026-03-04"):
        idx = pd.date_range(f"{day} 09:30", f"{day} 15:59", freq="1min", tz="America/New_York")
        close = 100 + np.sin(np.arange(len(idx)) / 20)
        pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close},
                     index=idx).to_parquet(folder / f"{day}.parquet")
    win = BacktesterWindow(Settings({}, [str(tmp_path)]))
    qtbot.addWidget(win)
    win._strategy.setCurrentText("example_first_hour_breakout")
    assert win._strategy.currentText() == "example_first_hour_breakout"
    assert not win._cancel_btn.isVisibleTo(win)
    win._on_run()
    assert win._cancel_btn.isVisibleTo(win) and not win._run_btn.isEnabled()
    qtbot.waitUntil(lambda: win._run_btn.isEnabled(), timeout=30000)
    _bar, now, detail = win._progress.current_text()
    assert now == "Finished", now
    assert detail.startswith("example_first_hour_breakout: 3 days in range, 3 processed")
    assert "day:process" in win._progress._timing_text.toPlainText()
    assert not win._cancel_btn.isVisibleTo(win)

@needs_data
def test_optimizer_cell_detail_constructs(qtbot, settings):
    """The Optimizer smoke test never reaches the drill-down (it needs a
    heatmap click), so construct it directly."""
    from modules.common.trade_report import DEFAULT_ORDER
    from modules.common.trade_report.ui import TradeReport
    report = TradeReport(settings, track_worker=lambda w: None,
                         header="Cell detail")
    qtbot.addWidget(report)
    assert sorted(report.panel.sections.keys()) == sorted(DEFAULT_ORDER)


@needs_data
def test_optimizer_combine_detail_constructs(qtbot, settings):
    """Same drill-down host as the cell detail, fed a combined set instead."""
    from modules.common.trade_report import DEFAULT_ORDER
    from modules.common.trade_report.ui import TradeReport
    from modules.optimizer.combine_detail import CombineReportSource
    report = TradeReport(settings, track_worker=lambda w: None,
                         header="Combined trade report")
    qtbot.addWidget(report)
    assert sorted(report.panel.sections.keys()) == sorted(DEFAULT_ORDER)
    # no set loaded yet -> no-op, never raises
    CombineReportSource(report).set_scope("is")


def test_combine_report_scope_switch_keeps_reappearing_trade_types(qtbot, tmp_path):
    """
    A member with no trades in one scope must not come back silently filtered
    out: the scope switch carries over what the USER decided, and a type the
    previous slice never offered arrives checked.
    """
    import pandas as pd

    from modules.common.backend.settings import Settings
    from modules.common.trade_report.ui import TradeReport
    from modules.optimizer.combine_detail import CombineReportSource

    def frame(trade_types):
        rows = []
        for n, ttype in enumerate(trade_types):
            day = f"2026-01-{5 + n:02d}"
            rows.append({
                "date": pd.Timestamp(day),
                "entry_time": pd.Timestamp(f"{day} 09:00", tz="America/New_York"),
                "exit_time": pd.Timestamp(f"{day} 10:00", tz="America/New_York"),
                "pnl_ticks": 10.0, "direction": "long",
                "entry_price": 5000.0, "exit_price": 5002.5,
                "exit_reason": "target", "trade_type": ttype,
                "day_bucket": "normal",
            })
        return pd.DataFrame(rows)

    slices = {"all": frame(["alpha", "beta"]), "is": frame(["alpha", "beta"]),
              "oos": frame(["alpha"])}          # beta has no out-of-sample trades
    report = TradeReport(Settings({}, [str(tmp_path)]),
                         track_worker=lambda w: None)
    qtbot.addWidget(report)
    panel = CombineReportSource(report)
    panel.show_set(resolve=lambda s: slices[s], scope="all",
                   header_stem="combined", save_stem=["ES", "run", "k2"],
                   ticker="ES", tick_size=0.25, ticks_per_point=4.0,
                   dataset="", root=tmp_path, regime_start="2026-01-05",
                   regime_end="2026-01-06", day_bucket_defaults={"normal"})
    assert set(report._tt_filter.selected()) == {"alpha", "beta"}

    panel.set_scope("oos")
    assert set(report._tt_filter.selected()) == {"alpha"}
    panel.set_scope("all")
    assert set(report._tt_filter.selected()) == {"alpha", "beta"}

    # a deliberate uncheck DOES survive the switch
    report._tt_filter._boxes["beta"].setChecked(False)
    panel.set_scope("is")
    assert set(report._tt_filter.selected()) == {"alpha"}


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
def test_regime_detector_window_two_tabs(qtbot, settings):
    from modules.regime_detector.window import RegimeDetectorWindow
    win = RegimeDetectorWindow(settings)
    qtbot.addWidget(win)
    win.show()
    labels = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert labels == ["New Run", "Explore"]
    assert win.run_tab._detector.count() > 0          # scaffold discovered
    assert win.run_tab._module is not None            # contract validated


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


def test_auto_params_are_read_only(qtbot):
    """tick_size (AUTO_PARAMS) is shown but disabled, follows set_value(), and
    the Optimizer never sweeps it nor records it among fixed params."""
    import types
    from modules.common.backend.asset_info import AUTO_PARAMS, auto_param_values
    from modules.common.ui.params_form import ParamsForm
    from modules.optimizer.sweep_panel import SweepPanel

    assert auto_param_values("ZN") == {"tick_size": 0.015625}
    form = ParamsForm({"tick_size": 0.25, "n": 3}, readonly=AUTO_PARAMS)
    qtbot.addWidget(form)
    assert not form._widgets["tick_size"].isEnabled()
    assert form._widgets["n"].isEnabled()
    form.set_value("tick_size", 0.015625)             # must not round to 0.02
    assert form.values() == {"tick_size": 0.015625, "n": 3}

    stub = types.SimpleNamespace(PARAMS={"tick_size": 0.25, "n": 3})
    panel = SweepPanel(stub)
    qtbot.addWidget(panel)
    cell = panel._cells["tick_size"]
    assert cell.readonly and cell.check is None       # not sweepable
    panel.set_auto_value("tick_size", 0.5)
    assert cell.fixed_value() == 0.5
    assert panel.fixed_params() == {"n": 3}           # engine injects tick_size


def test_additional_data_panel_auto_picks_by_name(qtbot, tmp_path):
    from modules.common.backend.data_roots import scan_structure
    from modules.common.ui.additional_data import AdditionalDataPanel
    for ds in ("ES_1m_advanced", "ES_1m_indicators", "ES_big_trades"):
        (tmp_path / "parquet" / "Futures" / "ES" / ds).mkdir(parents=True)
    (tmp_path / "parquet" / "Futures" / "NQ" / "NQ_1m_advanced").mkdir(parents=True)

    panel = AdditionalDataPanel()
    qtbot.addWidget(panel)
    panel.set_structure(scan_structure([tmp_path]))
    panel.set_slots(["indicators", "big_trades"])
    panel.follow_main("Futures", "ES", tmp_path)
    assert panel.ok()
    assert {k: v.name for k, v in panel.folders().items()} == {
        "indicators": "ES_1m_indicators", "big_trades": "ES_big_trades"}
    assert panel.descriptions()["indicators"] == "Futures/ES/ES_1m_indicators"

    panel.follow_main("Futures", "NQ", tmp_path)      # NQ has neither
    assert not panel.ok() and len(panel.problems()) == 2
    assert "indicators" in panel.problems()[0]

    panel.set_slots([])                               # strategy without extra data
    assert panel.ok() and panel.isHidden()


def test_worker_import_chain_is_qt_free():
    """Pool workers import the engine module by name — Qt must never come
    along (each worker would load ~100 MB of GUI)."""
    code = (
        "import sys; "
        "import modules.optimizer.backend.engine; "
        "import modules.engine; "
        "import modules.optimizer.backend.run_setup; "
        "import modules.common.backend.regime_join; "
        "import modules.regime_detector.backend.runner; "
        # the report's backend half, and the package __init__ that every one
        # of those imports executes on the way in
        "import modules.common.trade_report; "
        "import modules.common.trade_report.backend.trade_stats; "
        "import modules.common.trade_report.backend.trade_notes; "
        "import modules.common.trade_report.backend.layout; "
        "import modules.optimizer.cell_detail; "
        "import modules.optimizer.combine_detail; "
        "assert 'PySide6' not in sys.modules, 'engine import pulled in Qt'; "
        "assert 'pyqtgraph' not in sys.modules, 'engine import pulled in pyqtgraph'; "
        "print('CLEAN')"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout


def test_heatmap_label_font_fits_the_cell():
    """In-cell numbers get the largest font at which the widest one fits the
    cell (width and height); hidden only below the minimum readable size."""
    from modules.common.ui.charts.heatmap import (LABEL_FILL_W, LABEL_MAX_PT, LABEL_MIN_PT,
                                                  LABEL_REF_PT, label_font_pt)
    # widest label 30 px at the reference size, line 15 px
    pt = label_font_pt(36, 36, 30, 15)
    assert LABEL_MIN_PT <= pt < LABEL_MAX_PT
    assert 30 * pt / LABEL_REF_PT <= 36 * LABEL_FILL_W           # it really fits
    assert pt * 2 == int(pt * 2)                                  # 0.5-pt steps
    assert label_font_pt(500, 500, 30, 15) == LABEL_MAX_PT        # capped
    assert label_font_pt(12, 12, 30, 15) is None                  # too small: hidden
    assert label_font_pt(60, 10, 30, 15) is None                  # too flat: hidden
    assert label_font_pt(36, 36, 0, 15) is None                   # no labels measured


def test_heatmap_grows_for_many_rows(qtbot):
    import numpy as np
    from modules.common.ui.charts.heatmap import (BASE_MAX_HEIGHT, MAX_HEIGHT_PX,
                                                  MIN_CELL_PX, HeatmapChart)
    from modules.optimizer.backend.metrics import METRIC_ORDER

    def build(nx, ny):
        w = HeatmapChart()
        qtbot.addWidget(w)
        w.resize(1900, 600)
        w.show()
        qtbot.waitExposed(w)
        arrays = {m: np.full((ny, nx), 100.0) for m in METRIC_ORDER}
        w.set_data(arrays, "total_ticks", "Total Ticks", "x", list(range(nx)), "y",
                   list(range(ny)), "", 0)
        return w

    assert build(15, 30).height() == 30 * MIN_CELL_PX + 100      # was capped at 900
    assert build(6, 5).height() <= BASE_MAX_HEIGHT                # small grids unchanged
    assert build(10, 200).height() == MAX_HEIGHT_PX


def test_heatmap_filled_while_hidden_uses_its_full_size_when_shown(qtbot):
    """The Explore tab fills the heatmap while it is hidden, then shows it.
    The inner plot must then fill the whole widget (it used to stay at the
    pre-show ~355 px, leaving tiny cells and no numbers)."""
    import numpy as np
    from PySide6.QtWidgets import QVBoxLayout, QWidget
    from modules.common.ui.charts.heatmap import HeatmapChart
    from modules.optimizer.backend.metrics import METRIC_ORDER
    host = QWidget()
    qtbot.addWidget(host)
    lay = QVBoxLayout(host)
    hm = HeatmapChart()
    hm.setVisible(False)
    lay.addWidget(hm)
    host.resize(1900, 1000)
    host.show()
    qtbot.waitExposed(host)
    nx, ny = 30, 15
    arrays = {m: np.full((ny, nx), -2.53) for m in METRIC_ORDER}
    arrays["total_trades"] = np.full((ny, nx), 300.0)
    hm.set_data(arrays, "avg_trade", "Avg Trade (ticks)", "hold_bars", list(range(nx)),
                "delta_threshold", list(range(ny)), "", 200)
    hm.setVisible(True)
    qtbot.wait(50)
    assert hm._glw.height() == hm.height() > 600
    px_w, _ = hm._plot.getViewBox().viewPixelSize()
    assert 1 / px_w > 40                                  # cells, not ~18 px
    assert hm._labels_visible
