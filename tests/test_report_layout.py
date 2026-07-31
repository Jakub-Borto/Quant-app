"""Tests for the trade-report section registry, the section stack, the layout
dialog, and the settings ui_prefs round trip.

The highest-value test here is test_settings_to_json_keeps_ui_prefs: Settings.
to_json() rebuilds the document from scratch, so a key that isn't wired into
all four places is silently dropped the next time ANY dialog saves.
"""

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel, QWidget  # noqa: E402

from modules.common.backend.settings import (UI_PREF_TRADE_REPORT,  # noqa: E402
                                             Settings, load_settings)
from modules.common.ui.trade_report import sections as sec  # noqa: E402


# ── settings persistence ─────────────────────────────────────────────────────

def test_settings_to_json_keeps_ui_prefs(tmp_path):
    """The guard against the to_json() rebuild gotcha."""
    path = tmp_path / "settings.json"
    s = load_settings(path)
    s.set_ui_pref(UI_PREF_TRADE_REPORT,
                  {"order": ["metrics", "rr"], "modes": {"rr": "hidden"}})
    s.save(path)

    back = load_settings(path)
    pref = back.ui_pref(UI_PREF_TRADE_REPORT)
    assert pref["order"] == ["metrics", "rr"]
    assert pref["modes"]["rr"] == "hidden"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["ui_prefs"][UI_PREF_TRADE_REPORT]["modes"]["rr"] == "hidden"
    # and the pre-existing keys are untouched
    assert raw["version"] == 1 and "data_roots" in raw


def test_saving_an_unrelated_dialog_preserves_ui_prefs(tmp_path):
    """A folder edit elsewhere must not wipe the report layout."""
    path = tmp_path / "settings.json"
    s = load_settings(path)
    s.set_ui_pref(UI_PREF_TRADE_REPORT, {"order": ["rr"], "modes": {}})
    s.save(path)

    other = load_settings(path)          # e.g. the folders dialog
    other.data_roots_raw = ["D:/market_data", "E:/more"]
    other.save(path)

    assert load_settings(path).ui_pref(UI_PREF_TRADE_REPORT)["order"] == ["rr"]


def test_ui_pref_accessors_are_deep_copies():
    s = Settings({}, ["D:/market_data"])
    s.set_ui_pref("x", {"nested": [1, 2]})
    got = s.ui_pref("x")
    got["nested"].append(3)
    assert s.ui_pref("x")["nested"] == [1, 2]     # mutation didn't leak back
    assert s.ui_pref("missing", "fallback") == "fallback"


# ── resolve_layout ───────────────────────────────────────────────────────────

def test_resolve_layout_defaults():
    order, modes = sec.resolve_layout(None)
    assert order == sec.DEFAULT_ORDER
    assert modes["day_type_filter"] == sec.MODE_VISIBLE
    assert modes["news"] == sec.MODE_COLLAPSED
    # the shipped order puts every dropdown after the always-visible block
    first_collapsed = min(i for i, k in enumerate(order)
                          if modes[k] == sec.MODE_COLLAPSED)
    last_visible_locked = max(i for i, k in enumerate(order)
                              if sec.SPEC_BY_KEY[k].lockable)
    assert last_visible_locked < first_collapsed


def test_resolve_layout_drops_unknown_and_readds_missing_keys():
    saved = {"order": ["rr", "metrics", "section_from_the_future"],
             "modes": {"rr": "visible"}}
    order, modes = sec.resolve_layout(saved)
    assert "section_from_the_future" not in order   # unknown key dropped
    assert set(order) == set(sec.DEFAULT_ORDER)     # nothing lost
    assert order.index("rr") < order.index("metrics")   # user's order kept
    assert modes["rr"] == sec.MODE_VISIBLE
    assert len(order) == len(set(order))            # no duplicates


def test_new_section_lands_next_to_its_default_neighbour():
    """Shipping a new section must not scramble a saved order: it appears
    just after the section it follows by default."""
    full = list(sec.DEFAULT_ORDER)
    full.remove("rr")                  # pretend 'rr' is brand new
    saved = {"order": full, "modes": {}}
    order, _modes = sec.resolve_layout(saved)
    assert set(order) == set(sec.DEFAULT_ORDER)
    default_predecessor = sec.DEFAULT_ORDER[sec.DEFAULT_ORDER.index("rr") - 1]
    assert order.index("rr") == order.index(default_predecessor) + 1


def test_resolve_layout_ignores_duplicate_saved_keys():
    order, _modes = sec.resolve_layout({"order": ["rr", "rr", "metrics"]})
    assert order.count("rr") == 1


def test_locked_sections_are_forced_visible():
    saved = {"order": list(sec.DEFAULT_ORDER),
             "modes": {"day_type_filter": "hidden", "metrics": "collapsed",
                       "equity": "hidden", "trade_type_filter": "collapsed"}}
    _order, modes = sec.resolve_layout(saved)
    for key in ("day_type_filter", "trade_type_filter", "metrics", "equity"):
        assert modes[key] == sec.MODE_VISIBLE, key
    assert sec.SPEC_BY_KEY["news"].lockable is False


def test_resolve_layout_rejects_garbage_mode():
    _order, modes = sec.resolve_layout({"order": [], "modes": {"rr": "sideways"}})
    assert modes["rr"] == sec.SPEC_BY_KEY["rr"].default_mode


# ── SectionStack ─────────────────────────────────────────────────────────────

def _stack(qtbot, settings=None, keys=("metrics", "news", "rr", "trades_table")):
    stack = sec.SectionStack(settings or Settings({}, ["D:/market_data"]))
    widgets = {}
    for key in keys:
        w = QLabel(key)
        widgets[key] = w
        stack.register(key, w)
    stack.build()
    qtbot.addWidget(stack)
    return stack, widgets


def test_stack_orders_from_settings(qtbot):
    s = Settings({}, ["D:/market_data"])
    s.set_ui_pref(UI_PREF_TRADE_REPORT,
                  {"order": ["rr", "trades_table", "news", "metrics"],
                   "modes": {}})
    stack, _w = _stack(qtbot, s)
    assert stack.keys() == ["rr", "trades_table", "news", "metrics"]


def test_stack_reorder_preserves_widget_instances(qtbot):
    s = Settings({}, ["D:/market_data"])
    stack, widgets = _stack(qtbot, s)
    before = {k: stack.frame(k).content for k in widgets}

    s.set_ui_pref(UI_PREF_TRADE_REPORT,
                  {"order": ["rr", "metrics", "news", "trades_table"],
                   "modes": {"news": "visible"}})
    stack.reload_layout()

    assert stack.keys()[:2] == ["rr", "metrics"]
    for key, widget in before.items():
        assert stack.frame(key).content is widget      # never rebuilt
        assert widget.text() == key                    # state intact


def test_stack_mode_round_trip(qtbot):
    s = Settings({}, ["D:/market_data"])
    stack, widgets = _stack(qtbot, s)
    stack.show()
    frame = stack.frame("news")

    for mode in (sec.MODE_VISIBLE, sec.MODE_COLLAPSED, sec.MODE_HIDDEN,
                 sec.MODE_VISIBLE):
        s.set_ui_pref(UI_PREF_TRADE_REPORT,
                      {"order": list(sec.DEFAULT_ORDER), "modes": {"news": mode}})
        stack.reload_layout()
        assert frame.mode == mode
        assert frame.isVisible() == (mode != sec.MODE_HIDDEN)
        assert widgets["news"].parent() is not None    # still owned somewhere


def test_stack_report_visibility_spares_filters_and_breakdowns(qtbot):
    stack, _w = _stack(qtbot, keys=("day_type_filter", "news", "metrics", "rr"))
    stack.show()
    stack.set_report_visible(False)
    assert stack.frame("day_type_filter").isVisible()   # filters stay
    assert stack.frame("news").isVisible()              # breakdown stays
    assert not stack.frame("metrics").isVisible()       # report hides
    assert not stack.frame("rr").isVisible()
    stack.set_report_visible(True)
    assert stack.frame("metrics").isVisible()


def test_stack_host_override_survives_relayout(qtbot):
    s = Settings({}, ["D:/market_data"])
    stack, _w = _stack(qtbot, s)
    stack.show()
    stack.set_frame_visible("news", False)
    assert not stack.frame("news").isVisible()
    stack.reload_layout()
    assert not stack.frame("news").isVisible()          # host wins over layout
    stack.set_frame_visible("news", True)
    assert stack.frame("news").isVisible()


def test_stack_ignores_unknown_keys(qtbot):
    stack = sec.SectionStack(Settings({}, ["D:/market_data"]))
    qtbot.addWidget(stack)
    stack.register("not_a_section", QWidget())
    stack.register("metrics", QLabel("m"))
    stack.build()
    assert stack.keys() == ["metrics"]


def test_layout_bus_restacks_every_open_stack(qtbot):
    s = Settings({}, ["D:/market_data"])
    one, _a = _stack(qtbot, s)
    two, _b = _stack(qtbot, s)
    s.set_ui_pref(UI_PREF_TRADE_REPORT,
                  {"order": ["rr", "news", "metrics", "trades_table"],
                   "modes": {}})
    sec.LAYOUT_BUS.changed.emit()
    assert one.keys()[0] == "rr" and two.keys()[0] == "rr"


def test_layout_bus_survives_a_closed_report(qtbot):
    """Closing a report window and THEN saving a layout from another one must
    not reach into the dead stack. A direct signal->bound-method connection
    outlives the widget and crashes the process (access violation), so the bus
    keeps only weak references and revalidates before each call."""
    from shiboken6 import delete, isValid

    s = Settings({}, ["D:/market_data"])
    survivor, _w = _stack(qtbot, s)

    doomed = sec.SectionStack(s)
    doomed.register("metrics", QLabel("m"))
    doomed.build()
    delete(doomed)                 # what a closed+destroyed window looks like
    assert not isValid(doomed)

    s.set_ui_pref(UI_PREF_TRADE_REPORT,
                  {"order": ["rr", "news", "metrics", "trades_table"],
                   "modes": {}})
    sec.LAYOUT_BUS.changed.emit()            # must not crash
    assert survivor.keys()[0] == "rr"


# ── the layout dialog ────────────────────────────────────────────────────────

def test_layout_dialog_round_trip(tmp_path, qtbot):
    from modules.common.ui.trade_report.layout_dialog import ReportLayoutDialog
    path = tmp_path / "settings.json"
    s = load_settings(path)

    dlg = ReportLayoutDialog(s, settings_path=path)
    qtbot.addWidget(dlg)
    rows = [dlg._list.item(i).data(sec_role()) for i in range(dlg._list.count())]
    assert rows == list(sec.DEFAULT_ORDER)

    dlg._list.setCurrentRow(dlg._list.count() - 1)     # the last section
    moved = dlg._list.currentItem().data(sec_role())
    dlg._move(-1)
    dlg._on_ok()

    saved = load_settings(path).ui_pref(UI_PREF_TRADE_REPORT)
    assert saved["order"].index(moved) == len(sec.DEFAULT_ORDER) - 2


def test_layout_dialog_locked_rows_cannot_change_mode(tmp_path, qtbot):
    from modules.common.ui.trade_report.layout_dialog import ReportLayoutDialog
    s = load_settings(tmp_path / "settings.json")
    dlg = ReportLayoutDialog(s, settings_path=tmp_path / "settings.json")
    qtbot.addWidget(dlg)

    locked = sec.DEFAULT_ORDER.index("day_type_filter")
    dlg._list.setCurrentRow(locked)
    assert not dlg._mode.isEnabled()

    free = sec.DEFAULT_ORDER.index("news")
    dlg._list.setCurrentRow(free)
    assert dlg._mode.isEnabled()


def test_layout_dialog_reset(tmp_path, qtbot):
    from modules.common.ui.trade_report.layout_dialog import ReportLayoutDialog
    path = tmp_path / "settings.json"
    s = load_settings(path)
    s.set_ui_pref(UI_PREF_TRADE_REPORT,
                  {"order": ["rr"], "modes": {"news": "hidden"}})
    dlg = ReportLayoutDialog(s, settings_path=path)
    qtbot.addWidget(dlg)
    dlg._on_reset()
    dlg._on_ok()
    order, modes = sec.resolve_layout(load_settings(path).ui_pref(UI_PREF_TRADE_REPORT))
    assert order == sec.DEFAULT_ORDER
    assert modes["news"] == sec.MODE_COLLAPSED


def sec_role():
    from PySide6.QtCore import Qt
    return Qt.UserRole


# ── the panel actually renders every section ─────────────────────────────────

def _sample_trades():
    import numpy as np
    import pandas as pd
    n = 12
    entry = pd.date_range("2026-01-05 10:00", periods=n, freq="D",
                          tz="America/New_York")
    ticks = np.where(np.arange(n) % 3 == 0, -20.0, 15.0)
    return pd.DataFrame({
        "date": entry.tz_localize(None).normalize(),
        "direction": ["long", "short"] * (n // 2),
        "entry_time": entry, "exit_time": entry + pd.Timedelta(hours=2),
        "entry_price": 5000.0, "exit_price": 5010.0,
        "sl": 4990.0, "tp": 5030.0, "pnl_points": ticks / 4.0,
        "exit_reason": ["tp", "sl"] * (n // 2),
        "ticks": ticks, "cumulative_ticks": ticks.cumsum(),
        "day_type": "normal",
    })


def test_panel_set_trades_renders_every_section(qtbot, tmp_path, monkeypatch):
    """Push real trades through the panel so every section's refresh path
    executes. The Qt smoke tests only CONSTRUCT windows, so a NameError inside
    a rebuild (exactly what happened to ExposureSection during the section
    split) would otherwise only surface when a human ran a backtest."""
    import pandas as pd
    from modules.common.ui.trade_report import exposure_section as ex
    from modules.common.ui.trade_report.panel import TradeReportPanel

    # force the DEEP exposure path (the one that builds the 2x2 cell grid,
    # each cell a coefficient table plus a scatter+fit chart)
    import numpy as np
    cells = {(s, b): {"alpha": 0.1, "alpha_ann": 25.2, "beta": 0.5,
                      "t_alpha": 1.0, "t_beta": 2.0, "r2": 0.3, "n": 10,
                      "x": np.linspace(-10, 10, 10),
                      "y": np.linspace(-4, 6, 10)}
             for s in ("Days traded", "All days") for b in ("Settlement", "RTH")}
    monkeypatch.setattr(ex, "load_asset_statistics", lambda *_a, **_k: object())
    monkeypatch.setattr(ex, "market_exposure_data", lambda *_a, **_k: {
        "bench": {"n_roll": 1, "n_missing_settle": 0, "n_missing_rth": 0},
        "n_absent": 0, "cells": cells})

    panel = TradeReportPanel(Settings({}, [str(tmp_path)]))
    panel.build_sections()
    qtbot.addWidget(panel)
    panel.set_context("ES", 0.25, 4.0, candles_folder=tmp_path,
                      parquet_root=tmp_path)

    trades = _sample_trades()
    panel.set_trades(trades)                    # must not raise

    assert panel.metrics._tiles["Total Trades"]._value.text() == "12"
    exit_model = panel.exit_breakdown.table.model()
    assert exit_model.rowCount() > 0
    assert panel.rr.hist.isVisible() or panel.rr._trades is not None
    assert panel.exposure.content_layout.count() > 1     # the deep path ran


def test_exposure_cell_renders_table_and_chart(qtbot, monkeypatch, tmp_path):
    """The α/β cells used to be Markdown QLabels whose size hint ignored the
    rendered table, so cells drew on top of each other and the section was
    unreadable. Assert real widgets with real sizes now."""
    import numpy as np
    from modules.common.ui.charts.scatter_fit import ScatterFitChart
    from modules.common.ui.trade_report.exposure_section import _cell

    res = {"alpha": 1.5, "alpha_ann": 378.0, "beta": -0.02, "t_alpha": 2.1,
           "t_beta": -0.3, "r2": 0.001, "n": 149,
           "x": np.linspace(-30, 30, 40), "y": np.linspace(-10, 25, 40)}
    cell = _cell("Days traded × Settlement", res)
    qtbot.addWidget(cell)
    cell.resize(700, 260)
    cell.show()

    charts = cell.findChildren(ScatterFitChart)
    assert len(charts) == 1                       # a plot next to the numbers
    labels = [w.text() for w in cell.findChildren(QLabel)]
    assert "β" in labels and "-0.020" in labels   # real rows, real values
    assert "Days traded × Settlement" in labels
    # nothing overlaps: every label has a positive height and a sane position
    for w in cell.findChildren(QLabel):
        assert w.height() > 0

    empty = _cell("All days × RTH", None)         # insufficient data path
    qtbot.addWidget(empty)
    assert not empty.findChildren(ScatterFitChart)


def test_page_cannot_be_squashed_below_its_content(qtbot):
    """The dropdown-jitter fix.

    The scrolled page's minimumHeight used to be 0, so nothing stopped the
    scroll area from briefly sizing it to the viewport while it recomputed —
    and a QVBoxLayout asked to fit 2800px of content into 1000px crushes every
    child proportionally. That one frame is the visible squash: metric tiles
    clipping their text, the equity chart at half height. SetSizeConstraint
    makes the page's minimum track its content so the state is unreachable.
    """
    from PySide6.QtWidgets import QScrollArea, QVBoxLayout

    from modules.common.ui.module_window import ModuleWindowBase
    from modules.common.ui.widgets import CollapsibleSection

    win = ModuleWindowBase(Settings({}, ["D:/market_data"]), "T", "c")
    qtbot.addWidget(win)
    page = win.findChild(QScrollArea).widget()

    tall = QLabel("row\n" * 120)
    tall.setMinimumHeight(900)
    win.content.addWidget(tall)
    box = CollapsibleSection("Section")
    body = QLabel("body\n" * 60)
    body.setMinimumHeight(700)
    box.add_widget(body)
    win.content.addWidget(box)
    win.resize(900, 500)                       # viewport far shorter than content
    win.show()
    qtbot.waitExposed(win)

    collapsed_min = page.minimumHeight()
    assert collapsed_min >= 900, collapsed_min

    box._button.setChecked(True)
    qtbot.wait(50)
    expanded_min = page.minimumHeight()
    assert expanded_min >= collapsed_min + 700, (collapsed_min, expanded_min)

    # the squash: force the page to the viewport height while it holds the
    # expanded content. The minimum must refuse it, so children keep their size.
    page.resize(page.width(), 500)
    qtbot.wait(50)
    assert page.height() >= expanded_min
    assert tall.height() >= 900
    assert body.height() >= 700


def test_expanding_a_section_never_squeezes_its_siblings(qtbot):
    """The dropdown-jitter bug, at the level it actually happened.

    Traced order before the fix: the regime frame grew to 313, then the equity
    frame was crushed 572->313 and metrics 330->313, and only THEN did the
    stack grow 1432->1849. Qt absorbs that temporary deficit by violating the
    children's own minimums, which is the visible squash — clipped metric
    tiles, a half-height equity chart. Every container between the section and
    the scroll area needs SetMinimumSize so it grows first.
    """
    from PySide6.QtCore import QEvent, QObject
    from PySide6.QtWidgets import QVBoxLayout

    class ResizeTrace(QObject):
        def __init__(self):
            super().__init__()
            self.shrinks = []

        def eventFilter(self, obj, ev):
            if ev.type() == QEvent.Resize:
                old, new = ev.oldSize().height(), ev.size().height()
                floor = obj.minimumSizeHint().height()
                if old > 0 and new < floor - 4:      # ignore 1-2px rounding
                    self.shrinks.append((getattr(obj, "_key", obj), old, new,
                                         floor))
            return False

    from PySide6.QtWidgets import QScrollArea

    s = Settings({}, ["D:/market_data"])
    stack = sec.SectionStack(s)

    tall = {}
    for key in ("metrics", "equity", "rr", "news"):
        w = QLabel(f"{key}\n" * 30)
        w.setMinimumHeight(300)
        tall[key] = w
        stack.register(key, w)
    stack.register("regime", QLabel("regime body\n" * 20))
    stack.build()

    # the real configuration: the stack lives in a scrolled page whose
    # viewport is far SHORTER than the content, so the layout is under
    # pressure exactly as it is in a module window
    host = QScrollArea()
    host.setWidgetResizable(True)
    host.setWidget(stack)
    qtbot.addWidget(host)
    host.resize(800, 500)
    host.show()
    qtbot.waitExposed(host)

    trace = ResizeTrace()
    for key in stack.keys():
        frame = stack.frame(key)
        frame._key = key
        frame.installEventFilter(trace)

    before = {k: stack.frame(k).height() for k in stack.keys()}
    stack.frame("regime")._box._button.setChecked(True)
    qtbot.wait(80)

    assert not trace.shrinks, f"siblings squeezed below their minimum: {trace.shrinks}"
    for key in ("metrics", "equity", "rr", "news"):
        assert stack.frame(key).height() >= before[key] - 4, key
        assert tall[key].height() >= 300, f"{key} content clipped"

    # the mechanism: every container's minimum tracks its content, so it
    # grows before its children can be squeezed
    assert stack.minimumHeight() > 0
    assert stack.frame("equity").minimumHeight() > 0


def test_collapsing_a_section_never_inflates_its_siblings(qtbot):
    """The mirror of the squeeze, and what made CLOSING jitter.

    Collapsing frees its height inside a container that is still its old size
    for one pass, and QVBoxLayout hands that surplus to whichever child will
    take it — pyqtgraph plots are Expanding, so the equity chart ballooned by
    exactly the freed amount and snapped back (traced: 539 -> 958 -> 539). The
    stack's trailing stretch must win the surplus instead.
    """
    from PySide6.QtCore import QEvent, QObject
    from PySide6.QtWidgets import QScrollArea, QSizePolicy

    class Balloon(QObject):
        def __init__(self):
            super().__init__()
            self.hits = []

        def eventFilter(self, obj, ev):
            if ev.type() == QEvent.Resize:
                old, new = ev.oldSize().height(), ev.size().height()
                hint = obj.sizeHint().height()
                if old > 0 and hint > 0 and new > hint + 40 and new > old + 40:
                    self.hits.append((getattr(obj, "_key", "?"), old, new, hint))
            return False

    stack = sec.SectionStack(Settings({}, ["D:/market_data"]))
    greedy = QLabel("chart stand-in")
    greedy.setMinimumHeight(300)
    greedy.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
    stack.register("equity", greedy)
    stack.register("metrics", QLabel("m\n" * 20))
    body = QLabel("regime body\n" * 25)
    stack.register("regime", body)
    stack.build()

    host = QScrollArea()
    host.setWidgetResizable(True)
    host.setWidget(stack)
    qtbot.addWidget(host)
    host.resize(800, 600)
    host.show()
    qtbot.waitExposed(host)

    stack.frame("regime")._box._button.setChecked(True)
    qtbot.wait(60)
    open_height = stack.frame("equity").height()

    spy = Balloon()
    for key in stack.keys():
        f = stack.frame(key)
        f._key = key
        f.installEventFilter(spy)
        f.content._key = f"content:{key}"
        f.content.installEventFilter(spy)

    stack.frame("regime")._box._button.setChecked(False)     # the close
    qtbot.wait(80)

    assert not spy.hits, f"siblings inflated by the freed space: {spy.hits}"
    assert abs(stack.frame("equity").height() - open_height) <= 4


def test_equity_selection_marks_deselects_and_survives_replot(qtbot, tmp_path):
    """Clicking a trade must be visible on the chart, clickable again to
    deselect, and must not survive a filter change (the row index would point
    at a different trade)."""
    from modules.common.ui.trade_report.panel import TradeReportPanel

    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    def click_point(chart, row):
        """A REAL click routed through pyqtgraph's scene.

        Calling the clicked-handler directly bypasses hit-testing, which is
        how an overlay item silently swallowing the second click went
        unnoticed. Always drive selection through the scene in tests.
        """
        vb = chart._plot.getPlotItem().vb
        scene_pt = vb.mapViewToScene(QPointF(float(chart._x[row]),
                                             float(chart._y[row])))
        view_pt = chart._plot.mapFromScene(scene_pt)
        for etype, buttons in ((QMouseEvent.Type.MouseButtonPress, Qt.LeftButton),
                               (QMouseEvent.Type.MouseButtonRelease, Qt.NoButton)):
            QApplication.sendEvent(chart._plot.viewport(), QMouseEvent(
                etype, QPointF(view_pt), Qt.LeftButton, buttons, Qt.NoModifier))
        qtbot.wait(20)

    panel = TradeReportPanel(Settings({}, [str(tmp_path)]))
    panel.build_sections()
    qtbot.addWidget(panel)
    panel.resize(900, 700)
    panel.show()                      # isVisible() on children needs this
    qtbot.waitExposed(panel)
    panel.set_context("ES", 0.25, 4.0, candles_folder=tmp_path,
                      parquet_root=tmp_path)
    panel.set_trades(_sample_trades())
    eq = panel.equity

    def highlighted():
        """Indices drawn at the selected size, read back off the scatter."""
        sizes = eq._scatter.data["size"]
        return [i for i, s in enumerate(sizes) if s > 7]

    # exactly ONE scatter item: an overlay would swallow the deselect click
    plot_items = [i for i in eq._plot.getPlotItem().items
                  if type(i).__name__ == "ScatterPlotItem"]
    assert len(plot_items) == 1

    assert eq.selected() is None and highlighted() == []
    assert eq._selected_label.text() == ""

    click_point(eq, 3)                          # click a trade
    assert eq.selected() == 3
    assert highlighted() == [3]                 # white dot on that point
    assert "#4" in eq._selected_label.text()    # 1-based in the label

    click_point(eq, 3)                          # SAME point -> deselect
    assert eq.selected() is None, "clicking the selected point must deselect"
    assert highlighted() == []
    assert eq._selected_label.text() == ""

    click_point(eq, 6)                          # a different point selects
    assert eq.selected() == 6
    click_point(eq, 6)
    assert eq.selected() is None

    # the highlight must survive the chart being re-plotted by the axis toggle
    click_point(eq, 2)
    assert eq.selected() == 2
    eq._axis_btn.click()
    qtbot.wait(30)
    assert eq.selected() == 2 and highlighted() == [2]

    # new data invalidates the selection
    panel.set_trades(_sample_trades())
    qtbot.wait(20)
    assert eq.selected() is None and highlighted() == []


def test_regime_filter_has_its_own_timing(qtbot, tmp_path):
    """The performance table and the filter read labels at INDEPENDENT
    moments, so you can measure as-of-entry while slicing the equity path by
    the final label. annotate() therefore emits two columns, and both must be
    stripped before a save."""
    import numpy as np
    import pandas as pd

    from modules.common.backend.regime_join import (MODE_ASOF, MODE_EXACT,
                                                    MODE_FINAL)
    from modules.common.ui.trade_report.actions_row import DERIVED_COLUMNS
    from modules.common.ui.trade_report.regime_section import (FILTER_COLUMN,
                                                               RegimeSection)

    assert "regime" in DERIVED_COLUMNS and FILTER_COLUMN in DERIVED_COLUMNS

    ny = "America/New_York"
    # RTH date 2026-01-06: the file spans the prior evening 18:30 -> 17:00,
    # and the label flips from 'low' to 'high' at 12:30
    idx = pd.date_range(pd.Timestamp("2026-01-05 18:30", tz=ny),
                        periods=46, freq="30min")
    states = ["low"] * 36 + ["high"] * 10
    frames = {"2026-01-06": pd.DataFrame({"vol_state": states,
                                          "price": np.arange(46.0)}, index=idx)}

    section = RegimeSection(Settings({}, [str(tmp_path)]), lambda w: None)
    qtbot.addWidget(section)
    section._frames = frames
    section._meta = {"globex_session": {"start": "18:00", "end": "17:00"},
                     "snapshot_minutes": 30,
                     "states": {"vol_state": {"states": ["low", "high"],
                                              "colors": ["#1", "#2"]}}}
    section._column.addItem("vol_state")

    trades = pd.DataFrame({
        "date": [pd.Timestamp("2026-01-06")] * 2,
        "entry_time": [pd.Timestamp("2026-01-06 10:00", tz=ny),
                       pd.Timestamp("2026-01-06 15:00", tz=ny)],
        "ticks": [5.0, -5.0],
    })

    # same timing on both -> the filter column is a copy, no second join
    section._mode.setCurrentIndex(section._mode.findData(MODE_ASOF))
    section._filter_mode.setCurrentIndex(section._filter_mode.findData(MODE_ASOF))
    out = section.annotate(trades)
    assert not section.timings_differ()
    assert list(out["regime"]) == ["low", "high"]
    assert list(out[FILTER_COLUMN]) == ["low", "high"]

    # table as-of-entry, filter on the FINAL label -> the morning trade keeps
    # its honest 'low' while the filter sees the day's verdict, 'high'
    section._filter_mode.setCurrentIndex(section._filter_mode.findData(MODE_FINAL))
    out = section.annotate(trades)
    assert section.timings_differ()
    assert list(out["regime"]) == ["low", "high"]
    assert list(out[FILTER_COLUMN]) == ["high", "high"]

    # filtering by 'final' is lookahead — say so loudly
    assert "not tradeable" in section._banner.text().lower()

    # exact mode enables only the filter's own HH:MM box
    section._filter_mode.setCurrentIndex(section._filter_mode.findData(MODE_EXACT))
    assert section._filter_at.isEnabled() and not section._at.isEnabled()

    # no run loaded -> untouched frame, and no filtering
    section._frames = {}
    assert section.annotate(trades) is trades
    assert section.selected() is None


def test_tables_sort_numerically_and_do_not_stretch(qtbot):
    """Breakdown tables hold DISPLAY strings ("62.5%", "1.41", "N/A", "∞").
    Sorting them as text puts 100% before 62.5%, so the model parses them back
    to numbers. And the last column must not stretch: it inflated a narrow
    "Avg Ticks" to hundreds of pixels with its value marooned on the left.
    """
    import pandas as pd
    from PySide6.QtCore import Qt

    from modules.common.ui.dataframe_model import (make_table_view,
                                                   update_table_view)

    df = pd.DataFrame({
        "Entry": ["beta", "alpha", "gamma", "delta"],
        "Win Rate": ["62.5%", "100.0%", "9.0%", "N/A"],
        "Profit Factor": ["1.41", "∞", "0.43", "N/A"],
        "Total Ticks": [414, -8, 1662, 0],
    })
    view = make_table_view(df, height=200)
    qtbot.addWidget(view)
    header = view.horizontalHeader()

    assert view.isSortingEnabled()
    assert not header.stretchLastSection()
    # nothing is sorted until the user asks: row order can carry meaning
    assert header.sortIndicatorSection() < 0
    model = view.model()

    def column(name):
        c = [i for i in range(model.columnCount())
             if model.headerData(i, Qt.Horizontal) == name][0]
        return [model.data(model.index(r, c)) for r in range(model.rowCount())]

    assert column("Entry") == ["beta", "alpha", "gamma", "delta"]

    def sort(name, order=Qt.AscendingOrder):
        c = [i for i in range(model.columnCount())
             if model.headerData(i, Qt.Horizontal) == name][0]
        view.sortByColumn(c, order)
        qtbot.wait(10)
        return c

    sort("Win Rate", Qt.DescendingOrder)
    assert column("Win Rate") == ["100.0%", "62.5%", "9.0%", "N/A"], \
        "percentages sorted as text"

    sort("Profit Factor", Qt.DescendingOrder)
    assert column("Profit Factor")[:3] == ["∞", "1.41", "0.43"]
    assert column("Profit Factor")[-1] == "N/A"      # blanks always last

    sort("Total Ticks", Qt.AscendingOrder)
    assert [int(v) for v in column("Total Ticks")] == [-8, 0, 414, 1662]

    sort("Entry", Qt.AscendingOrder)
    assert column("Entry") == ["alpha", "beta", "delta", "gamma"]

    # a pinned benchmark row stays on top through ANY sort
    pinned = make_table_view(df.rename(columns={"Entry": "Entry"}),
                             pinned_labels={"gamma"})
    qtbot.addWidget(pinned)
    pm = pinned.model()

    def pinned_first_col():
        return [pm.data(pm.index(r, 0)) for r in range(pm.rowCount())]

    for c in range(pm.columnCount()):
        for order in (Qt.AscendingOrder, Qt.DescendingOrder):
            pinned.sortByColumn(c, order)
            qtbot.wait(5)
            assert pinned_first_col()[0] == "gamma", (c, order)
    # the unpinned rows are still genuinely sorted beneath it
    pinned.sortByColumn(3, Qt.AscendingOrder)          # Total Ticks
    rest = [pm.data(pm.index(r, 3)) for r in range(1, pm.rowCount())]
    assert [int(v) for v in rest] == sorted(int(v) for v in rest)

    # a data refresh keeps the user's chosen sort
    ticks_col = sort("Total Ticks", Qt.DescendingOrder)
    refreshed = df.copy()
    refreshed["Total Ticks"] = [10, 900, -50, 3]
    update_table_view(view, refreshed)
    qtbot.wait(10)
    assert header.sortIndicatorSection() == ticks_col
    assert [int(v) for v in column("Total Ticks")] == [900, 10, 3, -50]


def test_hover_tooltip_only_updates_when_the_point_changes(qtbot):
    """QToolTip.showText re-anchors the popup to the cursor every call, so
    calling it on every mouse move makes the box chase the mouse and flicker.
    It must only fire when the text (i.e. the hovered point) actually changes.
    """
    from PySide6.QtCore import QPointF

    from modules.common.ui.charts import base as chart_base

    shown = []

    class FakeToolTip:
        @staticmethod
        def showText(_pos, text, _w=None):
            shown.append(text)

        @staticmethod
        def hideText():
            shown.append(None)

        @staticmethod
        def isVisible():
            return bool(shown) and shown[-1] is not None

    monkey = chart_base.QToolTip
    chart_base.QToolTip = FakeToolTip
    try:
        plot = chart_base.make_plot("x", "y")
        qtbot.addWidget(plot)
        texts = iter(["point A"] * 5 + ["point B"] * 3 + [None] * 2)
        tip = chart_base.HoverTooltip(plot, lambda _x, _y: next(texts))
        # pretend the cursor is inside the plot for every call
        plot.sceneBoundingRect = lambda: type(
            "R", (), {"contains": lambda _s, _p: True})()

        for _ in range(10):
            tip._on_move([plot.getPlotItem().vb.mapViewToScene(QPointF(0, 0))])

        assert shown == ["point A", "point B", None], (
            f"tooltip fired {len(shown)} times for 3 distinct states: {shown}")
    finally:
        chart_base.QToolTip = monkey


def test_charts_in_collapsibles_have_one_unambiguous_height(qtbot):
    """A chart whose minimum is BELOW pyqtgraph's inherited 640x480 sizeHint
    gives the layout two valid answers: it lays out at the minimum, then jumps
    to the hint a pass later. That two-stage expand was the Optimizer's
    remaining jitter (the exposure section going 630 -> 1076, resizing the
    whole window twice). Charts must resolve to ONE height.
    """
    from modules.common.ui.charts.candlestick import TradeChart
    from modules.common.ui.charts.equity_curve import (EquityCurveChart,
                                                       MultiLineEquityChart)
    from modules.common.ui.charts.histogram import OverlaidHistogram
    from modules.common.ui.charts.path_chart import CombinePathChart
    from modules.common.ui.charts.scatter_fit import ScatterFitChart

    for cls in (EquityCurveChart, MultiLineEquityChart, OverlaidHistogram,
                ScatterFitChart, CombinePathChart, TradeChart):
        chart = cls()
        qtbot.addWidget(chart)
        plot = chart._plot
        assert plot.minimumHeight() == plot.maximumHeight(), (
            f"{cls.__name__} has a height RANGE, so the layout renegotiates "
            f"it on every pass (min {plot.minimumHeight()}, max "
            f"{plot.maximumHeight()}) — use set_chart_height()")


def test_toggling_a_section_stays_cheap(qtbot):
    """Expanding a dropdown must not repaint the world.

    Two things were measured on a real 1600x1000 Backtester: wrapping the
    toggle in window().setUpdatesEnabled() forces a full-window repaint on
    re-enable (849 paint events), and without WA_StaticContents on the
    scrolled page every child repaints on the page resize (438). With both
    fixed it is 72. This test guards the mechanism, not the exact number.
    """
    from PySide6.QtCore import QEvent, QObject, Qt
    from PySide6.QtWidgets import QScrollArea, QVBoxLayout

    from modules.common.ui.module_window import ModuleWindowBase
    from modules.common.ui.widgets import CollapsibleSection

    class PaintSpy(QObject):
        def __init__(self):
            super().__init__()
            self.count = 0

        def eventFilter(self, _obj, ev):
            if ev.type() == QEvent.Paint:
                self.count += 1
            return False

    win = ModuleWindowBase(Settings({}, ["D:/market_data"]), "T", "c")
    qtbot.addWidget(win)

    # the scrolled page must carry WA_StaticContents — this is the fix
    page = win.findChild(QScrollArea).widget()
    assert page.testAttribute(Qt.WA_StaticContents)

    box = CollapsibleSection("Section")
    body = QLabel("x\n" * 40)
    box.add_widget(body)
    win.content.addWidget(box)
    for _ in range(6):                       # neighbours that must NOT repaint
        win.content.addWidget(QLabel("neighbour\n" * 10))
    win.resize(900, 600)
    win.show()
    qtbot.waitExposed(win)

    spy = PaintSpy()
    for child in win.findChildren(QObject):
        child.installEventFilter(spy)

    box._button.setChecked(True)
    qtbot.wait(60)
    opened = spy.count

    spy.count = 0
    box._button.setChecked(False)
    qtbot.wait(60)
    closed = spy.count

    assert body.isVisible() is False              # collapse really collapsed
    # a toggle touches a handful of widgets, not every widget several times
    n_widgets = len(win.findChildren(QObject))
    assert opened < n_widgets, f"{opened} paints for {n_widgets} widgets"
    assert closed < n_widgets, f"{closed} paints for {n_widgets} widgets"


def test_exposure_rebuild_does_not_leak_widgets(qtbot, monkeypatch, tmp_path):
    """update_exposure runs on EVERY filter change. Clearing only top-level
    widgets left the nested 2x2 grid intact, so each rebuild stacked another
    four scatter charts (48 after a dozen filter toggles)."""
    import numpy as np
    from modules.common.ui.charts.scatter_fit import ScatterFitChart
    from modules.common.ui.trade_report import exposure_section as ex

    cells = {(s, b): {"alpha": 0.1, "alpha_ann": 25.2, "beta": 0.5,
                      "t_alpha": 1.0, "t_beta": 2.0, "r2": 0.3, "n": 10,
                      "x": np.linspace(-5, 5, 8), "y": np.linspace(-2, 3, 8)}
             for s in ("Days traded", "All days") for b in ("Settlement", "RTH")}
    monkeypatch.setattr(ex, "load_asset_statistics", lambda *_a, **_k: object())
    monkeypatch.setattr(ex, "market_exposure_data", lambda *_a, **_k: {
        "bench": {"n_roll": 0, "n_missing_settle": 0, "n_missing_rth": 0},
        "n_absent": 0, "cells": cells})

    section = ex.ExposureSection()
    qtbot.addWidget(section)
    trades = _sample_trades()
    for _ in range(5):
        section.update_exposure(trades, "ES", 0.25, tmp_path)
        qtbot.wait(1)                       # let deleteLater() run

    charts = [c for c in section.findChildren(ScatterFitChart)
              if c.parent() is not None]
    assert len(charts) == 4, f"leaked charts: {len(charts)}"


def test_panel_handles_trades_without_optional_columns(qtbot, tmp_path):
    """A strategy emitting neither exit_reason nor sl/tp must not crash the
    now-standalone exit and RR sections."""
    import pandas as pd
    from modules.common.ui.trade_report.panel import TradeReportPanel

    trades = _sample_trades().drop(columns=["exit_reason", "sl", "tp"])
    panel = TradeReportPanel(Settings({}, [str(tmp_path)]))
    panel.build_sections()
    qtbot.addWidget(panel)
    panel.set_context("ES", 0.25, 4.0, candles_folder=tmp_path,
                      parquet_root=tmp_path)
    panel.set_trades(trades)
    assert panel.exit_breakdown.table.model().rowCount() == 0
