"""
Optimizer "Explore" tab — browse a run's metric surface.

The PySide6 port of render_explore / _load_selected_run / _render_save_panel:
data-root + folder + run cascading selectors (the in-memory unsaved run is
pinned on top and discarded when a saved run is loaded), the optional save
panel, run caption + held-params viewer, the metric / data-half / min-trades
/ color-curve controls, per-slider-axis value sliders, day-bucket checkboxes,
the cached vectorized grid recompute (same cache-key tuple, including
created_at to disambiguate successive unsaved runs), the heatmap, and the
cell drill-down.
"""

import json

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDoubleSpinBox, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QPushButton, QRadioButton, QSlider, QSpinBox,
                               QVBoxLayout, QWidget)

import numpy as np
import pandas as pd

from modules.common.backend.data_roots import optimizations_root
from modules.common.ui import theme
from modules.common.ui.charts.heatmap import HeatmapChart
from modules.common.trade_report.ui.filters import CheckboxFilterRow
from modules.common.ui.widgets import (Banner, Caption, CollapsibleSection,
                                       SectionHeader, pin_minimum_height,
                                       wrap_card)
from modules.optimizer.backend import io as opt_io
from modules.optimizer.backend.buckets import BUCKET_ORDER
from modules.optimizer.backend.heatmap_model import (CURVE_LIMIT, METRIC_PIVOTS,
                                                     MIN_TRADES_DEFAULT,
                                                     _build_grid_arrays,
                                                     _fmt_axis_value)
from modules.optimizer.backend.metrics import (METRIC_LABELS, METRIC_ORDER,
                                               compute_metrics_by_cell)
from modules.common.trade_report.ui import TradeReport
from modules.optimizer import cell_detail

UNSAVED_LABEL = "● Unsaved run — switching away discards it"
NEW_FOLDER    = "── New folder ──"
NO_FOLDER     = "(no folder)"


# Dead space under the page: scrolled to the bottom, toggling a section
# changes the scroll range and the view lurches. A tall run-out makes the
# last sections behave like the ones above them.
BOTTOM_PADDING = 900


CURVE_STEPS = 100          # slider ticks per unit of curve exponent (0.01 precision)


class _CurveControl(QObject):
    """One color curve: a fine slider (-CURVE_LIMIT .. +CURVE_LIMIT in 0.01
    steps) and a number box for exact values, kept in sync."""
    changed = Signal()

    def __init__(self, tooltip: str, parent=None):
        super().__init__(parent)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(-CURVE_LIMIT * CURVE_STEPS), int(CURVE_LIMIT * CURVE_STEPS))
        self.slider.setValue(0)
        self.slider.setToolTip(tooltip)
        self.box = QDoubleSpinBox()
        self.box.setRange(-CURVE_LIMIT, CURVE_LIMIT)     # explicit range (CLAUDE.md)
        self.box.setDecimals(2)
        self.box.setSingleStep(0.05)
        self.box.setValue(0.0)
        self.box.setToolTip(tooltip)
        self.box.setFixedWidth(80)
        self.layout_row = QHBoxLayout()
        self.layout_row.addWidget(self.slider, 1)
        self.layout_row.addWidget(self.box)
        self.slider.valueChanged.connect(self._from_slider)
        self.box.valueChanged.connect(self._from_box)

    def _from_slider(self, v: int) -> None:
        self.box.blockSignals(True)
        self.box.setValue(v / CURVE_STEPS)
        self.box.blockSignals(False)
        self.changed.emit()

    def _from_box(self, v: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(round(v * CURVE_STEPS))
        self.slider.blockSignals(False)
        self.changed.emit()

    def value(self) -> float:
        return round(self.box.value(), 2)

    def set_value(self, v: float) -> None:
        self.box.setValue(v)            # syncs the slider and emits changed


SPLIT_STEPS = 1000         # split slider positions across the value range


class _SplitControl(QObject):
    """The split value: a slider across the values on screen (zmin .. zmax)
    and a number box for an exact value. set_range() never emits."""
    changed = Signal()

    def __init__(self, tooltip: str, parent=None):
        super().__init__(parent)
        self._lo, self._hi = 0.0, 1.0
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, SPLIT_STEPS)
        self.slider.setToolTip(tooltip)
        self.box = QDoubleSpinBox()
        self.box.setRange(-1e12, 1e12)                    # explicit range (CLAUDE.md)
        self.box.setToolTip(tooltip)
        self.box.setFixedWidth(80)
        self.layout_row = QHBoxLayout()
        self.layout_row.addWidget(self.slider, 1)
        self.layout_row.addWidget(self.box)
        self.slider.valueChanged.connect(self._from_slider)
        self.box.valueChanged.connect(self._from_box)

    def set_range(self, lo: float, hi: float, value: float) -> None:
        self._lo, self._hi = lo, hi
        span = hi - lo
        decimals = 0 if span >= 1000 else 1 if span >= 100 else 2 if span >= 1 else 4
        for w in (self.slider, self.box):
            w.blockSignals(True)
        self.box.setDecimals(decimals)
        self.box.setSingleStep(max(span / 100, 10 ** -decimals))
        self.box.setValue(value)
        self.slider.setValue(self._pos(value))
        for w in (self.slider, self.box):
            w.blockSignals(False)

    def _pos(self, value: float) -> int:
        span = self._hi - self._lo
        f = 0.5 if span <= 0 else (value - self._lo) / span
        return int(round(min(max(f, 0.0), 1.0) * SPLIT_STEPS))

    def _from_slider(self, i: int) -> None:
        value = self._lo + (self._hi - self._lo) * i / SPLIT_STEPS
        self.box.blockSignals(True)
        self.box.setValue(value)
        self.box.blockSignals(False)
        self.changed.emit()

    def _from_box(self, v: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self._pos(v))
        self.slider.blockSignals(False)
        self.changed.emit()

    def value(self) -> float:
        return float(self.box.value())


class ExploreTab(QWidget):
    """Run state (trades/meta/unsaved/loaded_run/run_root) lives on the
    OptimizerWindow (`state`) — New Run hands off through it."""

    def __init__(self, settings, state, track_worker=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.state = state
        # only the cell drill-down needs it (regime loading); passed through
        self._track_worker = track_worker
        self._grid = None
        self._grid_key = None
        self._slider_widgets: list[tuple[dict, QSlider | None, QLabel]] = []

        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        pin_minimum_height(self)      # tab pages need it too — see widgets.py

        self._success = Banner()
        lay.addWidget(self._success)

        # ── run selector row ──────────────────────────────────────────────────
        sel = QGridLayout()
        sel.setHorizontalSpacing(16)
        self._root = QComboBox()
        self._folder = QComboBox()
        self._run = QComboBox()
        sel.addWidget(QLabel("Data root"), 0, 0)
        sel.addWidget(self._root, 1, 0)
        sel.addWidget(QLabel("Folder"), 0, 1)
        sel.addWidget(self._folder, 1, 1)
        sel.addWidget(QLabel("Optimization run"), 0, 2)
        sel.addWidget(self._run, 1, 2)
        sel.setColumnStretch(2, 2)
        lay.addWidget(wrap_card(sel))
        self._empty_banner = Banner()
        lay.addWidget(self._empty_banner)

        # ── save panel (unsaved runs only) ────────────────────────────────────
        self._save_panel = CollapsibleSection("Save this run", expanded=True)
        save_box = QWidget()
        sgrid = QGridLayout(save_box)
        self._save_folder = QComboBox()
        self._save_new_folder = QLineEdit()
        self._save_new_folder.setPlaceholderText("e.g. ES_ivb_sweeps")
        self._save_name = QLineEdit()
        save_btn = QPushButton("Save run")
        save_btn.setProperty("primary", True)
        save_btn.clicked.connect(self._on_save)
        sgrid.addWidget(QLabel("Folder"), 0, 0)
        sgrid.addWidget(self._save_folder, 1, 0)
        self._new_folder_label = QLabel("New folder name")
        sgrid.addWidget(self._new_folder_label, 0, 1)
        sgrid.addWidget(self._save_new_folder, 1, 1)
        sgrid.addWidget(QLabel("Run name"), 0, 2)
        sgrid.addWidget(self._save_name, 1, 2)
        sgrid.addWidget(save_btn, 1, 3)
        sgrid.setColumnStretch(2, 2)
        self._save_panel.add_widget(save_box)
        self._save_panel.setVisible(False)
        lay.addWidget(self._save_panel)
        self._save_banner = Banner()
        lay.addWidget(self._save_banner)
        self._save_folder.currentIndexChanged.connect(self._sync_new_folder_edit)

        # ── run caption + held params ─────────────────────────────────────────
        self._caption = Caption("")
        lay.addWidget(self._caption)
        self._ff_warning = Banner()
        lay.addWidget(self._ff_warning)
        self._held = CollapsibleSection("Held parameters")
        self._held_json = QPlainTextEdit()
        self._held_json.setReadOnly(True)
        self._held_json.setFixedHeight(180)
        self._held.add_widget(self._held_json)
        lay.addWidget(self._held)

        # ── axes: which swept param goes on X / Y (every other one -> slider) ──
        axes_grid = QGridLayout()
        axes_grid.setHorizontalSpacing(16)
        self._x_combo = QComboBox()
        self._y_combo = QComboBox()
        self._swap_btn = QPushButton("⇄  Swap X / Y")
        self._swap_btn.setToolTip("Put the X param on Y and the Y param on X")
        axes_grid.addWidget(QLabel("X axis"), 0, 0)
        axes_grid.addWidget(self._x_combo, 1, 0)
        axes_grid.addWidget(QLabel("Y axis"), 0, 1)
        axes_grid.addWidget(self._y_combo, 1, 1)
        axes_grid.addWidget(self._swap_btn, 1, 2)
        self._axes_note = Caption("")
        axes_grid.addWidget(self._axes_note, 2, 0, 1, 4)
        axes_grid.setColumnStretch(3, 1)
        self._axes_card = wrap_card(axes_grid)
        self._axes_card.setVisible(False)
        lay.addWidget(self._axes_card)
        self._axis_choice: tuple | None = None      # (x_param, y_param | None)
        self._axis_run_key = None                   # resets the choice per run
        self._slider_pos: dict = {}                 # param -> slider index
        self._x_combo.activated.connect(lambda _=None: self._on_axis_picked("x"))
        self._y_combo.activated.connect(lambda _=None: self._on_axis_picked("y"))
        self._swap_btn.clicked.connect(self._on_swap_axes)

        # ── controls row ──────────────────────────────────────────────────────
        controls = QGridLayout()
        controls.setHorizontalSpacing(16)
        self._metric = QComboBox()
        for m in METRIC_ORDER:
            self._metric.addItem(METRIC_LABELS[m], m)
        controls.addWidget(QLabel("Metric"), 0, 0)
        controls.addWidget(self._metric, 1, 0)

        half_box = QHBoxLayout()
        self._half_group = QButtonGroup(self)
        self._half_buttons = {}
        for label in ("both", "1st", "2nd"):
            btn = QRadioButton(label)
            btn.setChecked(label == "both")
            self._half_group.addButton(btn)
            self._half_buttons[label] = btn
            half_box.addWidget(btn)
        half_label = QLabel("Data half")
        half_label.setToolTip("split at the median trading day of the full run")
        controls.addWidget(half_label, 0, 1)
        controls.addLayout(half_box, 1, 1)

        self._min_trades = QSpinBox()
        self._min_trades.setRange(0, 1_000_000)
        self._min_trades.setSingleStep(5)
        self._min_trades.setValue(MIN_TRADES_DEFAULT)
        self._min_trades.setToolTip("cells with fewer trades (after filtering) "
                                    "are hatched")
        controls.addWidget(QLabel("Min trades"), 0, 2)
        controls.addWidget(self._min_trades, 1, 2)

        # coloring: a split value and one curve on each side of it
        curves = QGridLayout()
        curves.setHorizontalSpacing(8)
        curves.setVerticalSpacing(4)
        self._split = _SplitControl(
            "The value where the color scale changes sides. Starts at the "
            "metric's natural point (0 ticks, profit factor 1, 50 % win rate, "
            "the middle of the range for trade count); the cells on each side "
            "are colored with their own curve. Recolor only.")
        self._curve_good = _CurveControl(
            "Bends the colors ABOVE the split — recolor only, values unchanged. "
            "0 = linear · > 0 = exp-like, only the highest cells reach red · "
            "< 0 = log-like, red already reaches cells just above the split.")
        self._curve_bad = _CurveControl(
            "Bends the colors BELOW the split — recolor only. 0 = linear · "
            "> 0 = only the lowest cells reach dark blue · < 0 = dark blue "
            "already reaches cells just below the split.")
        self._split_label = QLabel("Split at")
        self._curve_good_label = QLabel("Above split curve")
        self._curve_bad_label = QLabel("Below split curve")
        reset = QPushButton("Reset")
        reset.setToolTip("Split back to the metric's natural point, both curves to 0")
        reset.clicked.connect(self._reset_coloring)
        curves.addWidget(self._split_label, 0, 0)
        curves.addLayout(self._split.layout_row, 0, 1)
        curves.addWidget(self._curve_good_label, 1, 0)
        curves.addLayout(self._curve_good.layout_row, 1, 1)
        curves.addWidget(self._curve_bad_label, 2, 0)
        curves.addLayout(self._curve_bad.layout_row, 2, 1)
        curves.addWidget(reset, 0, 2, 3, 1)
        curves.setColumnStretch(1, 1)
        self._split_value: float | None = None    # None = the metric's natural point
        self._split_metric = None
        controls.addWidget(QLabel("Coloring"), 0, 3)
        controls.addLayout(curves, 1, 3)
        controls.setColumnStretch(3, 1)
        lay.addWidget(wrap_card(controls))
        # dragging a slider fires many changes: redraw once it settles
        self._recolor_timer = QTimer(self)
        self._recolor_timer.setSingleShot(True)
        self._recolor_timer.setInterval(60)
        self._recolor_timer.timeout.connect(self._refresh_heatmap)
        for c in (self._curve_good, self._curve_bad):
            c.changed.connect(self._recolor_timer.start)
        self._split.changed.connect(self._on_split_changed)

        # slider axes + bucket filter
        self._sliders_holder = QVBoxLayout()
        lay.addLayout(self._sliders_holder)
        lay.addWidget(Caption("Day types included"))
        self._buckets = CheckboxFilterRow(list(BUCKET_ORDER),
                                          per_row=len(BUCKET_ORDER))
        lay.addWidget(self._buckets)
        self._filter_banner = Banner()
        lay.addWidget(self._filter_banner)

        # ── heatmap + drill-down ──────────────────────────────────────────────
        self._heatmap = HeatmapChart()
        self._heatmap.setVisible(False)
        self._heatmap.cellClicked.connect(self._on_cell_clicked)
        lay.addWidget(self._heatmap)
        self._heatmap_caption = Caption(
            "Click a cell to open its full trade report below (click it again "
            "to close) · Hatched = fewer trades than the min-trades threshold "
            "· blank = no finite value (no trades, PF ∞, undefined Sharpe)")
        self._heatmap_caption.setVisible(False)
        lay.addWidget(self._heatmap_caption)
        self._reading = CollapsibleSection("How to read this surface")
        reading = QLabel(
            "- **Hatched cells** keep their color but sit below the min-trades "
            "threshold — read them with suspicion.\n"
            "- **Blank cells** have no finite metric value and are excluded "
            "from the color scale.\n"
            "- **Sharpe (daily, zero-filled)**: daily P&L over every business "
            "day between the first and last (filtered) trade — days without "
            "trades count as 0. **Sharpe (traded days)**: only days with at "
            "least one trade. Both ×√252 — approximate once day types are "
            "filtered out, comparative use only.\n"
            "- Look for **plateaus that survive both halves and day-type "
            "changes**, not the single brightest cell — the top cell of a "
            "large grid is selection-biased by construction.")
        reading.setTextFormat(Qt.MarkdownText)
        reading.setWordWrap(True)
        self._reading.add_widget(reading)
        self._reading.setVisible(False)
        lay.addWidget(self._reading)

        self.cell_report = TradeReport(
            settings, track_worker=track_worker, header="Cell detail",
            empty_message="No trades in this cell.")
        self.cell_report.setVisible(False)
        lay.addWidget(self.cell_report)
        # un-clicking the selected square closes its drill-down report
        self._heatmap.cellDeselected.connect(self.cell_report.clear)
        lay.addSpacing(BOTTOM_PADDING)
        lay.addStretch()

        # ── wiring ────────────────────────────────────────────────────────────
        self._root.currentIndexChanged.connect(self._refresh_folders)
        self._folder.currentIndexChanged.connect(self._refresh_runs)
        self._run.currentIndexChanged.connect(self._on_run_selected)
        self._metric.currentIndexChanged.connect(lambda _=None: self._refresh_heatmap())
        for btn in self._half_buttons.values():
            btn.toggled.connect(lambda on: self._refresh_heatmap() if on else None)
        self._min_trades.valueChanged.connect(lambda _=None: self._refresh_heatmap())
        self._buckets.selectionChanged.connect(self._refresh_heatmap)

        self.refresh_run_list(select_unsaved=False)

    # ══ run selection ═══════════════════════════════════════════════════════════
    def refresh_run_list(self, select_unsaved: bool) -> None:
        """Repopulate root/folder/run selectors (called on tab init, settings
        refresh, and the post-run/post-save hand-off)."""
        self._root.blockSignals(True)
        self._root.clear()
        for root in self.settings.data_roots:
            self._root.addItem(str(root), root)
        self._root.blockSignals(False)
        self._root.setVisible(self._root.count() > 1)
        self._select_unsaved_on_refresh = select_unsaved
        self._refresh_folders()

    def _runs_root(self):
        root = self._root.currentData() if self._root.count() else self.settings.data_roots[0]
        return optimizations_root(root)

    def _refresh_folders(self) -> None:
        runs = opt_io.list_runs(root=self._runs_root())
        by_folder: dict = {}
        for rel in runs:
            folder, _, name = rel.rpartition("/")
            by_folder.setdefault(folder, []).append(name)
        self._by_folder = by_folder

        folder_labels = ([NO_FOLDER] if "" in by_folder else []) \
            + sorted(f for f in by_folder if f)
        if not folder_labels:
            folder_labels = [NO_FOLDER]
        current = (self.state.loaded_run or "")
        current_folder, _, _name = current.rpartition("/")
        current_label = current_folder if current_folder else NO_FOLDER

        self._folder.blockSignals(True)
        self._folder.clear()
        self._folder.addItems(folder_labels)
        idx = folder_labels.index(current_label) if current_label in folder_labels else 0
        self._folder.setCurrentIndex(idx)
        self._folder.blockSignals(False)
        self._refresh_runs()

    def _refresh_runs(self) -> None:
        folder = "" if self._folder.currentText() == NO_FOLDER else self._folder.currentText()
        has_unsaved = self.state.unsaved and self.state.trades is not None
        options = ([UNSAVED_LABEL] if has_unsaved else []) \
            + self._by_folder.get(folder, [])

        self._empty_banner.clear_message()
        self._run.blockSignals(True)
        self._run.clear()
        if not options:
            self._run.addItem("— no runs in this folder —")
            self._run.setEnabled(False)
            self._run.blockSignals(False)
            if not self._by_folder and not has_unsaved:
                self._empty_banner.show_message(
                    "info", "No saved optimization runs yet — create one under "
                            "New Run.")
                self._show_run(None)
            return
        self._run.setEnabled(True)
        self._run.addItems(options)
        current_name = (self.state.loaded_run or "").rpartition("/")[2]
        default = UNSAVED_LABEL if has_unsaved else current_name
        idx = options.index(default) if default in options else 0
        self._run.setCurrentIndex(idx)
        self._run.blockSignals(False)
        self._on_run_selected()

    def _on_run_selected(self) -> None:
        if not self._run.isEnabled():
            return
        selected = self._run.currentText()
        if selected == UNSAVED_LABEL:
            self._show_run(unsaved=True)
            return
        if selected.startswith("—"):
            return
        # switching to a saved run discards any unsaved one
        self.state.unsaved = False
        folder = "" if self._folder.currentText() == NO_FOLDER else self._folder.currentText()
        rel = f"{folder}/{selected}" if folder else selected
        if rel != self.state.loaded_run or self.state.trades is None:
            trades, meta = opt_io.load_run(rel, root=self._runs_root())
            self.state.loaded_run = rel
            self.state.trades = trades
            self.state.meta = meta
            self.state.run_root = self._root.currentData() if self._root.count() \
                else self.settings.data_roots[0]
            self._grid_key = None
        self._show_run(unsaved=False)

    # ══ run display ═══════════════════════════════════════════════════════════
    def show_success(self, text: str) -> None:
        self._success.show_message("success", text)

    def _show_run(self, unsaved: bool | None) -> None:
        """Rebuild the caption/save-panel/controls for the current state."""
        self.cell_report.clear()
        if unsaved is None or self.state.trades is None:
            for w in (self._caption, self._held, self._heatmap, self._axes_card,
                      self._heatmap_caption, self._reading, self._save_panel):
                w.setVisible(False)
            return

        meta = self.state.meta
        be_band = meta.get("be_band_ticks", 0.0)
        self._caption.setVisible(True)
        self._caption.setText(
            f"{meta.get('strategy')} on {meta.get('dataset')} · "
            f"{meta.get('start_date')} → {meta.get('end_date')} · "
            f"{meta.get('n_combos')} combos · {meta.get('n_trades')} trades · "
            f"BE band {be_band:g} ticks")
        if not meta.get("ff_events_found", True):
            self._ff_warning.show_message(
                "warning", "This run was built without ff_usd_events.parquet — "
                           "every day is bucketed 'normal'.")
        else:
            self._ff_warning.clear_message()

        self._save_panel.setVisible(bool(unsaved))
        if unsaved:
            self._save_folder.blockSignals(True)
            self._save_folder.clear()
            self._save_folder.addItems([NO_FOLDER, NEW_FOLDER]
                                       + opt_io.list_folders(root=self._runs_root()))
            self._save_folder.blockSignals(False)
            self._sync_new_folder_edit()
            self._save_name.setText(
                f"{meta.get('ticker')}_{meta.get('strategy')}_"
                f"{meta.get('start_date')}_{meta.get('end_date')}")

        self._held.setVisible(True)
        self._held_json.setPlainText(
            json.dumps(meta.get("fixed_params", {}), indent=2))

        # data-half radio enabled only when the run has a split date
        split = meta.get("split_date")
        for btn in self._half_buttons.values():
            btn.setEnabled(split is not None)

        # min-trades default from the run's meta
        self._min_trades.blockSignals(True)
        self._min_trades.setValue(int(meta.get("min_trades_default",
                                                MIN_TRADES_DEFAULT)))
        self._min_trades.blockSignals(False)

        self._sync_axes()
        self._rebuild_sliders()
        self._refresh_heatmap()

    def _sync_new_folder_edit(self) -> None:
        is_new = self._save_folder.currentText() == NEW_FOLDER
        self._save_new_folder.setVisible(is_new)
        self._new_folder_label.setVisible(is_new)

    def _on_save(self) -> None:
        self._save_banner.clear_message()
        meta = self.state.meta
        choice = self._save_folder.currentText()
        if choice == NEW_FOLDER:
            folder = self._save_new_folder.text().strip()
            if not folder:
                self._save_banner.show_message("error", "Enter a folder name.")
                return
        else:
            folder = "" if choice == NO_FOLDER else choice
        run_name = self._save_name.text()
        if not run_name.strip():
            self._save_banner.show_message("error", "Enter a run name.")
            return
        root = self._runs_root()
        run_dir = opt_io.save_run(self.state.trades, meta, run_name=run_name,
                                  folder=folder, root=root)
        rel = run_dir.relative_to(root).as_posix()
        meta = dict(meta)
        meta["run_name"] = rel
        self.state.meta = meta
        self.state.loaded_run = rel
        self.state.unsaved = False
        self.show_success(f"Saved to {run_dir}")
        self.refresh_run_list(select_unsaved=False)

    # ══ sliders / heatmap ═══════════════════════════════════════════════════════
    def _swept_axes(self) -> list[dict]:
        """The run's swept params ({"param", "values"}) in their saved role
        order (x, y, slider, slider2) — the default layout."""
        axes = self.state.meta.get("axes", {}) if self.state.meta else {}
        return [axes[r] for r in ("x", "y", "slider", "slider2") if axes.get(r)]

    def _axes(self):
        """(x_axis, y_axis | None, slider_axes) for the CURRENT axis choice:
        the user's X / Y pick, every other swept param as a slider (in the
        saved order)."""
        swept = self._swept_axes()
        if not swept:
            return None, None, []
        by_param = {ax["param"]: ax for ax in swept}
        x, y = self._axis_choice or (swept[0]["param"],
                                     swept[1]["param"] if len(swept) > 1 else None)
        if x not in by_param:                       # stale choice: fall back
            x, y = swept[0]["param"], (swept[1]["param"] if len(swept) > 1 else None)
        sliders = [ax for ax in swept if ax["param"] not in (x, y)]
        return by_param[x], by_param.get(y), sliders

    # ── axis selection ───────────────────────────────────────────────────────
    def _sync_axes(self) -> None:
        """Fill the X / Y pickers for the shown run. A different run starts
        from its saved layout; re-showing the same run keeps the user's pick."""
        meta = self.state.meta or {}
        swept = self._swept_axes()
        run_key = (meta.get("run_name"), meta.get("created_at"),
                   tuple(ax["param"] for ax in swept))
        if run_key != self._axis_run_key:
            self._axis_run_key = run_key
            self._axis_choice = None
            self._slider_pos = {}
        x_axis, y_axis, _sliders = self._axes()
        if x_axis is None:
            self._axes_card.setVisible(False)
            return
        self._axis_choice = (x_axis["param"], y_axis["param"] if y_axis else None)
        params = [ax["param"] for ax in swept]
        for combo, current in ((self._x_combo, x_axis["param"]),
                               (self._y_combo, y_axis["param"] if y_axis else None)):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(params)
            if current is None:
                combo.addItem("—")
                combo.setCurrentIndex(combo.count() - 1)
            else:
                combo.setCurrentIndex(params.index(current))
            combo.blockSignals(False)
        two = len(params) > 1
        self._y_combo.setEnabled(two)
        self._swap_btn.setEnabled(two)
        others = [p for p in params if p not in self._axis_choice]
        self._axes_note.setText(
            "Only one param was swept — it is the X axis." if not two else
            ("Sliders below: " + ", ".join(others) + "." if others else
             "Pick any swept param for each axis — choosing the one already on the "
             "other axis swaps them."))
        self._axes_card.setVisible(True)

    def _on_axis_picked(self, which: str) -> None:
        x, y = self._axis_choice
        combo = self._x_combo if which == "x" else self._y_combo
        picked = combo.currentText()
        if which == "x":
            if picked == y:                    # taking Y's param: swap them
                y = x
            x = picked
        else:
            if picked == x:
                x = y
            y = picked
        self._set_axis_choice((x, y))

    def _on_swap_axes(self) -> None:
        x, y = self._axis_choice
        if y is not None:
            self._set_axis_choice((y, x))

    def _set_axis_choice(self, choice: tuple) -> None:
        if choice == self._axis_choice:
            return
        self._axis_choice = choice
        self.cell_report.clear()               # the open cell belongs to the old grid
        self._sync_axes()
        self._rebuild_sliders()
        self._refresh_heatmap()

    def _rebuild_sliders(self) -> None:
        while self._sliders_holder.count():
            item = self._sliders_holder.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()               # gone NOW, not when deleteLater runs
                w.deleteLater()
        self._slider_widgets = []
        _x, _y, slider_axes = self._axes()
        for idx, ax in enumerate(slider_axes, start=1):
            options = ax["values"]
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(QLabel(f"{ax['param']} (slider {idx})"))
            if len(options) > 1:
                slider = QSlider(Qt.Horizontal)
                slider.setRange(0, len(options) - 1)
                # a param that moves back from an axis to a slider keeps its
                # last position
                start = min(self._slider_pos.get(ax["param"], 0), len(options) - 1)
                slider.setValue(start)
                slider.setMaximumWidth(320)
                value_label = QLabel(_fmt_axis_value(options[start]))
                slider.valueChanged.connect(
                    lambda i, lab=value_label, opts=options, prm=ax["param"]:
                    (self._slider_pos.__setitem__(prm, i),
                     lab.setText(_fmt_axis_value(opts[i])),
                     self._refresh_heatmap()))
                h.addWidget(slider)
                h.addWidget(value_label)
                self._slider_widgets.append((ax, slider, value_label))
            else:
                value_label = QLabel(_fmt_axis_value(options[0]))
                h.addWidget(value_label)
                self._slider_widgets.append((ax, None, value_label))
            h.addStretch()
            self._sliders_holder.addWidget(row)

    def _slider_values(self) -> dict:
        values = {}
        for ax, slider, _label in self._slider_widgets:
            options = ax["values"]
            values[ax["param"]] = options[slider.value()] if slider is not None \
                else options[0]
        return values

    def _half(self) -> str:
        for label, btn in self._half_buttons.items():
            if btn.isChecked():
                return label
        return "both"

    def _on_split_changed(self) -> None:
        self._split_value = self._split.value()
        self._recolor_timer.start()

    def _reset_coloring(self) -> None:
        self._split_value = None                 # back to the metric's natural point
        for c in (self._curve_good, self._curve_bad):
            c.blockSignals(True)
            c.set_value(0.0)
            c.blockSignals(False)
        self._refresh_heatmap()

    def _refresh_heatmap(self) -> None:
        if self.state.trades is None or self.state.meta is None:
            return
        meta = self.state.meta
        x_axis, y_axis, slider_axes = self._axes()
        self._filter_banner.clear_message()
        if x_axis is None:
            self._filter_banner.show_message(
                "error", "Run has no X axis — meta.json is incomplete.")
            return
        selected_buckets = self._buckets.selected()
        if not selected_buckets:
            self._filter_banner.show_message("warning", "No day types selected.")
            self._heatmap.setVisible(False)
            return

        be_band = meta.get("be_band_ticks", 0.0)
        split = meta.get("split_date")
        half = self._half()
        slider_values = self._slider_values()

        # min_trades and the metric toggle are NOT part of the key: they only
        # re-mask / recolor the cached grid, no recompute. created_at
        # disambiguates successive unsaved runs (both have run_name None).
        grid_key = (meta.get("run_name"), meta.get("created_at"),
                    x_axis["param"], y_axis["param"] if y_axis else None,
                    tuple(sorted(slider_values.items(), key=lambda kv: kv[0])),
                    tuple(selected_buckets), half, float(be_band))
        if self._grid_key != grid_key:
            df = self.state.trades
            for ax in slider_axes:
                df = df[df[ax["param"]] == slider_values[ax["param"]]]
            if len(selected_buckets) < len(BUCKET_ORDER):
                df = df[df["day_bucket"].isin(selected_buckets)]
            if half != "both" and split is not None:
                dates = pd.to_datetime(df["date"])
                split_ts = pd.Timestamp(split)
                df = df[dates <= split_ts] if half == "1st" else df[dates > split_ts]
            cell_cols = [x_axis["param"]] + ([y_axis["param"]] if y_axis else [])
            self._grid = compute_metrics_by_cell(df, cell_cols, be_band)
            self._grid_key = grid_key

        x_values = x_axis["values"]
        y_param = y_axis["param"] if y_axis else None
        y_values = y_axis["values"] if y_axis else ["—"]
        arrays = _build_grid_arrays(self._grid, x_axis["param"], x_values,
                                    y_param, y_axis["values"] if y_axis else None)

        slider_desc = " · ".join(
            f"{ax['param']} = {_fmt_axis_value(v)}"
            for ax, v in ((ax, slider_values[ax["param"]]) for ax in slider_axes))
        metric = self._metric.currentData()
        if metric != self._split_metric:           # a new metric: its natural split
            self._split_metric = metric
            self._split_value = None
        # the split slider spans the values on screen (the heatmap's own range)
        z = arrays[metric]
        finite = z[np.isfinite(z)]
        zmin = float(finite.min()) if finite.size else 0.0
        zmax = float(finite.max()) if finite.size else 1.0
        if zmin == zmax:
            zmin, zmax = zmin - 0.5, zmax + 0.5
        pivot = self._split_value
        if pivot is None:
            natural = METRIC_PIVOTS.get(metric)
            pivot = zmin + (zmax - zmin) / 2 if natural is None else natural
        self._split.set_range(zmin, zmax, pivot)
        status = self._heatmap.set_data(
            arrays, metric, METRIC_LABELS[metric], x_axis["param"], x_values,
            y_param, y_values, slider_desc, int(self._min_trades.value()),
            color_curves=(self._curve_good.value(), self._curve_bad.value()),
            pivot=pivot)
        self._heatmap.setVisible(True)
        self._heatmap_caption.setVisible(True)
        self._reading.setVisible(True)
        if status["no_finite"]:
            self._filter_banner.show_message(
                "info", "No cell has a finite value for this metric under the "
                        "current filters.")
        if status["all_masked"]:
            self._filter_banner.show_message(
                "warning", "All cells are below the min-trades threshold under "
                           "the current filters.")

    def _on_cell_clicked(self, xi: int, yj: int) -> None:
        meta = self.state.meta
        x_axis, y_axis, slider_axes = self._axes()
        cell_detail.show_cell(
            self.cell_report, self.state.trades, meta, x_axis, y_axis, slider_axes,
            self._slider_values(), self._half(), meta.get("split_date"),
            (xi, yj), self._buckets.selected(), self.state.run_root)
