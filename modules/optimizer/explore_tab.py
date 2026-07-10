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

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QPushButton, QRadioButton, QSlider, QSpinBox,
                               QVBoxLayout, QWidget)

import pandas as pd

from modules.common.backend.data_roots import optimizations_root
from modules.common.ui import theme
from modules.common.ui.charts.heatmap import HeatmapChart
from modules.common.ui.trade_report.filters import CheckboxFilterRow
from modules.common.ui.widgets import (Banner, Caption, CollapsibleSection,
                                       SectionHeader, wrap_card)
from modules.optimizer.backend import io as opt_io
from modules.optimizer.backend.buckets import BUCKET_ORDER
from modules.optimizer.backend.heatmap_model import (MIN_TRADES_DEFAULT,
                                                     _build_grid_arrays,
                                                     _fmt_axis_value)
from modules.optimizer.backend.metrics import (METRIC_LABELS, METRIC_ORDER,
                                               compute_metrics_by_cell)
from modules.optimizer.cell_detail import CellDetailPanel

UNSAVED_LABEL = "● Unsaved run — switching away discards it"
NEW_FOLDER    = "── New folder ──"
NO_FOLDER     = "(no folder)"


class ExploreTab(QWidget):
    """Run state (trades/meta/unsaved/loaded_run/run_root) lives on the
    OptimizerWindow (`state`) — New Run hands off through it."""

    def __init__(self, settings, state, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.state = state
        self._grid = None
        self._grid_key = None
        self._slider_widgets: list[tuple[dict, QSlider | None, QLabel]] = []

        lay = QVBoxLayout(self)
        lay.setSpacing(10)

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

        curve_box = QHBoxLayout()
        self._color_curve = QSlider(Qt.Horizontal)
        self._color_curve.setRange(-8, 8)   # -2.0 .. 2.0 in 0.25 steps
        self._color_curve.setValue(0)
        self._color_curve.setToolTip(
            "Bends the value→color mapping (recolor only, values unchanged). "
            "0 = linear · < 0 = log-like, the yellow→red band arrives at lower "
            "values · > 0 = exp-like, only the top cells go red.")
        self._color_value = QLabel("0.0")
        curve_box.addWidget(self._color_curve)
        curve_box.addWidget(self._color_value)
        controls.addWidget(QLabel("Color curve"), 0, 3)
        controls.addLayout(curve_box, 1, 3)
        lay.addWidget(wrap_card(controls))

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
            "Click a cell to open its full trade report below · Hatched = "
            "fewer trades than the min-trades threshold · blank = no finite "
            "value (no trades, PF ∞, undefined Sharpe)")
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

        self.cell_detail = CellDetailPanel()
        lay.addWidget(self.cell_detail)
        lay.addStretch()

        # ── wiring ────────────────────────────────────────────────────────────
        self._root.currentIndexChanged.connect(self._refresh_folders)
        self._folder.currentIndexChanged.connect(self._refresh_runs)
        self._run.currentIndexChanged.connect(self._on_run_selected)
        self._metric.currentIndexChanged.connect(lambda _=None: self._refresh_heatmap())
        for btn in self._half_buttons.values():
            btn.toggled.connect(lambda on: self._refresh_heatmap() if on else None)
        self._min_trades.valueChanged.connect(lambda _=None: self._refresh_heatmap())
        self._color_curve.valueChanged.connect(self._on_color_changed)
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
        self.cell_detail.hide_detail()
        if unsaved is None or self.state.trades is None:
            for w in (self._caption, self._held, self._heatmap,
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
    def _axes(self):
        axes = self.state.meta.get("axes", {}) if self.state.meta else {}
        x_axis = axes.get("x")
        y_axis = axes.get("y")
        slider_axes = [ax for ax in (axes.get("slider"), axes.get("slider2")) if ax]
        return x_axis, y_axis, slider_axes

    def _rebuild_sliders(self) -> None:
        while self._sliders_holder.count():
            item = self._sliders_holder.takeAt(0)
            w = item.widget()
            if w is not None:
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
                slider.setValue(0)
                slider.setMaximumWidth(320)
                value_label = QLabel(_fmt_axis_value(options[0]))
                slider.valueChanged.connect(
                    lambda i, lab=value_label, opts=options:
                    (lab.setText(_fmt_axis_value(opts[i])),
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

    def _on_color_changed(self, v: int) -> None:
        self._color_value.setText(f"{v * 0.25:g}")
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
        color_gamma = 2.0 ** (self._color_curve.value() * 0.25)
        status = self._heatmap.set_data(
            arrays, metric, METRIC_LABELS[metric], x_axis["param"], x_values,
            y_param, y_values, slider_desc, int(self._min_trades.value()),
            color_gamma=color_gamma)
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
        self.cell_detail.show_cell(
            self.state.trades, meta, x_axis, y_axis, slider_axes,
            self._slider_values(), self._half(), meta.get("split_date"),
            (xi, yj), self._buckets.selected(), self.state.run_root)
