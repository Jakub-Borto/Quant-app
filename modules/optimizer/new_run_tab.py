"""
Optimizer "New Run" tab — configure and launch a grid run.

The PySide6 port of render_setup / execute_grid_run's UI half: dataset/
strategy/date controls, the SweepPanel, the live combo readout (sizes ×
counts, degenerate-axis warnings), run settings (BE band / min-trades /
workers / memory budget with the live per-worker estimate + clamp), the
>2000-combo confirmation gate, then Run on a worker with the throttled
progress log and a working Cancel (the engine's pool shutdown fires through
the on_progress exception).

On success the tab hands (trades, meta, data root) to the window
(adopt_unsaved_run) which switches to Explore — the old
opt_switch_to_explore rerun dance as a direct call.
"""

import os

from PySide6.QtCore import QDate, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDateEdit, QDoubleSpinBox,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

from modules.common.backend.asset_info import ASSET_INFO, HIDDEN_PARAMS
from modules.common.backend.data_roots import (DatasetRef, available_dates,
                                               resolve_ff_events,
                                               scan_structure)
from modules.common.backend.plugins import PluginRef, list_strategies, load_strategy
from modules.common.ui.widgets import (Banner, Caption, ProgressLogPanel,
                                       wrap_card)
from modules.common.ui.workers import FunctionWorker
from modules.optimizer.backend.engine import (check_param_columns,
                                              estimate_worker_memory,
                                              sibling_dataset_folders)
from modules.optimizer.backend.heatmap_model import (COMBO_CONFIRM_THRESHOLD,
                                                     MIN_TRADES_DEFAULT,
                                                     NON_US_CALENDAR_ASSETS)
from modules.optimizer.backend.param_space import (ROLES, combo_count,
                                                   sweep_kind)
from modules.optimizer.backend.run_setup import run_grid_job
from modules.optimizer.sweep_panel import SweepPanel

_SPEED_CAPTION = (
    "Serial grid speed rides on the strategy's internal day cache being "
    "param-independent (ivb_model_optimized: yes). With parallel workers, "
    "each worker pays its own cold start before running warm — serial can "
    "beat parallel on small grids. Stopping a parallel run waits for the "
    "in-flight combo on each worker before releasing."
)


class NewRunTab(QWidget):
    # trades, meta, data root — consumed by OptimizerWindow.adopt_unsaved_run
    runFinished = Signal(object, object, object)

    def __init__(self, settings, track_worker, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._track_worker = track_worker
        self._structure: dict = {}
        self._strategies: list[PluginRef] = []
        self._strategy_module = None
        self._strategy_ref: PluginRef | None = None
        self._sweep_panel: SweepPanel | None = None
        self._effective_workers = 1
        self._worker: FunctionWorker | None = None

        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        # ── setup controls ────────────────────────────────────────────────────
        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        self._type = QComboBox()
        self._asset = QComboBox()
        self._dataset = QComboBox()
        self._strategy = QComboBox()
        for r, (label, w) in enumerate([("Type", self._type),
                                        ("Asset", self._asset),
                                        ("Dataset", self._dataset),
                                        ("Strategy", self._strategy)]):
            grid.addWidget(QLabel(label), r, 0)
            grid.addWidget(w, r, 1)
        self._start = QDateEdit(calendarPopup=True)
        self._end = QDateEdit(calendarPopup=True)
        grid.addWidget(QLabel("Start date"), 0, 2)
        grid.addWidget(self._start, 0, 3)
        grid.addWidget(QLabel("End date"), 1, 2)
        grid.addWidget(self._end, 1, 3)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        lay.addWidget(wrap_card(grid))

        self._banner = Banner()
        lay.addWidget(self._banner)

        # ── sweep panel (rebuilt per strategy) ────────────────────────────────
        self._panel_holder = QVBoxLayout()
        lay.addLayout(self._panel_holder)

        # ── live readout ──────────────────────────────────────────────────────
        self._readout = Banner()
        lay.addWidget(self._readout)
        self._warnings = Caption("")
        lay.addWidget(self._warnings)

        # ── run settings ──────────────────────────────────────────────────────
        box = QGroupBox("Run settings")
        srow = QGridLayout(box)
        self._be_band = QDoubleSpinBox()
        self._be_band.setRange(0.0, 1e9)
        self._be_band.setSingleStep(0.5)
        self._be_band.setValue(0.0)
        self._be_band.setToolTip("win = pnl > band, breakeven = |pnl| <= band, "
                                 "loss = pnl < -band")
        self._min_trades = QSpinBox()
        self._min_trades.setRange(0, 1_000_000)
        self._min_trades.setSingleStep(5)
        self._min_trades.setValue(MIN_TRADES_DEFAULT)
        self._min_trades.setToolTip("default hatch threshold in the heatmap "
                                    "(changeable there)")
        max_workers = min(os.process_cpu_count() or 1, 61)
        self._workers = QSpinBox()
        self._workers.setRange(1, max_workers)
        self._workers.setValue(1)
        self._workers.setToolTip(
            f"1 = serial (in-process, reuses the warm day cache across runs). "
            f">1 = separate processes, each building its OWN day cache — pays "
            f"off from ~100s of combos. Capped at your {max_workers} logical "
            f"cores: backtests are CPU-bound, so extra processes only add "
            f"memory, not speed.")
        self._mem_budget = QDoubleSpinBox()
        self._mem_budget.setRange(0.5, 1e6)
        self._mem_budget.setSingleStep(0.5)
        self._mem_budget.setValue(4.0)
        self._mem_budget.setToolTip("caps the parallel worker count via the "
                                    "per-worker estimate")
        for c, (label, w) in enumerate([("BE band (ticks)", self._be_band),
                                        ("Min trades default", self._min_trades),
                                        ("Parallel workers", self._workers),
                                        ("Memory budget (GB)", self._mem_budget)]):
            srow.addWidget(QLabel(label), 0, c)
            srow.addWidget(w, 1, c)
        self._estimate = Caption("")
        srow.addWidget(self._estimate, 2, 0, 1, 4)
        lay.addWidget(box)
        lay.addWidget(Caption(_SPEED_CAPTION))

        # ── combo guard + run/cancel ──────────────────────────────────────────
        self._confirm = QCheckBox()
        self._confirm.setVisible(False)
        lay.addWidget(self._confirm)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run grid")
        self._run_btn.setProperty("primary", True)
        self._run_btn.setMinimumWidth(200)
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        refresh_btn = QPushButton("Refresh folders")
        refresh_btn.clicked.connect(self.rescan)
        btn_row.addStretch()
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(refresh_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self._progress = ProgressLogPanel(log_height=280)
        lay.addWidget(self._progress)
        lay.addStretch()

        # ── wiring ────────────────────────────────────────────────────────────
        self._type.currentIndexChanged.connect(self._on_type_changed)
        self._asset.currentIndexChanged.connect(self._on_asset_changed)
        self._dataset.currentIndexChanged.connect(self._on_dataset_changed)
        self._strategy.currentIndexChanged.connect(self._on_strategy_changed)
        for w in (self._workers, self._mem_budget):
            w.valueChanged.connect(lambda _=None: self._refresh_readout())
        self.rescan()

    # ── scanning / cascading pickers (backtester pattern) ─────────────────────
    def rescan(self) -> None:
        self._strategies = list_strategies(self.settings.plugin_dirs("strategies"))
        self._strategy.blockSignals(True)
        self._strategy.clear()
        self._strategy.addItems([s.label for s in self._strategies])
        self._strategy.blockSignals(False)

        self._structure = scan_structure(self.settings.data_roots, source="parquet")
        self._type.blockSignals(True)
        self._type.clear()
        self._type.addItems(list(self._structure.keys()))
        self._type.blockSignals(False)
        self._on_type_changed()
        self._on_strategy_changed()

    def _on_type_changed(self) -> None:
        assets = list(self._structure.get(self._type.currentText(), {}).keys())
        self._asset.blockSignals(True)
        self._asset.clear()
        self._asset.addItems(assets)
        self._asset.blockSignals(False)
        self._on_asset_changed()

    def _on_asset_changed(self) -> None:
        refs = self._structure.get(self._type.currentText(), {}) \
                              .get(self._asset.currentText(), [])
        self._dataset.blockSignals(True)
        self._dataset.clear()
        for ref in refs:
            self._dataset.addItem(ref.label, ref)
        self._dataset.blockSignals(False)
        self._on_dataset_changed()

    def _on_dataset_changed(self) -> None:
        ref: DatasetRef | None = self._dataset.currentData()
        if ref is None:
            return
        dates = available_dates(ref.path)
        if not dates:
            return
        lo, hi = dates[0].date(), dates[-1].date()
        for w, value in ((self._start, lo), (self._end, hi)):
            w.blockSignals(True)
            w.setDateRange(QDate(lo.year, lo.month, lo.day),
                           QDate(hi.year, hi.month, hi.day))
            w.setDate(QDate(value.year, value.month, value.day))
            w.blockSignals(False)
        self._refresh_readout()

    def _on_strategy_changed(self) -> None:
        while self._panel_holder.count():
            item = self._panel_holder.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._sweep_panel = None
        self._strategy_module = None
        self._strategy_ref = None
        self._banner.clear_message()
        if self._strategy.currentIndex() < 0:
            return
        self._strategy_ref = self._strategies[self._strategy.currentIndex()]
        try:
            self._strategy_module = load_strategy(self._strategy_ref)
        except Exception as e:  # noqa: BLE001
            self._banner.show_message("error", f"Could not load strategy "
                                               f"'{self._strategy_ref.name}': {e}")
            return

        visible = {k: v for k, v in
                   getattr(self._strategy_module, "PARAMS", {}).items()
                   if k not in HIDDEN_PARAMS}
        opts = getattr(self._strategy_module, "PARAMS_OPTIONS", {}) or {}
        if not any(sweep_kind(v, opts.get(k)) is not None
                   for k, v in visible.items()):
            self._banner.show_message(
                "info", f"{self._strategy_ref.name} has no sweepable params — "
                        f"it needs a PARAMS dict with int/float/str/bool defaults.")
            return

        self._sweep_panel = SweepPanel(self._strategy_module)
        self._sweep_panel.changed.connect(self._refresh_readout)
        self._panel_holder.addWidget(self._sweep_panel)
        self._refresh_readout()

    # ── live readout (the old inline info/warning block) ──────────────────────
    def _current_axes(self) -> list | None:
        """axes ordered x -> y -> slider, or None when the setup is incomplete
        / invalid (readout shows why)."""
        if self._sweep_panel is None:
            return None
        order = self._sweep_panel.sweep_order()
        if not order:
            self._readout.show_message("info", "Check at least one param to sweep.")
            return None
        swept = self._sweep_panel.swept_values()
        if any(swept.get(p) is None for p in order):
            self._readout.show_message("error", "Fix the sweep inputs flagged (⚠) above.")
            return None
        roles = self._sweep_panel.roles_by_param()
        if roles is None:
            self._readout.show_message("error", "Each swept param needs a "
                                                "distinct role.")
            return None
        param_by_role = {role: p for p, role in roles.items()}
        axes = [{"param": param_by_role[role],
                 "values": swept[param_by_role[role]],
                 "role": role}
                for role in ROLES if role in param_by_role]
        try:
            check_param_columns(axes)
        except ValueError as e:
            self._readout.show_message("error", str(e))
            return None
        return axes

    def _refresh_readout(self) -> None:
        self._warnings.setText("")
        axes = self._current_axes()
        if axes is None:
            self._confirm.setVisible(False)
            self._run_btn.setEnabled(False)
            return
        n_combos = combo_count(axes)
        sizes = " × ".join(f"|{a['param']}| = {len(a['values'])}" for a in axes)
        self._readout.show_message("info", f"{sizes}   →   {n_combos} backtests")

        warnings = []
        order = self._sweep_panel.sweep_order()
        if len(order) == 1:
            warnings.append("Only 1 swept param — the heatmap degenerates to a "
                            "single row.")
        if len(order) < 3:
            warnings.append("No sliders (the 3rd and 4th swept params become "
                            "sliders).")
        for a in axes:
            if len(a["values"]) == 1:
                warnings.append(f"Axis '{a['param']}' has a single value — "
                                f"degenerate axis.")
        self._warnings.setText("\n".join(warnings))

        # memory estimate + worker clamp (verbatim math)
        ref: DatasetRef | None = self._dataset.currentData()
        if ref is not None:
            siblings = sibling_dataset_folders(ref.path,
                                               self._sweep_panel.fixed_params())
            est = estimate_worker_memory(ref.path, self._start.date().toPython(),
                                         self._end.date().toPython(),
                                         extra_folders=siblings)
            allowed = max(1, int(self._mem_budget.value() * 1024 // est["est_mb"])) \
                if est["est_mb"] > 0 else 1
            self._effective_workers = min(int(self._workers.value()), allowed)
            datasets_desc = ref.dataset + "".join(f" + {p.name}" for p in siblings)
            text = (f"≈ {est['est_mb']:.0f} MB/worker for {est['n_days']} days "
                    f"({est['disk_mb']:.0f} MB on disk: {datasets_desc}) → "
                    f"budget allows {allowed} worker(s). Rough estimate.")
            if self._effective_workers < int(self._workers.value()):
                text += (f"  ⚠ Worker count clamped to "
                         f"{self._effective_workers} by the memory budget.")
            self._estimate.setText(text)

        # combo guard
        if n_combos > COMBO_CONFIRM_THRESHOLD:
            self._confirm.setText(f"Run all {n_combos} backtests anyway "
                                  f"(exceeds the {COMBO_CONFIRM_THRESHOLD} "
                                  f"combo guard — this can take a long time)")
            if not self._confirm.isVisible():
                self._confirm.setChecked(False)
            self._confirm.setVisible(True)
            self._confirm.toggled.connect(self._sync_run_enabled)
            self._sync_run_enabled()
        else:
            self._confirm.setVisible(False)
            self._run_btn.setEnabled(True)

    def _sync_run_enabled(self) -> None:
        self._run_btn.setEnabled(not self._confirm.isVisible()
                                 or self._confirm.isChecked())

    # ── run flow ──────────────────────────────────────────────────────────────
    def _on_run(self) -> None:
        self._banner.clear_message()
        ref: DatasetRef | None = self._dataset.currentData()
        axes = self._current_axes()
        if ref is None or axes is None or self._strategy_module is None:
            return
        asset = ref.asset
        if asset not in ASSET_INFO:
            self._banner.show_message("error", f"Unknown asset: {asset}. "
                                               f"Add it to ASSET_INFO.")
            return
        start_date = self._start.date().toPython()
        end_date = self._end.date().toPython()
        if start_date > end_date:
            self._banner.show_message("error", "Start date must be before end date.")
            return
        if asset in NON_US_CALENDAR_ASSETS:
            self._banner.show_message(
                "warning", f"{asset}: day buckets come from the USD calendar "
                           f"only — foreign-calendar events are not tagged.")

        info = ASSET_INFO[asset]
        ff = resolve_ff_events(ref.root, self.settings.data_roots)
        if ff is None:
            self._banner.show_message("warning", "ff_usd_events.parquet missing "
                                                 "— every day is bucketed 'normal'.")
        self._run_root = ref.root

        self._progress.reset()
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        worker = FunctionWorker(
            run_grid_job, self._strategy_module, ref.path.resolve(),
            start_date, end_date,
            fixed_params=self._sweep_panel.fixed_params(), axes=axes,
            asset_type=ref.asset_type, asset=asset, dataset=ref.dataset,
            strategy_name=self._strategy_ref.name,
            tick_size=info["tick_size"],
            ticks_per_point=info["ticks_per_point"],
            be_band_ticks=float(self._be_band.value()),
            min_trades_default=int(self._min_trades.value()),
            n_workers=self._effective_workers,
            ff_events_path=ff,
            strategies_dir=self._strategy_ref.dir,
            needs_progress=True,
        )
        worker.signals.progress.connect(self._progress.on_progress)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.error.connect(self._on_error)
        worker.signals.cancelled.connect(self._on_cancelled)
        self._worker = worker
        self._track_worker(worker)

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)

    def _reset_buttons(self) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._worker = None

    def _on_finished(self, result) -> None:
        trades, meta = result
        self._reset_buttons()
        self.runFinished.emit(trades, meta, self._run_root)

    def _on_error(self, message: str, _tb: str) -> None:
        self._reset_buttons()
        self._banner.show_message("error", message)

    def _on_cancelled(self) -> None:
        self._reset_buttons()
        self._banner.show_message("warning", "Run cancelled — in-flight combos "
                                             "were allowed to finish, results "
                                             "discarded.")
