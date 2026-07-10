"""
Analytics window — apply position sizing to saved backtest trades and
compare runs.

The PySide6 port of legacy_streamlit/views/analytics.py. Same pipeline:
shared defaults (default trades file / account size / slippage) -> stats
curve selector -> N instance editors -> Run (worker executes every instance:
loads trades fresh, applies the sizer, dollars_per_tick derived from the
FILENAME's first token) -> results (4-curve charts, tile grids, combined
overlay, comparison table). Changing account size / slippage / stats curve
re-enriches the existing runs without re-running the sizers.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDoubleSpinBox,
                               QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QRadioButton, QSlider, QSpinBox, QVBoxLayout)

from modules.analytics.backend.costs import DEFAULT_ACCOUNT_SIZE
from modules.analytics.backend.metrics import CURVE_LABELS, DEFAULT_CURVE
from modules.analytics.backend.sizing import run_instance
from modules.analytics.instance_editor import InstanceEditor
from modules.analytics.results_view import ResultsView
from modules.common.backend.asset_info import get_dollars_per_tick
from modules.common.backend.data_roots import TradesRef, list_trades_files
from modules.common.backend.plugins import PluginRef, list_plugins
from modules.common.ui.module_window import ModuleWindowBase
from modules.common.ui.widgets import Banner, Caption, SectionHeader
from modules.common.ui.workers import FunctionWorker

_SLIPPAGE_HELP = ("Entry-side ticks slipped per trade; market exits (losers) "
                  "slip 2×. Drives every instance.")


def _execute_all(configs: list[dict], on_progress=None) -> dict:
    """
    Worker-side loop — the old execute_all_instances + _execute_instance:
    per instance derive dollars_per_tick from the FILENAME, reload the trades
    parquet fresh, apply the sizer. Errors are collected (instance skipped);
    skipped-trade counts are summarized for the warning banner.
    """
    runs, errors = [], []
    total = max(len(configs), 1)
    for idx, cfg in enumerate(configs, start=1):
        ref: TradesRef = cfg["trades_ref"]
        try:
            dollars_per_tick = get_dollars_per_tick(ref.filename)
        except ValueError as e:
            errors.append(str(e))
            if on_progress:
                on_progress(idx, total, "")
            continue
        params = {**cfg["params"], "dollars_per_tick": dollars_per_tick}
        try:
            sized = run_instance(ref.path, cfg["sizer_module"], params)
        except Exception as e:  # noqa: BLE001 — surfaced per instance
            errors.append(f"Instance `{cfg['label']}` failed: {e}")
            if on_progress:
                on_progress(idx, total, "")
            continue
        runs.append({
            "label": cfg["label"],
            "trades_file": ref.filename,   # asset lookups key off the filename
            "trades_path": ref.path,
            "sizer": cfg["sizer"],
            "params": params,
            "trades": sized,
        })
        if on_progress:
            on_progress(idx, total, f"[{idx}/{total}] {cfg['label']}")

    offenders = [
        (r["label"], int(r["trades"].attrs.get("skipped_trades", 0)))
        for r in runs
        if int(r["trades"].attrs.get("skipped_trades", 0)) > 0
    ]
    return {"runs": runs, "errors": errors, "skipped": offenders}


class AnalyticsWindow(ModuleWindowBase):
    def __init__(self, settings, parent=None):
        super().__init__(settings, "Analytics",
                         "Apply position sizing to saved backtest trades and "
                         "compare runs.", parent)
        self._trades_refs: list[TradesRef] = []
        self._sizer_refs: list[PluginRef] = []
        self._editors: list[InstanceEditor] = []
        self._runs: list[dict] = []          # the old st.session_state.analytics_runs

        self._banner = Banner()
        self.content.addWidget(self._banner)

        # ── shared defaults ───────────────────────────────────────────────────
        self.content.addWidget(SectionHeader("Shared defaults"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.addWidget(QLabel("Default trades file"), 0, 0)
        self._default_file = QComboBox()
        grid.addWidget(self._default_file, 1, 0)
        grid.addWidget(QLabel("account_size"), 0, 1)
        self._account = QDoubleSpinBox()
        self._account.setRange(0.0, 1e12)
        self._account.setDecimals(2)
        self._account.setSingleStep(1000.0)
        self._account.setValue(DEFAULT_ACCOUNT_SIZE)
        grid.addWidget(self._account, 1, 1)
        slip_label = QLabel("Slippage (ticks/side)")
        slip_label.setToolTip(_SLIPPAGE_HELP)
        grid.addWidget(slip_label, 0, 2)
        slip_row = QHBoxLayout()
        self._slippage = QSlider(Qt.Horizontal)
        self._slippage.setRange(1, 5)
        self._slippage.setValue(1)
        self._slippage.setToolTip(_SLIPPAGE_HELP)
        self._slippage_value = QLabel("1")
        slip_row.addWidget(self._slippage)
        slip_row.addWidget(self._slippage_value)
        grid.addLayout(slip_row, 1, 2)
        for c in range(3):
            grid.setColumnStretch(c, 1)
        self.content.addLayout(grid)

        # ── stats-curve selector ──────────────────────────────────────────────
        curve_row = QHBoxLayout()
        curve_label = QLabel("Statistics based on")
        curve_label.setToolTip("Pick which curve the metrics below are "
                               "calculated on — raw (gross) or after "
                               "commissions, slippage, or both.")
        curve_row.addWidget(curve_label)
        self._curve_group = QButtonGroup(self)
        self._curve_buttons = {}
        for label in CURVE_LABELS:
            btn = QRadioButton(label)
            btn.setChecked(label == DEFAULT_CURVE)
            self._curve_group.addButton(btn)
            self._curve_buttons[label] = btn
            curve_row.addWidget(btn)
        curve_row.addStretch()
        self.content.addLayout(curve_row)

        # ── instance builder ──────────────────────────────────────────────────
        self.content.addWidget(SectionHeader("Instances"))
        n_row = QHBoxLayout()
        n_row.addWidget(QLabel("Number of instances"))
        self._n_instances = QSpinBox()
        self._n_instances.setRange(1, 99)
        self._n_instances.setValue(1)
        n_row.addWidget(self._n_instances)
        n_row.addStretch()
        refresh_btn = QPushButton("Refresh files")
        refresh_btn.clicked.connect(self._rescan)
        n_row.addWidget(refresh_btn)
        self.content.addLayout(n_row)

        self._editors_holder = QVBoxLayout()
        self.content.addLayout(self._editors_holder)

        # ── run ───────────────────────────────────────────────────────────────
        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.setProperty("primary", True)
        self._run_btn.setMinimumWidth(200)
        self._run_btn.clicked.connect(self._on_run)
        run_row.addStretch()
        run_row.addWidget(self._run_btn)
        run_row.addStretch()
        self.content.addLayout(run_row)
        self._status = Caption("")
        self.content.addWidget(self._status)
        self._run_banner = Banner()
        self.content.addWidget(self._run_banner)

        # ── results ───────────────────────────────────────────────────────────
        self._results = ResultsView()
        self.content.addWidget(self._results)
        self.content.addStretch()

        # ── wiring ────────────────────────────────────────────────────────────
        self._slippage.valueChanged.connect(
            lambda v: (self._slippage_value.setText(str(v)), self._refresh_results()))
        self._account.valueChanged.connect(lambda _=None: self._refresh_results())
        for btn in self._curve_buttons.values():
            btn.toggled.connect(lambda on: self._refresh_results() if on else None)
        self._n_instances.valueChanged.connect(self._sync_editor_count)
        self._default_file.currentIndexChanged.connect(self._push_default_file)

        self._rescan()

    # ── scanning ──────────────────────────────────────────────────────────────
    def _rescan(self) -> None:
        self._trades_refs = list_trades_files(self.settings.data_roots)
        self._sizer_refs = list_plugins(self.settings.plugin_dirs("position_sizing"))
        self._banner.clear_message()
        if not self._trades_refs:
            self._banner.show_message(
                "warning", "No trades files found in any data root's trades/ "
                           "folder. Run a backtest first.")
        elif not self._sizer_refs:
            self._banner.show_message(
                "warning", "No position sizers found in the configured folders.")

        self._default_file.blockSignals(True)
        self._default_file.clear()
        for ref in self._trades_refs:
            self._default_file.addItem(ref.label, ref)
        self._default_file.blockSignals(False)

        # rebuild editors against the fresh file/sizer lists
        for editor in self._editors:
            editor.deleteLater()
        self._editors = []
        self._sync_editor_count()

    def _sync_editor_count(self) -> None:
        want = int(self._n_instances.value())
        while len(self._editors) > want:
            self._editors.pop().deleteLater()
        while len(self._editors) < want:
            editor = InstanceEditor(len(self._editors), self._trades_refs,
                                    self._sizer_refs)
            self._editors_holder.addWidget(editor)
            self._editors.append(editor)
        self._push_default_file()

    def _push_default_file(self) -> None:
        ref = self._default_file.currentData()
        for editor in self._editors:
            editor.set_default_file(ref)

    def _stats_curve(self) -> str:
        for label, btn in self._curve_buttons.items():
            if btn.isChecked():
                return label
        return DEFAULT_CURVE

    # ── run flow ──────────────────────────────────────────────────────────────
    def _on_run(self) -> None:
        self._run_banner.clear_message()
        configs = [c for c in (e.config(float(self._account.value()))
                               for e in self._editors) if c is not None]
        if not configs:
            self._run_banner.show_message("error", "No runnable instances.")
            return
        # wipe first — prevents stale results when instance count shrank
        self._runs = []
        self._results.clear()
        self._run_btn.setEnabled(False)
        self._status.setText("Running instances…")

        worker = FunctionWorker(_execute_all, configs, needs_progress=True)
        worker.signals.progress.connect(
            lambda cur, total, _msg: self._status.setText(
                f"Running instances… {cur}/{total}"))
        worker.signals.finished.connect(self._on_finished)
        worker.signals.error.connect(self._on_error)
        self.track_worker(worker)

    def _on_error(self, message: str, _tb: str) -> None:
        self._run_btn.setEnabled(True)
        self._status.setText("")
        self._run_banner.show_message("error", message)

    def _on_finished(self, result: dict) -> None:
        self._run_btn.setEnabled(True)
        self._status.setText("")
        self._runs = result["runs"]
        messages = list(result["errors"])
        if result["skipped"]:
            summary = ", ".join(f"{label}: {count}"
                                for label, count in result["skipped"])
            messages.append(f"Some trades were skipped (size=0) — {summary}")
        if messages:
            self._run_banner.show_message("warning", "\n".join(messages))
        self._refresh_results()

    def _refresh_results(self) -> None:
        if not self._runs:
            return
        self._results.refresh(self._runs, float(self._account.value()),
                              int(self._slippage.value()), self._stats_curve())
