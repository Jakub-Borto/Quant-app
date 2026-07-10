"""
Monte Carlo window — stress-test saved trades with resampling simulations.

The PySide6 port of legacy_streamlit/views/monte_carlo.py. Same flow:
trades file -> position sizing (account size, sizer, numeric-only sizer
params) -> simulation method (drop-in scripts from modules/monte_carlo/
methods/). Methods flagged PROP_FIRM get the dedicated PropFirmPanel;
everything else gets the generic flow: ruin definition, simulation params,
costs, Run (worker), then the fan chart (with the cap-at-3x-account toggle)
and the metrics table.

The sizer-params dict is assembled verbatim: {**PARAMS, **numeric-widget
overrides, "account_size": ...}; dollars_per_tick is injected at run time
from the trades FILENAME's first token.
"""

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QSlider, QVBoxLayout, QWidget)

from modules.common.backend.asset_info import get_dollars_per_tick
from modules.common.backend.data_roots import TradesRef, list_trades_files
from modules.common.backend.plugins import PluginRef, list_plugins, load_module
from modules.common.ui.charts.fan_chart import FanChart
from modules.common.ui.dataframe_model import make_table_view, update_table_view
from modules.common.ui.module_window import ModuleWindowBase
from modules.common.ui.params_form import ParamsForm
from modules.common.ui.widgets import (Banner, Caption, CollapsibleSection,
                                       SectionHeader, wrap_card)
from modules.common.ui.workers import FunctionWorker
from modules.monte_carlo.backend.cost_ctx import build_cost_ctx
from modules.monte_carlo.backend.stats import (_compute_metrics,
                                               _select_featured_paths,
                                               metrics_table_rows)
from modules.monte_carlo.prop_firm_panel import PropFirmPanel

# MC methods are internal drop-in scripts (not a settings category) — drop a
# .py file into modules/monte_carlo/methods/ and it appears in the picker.
METHODS_DIR = Path(__file__).resolve().parent / "methods"

RUIN_OPTIONS = {
    "No threshold": None,
    "Ruin at 0% (account wiped)": 0.0,
    "Ruin at 50% loss":           0.5,
}

_COSTS_CAPTION = (
    "Equity is net of commissions & slippage. Note: slippage is applied "
    "post-hoc to recorded trades — a worse entry that would have prevented "
    "a take-profit fill is not modelled."
)


class MonteCarloWindow(ModuleWindowBase):
    def __init__(self, settings, parent=None):
        super().__init__(settings, "Monte Carlo Simulation",
                         "Resample saved trades into thousands of equity "
                         "paths and study the distribution.", parent)
        self._trades_refs: list[TradesRef] = []
        self._sizer_refs: list[PluginRef] = []
        self._method_refs: list[PluginRef] = []
        self._sizer_module = None
        self._mc_module = None
        self._sizer_form: ParamsForm | None = None
        self._mc_form: ParamsForm | None = None

        # results state (the old mc_* session keys)
        self._results = None
        self._result_account = None
        self._result_ruin = None
        self._result_costs = None
        self._chart_capped = True

        self._banner = Banner()
        self.content.addWidget(self._banner)

        # ── 1. trades ─────────────────────────────────────────────────────────
        self.content.addWidget(SectionHeader("Trades"))
        self._file = QComboBox()
        self.content.addWidget(self._file)

        # ── 2. position sizing ────────────────────────────────────────────────
        self.content.addWidget(SectionHeader("Position Sizing"))
        grid = QGridLayout()
        grid.addWidget(QLabel("Account size ($)"), 0, 0)
        self._account = QDoubleSpinBox()
        self._account.setRange(0.0, 1e12)
        self._account.setDecimals(2)
        self._account.setSingleStep(1000.0)
        self._account.setValue(100_000.0)
        grid.addWidget(self._account, 1, 0)
        grid.addWidget(QLabel("Sizer"), 0, 1)
        self._sizer = QComboBox()
        grid.addWidget(self._sizer, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        self.content.addWidget(wrap_card(grid))
        self._sizer_section = CollapsibleSection("Sizer parameters")
        self._sizer_section.setVisible(False)
        self.content.addWidget(self._sizer_section)

        # ── 3. simulation method ──────────────────────────────────────────────
        self.content.addWidget(SectionHeader("Simulation"))
        self._method = QComboBox()
        self.content.addWidget(self._method)

        # generic branch container (hidden for PROP_FIRM methods)
        self._generic = QWidget()
        glay = QVBoxLayout(self._generic)
        glay.setContentsMargins(0, 0, 0, 0)
        glay.setSpacing(8)

        ruin_row = QHBoxLayout()
        ruin_row.addWidget(QLabel("Ruin definition"))
        self._ruin = QComboBox()
        self._ruin.addItems(list(RUIN_OPTIONS.keys()))
        ruin_row.addWidget(self._ruin)
        ruin_row.addStretch()
        glay.addLayout(ruin_row)

        self._mc_section = CollapsibleSection("Simulation parameters")
        self._mc_section.setVisible(False)
        glay.addWidget(self._mc_section)

        self._apply_costs = QCheckBox("Apply commissions && slippage")
        self._apply_costs.setChecked(True)
        glay.addWidget(self._apply_costs)
        slip_row = QHBoxLayout()
        self._slip_label = QLabel("Slippage (ticks/side)")
        self._slippage = QSlider(Qt.Horizontal)
        self._slippage.setRange(1, 5)
        self._slippage.setValue(1)
        self._slippage.setMaximumWidth(220)
        self._slippage.setToolTip("Entry-side ticks slipped per trade; market "
                                  "exits (losers) slip 2×.")
        self._slip_value = QLabel("1")
        self._slippage.valueChanged.connect(lambda v: self._slip_value.setText(str(v)))
        for w in (self._slip_label, self._slippage, self._slip_value):
            self._apply_costs.toggled.connect(w.setVisible)
        slip_row.addWidget(self._slip_label)
        slip_row.addWidget(self._slippage)
        slip_row.addWidget(self._slip_value)
        slip_row.addStretch()
        glay.addLayout(slip_row)

        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Simulation")
        self._run_btn.setProperty("primary", True)
        self._run_btn.setMinimumWidth(220)
        self._run_btn.clicked.connect(self._on_run)
        run_row.addStretch()
        run_row.addWidget(self._run_btn)
        run_row.addStretch()
        glay.addLayout(run_row)
        self._status = Caption("")
        glay.addWidget(self._status)

        # generic results
        self._results_box = QWidget()
        self._results_box.setVisible(False)
        rlay = QVBoxLayout(self._results_box)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.addWidget(SectionHeader("Equity Fan Chart"))
        self._costs_caption = Caption(_COSTS_CAPTION)
        rlay.addWidget(self._costs_caption)
        cap_row = QHBoxLayout()
        self._cap_btn = QPushButton()
        self._cap_btn.clicked.connect(self._toggle_cap)
        cap_row.addWidget(self._cap_btn)
        cap_row.addStretch()
        rlay.addLayout(cap_row)
        self._fan = FanChart()
        rlay.addWidget(self._fan)
        rlay.addWidget(SectionHeader("Metrics"))
        self._metrics_table = make_table_view(pd.DataFrame(), height=440)
        rlay.addWidget(self._metrics_table)
        glay.addWidget(self._results_box)

        self.content.addWidget(self._generic)

        # prop-firm branch container (swapped in per method)
        self._prop_holder = QVBoxLayout()
        self.content.addLayout(self._prop_holder)
        self._prop_panel: PropFirmPanel | None = None

        self.content.addStretch()

        # ── wiring ────────────────────────────────────────────────────────────
        self._sizer.currentIndexChanged.connect(self._on_sizer_changed)
        self._method.currentIndexChanged.connect(self._on_method_changed)
        self._rescan()

    # ── scanning ──────────────────────────────────────────────────────────────
    def _rescan(self) -> None:
        self._trades_refs = list_trades_files(self.settings.data_roots)
        self._sizer_refs = list_plugins(self.settings.plugin_dirs("position_sizing"))
        self._method_refs = list_plugins([METHODS_DIR])
        self._banner.clear_message()
        if not self._trades_refs:
            self._banner.show_message("error", "No trade files found in any "
                                               "data root's trades/ folder.")
        elif not self._sizer_refs:
            self._banner.show_message("error", "No sizer scripts found in the "
                                               "configured position_sizing folders.")
        elif not self._method_refs:
            self._banner.show_message("error", "No Monte Carlo scripts found "
                                               "in modules/monte_carlo/methods/.")

        self._file.clear()
        for ref in self._trades_refs:
            self._file.addItem(ref.label, ref)
        self._sizer.blockSignals(True)
        self._sizer.clear()
        for ref in self._sizer_refs:
            self._sizer.addItem(ref.label, ref)
        self._sizer.blockSignals(False)
        self._method.blockSignals(True)
        self._method.clear()
        for ref in self._method_refs:
            self._method.addItem(ref.label, ref)
        self._method.blockSignals(False)
        self._on_sizer_changed()
        self._on_method_changed()

    # ── sizer / method selection ──────────────────────────────────────────────
    def _on_sizer_changed(self) -> None:
        while self._sizer_section.content_layout.count():
            item = self._sizer_section.content_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._sizer_form = None
        self._sizer_module = None
        ref: PluginRef | None = self._sizer.currentData()
        if ref is None:
            self._sizer_section.setVisible(False)
            return
        self._sizer_module = load_module(ref)
        raw = getattr(self._sizer_module, "PARAMS", {})
        ui_params = {k: v for k, v in raw.items()
                     if k not in ("account_size", "dollars_per_tick")}
        if ui_params:
            self._sizer_form = ParamsForm(ui_params, numeric_only=True)
            self._sizer_section.content_layout.addWidget(self._sizer_form)
            self._sizer_section.setVisible(True)
        else:
            self._sizer_section.setVisible(False)

    def _sizer_params(self) -> dict:
        """{**PARAMS, **widget overrides, account_size} — verbatim assembly."""
        raw = getattr(self._sizer_module, "PARAMS", {}) if self._sizer_module else {}
        specific = self._sizer_form.values() if self._sizer_form else {}
        return {**raw, **specific, "account_size": float(self._account.value())}

    def _on_method_changed(self) -> None:
        # tear down the prop panel if any
        if self._prop_panel is not None:
            self._prop_panel.deleteLater()
            self._prop_panel = None
        ref: PluginRef | None = self._method.currentData()
        if ref is None:
            return
        self._mc_module = load_module(ref)

        if getattr(self._mc_module, "PROP_FIRM", False):
            self._generic.setVisible(False)
            self._prop_panel = PropFirmPanel(self._mc_module,
                                             self._prop_context,
                                             self.track_worker)
            self._prop_holder.addWidget(self._prop_panel)
            return

        self._generic.setVisible(True)
        # rebuild the simulation-params form for this method
        while self._mc_section.content_layout.count():
            item = self._mc_section.content_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._mc_form = None
        raw = getattr(self._mc_module, "PARAMS", {})
        if raw:
            self._mc_form = ParamsForm(raw, numeric_only=True)
            self._mc_section.content_layout.addWidget(self._mc_form)
            self._mc_section.setVisible(True)
        else:
            self._mc_section.setVisible(False)

    def _prop_context(self) -> dict | None:
        ref: TradesRef | None = self._file.currentData()
        if ref is None or self._sizer_module is None:
            return None
        return {"trades_ref": ref, "sizer_module": self._sizer_module,
                "sizer_params": self._sizer_params(),
                "account_size": float(self._account.value())}

    # ── generic run flow ──────────────────────────────────────────────────────
    def _on_run(self) -> None:
        self._banner.clear_message()
        ref: TradesRef | None = self._file.currentData()
        if ref is None or self._mc_module is None or self._sizer_module is None:
            self._banner.show_message("error", "Pick a trades file, sizer and "
                                               "method first.")
            return
        try:
            dollars_per_tick = get_dollars_per_tick(ref.filename)
        except ValueError as e:
            self._banner.show_message("error", str(e))
            return
        try:
            trades = pd.read_parquet(ref.path)
        except Exception as e:  # noqa: BLE001
            self._banner.show_message("error", f"Could not load trades: {e}")
            return

        final_sizer_params = {**self._sizer_params(),
                              "dollars_per_tick": dollars_per_tick}
        apply_costs = self._apply_costs.isChecked()
        slippage_n = int(self._slippage.value()) if apply_costs else 1
        cost_ctx, warn_missing = build_cost_ctx(ref.filename, apply_costs,
                                                slippage_n)
        if warn_missing:
            self._banner.show_message(
                "warning",
                f"No commission rate for asset '{ref.filename.split('_')[0]}' "
                f"— commissions billed at 0; slippage still applies.")
        mc_raw = getattr(self._mc_module, "PARAMS", {})
        mc_specific = self._mc_form.values() if self._mc_form else {}
        mc_params = {**mc_raw, **mc_specific}
        run_params = {**mc_params, "cost_ctx": cost_ctx}

        self._pending = {"account": float(self._account.value()),
                         "ruin": RUIN_OPTIONS[self._ruin.currentText()],
                         "costs": apply_costs}
        self._run_btn.setEnabled(False)
        self._status.setText(f"Running {self._method.currentText()} — "
                             f"{mc_params.get('n_paths', '?')} paths…")
        worker = FunctionWorker(self._mc_module.run, trades=trades,
                                sizer_module=self._sizer_module,
                                sizer_params=final_sizer_params,
                                params=run_params)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.error.connect(self._on_error)
        self.track_worker(worker)

    def _on_error(self, message: str, _tb: str) -> None:
        self._run_btn.setEnabled(True)
        self._status.setText("")
        self._banner.show_message("error", f"Simulation error: {message}")

    def _on_finished(self, results: dict) -> None:
        self._run_btn.setEnabled(True)
        self._status.setText("")
        for w in results.get("warnings", []):
            self._banner.show_message("warning", w)
        self._results = results
        self._result_account = self._pending["account"]
        self._result_ruin = self._pending["ruin"]
        self._result_costs = self._pending["costs"]
        self._render_results()

    # ── generic results ───────────────────────────────────────────────────────
    def _toggle_cap(self) -> None:
        self._chart_capped = not self._chart_capped
        self._render_results()

    def _render_results(self) -> None:
        if self._results is None:
            return
        equity_matrix = self._results["equity_matrix"]   # (n_paths, n_trades+1)
        account_size = self._result_account
        ruin_thresh = self._result_ruin

        self._costs_caption.setVisible(bool(self._result_costs))
        self._cap_btn.setText("Show full equity curve" if self._chart_capped
                              else "Cap at 3× account")
        y_max = account_size * 3 if self._chart_capped else None

        featured = _select_featured_paths(equity_matrix)
        metrics = _compute_metrics(equity_matrix, account_size, ruin_thresh)
        self._fan.set_data(equity_matrix, account_size, featured, ruin_thresh,
                           y_max=y_max, band_finals=metrics["band_finals"])
        update_table_view(self._metrics_table,
                          metrics_table_rows(metrics, account_size))
        self._results_box.setVisible(True)
