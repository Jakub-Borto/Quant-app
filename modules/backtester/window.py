"""
Backtester window — run a strategy on a dataset and inspect the results.

This window owns the run: the pickers, the params form, the worker thread and
the day-type tagging.

    controls -> params form -> Run (worker thread)
    -> tag day types (FF events from the dataset's data root)
    -> TradeReport.show_trades()

Everything after that — the filter chain, every section, Save Trades and the
Analytics / Monte Carlo handoff — is modules.common.trade_report, the same
widget the Optimizer's cell and Combine drill-downs use. There is exactly one
implementation of it; if you are about to change filter behaviour, change it
there.

Save uses the dataset/strategy/dates captured AT RUN TIME (the old page read
the current widget values, which could drift after the run — same values in
every normal flow). day_types defaults to KEEP, so a day-type choice survives
a re-run — the one report behaviour this window relies on.
"""

import pandas as pd
from PySide6.QtCore import QDate
from PySide6.QtWidgets import (QComboBox, QDateEdit, QGridLayout, QHBoxLayout,
                               QLabel, QPushButton, QVBoxLayout, QWidget)

from modules.backtester.backend.day_types import (load_day_classifications,
                                                  tag_trades)
from modules.backtester.backend.run import run_backtest
from modules.common.backend.asset_info import ASSET_INFO, HIDDEN_PARAMS
from modules.common.backend.data_roots import (DatasetRef, available_dates,
                                               resolve_ff_events,
                                               scan_structure)
from modules.common.backend.plugins import PluginRef, list_strategies, load_strategy
from modules.common.trade_report.ui import (ReportContext, SaveTarget,
                                            TradeReport, attach_layout_gear)
from modules.common.ui.module_window import ModuleWindowBase
from modules.common.ui.params_form import ParamsForm
from modules.common.ui.widgets import (Banner, Caption, SectionHeader,
                                       wrap_card)
from modules.common.ui.workers import FunctionWorker


# Dead space under the report. Scrolled to the very bottom, expanding or
# collapsing a section changes the scroll range and the view lurches; a tall
# run-out means the last sections behave like the ones above them.
BOTTOM_PADDING = 900



class BacktesterWindow(ModuleWindowBase):
    def __init__(self, settings, parent=None):
        super().__init__(settings, "Backtester",
                         "Run a strategy on a dataset and inspect the results.",
                         parent)
        # run state (the old st.session_state.trades / folder_path)
        self._trades: pd.DataFrame | None = None      # raw run output
        self._tagged: pd.DataFrame | None = None      # + day_type column
        self._run_ref: DatasetRef | None = None
        self._run_asset: str | None = None
        self._run_inputs: dict = {}                   # dataset/strategy/dates at run time
        self._strategies: list[PluginRef] = []
        self._strategy_module = None
        self._structure: dict = {}
        self._params_form: ParamsForm | None = None
        attach_layout_gear(self, settings,
                           tooltip="Report layout — order and show/hide the "
                                   "report's sections")

        self._build_controls()
        self._build_results_area()
        self._rescan()

    # ══ controls ═══════════════════════════════════════════════════════════════
    def _build_controls(self) -> None:
        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(8)

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
        self.content.addWidget(wrap_card(grid))

        # params form placeholder (rebuilt per strategy)
        self._params_header = SectionHeader("Parameters")
        self._params_header.setVisible(False)
        self.content.addWidget(self._params_header)
        self._params_container = QVBoxLayout()
        self.content.addLayout(self._params_container)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.setProperty("primary", True)
        self._run_btn.setMinimumWidth(200)
        self._run_btn.clicked.connect(self._on_run)
        self._status = Caption("")
        refresh_btn = QPushButton("Refresh folders")
        refresh_btn.clicked.connect(self._rescan)
        btn_row.addStretch()
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(refresh_btn)
        btn_row.addStretch()
        self.content.addLayout(btn_row)
        self.content.addWidget(self._status)

        self._banner = Banner()
        self.content.addWidget(self._banner)

        self._type.currentIndexChanged.connect(self._on_type_changed)
        self._asset.currentIndexChanged.connect(self._on_asset_changed)
        self._dataset.currentIndexChanged.connect(self._on_dataset_changed)
        self._strategy.currentIndexChanged.connect(self._on_strategy_changed)

    def _build_results_area(self) -> None:
        """The whole report is one widget — see modules.common.trade_report."""
        self._report = TradeReport(self.settings,
                                   track_worker=self.track_worker)
        self._report.setVisible(False)
        self.content.addWidget(self._report)
        self.content.addSpacing(BOTTOM_PADDING)
        self.content.addStretch()

    # ══ scanning / cascading pickers ═══════════════════════════════════════════
    def _rescan(self) -> None:
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

        if not self._structure:
            self._banner.show_message("error", "No datasets found under any "
                                               "data root's parquet/ folder.")
        elif not self._strategies:
            self._banner.show_message("error", "No strategies found in the "
                                               "configured strategy folders.")

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

    def _on_strategy_changed(self) -> None:
        # clear the old form
        while self._params_container.count():
            item = self._params_container.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._params_form = None
        self._strategy_module = None
        if self._strategy.currentIndex() < 0:
            self._params_header.setVisible(False)
            return
        ref = self._strategies[self._strategy.currentIndex()]
        try:
            self._strategy_module = load_strategy(ref)
        except Exception as e:  # noqa: BLE001
            self._banner.show_message("error", f"Could not load strategy "
                                               f"'{ref.name}': {e}")
            self._params_header.setVisible(False)
            return
        self._banner.clear_message()
        params = getattr(self._strategy_module, "PARAMS", {})
        visible = {k: v for k, v in params.items() if k not in HIDDEN_PARAMS}
        self._params_header.setVisible(bool(visible))
        if visible:
            self._params_form = ParamsForm(
                params, sections=getattr(self._strategy_module, "PARAM_SECTIONS", None),
                hidden=HIDDEN_PARAMS,
                options=getattr(self._strategy_module, "PARAMS_OPTIONS", None))
            self._params_container.addWidget(self._params_form)

    # ══ run flow ═══════════════════════════════════════════════════════════════
    def _on_run(self) -> None:
        self._banner.clear_message()
        ref: DatasetRef | None = self._dataset.currentData()
        if ref is None or self._strategy_module is None:
            self._banner.show_message("error", "Pick a dataset and a strategy first.")
            return

        start_date = self._start.date().toPython()
        end_date = self._end.date().toPython()
        if start_date > end_date:
            self._banner.show_message("error", "Start date must be before end date.")
            return
        asset = ref.asset
        if asset not in ASSET_INFO:
            self._banner.show_message("error", f"Unknown asset: {asset}. "
                                               f"Add it to ASSET_INFO.")
            return

        info = ASSET_INFO[asset]
        params = self._params_form.values() if self._params_form else {}
        strategy_ref = self._strategies[self._strategy.currentIndex()]
        self._run_inputs = {"dataset": ref.dataset, "strategy": strategy_ref.name,
                            "start_date": start_date, "end_date": end_date}
        self._run_ref = ref
        self._run_asset = asset

        self._run_btn.setEnabled(False)
        self._status.setText("Running strategy…")
        worker = FunctionWorker(run_backtest, self._strategy_module, ref.path,
                                start_date, end_date, params,
                                info["tick_size"], info["ticks_per_point"])
        worker.signals.finished.connect(self._on_run_finished)
        worker.signals.error.connect(self._on_run_error)
        self.track_worker(worker)

    def _on_run_error(self, message: str, _tb: str) -> None:
        self._run_btn.setEnabled(True)
        self._status.setText("")
        self._banner.show_message("error", message)

    def _on_run_finished(self, trades: pd.DataFrame) -> None:
        self._run_btn.setEnabled(True)
        self._status.setText("")
        if trades.empty:
            self._report.setVisible(False)
            self._banner.show_message("warning", "Strategy produced no trades.")
            return

        self._trades = trades
        # tag every trade with day_type (always, before any filter)
        ff = resolve_ff_events(self._run_ref.root, self.settings.data_roots)
        self._tagged = tag_trades(trades, load_day_classifications(ff))

        info = ASSET_INFO[self._run_asset]
        dates = pd.to_datetime(self._tagged["date"])
        inputs = self._run_inputs
        self._report.set_context(ReportContext(
            ticker=self._run_asset, tick_size=info["tick_size"],
            ticks_per_point=info["ticks_per_point"],
            root=self._run_ref.root, candles_folder=self._run_ref.path,
            # narrow the regime run list to this asset and this run's dates
            regime_start=dates.min().strftime("%Y-%m-%d"),
            regime_end=dates.max().strftime("%Y-%m-%d")))
        # day_types defaults to KEEP, so a day-type choice survives a re-run
        self._report.show_trades(
            self._tagged,
            save_target=SaveTarget([inputs["dataset"], inputs["strategy"],
                                    inputs["start_date"], inputs["end_date"]]))
