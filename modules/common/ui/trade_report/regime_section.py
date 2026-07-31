"""
Strategy Performance by Regime — one section holding the regime source
pickers, the per-state performance table, and the state filter.

Pick a regime run produced by the Regime Detector, choose how each trade is
matched to a snapshot, then slice the backtest by the resulting state.

The three matching modes and why they differ:

  As of entry (default) — merge_asof backward on the trade's entry time. Each
      trade sees the latest snapshot at or before it, so the label was
      genuinely knowable when the trade was taken. This is the only mode safe
      for judging a strategy.
  Exact time HH:MM     — every trade of a day gets that day's snapshot at one
      clock time ("what did the 10:00 label predict?").
  Final (hindsight)    — the day's last row, which knows how the day ended.
      Useful for measuring how much of a signal is only visible afterwards;
      never for filtering a backtest you intend to believe. Flagged with a
      warning banner whenever selected.

MEMORY: a run holds one parquet per trading day and they are all read into
memory (the join needs every snapshot, not only the days that traded). The
section measures one file to estimate the total, shows it up front, and
refuses to exceed a budget you can raise. Frames are dropped when the run
changes or when the backtest moves to a different asset — but NOT when the
same asset is re-run, because the loaded days do not depend on which days the
strategy happened to trade.
"""

import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout, QWidget)

from modules.common.backend.asset_info import ASSET_INFO
from modules.common.backend.data_roots import RegimeRunRef, list_regime_runs
from modules.common.backend.regime_join import (MODE_ASOF, MODE_EXACT,
                                                MODE_FINAL, MODE_LABELS,
                                                UNKNOWN, attach_regime)
from modules.common.backend.trade_stats import regime_performance_rows
from modules.regime_detector.backend import io as rio
from ..dataframe_model import make_table_view, update_table_view
from ..widgets import Banner, Caption, hline
from ..workers import FunctionWorker
from .filters import make_regime_filter
from .sections import ReportSection

_DEFAULT_SESSION_START = "18:00"
_GB = 1024 ** 3

# The column the report is FILTERED by, kept separate from the displayed
# `regime` column so the two timings can differ (see RegimeSection.annotate).
FILTER_COLUMN = "regime_filter"


def _fmt_size(gb: float) -> str:
    """Human size — MB below a gigabyte, so a 50 MB run doesn't read 0.00 GB."""
    return f"{gb * 1024:.0f} MB" if gb < 1.0 else f"{gb:.2f} GB"


class RegimeSection(ReportSection):
    """Source pickers + per-regime performance table + state filter."""

    sourceChanged = Signal()      # run, column, mode, time, or frames changed
    selectionChanged = Signal()   # state checkboxes only

    def __init__(self, settings, track_worker=None, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._track_worker = track_worker or (lambda w: None)
        self._runs: list[RegimeRunRef] = []
        self._meta: dict = {}
        self._frames: dict = {}
        self._loaded_range: tuple[str | None, str | None] = (None, None)
        self._asset: str | None = None
        self._range: tuple[str | None, str | None] = (None, None)
        self._filter = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # ── source pickers ────────────────────────────────────────────────────
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)

        self._run = QComboBox()
        self._all_assets = QCheckBox("Show runs from all assets")
        self._column = QComboBox()
        self._mode = QComboBox()
        for mode in (MODE_ASOF, MODE_EXACT, MODE_FINAL):
            self._mode.addItem(MODE_LABELS[mode], mode)
        self._at = QComboBox()
        self._at.setEnabled(False)

        grid.addWidget(QLabel("Regime run"), 0, 0)
        grid.addWidget(self._run, 0, 1)
        grid.addWidget(self._all_assets, 0, 2)
        grid.addWidget(QLabel("Regime column"), 1, 0)
        grid.addWidget(self._column, 1, 1)
        grid.addWidget(QLabel("Table labels at"), 2, 0)
        grid.addWidget(self._mode, 2, 1)
        grid.addWidget(self._at, 2, 2)
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)

        # ── load + memory budget ──────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._load_btn = QPushButton("Load regimes")
        self._load_btn.setProperty("primary", True)
        self._load_btn.clicked.connect(self._on_load)
        self._unload_btn = QPushButton("Unload")
        self._unload_btn.setToolTip("Free the loaded regime files")
        self._unload_btn.clicked.connect(self._on_unload)
        refresh = QPushButton("Refresh runs")
        refresh.clicked.connect(self.rescan)
        btn_row.addWidget(self._load_btn)
        btn_row.addWidget(self._unload_btn)
        btn_row.addWidget(refresh)
        btn_row.addSpacing(16)
        btn_row.addWidget(QLabel("Memory budget"))
        self._budget = QDoubleSpinBox()
        self._budget.setRange(0.05, 64.0)
        self._budget.setSingleStep(0.5)
        self._budget.setDecimals(2)
        self._budget.setValue(2.0)
        self._budget.setSuffix(" GB")
        self._budget.valueChanged.connect(lambda _=None: self._refresh_estimate())
        btn_row.addWidget(self._budget)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self._status = Caption("")
        lay.addWidget(self._status)
        self._banner = Banner()
        lay.addWidget(self._banner)

        # ── performance table ─────────────────────────────────────────────────
        lay.addWidget(hline())
        self._table_caption = Caption("Load a regime run to slice performance "
                                      "by regime.")
        lay.addWidget(self._table_caption)
        self._table = make_table_view(pd.DataFrame(), height=200)
        lay.addWidget(self._table)

        # ── state filter, with its OWN timing ─────────────────────────────────
        lay.addWidget(hline())
        filter_row = QHBoxLayout()
        self._filter_caption = Caption("Filter the whole report by regime, "
                                       "labelled at:")
        filter_row.addWidget(self._filter_caption)
        self._filter_mode = QComboBox()
        for mode in (MODE_ASOF, MODE_EXACT, MODE_FINAL):
            self._filter_mode.addItem(MODE_LABELS[mode], mode)
        self._filter_at = QComboBox()
        self._filter_at.setEnabled(False)
        filter_row.addWidget(self._filter_mode)
        filter_row.addWidget(self._filter_at)
        filter_row.addStretch()
        self._filter_row_holder = QWidget()
        self._filter_row_holder.setLayout(filter_row)
        self._filter_row_holder.setVisible(False)
        lay.addWidget(self._filter_row_holder)
        self._filter_holder = QVBoxLayout()
        lay.addLayout(self._filter_holder)

        self._run.currentIndexChanged.connect(self._on_run_changed)
        self._all_assets.toggled.connect(self.rescan)
        self._column.currentIndexChanged.connect(self._emit_source)
        self._mode.currentIndexChanged.connect(self._on_mode_changed)
        self._at.currentIndexChanged.connect(self._emit_source)
        self._filter_mode.currentIndexChanged.connect(self._on_mode_changed)
        self._filter_at.currentIndexChanged.connect(self._emit_source)
        self.rescan()

    # ── context ───────────────────────────────────────────────────────────────
    def set_context(self, asset, start: str | None, end: str | None) -> None:
        """The backtest's asset and date range. A DIFFERENT asset drops the
        loaded frames (they label the wrong instrument); the same asset keeps
        them, since a re-run reads the same days regardless of which ones the
        strategy traded."""
        if asset != self._asset and self._frames:
            self._frames = {}
            self._loaded_range = (None, None)
        self._asset = asset
        self._range = (start, end)
        self.rescan()

    def _eligible(self, ref: RegimeRunRef) -> bool:
        if self._all_assets.isChecked() or not self._asset:
            return True
        parent = (ASSET_INFO.get(self._asset) or {}).get("parent")
        return ref.asset == self._asset or (parent and ref.asset == parent)

    def rescan(self) -> None:
        previous = self.current_run()
        self._runs = [r for r in list_regime_runs(self._settings.data_roots)
                      if self._eligible(r)]
        self._run.blockSignals(True)
        self._run.clear()
        for ref in self._runs:
            self._run.addItem(ref.label, ref)
        if previous is not None:
            index = next((i for i, r in enumerate(self._runs)
                          if r.path == previous.path), -1)
            if index >= 0:
                self._run.setCurrentIndex(index)
        self._run.blockSignals(False)
        self._on_run_changed()

    def current_run(self) -> RegimeRunRef | None:
        return self._run.currentData()

    # ── run / mode selection ──────────────────────────────────────────────────
    def _on_run_changed(self) -> None:
        ref = self.current_run()
        self._frames = {}                 # a different run's labels don't apply
        self._loaded_range = (None, None)
        self._meta = {}
        self._column.blockSignals(True)
        self._column.clear()
        self._column.blockSignals(False)
        self._set_states([])
        if ref is None:
            self._status.setText("No regime runs found for this asset — "
                                 "build one in the Regime Detector.")
            self._emit_source()
            return
        try:
            self._meta = rio.read_meta(ref.path)
        except (OSError, ValueError) as e:
            self._banner.show_message("error", f"Could not read meta.json: {e}")
            self._emit_source()
            return

        columns = list(rio.tiers(self._meta).get("regime", []))
        self._column.blockSignals(True)
        self._column.addItems(columns)
        self._column.blockSignals(False)

        session = self._meta.get("globex_session") or {}
        start = session.get("start") or _DEFAULT_SESSION_START
        end = session.get("end") or "17:00"
        minutes = int(self._meta.get("snapshot_minutes", 30))
        labels = rio.snapshot_grid_labels(start, end, minutes)
        for combo in (self._at, self._filter_at):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(labels)
            default = combo.findText("10:00")
            combo.setCurrentIndex(default if default >= 0 else 0)
            combo.blockSignals(False)

        self._refresh_estimate()
        self._refresh_banner()
        self._emit_source()

    def _on_mode_changed(self) -> None:
        self._at.setEnabled(self.mode() == MODE_EXACT)
        self._filter_at.setEnabled(self.filter_mode() == MODE_EXACT)
        self._refresh_banner()
        self._emit_source()

    def _refresh_banner(self) -> None:
        messages = []
        if self.filter_mode() == MODE_FINAL:
            messages.append(
                "FILTERING by ‘Final’ selects trades using a label that already "
                "knows how the day ended — the resulting equity curve is not "
                "tradeable. Fine for measuring how much of the edge is "
                "hindsight; never a result to believe.")
        elif self.mode() == MODE_FINAL:
            messages.append(
                "‘Final’ uses each day's LAST snapshot, which already knows how "
                "the day ended — useful for measuring hindsight, not for "
                "filtering a backtest you intend to believe.")
        ref = self.current_run()
        if ref is not None and self._asset and ref.asset != self._asset:
            messages.append(f"This run labels {ref.asset}, but the backtest is "
                            f"on {self._asset}.")
        if messages:
            self._banner.show_message("warning", "  ".join(messages))
        else:
            self._banner.clear_message()

    # ── memory estimate ───────────────────────────────────────────────────────
    def _estimate(self) -> tuple[int, float]:
        """(day count, estimated GB in memory) for the current run + range."""
        ref = self.current_run()
        if ref is None:
            return 0, 0.0
        start, end = self._range
        return rio.estimate_load_size(ref.path, start=start, end=end)

    def _refresh_estimate(self) -> None:
        if self._frames:
            return
        days, gb = self._estimate()
        if days == 0:
            self._status.setText("No regime files in the backtest's date range.")
            self._load_btn.setEnabled(False)
            return
        over = gb > float(self._budget.value())
        self._load_btn.setEnabled(not over)
        covered = self._meta.get("date_range", {})
        text = (f"{days} day files in range · ≈{_fmt_size(gb)} in memory · "
                f"run covers {covered.get('first')} → {covered.get('last')}")
        if over:
            text += (f"  —  OVER the {self._budget.value():.2f} GB budget. "
                     f"Narrow the backtest's date range or raise the budget.")
        self._status.setText(text)

    # ── loading ───────────────────────────────────────────────────────────────
    def _on_load(self) -> None:
        ref = self.current_run()
        if ref is None:
            return
        days, gb = self._estimate()
        if gb > float(self._budget.value()):
            self._banner.show_message(
                "error", f"Loading {days} days would take about {_fmt_size(gb)}, "
                         f"over the {self._budget.value():.2f} GB budget.")
            return
        start, end = self._range
        self._load_btn.setEnabled(False)
        self._status.setText(f"Loading {days} regime files (≈{_fmt_size(gb)})…")
        worker = FunctionWorker(rio.load_day_frames, ref.path,
                                start=start, end=end, needs_progress=True)
        worker.signals.progress.connect(
            lambda cur, total, msg: self._status.setText(
                f"Loading regime files… {cur}/{total}"))
        worker.signals.finished.connect(self._on_loaded)
        worker.signals.error.connect(self._on_load_error)
        self._track_worker(worker)

    def _on_loaded(self, frames: dict) -> None:
        self._load_btn.setEnabled(True)
        self._frames = frames or {}
        self._loaded_range = self._range
        gb = sum(f.memory_usage(deep=True).sum()
                 for f in self._frames.values()) / _GB if self._frames else 0.0
        self._status.setText(f"{len(self._frames)} regime days loaded "
                             f"({_fmt_size(gb)} in memory).")
        self._set_states(self.states())
        self._emit_source()

    def _on_load_error(self, message: str, _tb: str) -> None:
        self._load_btn.setEnabled(True)
        self._status.setText("")
        self._banner.show_message("error", f"Could not load regimes: {message}")

    def _on_unload(self) -> None:
        self._frames = {}
        self._loaded_range = (None, None)
        self._set_states([])
        self._refresh_estimate()
        self._emit_source()

    # ── states / filter row ───────────────────────────────────────────────────
    def states(self) -> list[str]:
        """The run's DECLARED states for the chosen column (stable regardless
        of what the loaded range happens to contain)."""
        column = self._column.currentText()
        spec = (self._meta.get("states") or {}).get(column) or {}
        return list(spec.get("states", []))

    def _set_states(self, states: list[str]) -> None:
        if self._filter is not None:
            self._filter.deleteLater()
            self._filter = None
        has_states = bool(states)
        self._filter_row_holder.setVisible(has_states)
        if not has_states:
            return
        self._filter = make_regime_filter(states)
        self._filter.selectionChanged.connect(self.selectionChanged)
        self._filter_holder.addWidget(self._filter)

    def selected(self) -> list[str] | None:
        """Checked states, or None when no regime source is active (which
        means 'do not filter')."""
        if self._filter is None or not self._frames:
            return None
        return self._filter.selected()

    # ── the join ──────────────────────────────────────────────────────────────
    def mode(self) -> str:
        """Timing behind the performance TABLE's labels."""
        return self._mode.currentData()

    def filter_mode(self) -> str:
        """Timing behind the FILTER — independent of the table's."""
        return self._filter_mode.currentData()

    def timings_differ(self) -> bool:
        table = (self.mode(), self._at.currentText())
        return table != (self.filter_mode(), self._filter_at.currentText())

    def active(self) -> bool:
        return bool(self._frames and self._column.currentText())

    def annotate(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Trades + TWO label columns, because the table and the filter can be
        read at different moments:

          regime         — the table's timing, shown in the trades table
          FILTER_COLUMN  — the filter's timing, what the report is sliced by

        Filtering by the final label while reading performance as of entry is
        a legitimate question ("how did the days that ENDED high behave in the
        morning?"), so the two are deliberately allowed to disagree. When both
        timings match, the second column is a copy rather than a second join.
        Unchanged when no run is loaded.
        """
        if not self.active():
            return trades
        session = self._meta.get("globex_session") or {}
        common = {
            "session_start": session.get("start") or _DEFAULT_SESSION_START,
            "snapshot_minutes": int(self._meta.get("snapshot_minutes", 30)),
        }
        column = self._column.currentText()
        out = attach_regime(trades, self._frames, column, self.mode(),
                            at=self._at.currentText() or None, **common)
        if self.timings_differ():
            out = attach_regime(out, self._frames, column, self.filter_mode(),
                                at=self._filter_at.currentText() or None,
                                out_col=FILTER_COLUMN, **common)
        else:
            out[FILTER_COLUMN] = out["regime"]
        return out

    def _emit_source(self) -> None:
        self.sourceChanged.emit()

    # ── the performance table ─────────────────────────────────────────────────
    def set_trades(self, trades: pd.DataFrame, column: str = "regime") -> None:
        """Render the per-state breakdown. Computed BEFORE the regime filter is
        applied, so it always shows every state rather than only the checked
        ones (same rule as the news/holiday table)."""
        rows = regime_performance_rows(trades, column, self.states() or None)
        if rows is None:
            update_table_view(self._table, pd.DataFrame())
            self._table_caption.setText("Load a regime run to slice "
                                        "performance by regime.")
            return
        update_table_view(self._table, pd.DataFrame(rows))
        note = {MODE_ASOF: "labels as of each trade's entry — no lookahead",
                MODE_EXACT: f"labels from {self._at.currentText()} each day",
                MODE_FINAL: "END-OF-DAY labels — hindsight"}[self.mode()]
        self._table_caption.setText(
            f"Performance per regime state ({note}). "
            f"‘{UNKNOWN}’ means no label applied — exclude it rather than "
            f"reading it as a regime.")
