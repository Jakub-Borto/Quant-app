"""
Regime-switching Monte Carlo panel — the dedicated UI shown when the selected
method module sets REGIME_PANEL = True (methods/regime_switching.py), mirroring
how PROP_FIRM routes to PropFirmPanel.

It needs a dedicated panel because this method takes an input no other MC method
does: a regime run. A params form cannot express "pick a run, pick a column,
pick how a trade is matched to a snapshot" — and, more importantly, cannot show
the model BEFORE it is simulated.

THE PREVIEW IS THE POINT. The transition matrix, its RAW counts and the
trades-per-state table all render before Run is enabled, and the estimated model
is handed to the method verbatim (params["model"]). So what you inspect is
exactly what gets simulated — no re-estimation, no drift between the two. A 0.70
built from 8 transitions and a 0.70 built from 90 look identical as
probabilities; the counts are how you tell them apart, which is why they sit in
the cell rather than in a tooltip.

TWO WINDOWS, TWO MEANINGS (kept as separate controls on purpose):
  Matrix window — days feeding the transition matrix. Widening it past the
      trades file is the point: ES regimes reach back to 2010, so a one-year
      backtest can still borrow a 4000-transition matrix instead of a 300-one.
  Pool window   — days feeding trade-probability and the trade pools. Cannot
      usefully exceed the trades file; widening it only adds days that could not
      have traded and deflates every trade-probability toward zero. Clamped to
      the trades span, and it warns rather than silently obeying.
  Both default to the trades span, so the plain path is the coherent one.

The equity curves this method produces are indexed by TRADING DAY, not by trade
— a no-trade day is a flat step that still advances the chain. The fan chart's
x axis says so.

LOADING: only the days the two windows span are read, and of those only the
regime column plus is_final — a one-year window costs ~250 small files, and even
a full 15-year matrix is ~4 MB rather than 57 MB. Frames are cached per
(run, column) with the range they cover, so widening the matrix reloads but
narrowing it does not. Between the range limit and the column limit there is
nothing left for a memory-budget dialog to negotiate; the load just runs on a
worker with progress.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                               QDoubleSpinBox, QGridLayout, QHBoxLayout,
                               QHeaderView, QLabel, QPushButton, QSizePolicy,
                               QSlider, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from modules.common.backend.asset_info import get_dollars_per_tick
from modules.common.backend.data_roots import RegimeRunRef, list_regime_runs
from modules.common.ui import theme
from modules.common.ui.charts.fan_chart import FanChart
from modules.common.ui.dataframe_model import make_table_view, update_table_view
from modules.common.ui.widgets import (Banner, Caption, SectionHeader, hline,
                                       wrap_card)
from modules.common.ui.workers import FunctionWorker
from modules.monte_carlo.backend import regime_switching as rs
from modules.monte_carlo.backend.cost_ctx import build_cost_ctx
from modules.monte_carlo.backend.stats import (_compute_metrics,
                                               _select_featured_paths,
                                               metrics_table_rows)
from modules.regime_detector.backend import io as rio

RUIN_OPTIONS = {
    "No threshold": None,
    "Ruin at 0% (account wiped)": 0.0,
    "Ruin at 50% loss":           0.5,
}

_METHOD_CAPTION = (
    "Draws trade outcomes conditional on the regime the chain is in, so runs of "
    "losers cluster the way volatility does instead of being scattered by "
    "independent resampling. Curves are indexed by trading day — a no-trade day "
    "is a flat step that still moves the chain."
)

_COSTS_CAPTION = (
    "Equity is net of commissions & slippage. Note: slippage is applied "
    "post-hoc to recorded trades — a worse entry that would have prevented "
    "a take-profit fill is not modelled."
)


# ── transition-matrix grid ───────────────────────────────────────────────────

class TransitionMatrixView(QTableWidget):
    """Rows = from-state, columns = to-state, cell = probability over the raw
    count, background shaded by probability.

    Purpose-built rather than the optimizer's HeatmapChart: that widget is
    wired to the optimizer's metric/`total_trades` array contract and its
    parameter-axis hover text, and a 3x3 matrix would have to impersonate a
    parameter sweep to use it. At this size an explicit grid also does the one
    thing the plan actually asks for better — probability AND count visible in
    the same cell, no hover required.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setAlternatingRowColors(False)
        self.verticalHeader().setDefaultAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def set_matrix(self, states: list[str], probs: np.ndarray,
                   counts: np.ndarray, empty_rows: np.ndarray) -> None:
        n = len(states)
        self.setRowCount(n)
        self.setColumnCount(n)
        self.setHorizontalHeaderLabels([f"to {s}" for s in states])
        self.setVerticalHeaderLabels([f"from {s}" for s in states])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        accent = QColor(theme.ACCENT_SOFT)
        for i in range(n):
            row_total = int(counts[i].sum())
            for j in range(n):
                if empty_rows[i]:
                    item = QTableWidgetItem("—\nnever observed")
                    item.setForeground(QColor(theme.TEXT_MUTED))
                else:
                    p = float(probs[i, j])
                    item = QTableWidgetItem(f"{p:.3f}\n{int(counts[i, j])} of {row_total}")
                    # Shade by probability; alpha only, so the theme still reads.
                    tint = QColor(accent)
                    tint.setAlpha(int(20 + 150 * p))
                    item.setBackground(tint)
                    item.setForeground(QColor(theme.TEXT))
                item.setTextAlignment(Qt.AlignCenter)
                self.setItem(i, j, item)

        # Fixed height: the shared convention is that no chart/table resizes the
        # page as data changes (see the "pin every chart height" commit).
        self.setFixedHeight(38 + 46 * n)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)


# ── the panel ────────────────────────────────────────────────────────────────

class RegimeSwitchingPanel(QWidget):
    """context_fn() -> {trades_ref, sizer_module, sizer_params, account_size}
    (provided by the Monte Carlo window; read at Build/Run click)."""

    def __init__(self, mc_module, context_fn, track_worker, settings,
                 parent=None):
        super().__init__(parent)
        self._mc_module = mc_module
        self._context_fn = context_fn
        self._track_worker = track_worker
        self._settings = settings

        self._runs: list[RegimeRunRef] = []
        self._meta: dict = {}
        self._frames: dict = {}
        self._frames_key: tuple | None = None    # (run path, column)
        self._loaded_range: tuple[str, str] | None = None   # what's in _frames
        self._model: dict | None = None
        self._trades_cache: tuple[Path, pd.DataFrame] | None = None
        self._pending_warnings: list[str] = []
        self._results: dict | None = None
        self._result_account = None
        self._result_ruin = None
        self._result_costs = False
        self._chart_capped = True

        defaults = getattr(mc_module, "PARAMS", {})
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addWidget(Caption(_METHOD_CAPTION))

        self._banner = Banner()
        lay.addWidget(self._banner)

        # ── regime source ─────────────────────────────────────────────────────
        lay.addWidget(SectionHeader("Regime source"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)

        self._run = QComboBox()
        self._all_assets = QCheckBox("Show runs from all assets")
        self._all_assets.setToolTip(
            "The asset is derived from the trades filename's first token. "
            "Tick this to override it and use a run from another asset.")
        self._column = QComboBox()
        self._tag = QComboBox()
        for tag in rs.TAGS:
            self._tag.addItem(rs.TAG_LABELS[tag], tag)
        self._tag.setToolTip(
            "As of entry — the snapshot at or before the trade's entry; the "
            "only version safe as a live filter.\n"
            "Final row — knows how the day ended. Legitimate for structural "
            "research, invalid as a filter.")
        self._no_trade = QComboBox()
        for mode in rs.NO_TRADE_MODES:
            self._no_trade.addItem(rs.NO_TRADE_LABELS[mode], mode)
        self._no_trade.setToolTip(
            "Which snapshot labels a day the strategy did NOT trade — those "
            "days are the denominator of every trade-probability.\n"
            "The day's final snapshot knows how the day ended, so it makes the "
            "denominator partly retrospective; the trade pools stay "
            "point-in-time either way.")
        self._decision_at = QComboBox()
        self._decision_at.setEnabled(False)
        self._decision_at.setToolTip(
            "Wall-clock time a no-trade day is labelled at. Defaults to the "
            "median entry time of the trades file — roughly when the strategy "
            "would have decided.")

        grid.addWidget(QLabel("Regime run"), 0, 0)
        grid.addWidget(self._run, 0, 1)
        grid.addWidget(self._all_assets, 0, 2)
        grid.addWidget(QLabel("Regime column"), 1, 0)
        grid.addWidget(self._column, 1, 1)
        grid.addWidget(QLabel("Label trades"), 2, 0)
        grid.addWidget(self._tag, 2, 1)
        grid.addWidget(QLabel("Label no-trade days"), 3, 0)
        grid.addWidget(self._no_trade, 3, 1)
        grid.addWidget(self._decision_at, 3, 2)
        grid.setColumnStretch(1, 1)
        lay.addWidget(wrap_card(grid))

        # ── estimation windows ────────────────────────────────────────────────
        lay.addWidget(SectionHeader("Estimation windows"))
        wgrid = QGridLayout()
        wgrid.setHorizontalSpacing(16)
        wgrid.setVerticalSpacing(6)
        self._matrix_start, self._matrix_end = QComboBox(), QComboBox()
        self._pool_start, self._pool_end = QComboBox(), QComboBox()
        for w in (self._matrix_start, self._matrix_end,
                  self._pool_start, self._pool_end):
            w.setEditable(False)
        self._full_history = QPushButton("Use full regime history")
        self._full_history.setToolTip(
            "Estimate the transition matrix over every day the regime run "
            "covers. The trade pools stay on the trades span — a matrix from "
            "15 years is far better estimated than one from a single year.")
        self._full_history.clicked.connect(self._on_full_history)
        self._match_trades = QPushButton("Match trades span")
        self._match_trades.clicked.connect(lambda: self._reset_windows())

        wgrid.addWidget(QLabel("Transition matrix (all days)"), 0, 0)
        wgrid.addWidget(self._matrix_start, 0, 1)
        wgrid.addWidget(self._matrix_end, 0, 2)
        wgrid.addWidget(self._full_history, 0, 3)
        wgrid.addWidget(QLabel("Trade pools (trade days)"), 1, 0)
        wgrid.addWidget(self._pool_start, 1, 1)
        wgrid.addWidget(self._pool_end, 1, 2)
        wgrid.addWidget(self._match_trades, 1, 3)
        wgrid.setColumnStretch(1, 1)
        wgrid.setColumnStretch(2, 1)
        lay.addWidget(wrap_card(wgrid))
        self._window_note = Caption(
            "All days define how regimes move; trade days define what happens "
            "when you trade in each. Widening the matrix window is useful; "
            "widening the pool window past the trades file is not.")
        lay.addWidget(self._window_note)

        # ── build the model ───────────────────────────────────────────────────
        build_row = QHBoxLayout()
        self._build_btn = QPushButton("Build model")
        self._build_btn.setProperty("primary", True)
        self._build_btn.setMinimumWidth(200)
        self._build_btn.clicked.connect(self._on_build)
        refresh = QPushButton("Refresh runs")
        refresh.clicked.connect(self.rescan)
        build_row.addWidget(self._build_btn)
        build_row.addWidget(refresh)
        build_row.addStretch()
        lay.addLayout(build_row)
        self._build_status = Caption("Pick a regime run, then build the model "
                                     "to see it before simulating.")
        lay.addWidget(self._build_status)

        # ── model preview ─────────────────────────────────────────────────────
        self._preview = QWidget()
        self._preview.setVisible(False)
        plav = QVBoxLayout(self._preview)
        plav.setContentsMargins(0, 0, 0, 0)
        plav.setSpacing(8)
        plav.addWidget(hline())
        plav.addWidget(SectionHeader("Transition matrix"))
        plav.addWidget(Caption(
            "Probability of tomorrow's regime given today's, with the raw "
            "transition count underneath. The counts are the trustworthiness "
            "tell — a 0.70 from 8 transitions is not a 0.70 from 90, and the "
            "simulation treats both as exact."))
        self._matrix_view = TransitionMatrixView()
        plav.addWidget(self._matrix_view)
        plav.addWidget(SectionHeader("Per-regime sample size and shape"))
        plav.addWidget(Caption(
            "The left columns say whether each regime is well measured. The "
            "right columns say whether the split matters: if every regime's "
            "mean, spread and worst trade look alike, clustering regimes "
            "cannot cluster losers and this method reduces to a slower "
            "bootstrap. States usually separate on SPREAD rather than mean, so "
            "the effect lands in tail risk, not the median path."))
        self._state_table = make_table_view(pd.DataFrame(), height=220)
        plav.addWidget(self._state_table)
        self._state_caption = Caption("")
        plav.addWidget(self._state_caption)
        lay.addWidget(self._preview)

        # ── simulation controls ───────────────────────────────────────────────
        lay.addWidget(SectionHeader("Simulation"))
        sgrid = QGridLayout()
        self._n_paths = QSpinBox()
        self._n_paths.setRange(1, 1_000_000)
        self._n_paths.setSingleStep(500)
        self._n_paths.setValue(int(defaults.get("n_paths", 1000)))
        self._seed = QSpinBox()
        self._seed.setRange(-1_000_000_000, 1_000_000_000)
        self._seed.setValue(int(defaults.get("seed", 42)))
        self._horizon_mode = QComboBox()
        self._horizon_mode.addItem("Match the pool window", rs.HORIZON_MATCH)
        self._horizon_mode.addItem("Fixed number of days", rs.HORIZON_FIXED)
        self._horizon_mode.setToolTip(
            "How long each simulated path runs. Independent of the estimation "
            "windows on purpose — building the matrix from 15 years and "
            "simulating one year is perfectly valid.")
        self._horizon_days = QSpinBox()
        self._horizon_days.setRange(1, 100_000)
        self._horizon_days.setValue(252)
        self._horizon_days.setEnabled(False)
        for col, (label, w) in enumerate([("Paths", self._n_paths),
                                          ("Seed", self._seed),
                                          ("Horizon", self._horizon_mode),
                                          ("Days", self._horizon_days)]):
            sgrid.addWidget(QLabel(label), 0, col)
            sgrid.addWidget(w, 1, col)
        lay.addWidget(wrap_card(sgrid))

        ruin_row = QHBoxLayout()
        ruin_row.addWidget(QLabel("Ruin definition"))
        self._ruin = QComboBox()
        self._ruin.addItems(list(RUIN_OPTIONS.keys()))
        ruin_row.addWidget(self._ruin)
        ruin_row.addStretch()
        lay.addLayout(ruin_row)

        self._apply_costs = QCheckBox("Apply commissions && slippage")
        self._apply_costs.setChecked(True)
        lay.addWidget(self._apply_costs)
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
        lay.addLayout(slip_row)

        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Simulation")
        self._run_btn.setProperty("primary", True)
        self._run_btn.setMinimumWidth(220)
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run)
        run_row.addStretch()
        run_row.addWidget(self._run_btn)
        run_row.addStretch()
        lay.addLayout(run_row)
        self._status = Caption("")
        lay.addWidget(self._status)

        # ── results ───────────────────────────────────────────────────────────
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
        lay.addWidget(self._results_box)

        # ── wiring ────────────────────────────────────────────────────────────
        self._all_assets.toggled.connect(self.rescan)
        self._run.currentIndexChanged.connect(self._on_run_changed)
        self._column.currentIndexChanged.connect(self._invalidate)
        self._tag.currentIndexChanged.connect(self._on_tag_changed)
        self._no_trade.currentIndexChanged.connect(self._on_no_trade_changed)
        self._decision_at.currentIndexChanged.connect(self._invalidate)
        for w in (self._matrix_start, self._matrix_end,
                  self._pool_start, self._pool_end):
            w.currentIndexChanged.connect(self._invalidate)
        self._horizon_mode.currentIndexChanged.connect(
            lambda: self._horizon_days.setEnabled(
                self._horizon_mode.currentData() == rs.HORIZON_FIXED))
        self.rescan()

    # ── scanning ──────────────────────────────────────────────────────────────
    def _trades_asset(self) -> str | None:
        ctx = self._context_fn()
        if ctx is None:
            return None
        return ctx["trades_ref"].filename.split("_")[0].upper()

    def rescan(self) -> None:
        """Regime runs for the trades file's asset (or all assets when
        overridden). Derivation matches Analytics/Monte Carlo elsewhere: the
        FIRST underscore token of the trades filename."""
        self._trades_cache = None       # the window calls this on file change
        asset = self._trades_asset()
        runs = list_regime_runs(self._settings.data_roots)
        if asset and not self._all_assets.isChecked():
            runs = [r for r in runs if r.asset.upper() == asset]
        self._runs = runs

        self._run.blockSignals(True)
        self._run.clear()
        for ref in runs:
            self._run.addItem(ref.label, ref)
        self._run.blockSignals(False)

        if not runs:
            self._build_status.setText(
                f"No regime runs found{f' for {asset}' if asset else ''} — "
                f"produce one in the Regime Detector, or tick 'all assets'.")
            self._build_btn.setEnabled(False)
        else:
            self._build_btn.setEnabled(True)
        self._on_run_changed()

    def _on_run_changed(self) -> None:
        ref: RegimeRunRef | None = self._run.currentData()
        self._meta = {}
        self._column.blockSignals(True)
        self._column.clear()
        if ref is not None:
            try:
                self._meta = rio.read_meta(ref.path)
            except (OSError, ValueError) as e:
                self._banner.show_message("error", f"Could not read the run's "
                                                   f"meta.json: {e}")
            columns = list(rio.tiers(self._meta).get("regime", []))
            for c in columns:
                self._column.addItem(c, c)
            primary = (self._meta.get("script_extras") or {}).get(
                "primary_regime_column")
            if primary in columns:
                self._column.setCurrentIndex(columns.index(primary))
        self._column.blockSignals(False)
        self._refresh_decision_times()
        self._reset_windows()
        self._invalidate()

    def _refresh_decision_times(self) -> None:
        """The decision-time dropdown holds the run's own snapshot grid, and
        defaults to the snapshot nearest the median entry time of the trades
        file — roughly the hour the strategy actually decides."""
        session = self._meta.get("globex_session") or {}
        minutes = int(self._meta.get("snapshot_minutes") or 30)
        labels = rio.snapshot_grid_labels(session.get("start", "18:00"),
                                          session.get("end", "17:00"), minutes)
        self._decision_at.blockSignals(True)
        self._decision_at.clear()
        self._decision_at.addItems(labels)
        median = self._median_entry_time()
        if median and labels:
            # The grid is in SESSION order (18:30 ... 00:00 ... 17:00), so
            # scanning for the first label >= the median would pick 18:30 —
            # "18:30" > "10:30" lexicographically even though it is the
            # evening BEFORE. Compare as clock times and take the latest
            # snapshot at or before the median instead.
            at_or_before = [t for t in labels if t <= median]
            pick = max(at_or_before) if at_or_before else labels[-1]
            self._decision_at.setCurrentIndex(labels.index(pick))
        self._decision_at.blockSignals(False)

    def _median_entry_time(self) -> str | None:
        trades = self._load_trades(quiet=True)
        if trades is None or trades.empty or "entry_time" not in trades:
            return None
        minutes = (pd.to_datetime(trades["entry_time"]).dt.hour * 60
                   + pd.to_datetime(trades["entry_time"]).dt.minute)
        med = int(minutes.median())
        return f"{med // 60:02d}:{med % 60:02d}"

    # ── windows ───────────────────────────────────────────────────────────────
    def _run_dates(self) -> list[str]:
        ref: RegimeRunRef | None = self._run.currentData()
        if ref is None:
            return []
        return list(rio.day_files(ref.path).keys())

    def _reset_windows(self) -> None:
        """Both windows default to the trades span — the coherent pairing. The
        matrix can be widened deliberately; the pool window cannot usefully be."""
        dates = self._run_dates()
        span = self._trades_span_dates()
        for combo in (self._matrix_start, self._matrix_end,
                      self._pool_start, self._pool_end):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(dates)
            combo.blockSignals(False)
        if not dates:
            return
        lo, hi = span if span else (dates[0], dates[-1])
        for combo, target in ((self._matrix_start, lo), (self._pool_start, lo),
                              (self._matrix_end, hi), (self._pool_end, hi)):
            combo.blockSignals(True)
            self._select_nearest(combo, target, dates)
            combo.blockSignals(False)
        self._invalidate()

    @staticmethod
    def _select_nearest(combo: QComboBox, target: str, dates: list[str]) -> None:
        """Trades can start on a day the regime run lacks; clamp into range."""
        if target in dates:
            combo.setCurrentIndex(dates.index(target))
            return
        after = [i for i, d in enumerate(dates) if d >= target]
        combo.setCurrentIndex(after[0] if after else len(dates) - 1)

    def _trades_span_dates(self) -> tuple[str, str] | None:
        trades = self._load_trades(quiet=True)
        if trades is None or trades.empty or "date" not in trades:
            return None
        dates = pd.to_datetime(trades["date"])
        return (str(dates.min().date()), str(dates.max().date()))

    def _on_full_history(self) -> None:
        dates = self._run_dates()
        if not dates:
            return
        self._matrix_start.setCurrentIndex(0)
        self._matrix_end.setCurrentIndex(len(dates) - 1)

    def _on_tag_changed(self) -> None:
        entry = self._tag.currentData() == rs.TAG_ENTRY
        self._no_trade.setEnabled(entry)
        self._decision_at.setEnabled(
            entry and self._no_trade.currentData() == rs.NO_TRADE_DECISION)
        self._invalidate()

    def _on_no_trade_changed(self) -> None:
        self._decision_at.setEnabled(
            self._tag.currentData() == rs.TAG_ENTRY
            and self._no_trade.currentData() == rs.NO_TRADE_DECISION)
        self._invalidate()

    def _invalidate(self) -> None:
        """Any source or window change retires the previewed model: the whole
        contract is that what you inspect is what you simulate."""
        self._model = None
        self._run_btn.setEnabled(False)
        self._preview.setVisible(False)
        self._build_status.setText("Model out of date — build it again.")

    def _show_warnings(self, kind: str, messages: list[str]) -> None:
        """Banner is a single QLabel and show_message REPLACES its text, so a
        loop over warnings would display only the last one. This method's
        warnings ARE its honesty mechanism — thin pools, retrospective
        labelling, chain breaks, a pool window past the trades file — and a run
        routinely raises several at once, so they are bulleted into one message
        instead of silently dropping all but the final."""
        msgs = [m for m in messages if m]
        if not msgs:
            self._banner.clear_message()
            return
        body = msgs[0] if len(msgs) == 1 else "\n".join(f"•  {m}" for m in msgs)
        self._banner.show_message(kind, body)

    # ── trades ────────────────────────────────────────────────────────────────
    def _load_trades(self, quiet: bool = False) -> pd.DataFrame | None:
        """Cached per path: the trades span, the median entry time and the
        build all want the same frame, and re-reading it on every combo change
        is pure waste."""
        ctx = self._context_fn()
        if ctx is None:
            if not quiet:
                self._banner.show_message("error", "Pick a trades file and "
                                                   "sizer first.")
            return None
        path = ctx["trades_ref"].path
        if self._trades_cache is not None and self._trades_cache[0] == path:
            return self._trades_cache[1]
        try:
            trades = pd.read_parquet(path)
        except Exception as e:  # noqa: BLE001
            if not quiet:
                self._banner.show_message("error", f"Could not load trades: {e}")
            return None
        self._trades_cache = (path, trades)
        return trades

    # ── build ─────────────────────────────────────────────────────────────────
    def _on_build(self) -> None:
        self._banner.clear_message()
        ref: RegimeRunRef | None = self._run.currentData()
        column = self._column.currentData()
        if ref is None or not column:
            self._banner.show_message("error", "Pick a regime run and column.")
            return
        trades = self._load_trades()
        if trades is None:
            return
        if "entry_time" not in trades.columns or "date" not in trades.columns:
            self._banner.show_message(
                "error", "This trades file has no entry_time/date column, so "
                         "trades cannot be matched to a regime snapshot.")
            return

        matrix_range = (self._matrix_start.currentText(),
                        self._matrix_end.currentText())
        pool_range = (self._pool_start.currentText(), self._pool_end.currentText())
        # Held rather than shown now: the estimate raises its own warnings and
        # they all surface together when the build lands.
        self._pending_warnings = []
        span = self._trades_span_dates()
        if span and (pool_range[0] < span[0] or pool_range[1] > span[1]):
            self._pending_warnings.append(
                f"The pool window ({pool_range[0]} → {pool_range[1]}) reaches "
                f"outside the trades file ({span[0]} → {span[1]}). Days with no "
                f"trades to draw from still count in every trade-probability "
                f"denominator, so activity will be understated.")

        states = list((self._meta.get("states") or {}).get(column, {})
                      .get("states", []))
        if not states:
            self._banner.show_message(
                "error", f"The run's meta.json declares no states for "
                         f"'{column}'.")
            return

        # Read only the days the two windows actually span. estimate() slices to
        # them anyway, and the as-of join never reaches outside a trade's own
        # day (the cross-session guard requires the snapshot's RTH date to equal
        # the trade's), so a narrower load gives identical numbers — it just
        # doesn't make a one-year window pay for fifteen years of files.
        need = (min(matrix_range[0], pool_range[0]),
                max(matrix_range[1], pool_range[1]))
        covered = (self._frames_key == (str(ref.path), column)
                   and self._loaded_range is not None
                   and self._loaded_range[0] <= need[0]
                   and self._loaded_range[1] >= need[1])
        self._build_btn.setEnabled(False)
        self._build_status.setText("Estimating…" if covered
                                   else "Loading regime files…")

        worker = FunctionWorker(
            self._build_model,
            needs_progress = True,      # a full-history load is thousands of files
            run_path   = ref.path,
            column     = column,
            states     = states,
            trades     = trades,
            frames     = self._frames if covered else None,
            load_range = need,
            tag        = self._tag.currentData(),
            no_trade   = self._no_trade.currentData(),
            decision   = self._decision_at.currentText(),
            matrix_range = matrix_range,
            pool_range = pool_range,
            meta       = self._meta,
        )
        worker.signals.progress.connect(self._on_build_progress)
        worker.signals.finished.connect(self._on_built)
        worker.signals.error.connect(self._on_build_error)
        self._track_worker(worker)

    @staticmethod
    def _build_model(run_path: Path, column: str, states: list, trades,
                     frames, load_range, tag: str, no_trade: str,
                     decision: str, matrix_range, pool_range, meta: dict,
                     on_progress=None) -> dict:
        """Load the days the windows span, then estimate — on a worker thread.

        Two things keep this cheap. Only the days in `load_range` are read, so a
        one-year window costs ~250 files rather than the run's whole history;
        and only the regime column plus is_final are read, so even a full
        15-year load is ~4 MB instead of 57 MB. Between them there is nothing
        left for a memory-budget dialog to negotiate."""
        loaded_from_cache = frames is not None
        if frames is None:
            files = rio.files_in_range(run_path, load_range[0], load_range[1])
            frames, total = {}, len(files)
            for i, (date, path) in enumerate(files.items()):
                if on_progress is not None and i % 50 == 0:
                    on_progress(i, total, f"Loading regime files… {i}/{total}")
                frames[date] = pd.read_parquet(path, columns=[column, "is_final"])
        if on_progress is not None:
            on_progress(1, 1, "Estimating the model…")

        session = meta.get("globex_session") or {}
        model = rs.estimate(
            trades, frames, column, states,
            tag               = tag,
            no_trade_snapshot = no_trade,
            decision_time     = decision or None,
            session_start     = session.get("start", rs.DEFAULT_SESSION_START),
            snapshot_minutes  = int(meta.get("snapshot_minutes") or 30),
            matrix_start = matrix_range[0], matrix_end = matrix_range[1],
            pool_start   = pool_range[0],   pool_end   = pool_range[1])
        # range=None means "reused the cache" — the caller keeps the range it
        # already has, which may be WIDER than what this build needed. Reporting
        # the narrower request would make a later re-widening reload files that
        # are still in memory.
        return {"frames": frames, "model": model, "key": (str(run_path), column),
                "range": None if loaded_from_cache else tuple(load_range)}

    def _on_build_progress(self, _cur: int, _total: int, message: str) -> None:
        self._build_status.setText(message)

    def _on_build_error(self, message: str, _tb: str) -> None:
        self._build_btn.setEnabled(True)
        self._build_status.setText("")
        self._banner.show_message("error", f"Could not build the model: {message}")

    def _on_built(self, payload: dict) -> None:
        self._build_btn.setEnabled(True)
        self._frames = payload["frames"]
        self._frames_key = payload["key"]
        if payload["range"] is not None:      # None = cache reused, range stands
            self._loaded_range = payload["range"]
        self._model = payload["model"]
        model = self._model

        self._show_warnings("warning",
                            self._pending_warnings + list(model.get("warnings", [])))

        self._matrix_view.set_matrix(model["states"], model["probs"],
                                     model["counts"], model["empty_rows"])
        update_table_view(self._state_table, self._state_rows(model))
        thin = model["min_trades_per_state"]
        self._state_caption.setText(
            f"Trade pools under {thin} trades are flagged: the simulation "
            f"redraws the same few extremes across every path, so the fan "
            f"chart's tails look more precise than the data supports. "
            f"Matrix window {model['matrix_days']:,} days "
            f"({int(model['counts'].sum()):,} transitions); "
            f"pool window {model['pool_days']:,} days.")
        self._preview.setVisible(True)
        self._run_btn.setEnabled(True)
        self._build_status.setText(
            f"Model built from {model['matrix_days']:,} days — "
            f"inspect the matrix, then run.")

    @staticmethod
    def _state_rows(model: dict) -> pd.DataFrame:
        """Sample size AND shape. The left columns say whether each state is
        well measured; the right columns say whether the split matters at all —
        if every state's mean/sd/worst look alike, clustering regimes cannot
        cluster losers and this method has nothing to bite on."""
        rows = []
        thin = model["min_trades_per_state"]
        profile = model.get("pool_profile", {})
        for s in model["states"]:
            pool = model["pool_size"][s]
            flag = ("—" if pool >= thin
                    else ("NO TRADES" if pool == 0 else "THIN"))
            prof = profile.get(s) or {}
            def fmt(key, spec):
                v = prof.get(key)
                return "—" if v is None else format(v, spec)
            rows.append((
                s,
                f"{model['n_days'][s]:,}",
                f"{model['n_trade_days'][s]:,}",
                f"{model['p_trade'][s] * 100:.1f}%",
                f"{pool:,}",
                flag,
                fmt("mean", "+.2f"),
                fmt("sd", ".2f"),
                "—" if prof.get("win_rate") is None
                else f"{prof['win_rate'] * 100:.1f}%",
                fmt("worst", "+.1f"),
            ))
        return pd.DataFrame(rows, columns=[
            "Regime", "Days", "Trade days", "P(trade)", "Trades in pool",
            "Sample size", "Mean (ticks)", "SD", "Win rate", "Worst"])

    # ── run ───────────────────────────────────────────────────────────────────
    def _on_run(self) -> None:
        self._banner.clear_message()
        ctx = self._context_fn()
        if ctx is None or self._model is None:
            self._banner.show_message("error", "Build the model first.")
            return
        trades = self._load_trades()
        if trades is None:
            return
        trades_ref = ctx["trades_ref"]
        try:
            dollars_per_tick = get_dollars_per_tick(trades_ref.filename)
        except ValueError as e:
            self._banner.show_message("error", str(e))
            return

        apply_costs = self._apply_costs.isChecked()
        slippage_n = int(self._slippage.value()) if apply_costs else 1
        cost_ctx, warn_missing = build_cost_ctx(trades_ref.filename,
                                                apply_costs, slippage_n)
        if warn_missing:
            self._banner.show_message(
                "warning",
                f"No commission rate for asset "
                f"'{trades_ref.filename.split('_')[0]}' — commissions billed "
                f"at 0; slippage still applies.")

        horizon = rs.resolve_horizon(self._model,
                                     self._horizon_mode.currentData(),
                                     int(self._horizon_days.value()))
        params = {
            "n_paths":  int(self._n_paths.value()),
            "seed":     int(self._seed.value()),
            "cost_ctx": cost_ctx,
            "model":    self._model,
            "horizon":  horizon,
        }
        final_sizer_params = {**ctx["sizer_params"],
                              "dollars_per_tick": dollars_per_tick}

        self._pending = {"account": float(ctx["account_size"]),
                         "ruin": RUIN_OPTIONS[self._ruin.currentText()],
                         "costs": apply_costs}
        self._run_btn.setEnabled(False)
        self._status.setText(
            f"Running regime-switching MC — {int(self._n_paths.value()):,} "
            f"paths × {horizon:,} trading days…")
        worker = FunctionWorker(self._mc_module.run, trades=trades,
                                sizer_module=ctx["sizer_module"],
                                sizer_params=final_sizer_params, params=params)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.error.connect(self._on_error)
        self._track_worker(worker)

    def _on_error(self, message: str, _tb: str) -> None:
        self._run_btn.setEnabled(True)
        self._status.setText("")
        self._banner.show_message("error", f"Simulation error: {message}")

    def _on_finished(self, results: dict) -> None:
        self._run_btn.setEnabled(True)
        self._status.setText("")
        self._results = results
        self._result_account = self._pending["account"]
        self._result_ruin = self._pending["ruin"]
        self._result_costs = self._pending["costs"]
        self._render_results()

    # ── results ───────────────────────────────────────────────────────────────
    def _toggle_cap(self) -> None:
        self._chart_capped = not self._chart_capped
        self._render_results()

    def _render_results(self) -> None:
        if self._results is None:
            return
        equity_matrix = self._results["equity_matrix"]
        account_size = self._result_account

        self._costs_caption.setVisible(bool(self._result_costs))
        self._cap_btn.setText("Show full equity curve" if self._chart_capped
                              else "Cap at 3× account")
        y_max = account_size * 3 if self._chart_capped else None

        featured = _select_featured_paths(equity_matrix)
        metrics = _compute_metrics(equity_matrix, account_size,
                                   self._result_ruin)
        self._fan.set_data(
            equity_matrix, account_size, featured, self._result_ruin,
            y_max=y_max, band_finals=metrics["band_finals"],
            x_label="Trading day # (flat steps are no-trade days)")
        update_table_view(self._metrics_table,
                          metrics_table_rows(metrics, account_size))
        self._results_box.setVisible(True)
