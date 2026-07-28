"""
Regime Detector — Explore tab. Loads an existing run folder (meta.json +
daily parquets, nothing is ever recomputed) and renders the daily price line
shaded by one regime column at a time.

The two selectors:
- Regime axis — which declared regime column colours the chart.
- As of — which snapshot to read per day: 'final' shows what the day turned
  out to be; an HH:MM shows only what was knowable at that moment. Flipping
  between them shows how much of the signal is hindsight.

Daily frames load fully into memory on a worker (date-range-limited), so
switching selectors re-renders instantly without re-reading files. Clicking
a chart point opens that day's full parquet as a table (diagnostic-tier
columns behind a toggle — the split comes from meta.json `schema`, not code).
"""

from PySide6.QtCore import QDate
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDateEdit, QGridLayout,
                               QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                               QWidget)

from modules.common.backend.data_roots import RegimeRunRef, list_regime_runs
from modules.common.ui.dataframe_model import make_table_view, update_table_view
from modules.common.ui.widgets import (Banner, Caption, ProgressLogPanel,
                                       SectionHeader, pin_minimum_height,
                                       wrap_card)
from modules.common.ui.workers import FunctionWorker
from modules.regime_detector.backend import io as rio
from modules.regime_detector.regime_chart import RegimeChart

_DEFAULT_SESSION = {"start": "18:00", "end": "17:00"}


class ExploreTab(QWidget):
    def __init__(self, settings, track_worker, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._track_worker = track_worker
        self._runs: list[RegimeRunRef] = []
        self._meta: dict | None = None
        self._frames: dict = {}
        self._detail_df = None
        self._worker: FunctionWorker | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 8, 0, 0)
        outer.setSpacing(10)
        pin_minimum_height(self)

        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(8)
        self._run = QComboBox()
        self._from = QDateEdit()
        self._to = QDateEdit()
        for de in (self._from, self._to):
            de.setCalendarPopup(True)
            de.setDisplayFormat("yyyy-MM-dd")
        self._load_btn = QPushButton("Load")
        self._load_btn.setProperty("primary", True)
        refresh_btn = QPushButton("Refresh runs")
        refresh_btn.clicked.connect(self._rescan)

        grid.addWidget(QLabel("Regime run"), 0, 0)
        grid.addWidget(self._run, 0, 1)
        grid.addWidget(QLabel("From"), 0, 2)
        grid.addWidget(self._from, 0, 3)
        grid.addWidget(QLabel("To"), 0, 4)
        grid.addWidget(self._to, 0, 5)
        grid.addWidget(self._load_btn, 0, 6)
        grid.addWidget(refresh_btn, 0, 7)
        grid.setColumnStretch(1, 1)
        self._run_info = Caption("")
        grid.addWidget(self._run_info, 1, 0, 1, 8)
        outer.addWidget(wrap_card(grid))

        self._banner = Banner()
        outer.addWidget(self._banner)
        self._progress = ProgressLogPanel(log_height=90)
        self._progress.setVisible(False)
        outer.addWidget(self._progress)

        # ── selectors + chart ─────────────────────────────────────────────────
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Regime axis"))
        self._axis = QComboBox()
        sel_row.addWidget(self._axis)
        sel_row.addSpacing(24)
        sel_row.addWidget(QLabel("As of"))
        self._asof = QComboBox()
        sel_row.addWidget(self._asof)
        sel_row.addStretch()
        self._selectors = QWidget()
        self._selectors.setLayout(sel_row)
        self._selectors.setVisible(False)
        outer.addWidget(self._selectors)

        self._chart = RegimeChart()
        self._chart.setVisible(False)
        outer.addWidget(self._chart)

        # ── day detail ────────────────────────────────────────────────────────
        self._detail_header = SectionHeader("Day detail")
        self._detail_caption = Caption("Click a day on the chart to see every "
                                       "snapshot row.")
        self._show_diag = QCheckBox("Show diagnostic columns")
        self._table = make_table_view(_empty_frame(), height=320)
        for w in (self._detail_header, self._detail_caption, self._show_diag,
                  self._table):
            w.setVisible(False)
            outer.addWidget(w)
        outer.addStretch()

        # ── wiring ────────────────────────────────────────────────────────────
        self._run.currentIndexChanged.connect(self._on_run_selected)
        self._load_btn.clicked.connect(self._on_load)
        self._axis.currentIndexChanged.connect(self._render)
        self._asof.currentIndexChanged.connect(self._render)
        self._chart.dayClicked.connect(self._on_day_clicked)
        self._show_diag.toggled.connect(self._refresh_detail)
        self._rescan()

    # ── run list / date bounds ────────────────────────────────────────────────
    def _rescan(self) -> None:
        self._runs = list_regime_runs(self.settings.data_roots)
        self._run.blockSignals(True)
        self._run.clear()
        self._run.addItems([r.label for r in self._runs])
        self._run.blockSignals(False)
        self._on_run_selected()

    def _current_run(self) -> RegimeRunRef | None:
        i = self._run.currentIndex()
        return self._runs[i] if 0 <= i < len(self._runs) else None

    def _on_run_selected(self) -> None:
        ref = self._current_run()
        self._run_info.setText("")
        if ref is None:
            return
        try:
            meta = rio.read_meta(ref.path)
        except (OSError, ValueError) as e:
            self._banner.show_message("error", f"Could not read meta.json: {e}")
            return
        dates = list(rio.day_files(ref.path))
        if dates:
            self._from.setDate(QDate.fromString(dates[0], "yyyy-MM-dd"))
            self._to.setDate(QDate.fromString(dates[-1], "yyyy-MM-dd"))
        counts = meta.get("counts", {})
        self._run_info.setText(
            f"{meta.get('script_name')} v{meta.get('script_version')} on "
            f"{meta.get('input_dataset_path', '?')} — {len(dates)} day files, "
            f"{counts.get('processed', '?')} processed / "
            f"{counts.get('skipped', 0)} skipped")

    # ── loading ───────────────────────────────────────────────────────────────
    def _on_load(self) -> None:
        ref = self._current_run()
        if ref is None:
            self._banner.show_message("error", "No regime run selected.")
            return
        self._banner.clear_message()
        try:
            self._meta = rio.read_meta(ref.path)
        except (OSError, ValueError) as e:
            self._banner.show_message("error", f"Could not read meta.json: {e}")
            return
        self._progress.setVisible(True)
        self._progress.reset()
        worker = FunctionWorker(
            rio.load_day_frames, ref.path,
            start=self._from.date().toString("yyyy-MM-dd"),
            end=self._to.date().toString("yyyy-MM-dd"),
            needs_progress=True)
        worker.signals.progress.connect(self._progress.on_progress)
        worker.signals.finished.connect(self._on_loaded)
        worker.signals.error.connect(
            lambda m, _tb: self._banner.show_message("error", m))
        self._worker = worker
        self._load_btn.setEnabled(False)
        worker.signals.finished.connect(lambda *_: self._load_btn.setEnabled(True))
        worker.signals.error.connect(lambda *_: self._load_btn.setEnabled(True))
        worker.signals.cancelled.connect(lambda: self._load_btn.setEnabled(True))
        self._track_worker(worker)

    def _on_loaded(self, frames: dict) -> None:
        self._frames = frames or {}
        self._progress.setVisible(False)
        if not self._frames:
            self._banner.show_message("warning", "No daily files in the "
                                                 "selected date range.")
            return
        meta = self._meta or {}
        regime_cols = list(_tiers(meta).get("regime", []))
        session = meta.get("globex_session") or {}
        start = session.get("start") or _DEFAULT_SESSION["start"]
        end = session.get("end") or _DEFAULT_SESSION["end"]
        minutes = int(meta.get("snapshot_minutes", 30))

        self._axis.blockSignals(True)
        self._axis.clear()
        self._axis.addItems(regime_cols)
        self._axis.blockSignals(False)

        self._asof.blockSignals(True)
        self._asof.clear()
        self._asof.addItem(rio.FINAL)
        self._asof.addItems(rio.snapshot_grid_labels(start, end, minutes))
        self._asof.setCurrentIndex(0)
        self._asof.blockSignals(False)

        self._selectors.setVisible(True)
        self._chart.setVisible(True)
        self._detail_header.setVisible(True)
        self._detail_caption.setVisible(True)
        self._render()

    # ── rendering ─────────────────────────────────────────────────────────────
    def _session_start(self) -> str:
        session = (self._meta or {}).get("globex_session") or {}
        return session.get("start") or _DEFAULT_SESSION["start"]

    def _render(self) -> None:
        if not self._frames or self._axis.currentIndex() < 0:
            return
        col = self._axis.currentText()
        at = self._asof.currentText() or rio.FINAL
        picked = rio.pick_rows(self._frames, at, self._session_start())
        if picked.empty or col not in picked.columns:
            self._chart.clear()
            self._banner.show_message(
                "warning", f"No rows for as-of {at} in the loaded range.")
            return
        self._banner.clear_message()
        states_meta = (self._meta or {}).get("states", {}).get(col, {})
        colors = dict(zip(states_meta.get("states", []),
                          states_meta.get("colors", [])))
        self._chart.set_data(picked.index, picked["price"], picked[col], colors)

    # ── day detail ────────────────────────────────────────────────────────────
    def _on_day_clicked(self, date: str) -> None:
        df = self._frames.get(date)
        if df is None:
            return
        self._detail_df = df.reset_index(names="snapshot")
        self._detail_caption.setText(f"{date} — {len(df)} snapshots")
        self._show_diag.setVisible(True)
        self._table.setVisible(True)
        self._refresh_detail()

    def _refresh_detail(self) -> None:
        if self._detail_df is None:
            return
        df = self._detail_df
        if not self._show_diag.isChecked():
            diag = set(_tiers(self._meta or {}).get("diagnostic", []))
            df = df[[c for c in df.columns if c not in diag]]
        update_table_view(self._table, df)


def _tiers(meta: dict) -> dict:
    """The meta schema's tier split (io.tiers owns the v1/v2 shim)."""
    return rio.tiers(meta)


def _empty_frame():
    import pandas as pd
    return pd.DataFrame()
