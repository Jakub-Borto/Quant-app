"""
TradeReportPanel — the full shared report, composed in the old page order:

    Performance tiles + exit breakdown
    Market Exposure (α/β) collapsed section
    Equity curve (click a point -> trade detail)
    Chart View Settings
    Trade detail (hidden until a click)
    RR Distribution (bin-width input + overlaid histogram)

Used by the Backtester window and the Optimizer's cell drill-down. The
news/holiday table and the trade/day-type filter rows are deliberately NOT
part of the panel — the calling window places them BEFORE it, preserving the
old filter ordering (news table sees pre-day-filter trades).

API:
    set_context(asset, tick_size, ticks_per_point, candles_folder, parquet_root)
    set_trades(filtered_trades)      # recomputes every section
"""

import pandas as pd
from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from modules.common.backend.trade_stats import rr_bin_edges, rr_distribution_series
from ..charts.equity_curve import EquityCurveChart
from ..charts.histogram import OverlaidHistogram
from ..widgets import Banner, SectionHeader
from .chart_controls import ChartViewControls
from .exposure_section import ExposureSection
from .metrics_section import MetricsSection
from .trade_detail import TradeDetailView


class TradeReportPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._trades: pd.DataFrame | None = None
        self._asset = None
        self._tick_size = None
        self._ticks_per_point = None
        self._candles_folder = None
        self._parquet_root = None
        self._selected_row: int | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        self.metrics = MetricsSection()
        self.exposure = ExposureSection()
        lay.addWidget(self.metrics)
        lay.addWidget(self.exposure)

        lay.addWidget(SectionHeader("Equity Curve"))
        self.equity = EquityCurveChart()
        self.equity.pointClicked.connect(self._on_point_clicked)
        lay.addWidget(self.equity)

        self.chart_controls = ChartViewControls()
        self.chart_controls.settingsChanged.connect(self._refresh_detail)
        lay.addWidget(self.chart_controls)

        self.trade_detail = TradeDetailView()
        lay.addWidget(self.trade_detail)

        # ── RR distribution ───────────────────────────────────────────────────
        lay.addWidget(SectionHeader("RR Distribution"))
        rr_row = QHBoxLayout()
        rr_row.addWidget(QLabel("RR bin width"))
        self._rr_width = QDoubleSpinBox()
        self._rr_width.setRange(0.1, 1e6)
        self._rr_width.setSingleStep(0.1)
        self._rr_width.setDecimals(2)
        self._rr_width.setValue(0.5)
        self._rr_width.valueChanged.connect(lambda _=None: self._refresh_rr())
        rr_row.addWidget(self._rr_width)
        rr_row.addStretch()
        lay.addLayout(rr_row)
        self._rr_banner = Banner("info", "")
        lay.addWidget(self._rr_banner)
        self.rr_hist = OverlaidHistogram()
        lay.addWidget(self.rr_hist)

    # ── context / data ────────────────────────────────────────────────────────
    def set_context(self, asset: str, tick_size: float, ticks_per_point: float,
                    candles_folder, parquet_root) -> None:
        self._asset = asset
        self._tick_size = tick_size
        self._ticks_per_point = ticks_per_point
        self._candles_folder = candles_folder
        self._parquet_root = parquet_root

    def set_trades(self, trades: pd.DataFrame) -> None:
        self._trades = trades
        self._selected_row = None
        self.trade_detail.clear()

        self.metrics.set_trades(trades)
        self.exposure.update_exposure(trades, self._asset, self._tick_size,
                                      self._parquet_root)
        self.equity.set_trades(trades)
        self._refresh_rr()

    # ── internals ─────────────────────────────────────────────────────────────
    def _refresh_rr(self) -> None:
        if self._trades is None:
            return
        series = rr_distribution_series(self._trades)
        if series is None:
            self._rr_banner.show_message(
                "info", "RR distribution needs entry_price + sl (+ tp / pnl_points).")
            self.rr_hist.setVisible(False)
            return
        self._rr_banner.clear_message()
        self.rr_hist.setVisible(True)
        w = float(self._rr_width.value())
        start, end = rr_bin_edges(series, w)
        self.rr_hist.set_series(
            {"Planned RR": series["planned"], "Realised RR": series["realised"],
             "Realised RR (wins)": series["won"], "Break even": series["be"]},
            start, end, w)

    def _on_point_clicked(self, row: int) -> None:
        self._selected_row = row
        self._refresh_detail()

    def _refresh_detail(self) -> None:
        if self._trades is None or self._selected_row is None:
            return
        if self._selected_row >= len(self._trades):
            return   # stale click after a filter change
        self.trade_detail.show_trade(
            self._trades.iloc[self._selected_row],
            self.chart_controls.settings(),
            self._candles_folder, self._ticks_per_point)
