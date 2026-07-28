"""
TradeReportPanel — the full shared report, rendered as an ordered stack of
sections whose order and visibility are a user preference (see sections.py and
layout_dialog.py). Used by the Backtester window and the Optimizer's cell
drill-down; both share one saved layout.

═══════════════════════════════════════════════════════════════════════════
 SECTION ORDER IS A VIEW CONCERN. IT NEVER MOVES A COMPUTATION.
═══════════════════════════════════════════════════════════════════════════

The host's _apply_filters is the model, and its order is fixed:

    trade-type filter -> recompute cumulative_ticks
      -> News table (deliberately sees PRE-day-filter trades)
      -> day-type filter -> recompute cumulative_ticks
      -> regime filter -> recompute cumulative_ticks
      -> panel.set_trades()

Moving News below the trades table in the layout dialog moves a widget. The
News table still shows pre-day-filter trades, because that is decided in
_apply_filters, not here.

The panel owns the VIEW for every section, including the ones whose widgets
the host builds (filter rows, news, trades table, actions, regimes). The host
constructs those, hands them over with attach_host_section(), then calls
build_sections(). That is what lets one user-ordered list span widgets that
live in different files.

API:
    attach_host_section(key, widget)   # before build_sections()
    build_sections()
    set_context(asset, tick_size, ticks_per_point, candles_folder, parquet_root)
    set_trades(filtered_trades)        # recomputes every panel-owned section
    set_report_visible(bool)           # what the host calls on an empty filter
"""

import pandas as pd
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..charts.equity_curve import EquityCurveChart
from ..widgets import pin_minimum_height
from .chart_controls import ChartViewControls
from .exit_section import ExitBreakdownSection
from .exposure_section import ExposureSection
from .metrics_section import MetricsSection
from .rr_section import RRDistributionSection
from .sections import SectionStack
from .trade_detail import TradeDetailView


class TradeReportPanel(QWidget):
    def __init__(self, settings=None, parent=None):
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
        lay.setSpacing(0)
        # same reason as SectionStack: the panel must grow with its stack
        # rather than let the stack be squeezed inside a stale height
        pin_minimum_height(self)

        self.sections = SectionStack(settings)
        lay.addWidget(self.sections)

        # panel-owned sections
        self.metrics = MetricsSection()
        self.exit_breakdown = ExitBreakdownSection()
        self.exposure = ExposureSection()
        self.equity = EquityCurveChart()
        self.equity.pointClicked.connect(self._on_point_clicked)
        self.chart_controls = ChartViewControls()
        self.chart_controls.settingsChanged.connect(self._refresh_detail)
        self.trade_detail = TradeDetailView()
        self.rr = RRDistributionSection()

        for key, widget in (("metrics", self.metrics),
                            ("exit_breakdown", self.exit_breakdown),
                            ("exposure", self.exposure),
                            ("equity", self.equity),
                            ("chart_controls", self.chart_controls),
                            ("trade_detail", self.trade_detail),
                            ("rr", self.rr)):
            self.sections.register(key, widget)

    # ── composition ───────────────────────────────────────────────────────────
    def attach_host_section(self, key: str, widget: QWidget) -> None:
        """Contribute a host-owned section (filters, news, trades table, …)."""
        self.sections.register(key, widget)

    def build_sections(self) -> None:
        """Materialise the stack. Call once, after every attach_host_section."""
        self.sections.build()
        self.sections.set_frame_visible("trade_detail", False)

    def set_report_visible(self, visible: bool) -> None:
        self.sections.set_report_visible(visible)

    def set_section_visible(self, key: str, visible: bool) -> None:
        self.sections.set_frame_visible(key, visible)

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
        self.sections.set_frame_visible("trade_detail", False)
        self.equity.select(None)        # a new filter invalidates the pick

        self.metrics.set_trades(trades)
        self.exit_breakdown.set_trades(trades)
        self.exposure.update_exposure(trades, self._asset, self._tick_size,
                                      self._parquet_root)
        self.equity.set_trades(trades)
        self.rr.set_trades(trades)

    # ── internals ─────────────────────────────────────────────────────────────
    def _on_point_clicked(self, row: int) -> None:
        """row < 0 means the selection was cleared (clicking the highlighted
        point again, or the chart's Clear button)."""
        if row < 0:
            self._selected_row = None
            self.trade_detail.clear()
            self.sections.set_frame_visible("trade_detail", False)
            return
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
        self.sections.set_frame_visible("trade_detail", True)
