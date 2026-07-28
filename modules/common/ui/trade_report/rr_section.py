"""
RR Distribution — bin-width control + the overlaid planned/realised histogram.

Moved verbatim out of TradeReportPanel so it becomes an independently
orderable, collapsible section. The bin width is a live control: changing it
re-bins the currently held trades without touching the filter pipeline.
"""

import pandas as pd
from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QVBoxLayout

from modules.common.backend.trade_stats import (rr_bin_edges,
                                                rr_distribution_series)
from ..charts.histogram import OverlaidHistogram
from ..widgets import Banner
from .sections import ReportSection


class RRDistributionSection(ReportSection):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._trades: pd.DataFrame | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        row = QHBoxLayout()
        row.addWidget(QLabel("RR bin width"))
        self._width = QDoubleSpinBox()
        self._width.setRange(0.1, 1e6)
        self._width.setSingleStep(0.1)
        self._width.setDecimals(2)
        self._width.setValue(0.5)
        self._width.valueChanged.connect(lambda _=None: self.refresh())
        row.addWidget(self._width)
        row.addStretch()
        lay.addLayout(row)

        self._banner = Banner("info", "")
        lay.addWidget(self._banner)
        self.hist = OverlaidHistogram()
        lay.addWidget(self.hist)

    def set_trades(self, trades: pd.DataFrame) -> None:
        self._trades = trades
        self.refresh()

    def refresh(self) -> None:
        if self._trades is None:
            return
        series = rr_distribution_series(self._trades)
        if series is None:
            self._banner.show_message(
                "info", "RR distribution needs entry_price + sl (+ tp / pnl_points).")
            self.hist.setVisible(False)
            return
        self._banner.clear_message()
        self.hist.setVisible(True)
        w = float(self._width.value())
        start, end = rr_bin_edges(series, w)
        self.hist.set_series(
            {"Planned RR": series["planned"], "Realised RR": series["realised"],
             "Realised RR (wins)": series["won"], "Break even": series["be"]},
            start, end, w)
