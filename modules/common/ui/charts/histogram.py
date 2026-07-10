"""
Overlaid histogram — the RR-distribution chart.

Reproduces the old Plotly overlay: shared bins across all series (bin edges
from trade_stats.rr_bin_edges — verbatim math), one translucent bar series
per RR flavor with the same colors, dashed vline at 0, dotted red vline at
-1 labeled "full stop", legend, and a hover tooltip showing every series'
count in the hovered bin.
"""

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .base import HoverTooltip, make_plot

SERIES_COLORS = {
    "Planned RR":         "#1f77b4",
    "Realised RR":        "#ff7f0e",
    "Realised RR (wins)": "#2ca02c",
    "Break even":         "#9467bd",
}


class OverlaidHistogram(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._plot = make_plot("RR (R-multiple)", "Number of trades")
        self._plot.setMinimumHeight(420)
        self._plot.addLegend(offset=(10, 10))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._plot)
        self._edges = np.array([])
        self._counts: dict[str, np.ndarray] = {}
        HoverTooltip(self._plot, self._hover_text)

    def set_series(self, labeled_series: dict, start: float, end: float,
                   w: float) -> None:
        """labeled_series: {label: pandas Series or None}; start/end/w define
        the shared bins (from rr_bin_edges + the bin-width input)."""
        self._plot.clear()
        legend = self._plot.getPlotItem().legend
        if legend is not None:
            legend.clear()

        n_bins = max(1, int(round((end - start) / w)))
        self._edges = start + np.arange(n_bins + 1) * w
        self._counts = {}

        for label, series in labeled_series.items():
            if series is None or len(series) == 0:
                continue
            counts, _ = np.histogram(series.to_numpy(dtype=float),
                                     bins=self._edges)
            self._counts[label] = counts
            color = pg.mkColor(SERIES_COLORS.get(label, "#cccccc"))
            color.setAlphaF(0.6)
            # NOTE: name= makes plot.addItem register the legend entry itself —
            # never ALSO call legend.addItem or every series shows up twice
            bars = pg.BarGraphItem(
                x=(self._edges[:-1] + self._edges[1:]) / 2.0,
                height=counts, width=w * 0.97,
                brush=pg.mkBrush(color), pen=pg.mkPen(None), name=label)
            self._plot.addItem(bars)

        self._plot.addItem(pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen("gray", width=1, style=pg.QtCore.Qt.DashLine)))
        self._plot.addItem(pg.InfiniteLine(
            pos=-1, angle=90,
            pen=pg.mkPen("#d64545", width=1, style=pg.QtCore.Qt.DotLine),
            label="full stop",
            labelOpts={"color": "#d64545", "position": 0.95}))
        self._plot.autoRange()

    def _hover_text(self, x: float, y: float) -> str | None:
        if self._edges.size < 2 or not self._counts:
            return None
        if x < self._edges[0] or x > self._edges[-1]:
            return None
        i = min(int((x - self._edges[0]) // (self._edges[1] - self._edges[0])),
                self._edges.size - 2)
        lo, hi = self._edges[i], self._edges[i + 1]
        lines = [f"<b>RR {lo:g} … {hi:g}</b>"]
        for label, counts in self._counts.items():
            lines.append(f"{label}: {int(counts[i])}")
        return "<br>".join(lines)
