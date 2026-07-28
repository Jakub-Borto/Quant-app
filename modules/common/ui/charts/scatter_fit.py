"""
ScatterFitChart — daily strategy P&L against a benchmark move, with the fitted
regression line drawn through it.

This is the picture behind the α/β table: every point is one trading day, the
line's SLOPE is β (how much the strategy moves with the market) and its
INTERCEPT at x=0 is α (what the strategy makes on a flat market). A cloud with
no tilt means the strategy is market-neutral; a visible tilt means it is
partly just long or short exposure.
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .base import HoverTooltip, make_plot, set_chart_height


class ScatterFitChart(QWidget):
    def __init__(self, height: int = 190, parent=None):
        super().__init__(parent)
        self._plot = make_plot("Benchmark move (ticks/day)",
                               "Strategy (ticks/day)")
        # FIXED, not just a minimum. pyqtgraph's PlotWidget inherits
        # QGraphicsView's 640x480 sizeHint, so a plot whose minimum is BELOW
        # 480 gives the layout two valid answers — it lays out at the minimum
        # first and jumps to the hint a pass later, which is a visible
        # two-stage expand when this section is opened (measured: the exposure
        # section going 630 -> 1076 and the whole window resizing twice).
        # Every other chart in the app has a minimum above 480, so this is the
        # only one that needs pinning.
        set_chart_height(self._plot, height)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._plot)
        self._x = np.array([])
        self._y = np.array([])
        HoverTooltip(self._plot, self._hover_text)

    def set_fit(self, x, y, alpha: float, beta: float) -> None:
        self._plot.clear()
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        if self._x.size == 0:
            return

        # zero lines: where the market is flat / the strategy breaks even
        pen = pg.mkPen(120, 128, 145, 90, width=1, style=pg.QtCore.Qt.DashLine)
        self._plot.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pen))
        self._plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pen))

        self._plot.addItem(pg.ScatterPlotItem(
            x=self._x, y=self._y, size=6, pen=None,
            brush=pg.mkBrush(91, 120, 240, 130)))

        lo, hi = float(self._x.min()), float(self._x.max())
        if hi == lo:
            lo, hi = lo - 1.0, hi + 1.0
        fit_x = np.array([lo, hi])
        self._plot.plot(fit_x, alpha + beta * fit_x,
                        pen=pg.mkPen("#e8b04b", width=2))
        self._plot.autoRange()

    def _hover_text(self, x: float, _y: float) -> str | None:
        if self._x.size == 0:
            return None
        i = int(np.argmin(np.abs(self._x - x)))
        return (f"benchmark {self._x[i]:+.0f} ticks<br>"
                f"strategy {self._y[i]:+.0f} ticks")
