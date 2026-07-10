"""
Equity-curve charts.

EquityCurveChart — the backtester/optimizer report curve: cumulative_ticks vs
entry_time as line + clickable scatter (the old Plotly on_select click becomes
the pointClicked(int) signal → trade-detail drill-down), dashed zero line,
hover tooltip with date + cumulative ticks.

MultiLineEquityChart — analytics' dollar-equity charts: any number of labeled
curves (the 4-curve per-instance figure and the combined overlay), a dotted
"Starting equity" hline, legend, hover tooltip with the nearest point of the
nearest curve.
"""

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .base import HoverTooltip, make_plot, nearest_index, ny_epoch_seconds


class EquityCurveChart(QWidget):
    pointClicked = Signal(int)   # row index into the trades frame

    def __init__(self, parent=None):
        super().__init__(parent)
        self._plot = make_plot("Date", "Cumulative Ticks", datetime_x=True)
        self._plot.setMinimumHeight(360)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._plot)

        self._x = np.array([])
        self._y = np.array([])
        self._dates: list[str] = []
        self._scatter: pg.ScatterPlotItem | None = None
        HoverTooltip(self._plot, self._hover_text)

    def set_trades(self, trades: pd.DataFrame) -> None:
        self._plot.clear()
        self._x = np.asarray(ny_epoch_seconds(trades["entry_time"]), dtype=float)
        self._y = trades["cumulative_ticks"].to_numpy(dtype=float)
        self._dates = [str(pd.Timestamp(t)) for t in trades["entry_time"]]

        self._plot.addItem(pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen("gray", width=1, style=pg.QtCore.Qt.DashLine)))
        self._plot.plot(self._x, self._y, pen=pg.mkPen("#5b78f0", width=2))
        self._scatter = pg.ScatterPlotItem(
            x=self._x, y=self._y, size=7,
            brush=pg.mkBrush(91, 120, 240, 160), pen=None)
        self._scatter.sigClicked.connect(self._on_clicked)
        self._plot.addItem(self._scatter)
        self._plot.autoRange()

    # ── interactions ──────────────────────────────────────────────────────────
    def _on_clicked(self, _item, points) -> None:
        if len(points):
            self.pointClicked.emit(int(points[0].index()))

    def _hover_text(self, x: float, y: float) -> str | None:
        i = nearest_index(self._x, x)
        if i is None:
            return None
        return (f"<b>{self._dates[i]}</b><br>"
                f"Cumulative: {self._y[i]:.0f} ticks<br>"
                f"<span style='color:#98a0b3'>trade #{i + 1} — click for detail</span>")


class MultiLineEquityChart(QWidget):
    """Labeled dollar-equity curves + start-equity reference line."""

    def __init__(self, height: int = 360, parent=None):
        super().__init__(parent)
        self._plot = make_plot("Date", "Equity ($)", datetime_x=True)
        self._plot.setMinimumHeight(height)
        self._plot.addLegend(offset=(10, 10))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._plot)
        self._series: list[tuple[str, np.ndarray, np.ndarray]] = []
        HoverTooltip(self._plot, self._hover_text)

    def clear(self) -> None:
        self._plot.clear()
        legend = self._plot.getPlotItem().legend
        if legend is not None:
            legend.clear()
        self._series = []

    def add_series(self, label: str, x_datetimes, y_values, color: str,
                   width: float = 2.0, dash: bool = False) -> None:
        x = np.asarray(ny_epoch_seconds(x_datetimes), dtype=float)
        y = np.asarray(y_values, dtype=float)
        style = pg.QtCore.Qt.DashLine if dash else pg.QtCore.Qt.SolidLine
        self._plot.plot(x, y, pen=pg.mkPen(color, width=width, style=style),
                        name=label)
        self._series.append((label, x, y))

    def add_start_line(self, account_size: float) -> None:
        line = pg.InfiniteLine(
            pos=account_size, angle=0,
            pen=pg.mkPen("#888888", width=1, style=pg.QtCore.Qt.DotLine),
            label="Starting equity",
            labelOpts={"color": "#98a0b3", "position": 0.02})
        self._plot.addItem(line)

    def finish(self) -> None:
        self._plot.autoRange()

    def _hover_text(self, x: float, y: float) -> str | None:
        best = None
        for label, xs, ys in self._series:
            i = nearest_index(xs, x)
            if i is None:
                continue
            d = abs(ys[i] - y)
            if best is None or d < best[0]:
                best = (d, label, xs[i], ys[i])
        if best is None:
            return None
        _d, label, xi, yi = best
        stamp = pd.Timestamp(xi, unit="s")
        return f"<b>{label}</b><br>{stamp}<br>${yi:,.2f}"
