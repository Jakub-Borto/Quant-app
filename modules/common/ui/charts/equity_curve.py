"""
Equity-curve charts.

EquityCurveChart — the backtester/optimizer report curve: cumulative_ticks vs
entry_time as line + clickable scatter (the old Plotly on_select click becomes
the pointClicked(int) signal → trade-detail drill-down), dashed zero line,
hover tooltip with date + cumulative ticks. A toggle button switches the
X axis between calendar time and plain trade number (no calendar gaps).

MultiLineEquityChart — analytics' dollar-equity charts: any number of labeled
curves (the 4-curve per-instance figure and the combined overlay), a dotted
"Starting equity" hline, legend, hover tooltip with the nearest point of the
nearest curve.
"""

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from .base import HoverTooltip, date_axis, make_plot, nearest_index, ny_epoch_seconds


class EquityCurveChart(QWidget):
    pointClicked = Signal(int)   # row index into the trades frame

    def __init__(self, parent=None):
        super().__init__(parent)
        self._plot = make_plot("Date", "Cumulative Ticks", datetime_x=True)
        self._plot.setMinimumHeight(360)

        # X-axis mode toggle: calendar time vs plain trade number (no gaps)
        self._trade_number_mode = False
        self._axis_btn = QPushButton("X axis: Date")
        self._axis_btn.setToolTip("Toggle the X axis between calendar time "
                                  "and trade number (removes calendar gaps)")
        self._axis_btn.clicked.connect(self._toggle_axis_mode)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self._axis_btn)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addLayout(btn_row)
        lay.addWidget(self._plot)

        self._trades: pd.DataFrame | None = None
        self._x = np.array([])
        self._y = np.array([])
        self._dates: list[str] = []
        self._scatter: pg.ScatterPlotItem | None = None
        HoverTooltip(self._plot, self._hover_text)

    def set_trades(self, trades: pd.DataFrame) -> None:
        self._trades = trades
        self._plot.clear()
        if self._trade_number_mode:
            self._x = np.arange(1, len(trades) + 1, dtype=float)
        else:
            self._x = np.asarray(ny_epoch_seconds(trades["entry_time"]),
                                 dtype=float)
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

    def _toggle_axis_mode(self) -> None:
        self._trade_number_mode = not self._trade_number_mode
        self._axis_btn.setText("X axis: Trade #" if self._trade_number_mode
                               else "X axis: Date")
        # swap the bottom axis item, then re-plot the stored trades
        if self._trade_number_mode:
            axis, label = pg.AxisItem(orientation="bottom"), "Trade #"
        else:
            axis, label = date_axis(), "Date"
        self._plot.getPlotItem().setAxisItems({"bottom": axis})
        self._plot.setLabel("bottom", label)
        self._plot.showGrid(x=True, y=True, alpha=0.18)   # grid lives on the axes
        if self._trades is not None:
            self.set_trades(self._trades)

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
        # (label, x, y, per-point extra hover lines or None)
        self._series: list[tuple[str, np.ndarray, np.ndarray, list | None]] = []
        HoverTooltip(self._plot, self._hover_text)

    def clear(self) -> None:
        self._plot.clear()
        legend = self._plot.getPlotItem().legend
        if legend is not None:
            legend.clear()
        self._series = []

    _STYLES = {"solid": pg.QtCore.Qt.SolidLine,
               "dash":  pg.QtCore.Qt.DashLine,
               "dot":   pg.QtCore.Qt.DotLine}

    # plotly's default categorical cycle — used when no color is given
    _PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    def add_series(self, label: str, x_datetimes, y_values, color: str | None = None,
                   width: float = 2.0, style: str = "solid",
                   hover_extra: list | None = None) -> None:
        """hover_extra: optional per-point extra tooltip line (e.g. contracts)."""
        if color is None:
            color = self._PALETTE[len(self._series) % len(self._PALETTE)]
        x = np.asarray(ny_epoch_seconds(x_datetimes), dtype=float)
        y = np.asarray(y_values, dtype=float)
        self._plot.plot(x, y, pen=pg.mkPen(color, width=width,
                                           style=self._STYLES[style]),
                        name=label)
        self._series.append((label, x, y, hover_extra))

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
        for label, xs, ys, extra in self._series:
            i = nearest_index(xs, x)
            if i is None:
                continue
            d = abs(ys[i] - y)
            if best is None or d < best[0]:
                best = (d, label, xs[i], ys[i], extra[i] if extra is not None else None)
        if best is None:
            return None
        _d, label, xi, yi, extra_line = best
        stamp = pd.Timestamp(xi, unit="s")
        text = f"<b>{label}</b><br>{stamp}<br>Equity: ${yi:,.2f}"
        if extra_line is not None:
            text += f"<br>{extra_line}"
        return text
