"""
Equity-curve charts.

EquityCurveChart — the backtester/optimizer report curve: cumulative_ticks vs
entry_time as line + clickable scatter (the old Plotly on_select click becomes
the pointClicked(int) signal → trade-detail drill-down), dashed zero line,
hover tooltip with date + cumulative ticks. A toggle button switches the
X axis between calendar time and plain trade number (no calendar gaps).
Joined underneath it (vertical splitter, shared X axis, one padlock) is the
drawdown chart: ticks below the running peak (peak starts at 0 — the one
definition is trade_stats.drawdown_series), red line over a translucent red
fill, max-drawdown marker, hover with the drawdown in ticks. A vertical
crosshair snapped to the nearest trade spans both charts, and the clicked
trade is marked on both.

MultiLineEquityChart — analytics' dollar-equity charts: any number of labeled
curves (the 4-curve per-instance figure and the combined overlay), a dotted
"Starting equity" hline, legend, hover tooltip with the nearest point of the
nearest curve.
"""

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QSplitter,
                               QVBoxLayout, QWidget)

from modules.common.trade_report.backend.trade_stats import drawdown_series

from .. import theme
from .base import (HoverTooltip, attach_lock_button, chart_min_height,
                   date_axis, make_plot, nearest_index, ny_epoch_seconds,
                   set_chart_height)

# design heights (scaled by chart_min_height): equity on top, drawdown below
EQUITY_DESIGN_PX = 360
DRAWDOWN_DESIGN_PX = 130

DD_COLOR = "#d64545"                  # = candlestick.DOWN_COLOR
DD_FILL = (214, 69, 69, 55)           # same red, much more transparent
_LEFT_AXIS_WIDTH = 64                 # both plots -> plot areas line up exactly


def _fmt_ticks(v: float) -> str:
    """-42.0 -> '−42' (true minus sign); never '−0'."""
    r = round(float(v))
    return f"{r:,}".replace("-", "−") if r else "0"


class EquityCurveChart(QWidget):
    pointClicked = Signal(int)   # row index into the trades frame, -1 = cleared

    def __init__(self, parent=None):
        super().__init__(parent)
        self._plot = make_plot("", "Cumulative Ticks", datetime_x=True, lock=False)
        self._dd_plot = make_plot("Date", "Drawdown (ticks)", datetime_x=True,
                                  lock=False)
        self._dd_plot.setXLink(self._plot)
        for p in (self._plot, self._dd_plot):
            p.getAxis("left").setWidth(_LEFT_AXIS_WIDTH)
            p.setMinimumHeight(chart_min_height(60))
        self._hide_equity_x_values()
        # one padlock drives both viewboxes (they are X-linked anyway)
        attach_lock_button(self._plot, [self._plot.getPlotItem().getViewBox(),
                                        self._dd_plot.getPlotItem().getViewBox()])

        # The SPLITTER carries the one fixed height (see set_chart_height for
        # why charts must resolve to a single height); dragging its handle only
        # moves the split between the two plots.
        self._splitter = QSplitter(Qt.Vertical)
        self._splitter.addWidget(self._plot)
        self._splitter.addWidget(self._dd_plot)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(6)
        self._splitter.setFixedHeight(
            chart_min_height(EQUITY_DESIGN_PX + DRAWDOWN_DESIGN_PX))
        self._splitter.setSizes([chart_min_height(EQUITY_DESIGN_PX),
                                 chart_min_height(DRAWDOWN_DESIGN_PX)])

        # X-axis mode toggle: calendar time vs plain trade number (no gaps)
        self._trade_number_mode = False
        self._axis_btn = QPushButton("X axis: Date")
        self._axis_btn.setToolTip("Toggle the X axis between calendar time "
                                  "and trade number (removes calendar gaps)")
        self._axis_btn.clicked.connect(self._toggle_axis_mode)
        # names the highlighted trade; clicking that point again clears it
        self._selected_label = QLabel("")
        self._selected_label.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: 12px;")
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._selected_label)
        btn_row.addStretch()
        btn_row.addWidget(self._axis_btn)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addLayout(btn_row)
        lay.addWidget(self._splitter)

        self._trades: pd.DataFrame | None = None
        self._x = np.array([])
        self._y = np.array([])
        self._dd = np.array([])
        self._dates: list[str] = []
        self._scatter: pg.ScatterPlotItem | None = None
        self._dd_marker: pg.ScatterPlotItem | None = None
        self._max_dd_text: pg.TextItem | None = None
        self._selected: int | None = None
        HoverTooltip(self._plot, self._hover_text)
        HoverTooltip(self._dd_plot, self._dd_hover_text)

        # crosshair: one vertical line per plot, both moved together
        self._vlines: list[pg.InfiniteLine] = []
        self._crosshair_proxies = [
            pg.SignalProxy(p.scene().sigMouseMoved, rateLimit=60,
                           slot=lambda ev, _p=p: self._on_mouse_moved(_p, ev[0]))
            for p in (self._plot, self._dd_plot)]
        for p in (self._plot, self._dd_plot):
            p.installEventFilter(self)          # Leave -> hide the crosshair

    def set_trades(self, trades: pd.DataFrame) -> None:
        self._trades = trades
        self._plot.clear()
        self._dd_plot.clear()
        self._max_dd_text = None
        if self._trade_number_mode:
            self._x = np.arange(1, len(trades) + 1, dtype=float)
        else:
            self._x = np.asarray(ny_epoch_seconds(trades["entry_time"]),
                                 dtype=float)
        self._y = trades["cumulative_ticks"].to_numpy(dtype=float)
        self._dd = drawdown_series(self._y)
        self._dates = [str(pd.Timestamp(t)) for t in trades["entry_time"]]

        zero_pen = pg.mkPen("gray", width=1, style=pg.QtCore.Qt.DashLine)
        self._plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=zero_pen))
        self._plot.plot(self._x, self._y, pen=pg.mkPen("#5b78f0", width=2))
        self._scatter = pg.ScatterPlotItem(
            x=self._x, y=self._y, size=7,
            brush=pg.mkBrush(91, 120, 240, 160), pen=None)
        self._scatter.sigClicked.connect(self._on_clicked)
        self._plot.addItem(self._scatter)

        # NOTE: the selection is drawn by recolouring the point INSIDE this
        # scatter (see _refresh_marker) rather than by adding a marker item on
        # top. pyqtgraph's ScatterPlotItem.mouseClickEvent accepts the event
        # whenever a point is under the cursor, so an overlay scatter swallows
        # the click on the highlighted trade and deselecting becomes
        # impossible — setAcceptedMouseButtons does not help, because the
        # scene's own hit-testing never consults it. (The drawdown plot has no
        # clickable items, so its selection marker CAN be an overlay.)

        self._draw_drawdown()
        self._add_crosshair_lines()

        self._plot.autoRange()
        self._fit_drawdown_y()
        self._refresh_marker()          # survives a re-plot (axis toggle)

    # ── drawdown pane ─────────────────────────────────────────────────────────
    def _draw_drawdown(self) -> None:
        dd_plot = self._dd_plot
        dd_plot.addItem(pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen("gray", width=1, style=pg.QtCore.Qt.DashLine)))
        if len(self._x):
            dd_plot.plot(self._x, self._dd, pen=pg.mkPen(DD_COLOR, width=2),
                         fillLevel=0, brush=pg.mkBrush(*DD_FILL))

        # clicked-trade mirror (overlay is safe here — see NOTE in set_trades)
        self._dd_marker = pg.ScatterPlotItem(
            size=11, brush=pg.mkBrush("#ffffff"), pen=pg.mkPen(DD_COLOR, width=1))
        dd_plot.addItem(self._dd_marker)

        if len(self._dd) and self._dd.min() < 0:
            i = int(np.argmin(self._dd))
            worst = self._dd[i]
            dd_plot.addItem(pg.ScatterPlotItem(
                x=[self._x[i]], y=[worst], size=9,
                brush=pg.mkBrush(DD_COLOR), pen=pg.mkPen("#ffffff", width=1)))
            # label on the side with more room, just above the deepest point
            anchor = (1.05, 1.0) if i > len(self._dd) / 2 else (-0.05, 1.0)
            self._max_dd_text = pg.TextItem(
                f"Max DD {_fmt_ticks(worst)} ticks",
                color=DD_COLOR, anchor=anchor,
                fill=pg.mkBrush(17, 20, 28, 215))  # readable over the line
            self._max_dd_text.setPos(self._x[i], worst)
            dd_plot.addItem(self._max_dd_text, ignoreBounds=True)

    def _fit_drawdown_y(self) -> None:
        lo = float(self._dd.min()) if len(self._dd) else 0.0
        lo = min(lo, -1.0)                      # never-in-drawdown: keep a band
        self._dd_plot.setYRange(lo * 1.1, -lo * 0.08, padding=0)

    # ── crosshair ─────────────────────────────────────────────────────────────
    def _add_crosshair_lines(self) -> None:
        pen = pg.mkPen((200, 205, 216, 110), width=1)
        self._vlines = []
        for p in (self._plot, self._dd_plot):
            line = pg.InfiniteLine(angle=90, movable=False, pen=pen)
            line.setVisible(False)
            line.setZValue(-5)                  # under the data
            p.addItem(line, ignoreBounds=True)
            self._vlines.append(line)

    def _set_crosshair(self, x: float | None) -> None:
        for line in self._vlines:
            if x is None:
                line.setVisible(False)
            else:
                line.setPos(x)
                line.setVisible(True)

    def _on_mouse_moved(self, plot: pg.PlotWidget, pos) -> None:
        if not plot.sceneBoundingRect().contains(pos) or not len(self._x):
            self._set_crosshair(None)
            return
        x = plot.getPlotItem().vb.mapSceneToView(pos).x()
        i = nearest_index(self._x, x)
        self._set_crosshair(None if i is None else float(self._x[i]))

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Leave and obj in (self._plot, self._dd_plot):
            self._set_crosshair(None)
        return False

    # ── axes ──────────────────────────────────────────────────────────────────
    def _hide_equity_x_values(self) -> None:
        """The equity plot keeps its bottom axis (it draws the vertical grid)
        but shows no values or label — the dates live on the drawdown pane."""
        axis = self._plot.getAxis("bottom")
        axis.setStyle(showValues=False)
        axis.setLabel("")
        axis.setHeight(6)

    def _toggle_axis_mode(self) -> None:
        self._trade_number_mode = not self._trade_number_mode
        self._axis_btn.setText("X axis: Trade #" if self._trade_number_mode
                               else "X axis: Date")
        # swap the bottom axis items of BOTH plots, then re-plot the trades
        label = "Trade #" if self._trade_number_mode else "Date"
        for p in (self._plot, self._dd_plot):
            axis = (pg.AxisItem(orientation="bottom") if self._trade_number_mode
                    else date_axis())
            p.getPlotItem().setAxisItems({"bottom": axis})
            p.showGrid(x=True, y=True, alpha=0.18)   # grid lives on the axes
        self._dd_plot.setLabel("bottom", label)
        self._hide_equity_x_values()
        if self._trades is not None:
            self.set_trades(self._trades)

    # ── selection ─────────────────────────────────────────────────────────────
    def select(self, row: int | None) -> None:
        """Highlight a trade (None clears). Emits pointClicked; -1 = cleared."""
        self._selected = row
        self._refresh_marker()
        self.pointClicked.emit(-1 if row is None else int(row))

    def selected(self) -> int | None:
        return self._selected

    def _refresh_marker(self) -> None:
        """Recolour the picked point white, in place, inside the data scatter,
        and mirror it on the drawdown pane."""
        if self._scatter is None:
            return
        n = len(self._x)
        row = self._selected
        valid = row is not None and 0 <= row < n

        brushes = [pg.mkBrush(91, 120, 240, 160)] * n
        sizes = [7] * n
        if valid:
            brushes[row] = pg.mkBrush("#ffffff")
            sizes[row] = 11
        self._scatter.setBrush(brushes)
        self._scatter.setSize(sizes)

        if self._dd_marker is not None:
            if valid:
                self._dd_marker.setData(x=[self._x[row]], y=[self._dd[row]])
            else:
                self._dd_marker.setData(x=[], y=[])

        self._selected_label.setText(
            "" if not valid else
            f"Trade #{row + 1} — {self._dates[row]}  ·  click it again to clear")

    # ── interactions ──────────────────────────────────────────────────────────
    def _on_clicked(self, _item, points) -> None:
        if not len(points):
            return
        row = int(points[0].index())
        # clicking the highlighted trade again deselects it
        self.select(None if row == self._selected else row)

    def _hover_text(self, x: float, y: float) -> str | None:
        i = nearest_index(self._x, x)
        if i is None:
            return None
        return (f"<b>{self._dates[i]}</b><br>"
                f"Cumulative: {self._y[i]:.0f} ticks<br>"
                f"<span style='color:#98a0b3'>trade #{i + 1} — click for detail</span>")

    def _dd_hover_text(self, x: float, y: float) -> str | None:
        i = nearest_index(self._x, x)
        if i is None:
            return None
        return f"Drawdown: {_fmt_ticks(self._dd[i])} ticks"


class MultiLineEquityChart(QWidget):
    """Labeled dollar-equity curves + start-equity reference line."""

    def __init__(self, height: int = 360, parent=None):
        super().__init__(parent)
        self._plot = make_plot("Date", "Equity ($)", datetime_x=True)
        set_chart_height(self._plot, height)
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
