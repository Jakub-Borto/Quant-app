"""
Single-trade candlestick chart (the trade-detail drill-down).

CandlestickItem draws OHLC candles from one QPicture (fast, standard
pyqtgraph pattern). TradeChart composes: candles + entry (blue solid) / SL
(red dash) / TP (green dash) lines spanning [entry, exit], translucent
red/green SL/TP zone rectangles, entry triangle / exit X markers, and a
hover tooltip showing the hovered candle's OHLC — the same elements as the
old Plotly build_trade_figure.
"""

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QPainter, QPicture
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .base import HoverTooltip, make_plot, nearest_index, ny_epoch_seconds

UP_COLOR   = "#26a269"
DOWN_COLOR = "#d64545"


class CandlestickItem(pg.GraphicsObject):
    """OHLC candles at x positions (epoch seconds), width in seconds."""

    def __init__(self, x, opens, highs, lows, closes, width: float):
        super().__init__()
        self._picture = QPicture()
        painter = QPainter(self._picture)
        w = width * 0.35
        for xi, o, h, l, c in zip(x, opens, highs, lows, closes):
            color = QColor(UP_COLOR if c >= o else DOWN_COLOR)
            painter.setPen(pg.mkPen(color, width=1))
            painter.drawLine(pg.QtCore.QPointF(xi, l), pg.QtCore.QPointF(xi, h))
            painter.setBrush(pg.mkBrush(color))
            top, bottom = max(o, c), min(o, c)
            painter.drawRect(QRectF(xi - w, bottom, 2 * w, top - bottom))
        painter.end()
        self._bounds = QRectF(self._picture.boundingRect())

    def paint(self, painter, *args):
        painter.drawPicture(0, 0, self._picture)

    def boundingRect(self):
        return self._bounds


class _ZoneRect(pg.GraphicsObject):
    """Translucent rectangle in data coordinates (the SL/TP zones)."""

    def __init__(self, x0, x1, y0, y1, color: str, alpha: int = 13):
        super().__init__()
        self._rect = QRectF(min(x0, x1), min(y0, y1),
                            abs(x1 - x0), abs(y1 - y0))
        c = QColor(color)
        c.setAlpha(alpha)   # ≈ the old 0.05 plotly fill opacity
        self._brush = pg.mkBrush(c)

    def paint(self, painter, *args):
        painter.setPen(pg.mkPen(None))
        painter.setBrush(self._brush)
        painter.drawRect(self._rect)

    def boundingRect(self):
        return self._rect


class TradeChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._plot = make_plot("Time", "Price", datetime_x=True)
        self._plot.setMinimumHeight(480)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._plot)
        self._x = np.array([])
        self._ohlc = None
        HoverTooltip(self._plot, self._hover_text)

    def set_trade(self, trade, chart_candles: pd.DataFrame,
                  entry_ts, exit_ts) -> None:
        self._plot.clear()
        x = np.asarray(ny_epoch_seconds(chart_candles.index), dtype=float)
        self._x = x
        self._ohlc = chart_candles[["open", "high", "low", "close"]]

        # candle width from the median bar spacing (1m data -> 60s)
        width = float(np.median(np.diff(x))) if len(x) > 1 else 60.0
        self._plot.addItem(CandlestickItem(
            x, chart_candles["open"].to_numpy(float),
            chart_candles["high"].to_numpy(float),
            chart_candles["low"].to_numpy(float),
            chart_candles["close"].to_numpy(float), width))

        ex0 = float(ny_epoch_seconds([entry_ts])[0])
        ex1 = float(ny_epoch_seconds([exit_ts])[0])

        def hseg(y, color, dash):
            style = pg.QtCore.Qt.DashLine if dash else pg.QtCore.Qt.SolidLine
            self._plot.plot([ex0, ex1], [y, y],
                            pen=pg.mkPen(color, width=1.4, style=style))

        # zones under the lines: red entry↔SL, green entry↔TP
        self._plot.addItem(_ZoneRect(ex0, ex1, trade["sl"], trade["entry_price"], "#ff0000"))
        self._plot.addItem(_ZoneRect(ex0, ex1, trade["entry_price"], trade["tp"], "#00ff00"))
        hseg(trade["entry_price"], "#4d7cff", dash=False)
        hseg(trade["sl"], "#d64545", dash=True)
        hseg(trade["tp"], "#26a269", dash=True)

        entry_symbol = "t1" if trade["direction"] == "long" else "t"  # ▲ / ▼
        self._plot.addItem(pg.ScatterPlotItem(
            x=[ex0], y=[float(trade["entry_price"])], symbol=entry_symbol,
            size=15, brush=pg.mkBrush("#4d7cff"), pen=pg.mkPen("#ffffff", width=0.5)))
        self._plot.addItem(pg.ScatterPlotItem(
            x=[ex1], y=[float(trade["exit_price"])], symbol="x",
            size=15, brush=pg.mkBrush("#ff9f1a"), pen=pg.mkPen("#ff9f1a")))

        self._plot.autoRange()

    def _hover_text(self, x: float, y: float) -> str | None:
        if self._ohlc is None:
            return None
        i = nearest_index(self._x, x)
        if i is None:
            return None
        row = self._ohlc.iloc[i]
        stamp = pd.Timestamp(self._x[i], unit="s").strftime("%H:%M")
        return (f"<b>{stamp}</b><br>O {row['open']:,.2f}  H {row['high']:,.2f}"
                f"<br>L {row['low']:,.2f}  C {row['close']:,.2f}")
