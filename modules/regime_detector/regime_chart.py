"""
RegimeChart — daily price line with background shading per contiguous run of
the selected regime state (one regime axis colours the chart at a time).

X is the RTH date as NY-wall-clock epoch (repo chart convention, display
only). Regions span ±half a day around each date so consecutive same-state
days merge into one continuous band. Clicking a point emits dayClicked(date)
for the Explore tab's day-detail table.
"""

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Signal
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from modules.common.ui.charts.base import (HoverTooltip, make_plot,
                                           nearest_index, ny_epoch_seconds,
                                           set_chart_height)
from modules.regime_detector.backend.schema import UNKNOWN_COLOR, UNKNOWN_STATE

_HALF_DAY_S = 43200.0
_REGION_ALPHA = 55          # background band opacity (0-255)


class _RegimeBands(pg.GraphicsObject):
    """ALL background bands as ONE graphics item, painted as ONE image.

    A per-run pg.LinearRegionItem looks natural but is an interactive item
    (two edge lines + scene event handling each) — a 16-year daily regime
    produces ~2000 of them and pan/hover grinds to a halt. Per-band
    fillRect() worked but still cost ~30 ms/frame at that count, and a
    batched QPainterPath was WORSE (Qt's rasterizer scans every subpath,
    ~1.4 s/frame). So: numpy rasterizes the visible bands into a 1-px-tall
    RGBA strip (one column per screen pixel) and paint() stretches it over
    the viewport in a single drawImage — ~1 ms regardless of band count."""

    def __init__(self):
        super().__init__()
        self._x0 = np.array([])            # band starts, sorted
        self._x1 = np.array([])            # band ends
        self._rgba = np.zeros((0, 4), dtype=np.uint8)
        self._strip = None                 # keeps the QImage buffer alive
        self.setZValue(-10)

    def set_bands(self, bands) -> None:
        """bands: sorted, non-overlapping (x0, x1, (r, g, b, a)) tuples."""
        self._x0 = np.array([b[0] for b in bands], dtype=float)
        self._x1 = np.array([b[1] for b in bands], dtype=float)
        self._rgba = np.array([b[2] for b in bands], dtype=np.uint8)
        self.prepareGeometryChange()
        self.update()

    # bands span the full visible y-range, whatever it currently is
    def _view_rect(self) -> QRectF:
        vb = self.getViewBox()
        return QRectF() if vb is None else vb.viewRect()

    def boundingRect(self) -> QRectF:
        return self._view_rect()

    def viewRangeChanged(self) -> None:
        self.prepareGeometryChange()
        self.update()

    def paint(self, painter, *_args) -> None:
        rect = self._view_rect()
        vb = self.getViewBox()
        if rect.isEmpty() or vb is None or not len(self._x0):
            return
        width_px = max(1, int(vb.width()))
        left, right = rect.left(), rect.right()
        # color of the band covering each pixel column's center (RGBA=0 gaps)
        centers = left + (np.arange(width_px) + 0.5) * (right - left) / width_px
        idx = np.searchsorted(self._x0, centers, side="right") - 1
        valid = (idx >= 0) & (centers <= self._x1[np.clip(idx, 0, None)])
        buf = np.zeros((1, width_px, 4), dtype=np.uint8)
        buf[0, valid] = self._rgba[idx[valid]]
        self._strip_bytes = buf.tobytes()      # QImage borrows, doesn't copy
        self._strip = QImage(self._strip_bytes, width_px, 1, 4 * width_px,
                             QImage.Format_RGBA8888)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(rect, self._strip)


class RegimeChart(QWidget):
    dayClicked = Signal(str)                     # YYYY-MM-DD

    def __init__(self, parent=None):
        super().__init__(parent)
        self._plot = make_plot("Date", "Price", datetime_x=True)
        set_chart_height(self._plot, 420)

        self._legend = QLabel("")
        self._legend.setTextFormat(pg.QtCore.Qt.TextFormat.RichText)
        legend_row = QHBoxLayout()
        legend_row.addWidget(self._legend)
        legend_row.addStretch()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addLayout(legend_row)
        lay.addWidget(self._plot)

        self._x = np.array([])
        self._y = np.array([])
        self._dates: list[str] = []
        self._states: list[str] = []
        HoverTooltip(self._plot, self._hover_text)

    def clear(self) -> None:
        self._plot.clear()
        self._legend.setText("")
        self._x, self._y = np.array([]), np.array([])
        self._dates, self._states = [], []

    def set_data(self, dates: pd.DatetimeIndex, prices, states,
                 colors: dict[str, str]) -> None:
        """dates: one Timestamp per day; states: that day's label for the
        selected regime column; colors: state -> hex from meta.json (the
        implicit 'unknown' grey is appended here)."""
        self._plot.clear()
        self._dates = [str(pd.Timestamp(d).date()) for d in dates]
        self._states = [str(s) for s in states]
        self._x = np.asarray(ny_epoch_seconds(dates), dtype=float)
        self._y = np.asarray(prices, dtype=float)
        colors = {**colors, UNKNOWN_STATE: UNKNOWN_COLOR}

        # background bands: one contiguous run per band, ALL drawn by a
        # single _RegimeBands item (kept out of autoRange — it always spans
        # the whole viewport)
        bands = []
        n = len(self._x)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and self._states[j + 1] == self._states[i]:
                j += 1
            color = colors.get(self._states[i], UNKNOWN_COLOR)
            bands.append((self._x[i] - _HALF_DAY_S, self._x[j] + _HALF_DAY_S,
                          _with_alpha(color)))
            i = j + 1
        band_item = _RegimeBands()
        band_item.set_bands(bands)
        self._plot.addItem(band_item, ignoreBounds=True)

        self._plot.plot(self._x, self._y, pen=pg.mkPen("#d5d9e4", width=1.6))
        scatter = pg.ScatterPlotItem(x=self._x, y=self._y, size=6, pen=None,
                                     brush=pg.mkBrush(213, 217, 228, 170))
        scatter.sigClicked.connect(self._on_clicked)
        self._plot.addItem(scatter)
        self._plot.autoRange()

        seen = list(dict.fromkeys(self._states))
        self._legend.setText("&nbsp;&nbsp;".join(
            f"<span style='color:{colors.get(s, UNKNOWN_COLOR)}'>■</span> {s}"
            for s in seen))

    # ── interactions ──────────────────────────────────────────────────────────
    def _on_clicked(self, _item, points) -> None:
        if len(points):
            self.dayClicked.emit(self._dates[int(points[0].index())])

    def _hover_text(self, x: float, y: float) -> str | None:
        i = nearest_index(self._x, x)
        if i is None:
            return None
        return (f"<b>{self._dates[i]}</b><br>"
                f"State: {self._states[i]}<br>"
                f"Price: {self._y[i]:g}<br>"
                f"<span style='color:#98a0b3'>click for day detail</span>")


def _with_alpha(hex_color: str):
    """'#rrggbb' -> (r, g, b, alpha) for the background bands."""
    h = hex_color.lstrip("#")
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        r, g, b = 106, 111, 122
    return (r, g, b, _REGION_ALPHA)
