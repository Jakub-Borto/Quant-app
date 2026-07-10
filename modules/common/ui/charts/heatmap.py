"""
Optimizer heatmap — the pyqtgraph port of the old Plotly surface.

Pixel-identical colors: the RGBA image is precomputed per cell with the
VERBATIM colorscale math from modules.optimizer.backend.heatmap_model
(_colorscale_rgb + the gamma curve), so a cell's color matches the old chart
exactly. Everything else mirrors the old surface too:

- NaN/inf cells transparent (excluded from the scale);
- diagonal hatch overlay on cells below the min-trades threshold (3 segments
  per cell, degrading to 1 on heavily masked grids — same 900-segment rule);
- in-cell value labels with luminance-based contrast, using the same
  cell-size/count thresholds (cell >= 34 px, nx*ny <= 600);
- axis tick labels at cell centers via _fmt_axis_value;
- square cells (aspect-locked viewbox);
- hover tooltip = the verbatim 8-metric text (heatmap_model.build_hover_texts);
- left-click on a cell -> cellClicked(i, j) for the drill-down;
- a color bar labeled with the metric.

Cell (i, j) spans x [i, i+1] x y [j, j+1]; centers at +0.5. (The old Plotly
chart centered cells ON integer coords; this is a pure coordinate shift.)
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QCursor, QPainterPath
from PySide6.QtWidgets import QGraphicsPathItem, QToolTip, QVBoxLayout, QWidget

from modules.optimizer.backend.heatmap_model import (_cell_text_color,
                                                     _colorscale_rgb,
                                                     _curved_colorscale,
                                                     _fmt_axis_value,
                                                     _fmt_cell,
                                                     build_hover_texts)
from .base import attach_lock_button


class HeatmapChart(QWidget):
    cellClicked = Signal(int, int)   # (i, j) = (x index, y index)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._glw = pg.GraphicsLayoutWidget()
        self._plot = self._glw.addPlot(row=0, col=0)
        self._plot.setMenuEnabled(False)
        self._plot.getViewBox().setAspectLocked(True)
        # default LOCKED like every chart; the padlock overlay can unlock it
        attach_lock_button(self._glw, self._plot.getViewBox())
        self._colorbar: pg.ColorBarItem | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._glw)

        self._nx = self._ny = 0
        self._hover = None
        self._proxy = pg.SignalProxy(self._glw.scene().sigMouseMoved,
                                     rateLimit=30, slot=self._on_move)
        self._glw.scene().sigMouseClicked.connect(self._on_click)

    # ── data ──────────────────────────────────────────────────────────────────
    def set_data(self, arrays: dict, metric: str, metric_label: str,
                 x_param: str, x_values: list, y_param, y_values: list,
                 slider_desc: str, min_trades: int,
                 color_gamma: float = 1.0) -> dict:
        """
        Rebuild the surface. Returns {"no_finite": bool, "all_masked": bool}
        so the window can show the old info/warning banners.
        """
        nx, ny = len(x_values), len(y_values)
        self._nx, self._ny = nx, ny
        counts = np.nan_to_num(arrays["total_trades"], nan=0.0)
        masked = counts < min_trades

        z = arrays[metric].copy()
        z[~np.isfinite(z)] = np.nan          # NaN AND inf sit outside the scale

        finite = z[np.isfinite(z)]
        status = {"no_finite": finite.size == 0, "all_masked": bool(masked.all())}

        zmin = float(finite.min()) if finite.size else 0.0
        zmax = float(finite.max()) if finite.size else 1.0
        if zmin == zmax:
            zmin, zmax = zmin - 0.5, zmax + 0.5
        span = zmax - zmin

        # ── RGBA image (verbatim color math incl. the gamma curve) ────────────
        rgba = np.zeros((ny, nx, 4), dtype=np.uint8)
        for j in range(ny):
            for i in range(nx):
                v = z[j, i]
                if not np.isfinite(v):
                    continue                          # transparent blank cell
                norm = max(0.0, min(1.0, (v - zmin) / span)) ** color_gamma
                r, g, b = _colorscale_rgb(norm)
                rgba[j, i] = (r, g, b, 255)

        plot = self._plot
        plot.clear()
        img = pg.ImageItem(rgba, axisOrder="row-major")
        img.setRect(QRectF(0, 0, nx, ny))
        plot.addItem(img)

        # ── hatch overlay for masked cells ────────────────────────────────────
        masked_ij = [(i, j) for j in range(ny) for i in range(nx) if masked[j, i]]
        full = [((0.0, 0.0), (1.0, 1.0)),
                ((0.0, 0.5), (0.5, 1.0)),
                ((0.5, 0.0), (1.0, 0.5))]
        segments = full if len(masked_ij) * 3 <= 900 else full[:1]
        if masked_ij:
            path = QPainterPath()
            for i, j in masked_ij:
                for (u0, v0), (u1, v1) in segments:
                    path.moveTo(i + u0, j + v0)
                    path.lineTo(i + u1, j + v1)
            hatch = QGraphicsPathItem(path)
            pen = pg.mkPen(255, 255, 255, 140, width=1.2)
            pen.setCosmetic(True)
            hatch.setPen(pen)
            plot.addItem(hatch)

        # ── in-cell labels (same visibility thresholds as the old chart) ─────
        cell = int(max(26, min(96, 1150 / nx, 640 / ny)))
        if cell >= 34 and nx * ny <= 600:
            font_size = int(max(9, min(15, cell * 0.26)))
            for j in range(ny):
                for i in range(nx):
                    raw = arrays[metric][j, i]
                    text = _fmt_cell(raw, metric)
                    if not text:
                        continue
                    color = ("rgba(150,150,150,0.85)" if not np.isfinite(raw)
                             else _cell_text_color(
                                 max(0.0, (z[j, i] - zmin) / span) ** color_gamma))
                    item = pg.TextItem(text, color=_rgba_to_qcolor(color),
                                       anchor=(0.5, 0.5))
                    item.setPos(i + 0.5, j + 0.5)
                    f = item.textItem.font()
                    f.setPointSize(font_size)
                    item.textItem.setFont(f)
                    plot.addItem(item)

        # ── axes ──────────────────────────────────────────────────────────────
        plot.setLabel("bottom", x_param)
        plot.setLabel("left", y_param or "")
        plot.getAxis("bottom").setTicks(
            [[(i + 0.5, _fmt_axis_value(v)) for i, v in enumerate(x_values)]])
        if y_param is not None:
            plot.getAxis("left").setTicks(
                [[(j + 0.5, _fmt_axis_value(v)) for j, v in enumerate(y_values)]])
        else:
            plot.getAxis("left").setTicks([[]])
        plot.setRange(xRange=(0, nx), yRange=(0, ny), padding=0)
        plot.setTitle(metric_label + (f"  ·  {slider_desc}" if slider_desc else ""),
                      size="12pt")

        # ── color bar ─────────────────────────────────────────────────────────
        stops = _curved_colorscale(color_gamma)
        cmap = pg.ColorMap(
            pos=[float(p) for p, _c in stops],
            color=[tuple(int(x) for x in c[4:-1].split(",")) for _p, c in stops])
        if self._colorbar is not None:
            self._glw.removeItem(self._colorbar)
        self._colorbar = pg.ColorBarItem(values=(zmin, zmax), colorMap=cmap,
                                         label=metric_label, interactive=False,
                                         width=14)
        self._glw.addItem(self._colorbar, row=0, col=1)

        # hover text matrix (verbatim 8-metric builder)
        self._hover = build_hover_texts(arrays, x_param, x_values, y_param,
                                        y_values, slider_desc, masked, min_trades)

        # size the widget so cells stay readable (aspect lock keeps them square)
        self.setMinimumHeight(min(max(320, cell * ny + 150), 860))
        return status

    # ── interactions ──────────────────────────────────────────────────────────
    def _cell_at(self, scene_pos) -> tuple[int, int] | None:
        if self._nx == 0:
            return None
        if not self._plot.sceneBoundingRect().contains(scene_pos):
            return None
        p = self._plot.getViewBox().mapSceneToView(scene_pos)
        i, j = int(np.floor(p.x())), int(np.floor(p.y()))
        if 0 <= i < self._nx and 0 <= j < self._ny:
            return i, j
        return None

    def _on_move(self, event) -> None:
        cell = self._cell_at(event[0])
        if cell is None or self._hover is None:
            QToolTip.hideText()
            return
        i, j = cell
        QToolTip.showText(QCursor.pos(), self._hover[j, i], self._glw)

    def _on_click(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        cell = self._cell_at(event.scenePos())
        if cell is not None:
            self.cellClicked.emit(cell[0], cell[1])


def _rgba_to_qcolor(css: str):
    """'rgba(r,g,b,a)' (a in 0..1) -> QColor."""
    from PySide6.QtGui import QColor
    parts = css[5:-1].split(",")
    r, g, b = (int(p) for p in parts[:3])
    a = int(float(parts[3]) * 255)
    c = QColor(r, g, b)
    c.setAlpha(a)
    return c
