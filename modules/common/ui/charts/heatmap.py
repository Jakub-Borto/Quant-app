"""
Optimizer heatmap — the pyqtgraph port of the old Plotly surface.

Pixel-identical colors: the RGBA image is precomputed per cell with the
VERBATIM colorscale math from modules.optimizer.backend.heatmap_model
(_colorscale_rgb + the gamma curve), so a cell's color matches the old chart
exactly. Everything else mirrors the old surface too:

- NaN/inf cells transparent (excluded from the scale);
- diagonal hatch overlay on cells below the min-trades threshold (3 segments
  per cell, degrading to 1 on heavily masked grids — same 900-segment rule);
- in-cell value labels with luminance-based contrast, shown/hidden and
  font-sized from the REAL on-screen cell size (re-evaluated on resize and
  zoom — the old fixed 1150-px width estimate hid labels on wide windows
  where they easily fit);
- axis tick labels at cell centers via _fmt_axis_value;
- square cells (aspect-locked viewbox);
- hover info = the verbatim 8-metric text (heatmap_model.build_hover_texts)
  in a CUSTOM overlay panel, NOT QToolTip — Qt tooltips auto-hide on a
  text-length timeout and on mouse moves over a QGraphicsView, which made the
  old hover flicker and vanish. The panel is a plain child QLabel that we
  show/move/hide ourselves, so it stays up exactly as long as a cell is
  hovered;
- the hovered cell pops (grows ~12% with an overshoot ease — an animated
  overlay rect filled with the cell's color);
- left-click on a cell -> cellClicked(i, j) for the drill-down, and the cell
  keeps a white-outlined enlarged marker; clicking it again deselects (no
  re-emit), and any rebuild clears it;
- a color bar labeled with the metric.

Cell (i, j) spans x [i, i+1] x y [j, j+1]; centers at +0.5. (The old Plotly
chart centered cells ON integer coords; this is a pure coordinate shift.)

Z-order: image 0, hatch 2, hover pop 6, selection 7, in-cell labels 10.
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEasingCurve, QEvent, QRectF, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QPainterPath
from PySide6.QtWidgets import (QGraphicsPathItem, QGraphicsRectItem, QLabel,
                               QVBoxLayout, QWidget)

from modules.optimizer.backend.heatmap_model import (_cell_text_color,
                                                     _colorscale_rgb,
                                                     _curved_colorscale,
                                                     _fmt_axis_value,
                                                     _fmt_cell,
                                                     build_hover_texts)
from .. import theme
from .base import attach_lock_button

HOVER_SCALE = 1.12    # how much the hovered cell grows
POP_MS      = 150     # pop animation duration
LABEL_MIN_CELL_PX = 30    # hide in-cell values below this on-screen cell size
LABEL_MAX_CELLS   = 2000  # don't build label items for absurdly large grids


class HeatmapChart(QWidget):
    cellClicked = Signal(int, int)   # (i, j) = (x index, y index)
    cellDeselected = Signal()        # the selected cell was clicked again

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
        self._rgba = None                          # cell colors (pop-rect fill)
        self._labels: list[pg.TextItem] = []       # in-cell value labels
        self._labels_visible: bool | None = None
        self._label_font_pt: int | None = None
        self._hover_cell: tuple[int, int] | None = None
        self._selected: tuple[int, int] | None = None
        self._hover_item: QGraphicsRectItem | None = None
        self._sel_item: QGraphicsRectItem | None = None

        # hover info panel — a plain child label we control (see module docstring)
        self._panel = QLabel(self._glw)
        self._panel.setTextFormat(Qt.RichText)
        self._panel.setAttribute(Qt.WA_TransparentForMouseEvents)  # never steal hover
        self._panel.setStyleSheet(
            f"background-color: {theme.SURFACE_2}; color: {theme.TEXT}; "
            f"border: 1px solid {theme.BORDER_LIGHT}; border-radius: 6px; "
            f"padding: 8px 10px; font-size: 12px;")
        self._panel.hide()

        # the pop: 0 -> 1 with an overshoot ease, applied as cell scale
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(POP_MS)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.OutBack)
        self._anim.valueChanged.connect(self._on_anim_tick)

        self._proxy = pg.SignalProxy(self._glw.scene().sigMouseMoved,
                                     rateLimit=30, slot=self._on_move)
        self._glw.scene().sigMouseClicked.connect(self._on_click)
        self._glw.viewport().installEventFilter(self)   # hide hover on Leave
        # label visibility depends on the on-screen cell size -> track zoom
        self._plot.getViewBox().sigRangeChanged.connect(self._update_labels)

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
        plot.clear()                               # also drops old highlight items
        self._rgba = rgba
        self._hover_cell = None
        self._selected = None
        self._panel.hide()
        self._anim.stop()

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
            hatch.setZValue(2)
            plot.addItem(hatch)

        # ── hover pop + selection marker (recreated: plot.clear() ate them) ──
        hover_pen = pg.mkPen(255, 255, 255, 170, width=1.4)
        hover_pen.setCosmetic(True)
        self._hover_item = QGraphicsRectItem()
        self._hover_item.setPen(hover_pen)
        self._hover_item.setZValue(6)
        self._hover_item.hide()
        plot.addItem(self._hover_item)

        sel_pen = pg.mkPen(255, 255, 255, 255, width=2.2)
        sel_pen.setCosmetic(True)
        self._sel_item = QGraphicsRectItem()
        self._sel_item.setPen(sel_pen)
        self._sel_item.setZValue(7)
        self._sel_item.hide()
        plot.addItem(self._sel_item)

        # ── in-cell labels: built for every cell (up to LABEL_MAX_CELLS), then
        #    shown/hidden + font-sized by _update_labels from the actual
        #    on-screen cell size (resize/zoom re-evaluate) ─────────────────────
        self._labels = []
        self._labels_visible = None                # force the first update
        self._label_font_pt = None
        if nx * ny <= LABEL_MAX_CELLS:
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
                    item.setZValue(10)             # stays readable over the pop
                    item.setVisible(False)         # _update_labels decides
                    plot.addItem(item)
                    self._labels.append(item)

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

        self._update_height()
        self._update_labels()
        return status

    def _update_height(self) -> None:
        """Pin the widget to the height the aspect-locked grid actually needs
        at the current width. A FIXED height (not a minimum) keeps the chart
        stable inside the scrollable page — with a minimum it would stretch
        into free space and then visibly shrink back the moment a sibling
        (the drill-down report) appears below it."""
        if self._nx == 0:
            return
        grid_w = max(200, self._glw.width() - 130)   # left axis + colorbar
        grid_h = grid_w * self._ny / self._nx        # aspect-locked cells
        h = int(min(max(320, grid_h + 100), 900))    # + title + bottom axis
        if self.minimumHeight() != h or self.maximumHeight() != h:
            self.setFixedHeight(h)

    def _update_labels(self, *_args) -> None:
        """Show/hide + font-size the in-cell labels from the REAL on-screen
        cell size. Hooked to viewbox range changes and widget resizes."""
        if not self._labels:
            return
        vb = self._plot.getViewBox()
        px_w, px_h = vb.viewPixelSize()            # view units per screen pixel
        if not px_w or not px_h:
            return
        cell_px = min(1.0 / px_w, 1.0 / px_h)      # cells are 1x1 view units
        visible = cell_px >= LABEL_MIN_CELL_PX
        font_pt = int(max(8, min(15, cell_px * 0.26)))
        if visible == self._labels_visible and \
                (not visible or font_pt == self._label_font_pt):
            return                                 # nothing changed — cheap out
        self._labels_visible = visible
        self._label_font_pt = font_pt
        for item in self._labels:
            if visible:
                f = item.textItem.font()
                f.setPointSize(font_pt)
                item.textItem.setFont(f)
            item.setVisible(visible)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_height()      # width changed -> recompute (no-op otherwise)
        self._update_labels()

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

    @staticmethod
    def _cell_rect(cell: tuple[int, int], scale: float) -> QRectF:
        """The cell square scaled around its own center (scale 1.0 = exact cell)."""
        i, j = cell
        half = scale / 2.0
        return QRectF(i + 0.5 - half, j + 0.5 - half, scale, scale)

    def _cell_color(self, cell: tuple[int, int]) -> QColor:
        i, j = cell
        r, g, b, a = (int(v) for v in self._rgba[j, i])
        return QColor(r, g, b) if a else QColor(38, 44, 61)   # blank cell -> surface gray

    def _on_anim_tick(self, t: float) -> None:
        if self._hover_item is not None and self._hover_cell is not None:
            scale = 1.0 + (HOVER_SCALE - 1.0) * float(t)
            self._hover_item.setRect(self._cell_rect(self._hover_cell, scale))

    def _set_hover_cell(self, cell: tuple[int, int] | None) -> None:
        if cell == self._hover_cell:
            return
        self._hover_cell = cell
        self._anim.stop()
        if self._hover_item is None:
            return
        # the selected cell already wears its own (bigger) marker — no pop on top
        if cell is None or cell == self._selected:
            self._hover_item.hide()
            return
        self._hover_item.setBrush(self._cell_color(cell))
        self._hover_item.setRect(self._cell_rect(cell, 1.0))
        self._hover_item.show()
        self._anim.start()

    def _move_panel(self, scene_pos) -> None:
        """Keep the info panel next to the cursor but inside the chart."""
        vp = self._glw.mapFromScene(scene_pos)
        self._panel.adjustSize()
        x, y = vp.x() + 16, vp.y() + 16
        if x + self._panel.width() > self._glw.width() - 4:
            x = vp.x() - self._panel.width() - 12
        if y + self._panel.height() > self._glw.height() - 4:
            y = vp.y() - self._panel.height() - 12
        self._panel.move(max(4, x), max(4, y))

    def _clear_hover(self) -> None:
        self._set_hover_cell(None)
        self._panel.hide()

    def _on_move(self, event) -> None:
        cell = self._cell_at(event[0])
        if cell is None or self._hover is None:
            self._clear_hover()
            return
        if cell != self._hover_cell:
            i, j = cell
            self._panel.setText(self._hover[j, i])
        self._set_hover_cell(cell)
        self._move_panel(event[0])
        self._panel.show()
        self._panel.raise_()

    def _set_selected(self, cell: tuple[int, int] | None) -> None:
        self._selected = cell
        if self._sel_item is None:
            return
        if cell is None:
            self._sel_item.hide()
            return
        self._sel_item.setBrush(self._cell_color(cell))
        self._sel_item.setRect(self._cell_rect(cell, HOVER_SCALE))
        self._sel_item.show()
        if self._hover_cell == cell and self._hover_item is not None:
            self._anim.stop()
            self._hover_item.hide()                # hand the pop over to the marker

    def _on_click(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        cell = self._cell_at(event.scenePos())
        if cell is None:
            return
        if cell == self._selected:             # click again -> deselect
            self._set_selected(None)
            self._hover_cell = None            # cursor is still on the cell:
            self._set_hover_cell(cell)         # give it back the hover pop
            self.cellDeselected.emit()         # listeners close the drill-down
            return
        self._set_selected(cell)
        self.cellClicked.emit(cell[0], cell[1])

    def eventFilter(self, obj, event) -> bool:
        # the scene stops reporting moves once the mouse leaves the view —
        # without this the panel + pop would freeze in place
        if obj is self._glw.viewport() and event.type() == QEvent.Leave:
            self._clear_hover()
        return False


def _rgba_to_qcolor(css: str):
    """'rgba(r,g,b,a)' (a in 0..1) -> QColor."""
    from PySide6.QtGui import QColor
    parts = css[5:-1].split(",")
    r, g, b = (int(p) for p in parts[:3])
    a = int(float(parts[3]) * 255)
    c = QColor(r, g, b)
    c.setAlpha(a)
    return c
