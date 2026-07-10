"""
Chart plumbing shared by every pyqtgraph widget:

- make_plot(): a configured dark PlotWidget (grid, no default context menu
  surprises), optionally with a NY-time date axis;
- HoverTooltip: one SignalProxy-per-plot mouse-move hook that maps the cursor
  to data coordinates and shows the HTML your callback returns via QToolTip —
  this is how every chart keeps the old Plotly hover behavior;
- time-axis helpers: candle indexes / entry_time columns are tz-aware
  America/New_York; pyqtgraph needs floats. We plot NY WALL-CLOCK time as if
  it were UTC epoch (tz_localize(None) first) and pair it with
  DateAxisItem(utcOffset=0), so axis labels read NY session times (09:30 …).
  Display-only — persisted data is never touched.
"""

import pandas as pd
import pyqtgraph as pg
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QToolTip


# ── time helpers ──────────────────────────────────────────────────────────────

def ny_epoch_seconds(values) -> "pd.Index":
    """tz-aware (or naive) datetimes -> NY-wall-clock-as-epoch float seconds."""
    idx = pd.DatetimeIndex(values)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.astype("int64") / 1e9


def date_axis() -> pg.DateAxisItem:
    """Bottom axis that renders our NY-wall-clock epoch values as NY time."""
    return pg.DateAxisItem(orientation="bottom", utcOffset=0)


# ── plot factory ──────────────────────────────────────────────────────────────

def make_plot(x_label: str = "", y_label: str = "",
              datetime_x: bool = False) -> pg.PlotWidget:
    axis_items = {"bottom": date_axis()} if datetime_x else None
    plot = pg.PlotWidget(axisItems=axis_items)
    plot.showGrid(x=True, y=True, alpha=0.18)
    if x_label:
        plot.setLabel("bottom", x_label)
    if y_label:
        plot.setLabel("left", y_label)
    plot.setMenuEnabled(False)
    return plot


# ── hover tooltip hook ────────────────────────────────────────────────────────

class HoverTooltip:
    """
    Attach to a PlotWidget; `text_fn(x, y) -> str | None` receives the cursor
    position in DATA coordinates and returns the HTML to show (None hides).
    Rate-limited to 30 Hz. Keep a reference (attaching stores it on the plot).
    """

    def __init__(self, plot: pg.PlotWidget, text_fn):
        self._plot = plot
        self._text_fn = text_fn
        self._proxy = pg.SignalProxy(plot.scene().sigMouseMoved, rateLimit=30,
                                     slot=self._on_move)
        # keep this object alive as long as the plot lives
        plot._hover_tooltip_ref = self

    def _on_move(self, event) -> None:
        pos = event[0]
        vb = self._plot.getPlotItem().vb
        if not self._plot.sceneBoundingRect().contains(pos):
            QToolTip.hideText()
            return
        point = vb.mapSceneToView(pos)
        html = self._text_fn(point.x(), point.y())
        if html:
            QToolTip.showText(QCursor.pos(), html, self._plot)
        else:
            QToolTip.hideText()


def nearest_index(x_array, x: float) -> int | None:
    """Index of the value in a sorted float array nearest to x (None if empty)."""
    import numpy as np
    arr = np.asarray(x_array, dtype=float)
    if arr.size == 0:
        return None
    i = int(np.searchsorted(arr, x))
    if i <= 0:
        return 0
    if i >= arr.size:
        return arr.size - 1
    return i if abs(arr[i] - x) < abs(arr[i - 1] - x) else i - 1
