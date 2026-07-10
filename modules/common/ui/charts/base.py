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
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QToolButton, QToolTip


# ── time helpers ──────────────────────────────────────────────────────────────

def ny_epoch_seconds(values) -> "pd.Index":
    """tz-aware (or naive) datetimes -> NY-wall-clock-as-epoch float seconds.

    as_unit("ns") first: datetimes loaded from parquet come back at
    MICROsecond resolution (pandas 3 keeps arrow's unit), and int64 on a
    us-resolution index would yield epoch/1000 — a 1970s date axis."""
    idx = pd.DatetimeIndex(values)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.as_unit("ns").astype("int64") / 1e9


def date_axis() -> pg.DateAxisItem:
    """Bottom axis that renders our NY-wall-clock epoch values as NY time."""
    return pg.DateAxisItem(orientation="bottom", utcOffset=0)


# ── pan/zoom lock ─────────────────────────────────────────────────────────────

class ChartLockButton(QToolButton):
    """
    Padlock overlay in a chart's top-right corner. Every chart starts LOCKED
    (mouse pan/zoom disabled — accidental drags/wheels can't wreck the view;
    the wheel falls through to the page scroll instead). Clicking the padlock
    toggles interaction on the given ViewBox(es).
    """

    def __init__(self, host, viewboxes):
        super().__init__(host)
        self._viewboxes = viewboxes if isinstance(viewboxes, (list, tuple)) \
            else [viewboxes]
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            "QToolButton { background: rgba(23, 26, 33, 0.75); "
            "border: 1px solid #2a2f3a; border-radius: 6px; padding: 3px 7px; "
            "font-size: 13px; } "
            "QToolButton:hover { border-color: #3450e0; }")
        self.setChecked(True)          # checked == locked
        self.toggled.connect(self._apply)
        self._apply(True)
        host.installEventFilter(self)  # reposition on host resizes
        self.raise_()
        self.show()

    def _apply(self, locked: bool) -> None:
        for vb in self._viewboxes:
            vb.setMouseEnabled(x=not locked, y=not locked)
        self.setText("🔒" if locked else "🔓")
        self.setToolTip("Chart locked — click to enable pan/zoom" if locked
                        else "Chart unlocked — drag/wheel moves the view; "
                             "click to lock")

    def _reposition(self) -> None:
        host = self.parentWidget()
        self.adjustSize()
        self.move(host.width() - self.width() - 8, 8)
        self.raise_()

    def eventFilter(self, obj, event) -> bool:
        if event.type() in (QEvent.Resize, QEvent.Show):
            self._reposition()
        return False


def attach_lock_button(host, viewboxes) -> ChartLockButton:
    """Overlay a pan/zoom padlock (default locked) on a chart widget."""
    button = ChartLockButton(host, viewboxes)
    host._chart_lock_ref = button   # keep alive alongside the host
    return button


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
    attach_lock_button(plot, plot.getPlotItem().getViewBox())
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
