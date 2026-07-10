"""
Strategy Combiner IS/OOS path chart — the pyqtgraph port of the optimizer's
_render_path_chart: total merged ticks per set size k for the forward-
selection stage, one line+markers series for in-sample, one for the sealed
out-of-sample, a star + label at the OOS peak, hover tooltip with k/IS/OOS.
Both series share one unit (ticks) — one axis, never dual.
"""

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .base import HoverTooltip, make_plot

IS_COLOR, OOS_COLOR = "#1f77b4", "#ff7f0e"   # app's two-series pair (CVD-safe)


class CombinePathChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._plot = make_plot("set size k", "total ticks (merged, no-overlap)")
        self._plot.setMinimumHeight(340)
        self._plot.addLegend(offset=(10, 10))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._plot)
        self._fwd: pd.DataFrame | None = None
        HoverTooltip(self._plot, self._hover_text)

    def set_path(self, path_df: pd.DataFrame) -> None:
        """path_df: the combine run's path table (k / stage / is_ticks /
        oos_ticks / is_oos_peak columns) — forward stage is charted."""
        fwd = path_df[path_df["stage"] == "forward"].reset_index(drop=True)
        self._fwd = fwd
        plot = self._plot
        plot.clear()
        legend = plot.getPlotItem().legend
        if legend is not None:
            legend.clear()
        if fwd.empty:
            return

        k = fwd["k"].to_numpy(dtype=float)
        for col, color, name in (("is_ticks", IS_COLOR, "In-sample"),
                                 ("oos_ticks", OOS_COLOR, "Out-of-sample")):
            y = fwd[col].to_numpy(dtype=float)
            plot.plot(k, y, pen=pg.mkPen(color, width=2), name=name,
                      symbol="o", symbolSize=8, symbolBrush=pg.mkBrush(color),
                      symbolPen=None)

        peak = path_df[path_df["is_oos_peak"]]
        if len(peak):
            star = pg.ScatterPlotItem(
                x=peak["k"].to_numpy(dtype=float),
                y=peak["oos_ticks"].to_numpy(dtype=float),
                symbol="star", size=17, brush=pg.mkBrush(OOS_COLOR),
                pen=pg.mkPen(255, 255, 255, 230, width=2))
            plot.addItem(star)
            text = pg.TextItem("OOS peak", color=OOS_COLOR, anchor=(0.5, 1.4))
            text.setPos(float(peak["k"].iloc[0]), float(peak["oos_ticks"].iloc[0]))
            plot.addItem(text)
        plot.autoRange()

    def _hover_text(self, x: float, y: float) -> str | None:
        if self._fwd is None or self._fwd.empty:
            return None
        ks = self._fwd["k"].to_numpy(dtype=float)
        i = int(np.argmin(np.abs(ks - x)))
        row = self._fwd.iloc[i]
        return (f"<b>k = {int(row['k'])}</b><br>"
                f"IS: {row['is_ticks']:.0f} ticks<br>"
                f"OOS: {row['oos_ticks']:.0f} ticks")
