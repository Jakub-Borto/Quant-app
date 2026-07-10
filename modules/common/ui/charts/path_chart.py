"""
Strategy Combiner IS/OOS path chart — the pyqtgraph port of the optimizer's
_render_path_chart: cumulative merged ticks per selection step k, one line for
the in-sample path, one for the sealed out-of-sample path, a star marker +
label at the OOS peak, hover tooltip with k / IS / OOS values.
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
        self._plot = make_plot("k (selection step)", "Total ticks")
        self._plot.setMinimumHeight(340)
        self._plot.addLegend(offset=(10, 10))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._plot)
        self._df: pd.DataFrame | None = None
        HoverTooltip(self._plot, self._hover_text)

    def set_path(self, path_df: pd.DataFrame) -> None:
        """path_df: the combine run's path table with k / is_total_ticks /
        oos_total_ticks columns."""
        self._df = path_df.reset_index(drop=True)
        plot = self._plot
        plot.clear()
        legend = plot.getPlotItem().legend
        if legend is not None:
            legend.clear()

        k = self._df["k"].to_numpy(dtype=float)
        is_y = self._df["is_total_ticks"].to_numpy(dtype=float)
        oos_y = self._df["oos_total_ticks"].to_numpy(dtype=float)

        plot.plot(k, is_y, pen=pg.mkPen(IS_COLOR, width=2), name="IS total ticks")
        plot.plot(k, oos_y, pen=pg.mkPen(OOS_COLOR, width=2), name="OOS total ticks")

        peak = int(np.argmax(oos_y))
        star = pg.ScatterPlotItem(x=[k[peak]], y=[oos_y[peak]], symbol="star",
                                  size=16, brush=pg.mkBrush(OOS_COLOR),
                                  pen=pg.mkPen("#ffffff", width=0.5))
        plot.addItem(star)
        text = pg.TextItem("OOS peak", color=OOS_COLOR, anchor=(0.5, 1.3))
        text.setPos(float(k[peak]), float(oos_y[peak]))
        plot.addItem(text)
        plot.autoRange()

    def _hover_text(self, x: float, y: float) -> str | None:
        if self._df is None or self._df.empty:
            return None
        i = int(np.clip(round(x - self._df["k"].iloc[0]), 0, len(self._df) - 1))
        row = self._df.iloc[i]
        return (f"<b>k = {int(row['k'])}</b><br>"
                f"IS: {row['is_total_ticks']:.0f} ticks<br>"
                f"OOS: {row['oos_total_ticks']:.0f} ticks")
