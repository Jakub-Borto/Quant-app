"""
Market Exposure (alpha/beta regression) — the 2x2 grid of regression cells,
each showing the coefficient table AND the scatter of daily strategy P&L
against the benchmark move with the fitted line through it.

All computation comes from modules.common.backend.benchmark (verbatim).

The coefficient tables are built as real QGridLayout rows, NOT Markdown
QLabels: QLabel's Markdown table renderer reports a size hint that ignores the
rendered table, so cells drew on top of each other and the whole section was
unreadable. Plain labels in a grid measure correctly at any width.
"""

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QVBoxLayout, QWidget)

from modules.common.backend.benchmark import (_ALPHA_BETA_TOOLTIP,
                                              _TSTAT_TOOLTIP,
                                              load_asset_statistics,
                                              market_exposure_data)
from .. import theme
from ..charts.scatter_fit import ScatterFitChart
from ..widgets import Caption
from .sections import ReportSection

_ROWS = (("α (ticks/day)", "alpha", "{:.2f}"),
         ("α annualized", "alpha_ann", "{:.0f}"),
         ("β", "beta", "{:.3f}"),
         ("t(α) HAC", "t_alpha", "{:.2f}"),
         ("t(β) HAC", "t_beta", "{:.2f}"),
         ("R²", "r2", "{:.3f}"),
         ("n", "n", "{:.0f}"))


def _stat_table(res: dict) -> QWidget:
    """The coefficient table as a real grid of labels."""
    box = QWidget()
    grid = QGridLayout(box)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(14)
    grid.setVerticalSpacing(3)
    for row, (label, key, fmt) in enumerate(_ROWS):
        name = QLabel(label)
        name.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 12px;")
        value = QLabel(fmt.format(res[key]))
        value.setStyleSheet("font-size: 12px; font-weight: 600;")
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(name, row, 0)
        grid.addWidget(value, row, 1)
    grid.setColumnStretch(1, 1)
    box.setFixedWidth(190)
    return box


def _cell(title: str, res: dict | None) -> QWidget:
    """One regression cell: title, then the table beside its scatter chart."""
    box = QFrame()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)

    heading = QLabel(title)
    heading.setStyleSheet("font-size: 13px; font-weight: 600;")
    lay.addWidget(heading)

    if res is None:
        note = Caption("insufficient data for a regression")
        lay.addWidget(note)
        return box

    row = QHBoxLayout()
    row.setSpacing(12)
    row.addWidget(_stat_table(res), alignment=Qt.AlignTop)
    chart = ScatterFitChart()
    chart.set_fit(res["x"], res["y"], res["alpha"], res["beta"])
    row.addWidget(chart, 1)
    lay.addLayout(row)
    return box


def _clear_layout(layout) -> None:
    """Delete every widget in a layout, DESCENDING into nested layouts.

    The obvious version only deletes `item.widget()` and silently skips items
    that are themselves layouts — so the 2x2 grid's contents survived every
    rebuild and stacked up (four more scatter charts per filter change).
    """
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        child = item.layout()
        if child is not None:
            _clear_layout(child)
            child.deleteLater()


class ExposureSection(ReportSection):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.content_layout = QVBoxLayout(self)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(10)

    def update_exposure(self, trades: pd.DataFrame, asset: str,
                        tick_size: float, parquet_root) -> None:
        _clear_layout(self.content_layout)      # rebuild from scratch

        stats = load_asset_statistics(parquet_root, asset)
        if stats is None:
            self.content_layout.addWidget(Caption(
                f"No statistics file for {asset}; regression unavailable."))
            return

        try:
            import statsmodels.api as sm  # noqa: F401 — availability check only
        except ImportError:
            self.content_layout.addWidget(Caption(
                "`statsmodels` is not installed; regression unavailable."))
            return

        data = market_exposure_data(trades, stats, tick_size)
        if data is None:
            self.content_layout.addWidget(Caption(
                "Statistics file has no overlap with the backtest window."))
            return

        bench = data["bench"]
        excluded = Caption(
            f"Excluded: {bench['n_roll']} roll days · "
            f"{bench['n_missing_settle']} missing settlement move · "
            f"{bench['n_missing_rth']} missing RTH · "
            f"{data['n_absent']} traded days absent from stats file")
        excluded.setToolTip(_ALPHA_BETA_TOOLTIP)
        self.content_layout.addWidget(excluded)

        legend = Caption("Each point is one trading day: benchmark move on X, "
                         "strategy P&L on Y. The line's slope is β and its "
                         "height at X=0 is α — a flat line means the returns "
                         "are not just market exposure.")
        legend.setToolTip(_TSTAT_TOOLTIP)
        self.content_layout.addWidget(legend)

        grid = QGridLayout()
        grid.setHorizontalSpacing(28)
        grid.setVerticalSpacing(16)
        for r, sample_label in enumerate(["Days traded", "All days"]):
            for c, bench_label in enumerate(["Settlement", "RTH"]):
                res = data["cells"][(sample_label, bench_label)]
                grid.addWidget(_cell(f"{sample_label} × {bench_label}", res),
                               r, c)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        self.content_layout.addLayout(grid)
