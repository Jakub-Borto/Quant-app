"""
Market Exposure (alpha/beta regression) — collapsed section with the 2x2
regression grid, the Qt analog of the old render_market_exposure. All
computation comes from modules.common.backend.benchmark (verbatim); the
per-cell markdown tables render via QLabel's Markdown support.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QWidget

import pandas as pd

from modules.common.backend.benchmark import (_ALPHA_BETA_TOOLTIP,
                                              _TSTAT_TOOLTIP,
                                              _regression_cell_md,
                                              load_asset_statistics,
                                              market_exposure_data)
from ..widgets import Caption, CollapsibleSection


def _md_label(text: str) -> QLabel:
    lbl = QLabel()
    lbl.setTextFormat(Qt.MarkdownText)
    lbl.setText(text)
    lbl.setWordWrap(True)
    return lbl


class ExposureSection(CollapsibleSection):
    def __init__(self, parent=None):
        super().__init__("Market Exposure (α/β regression)", expanded=False,
                         parent=parent)

    def update_exposure(self, trades: pd.DataFrame, asset: str,
                        tick_size: float, parquet_root) -> None:
        # rebuild content from scratch
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

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
        self.content_layout.addWidget(Caption(
            f"Excluded: {bench['n_roll']} roll days · "
            f"{bench['n_missing_settle']} missing settlement move · "
            f"{bench['n_missing_rth']} missing RTH · "
            f"{data['n_absent']} traded days absent from stats file"))

        hdr = QGridLayout()
        h1 = _md_label("**α / β — what they mean**")
        h1.setToolTip(_ALPHA_BETA_TOOLTIP)
        h2 = _md_label("**t-stats & the 2×2 grid**")
        h2.setToolTip(_TSTAT_TOOLTIP)
        hdr.addWidget(h1, 0, 0)
        hdr.addWidget(h2, 0, 1)
        self.content_layout.addLayout(hdr)

        grid = QGridLayout()
        grid.setHorizontalSpacing(28)
        grid.setVerticalSpacing(12)
        for r, sample_label in enumerate(["Days traded", "All days"]):
            for c, bench_label in enumerate(["Settlement", "RTH"]):
                cell = QWidget()
                from PySide6.QtWidgets import QVBoxLayout
                v = QVBoxLayout(cell)
                v.setContentsMargins(0, 0, 0, 0)
                v.addWidget(_md_label(f"**{sample_label} × {bench_label}**"))
                v.addWidget(_md_label(
                    _regression_cell_md(data["cells"][(sample_label, bench_label)])))
                grid.addWidget(cell, r, c)
        self.content_layout.addLayout(grid)
