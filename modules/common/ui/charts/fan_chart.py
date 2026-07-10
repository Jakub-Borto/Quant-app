"""
Monte-Carlo fan chart — the pyqtgraph port of the old _build_fan_chart.

Same visual grammar and the same data conventions:
- 1σ/2σ/3σ percentile bands (constants from modules.monte_carlo.backend.stats),
  outermost drawn first, computed from the FULL equity_matrix;
- up to SAMPLE_PATH_COUNT faded sample paths (same rng seed-0 selection over
  the non-featured pool), drawn as ONE NaN-separated PlotDataItem
  (connect="finite") — fast for 200×N;
- featured paths (colored) + an end label at each path's LAST FINITE point
  (prop-firm line matrices are NaN-truncated at pass/fail);
- median (white, 2.5), band-final dotted hlines, optional ruin floor /
  target hlines, dotted starting-equity line;
- optional y cap (0 .. y_max) for the "Cap at 3× account" toggle;
- hover tooltip: trade step + median and band values at that step.
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

from modules.monte_carlo.backend.stats import (BAND_ALPHAS, BAND_COLOR,
                                               BAND_LABELS, BAND_PERCENTILES,
                                               FEATURED_COLORS,
                                               SAMPLE_PATH_COUNT)
from .base import HoverTooltip, make_plot

_BAND_RGB = tuple(int(c) for c in BAND_COLOR.split(","))

# band-final hline styles (color alpha, dash, label) — mirrors the old chart
_BAND_LINE_STYLES = [
    (0.7, pg.QtCore.Qt.DotLine,     "1σ"),
    (0.5, pg.QtCore.Qt.DashLine,    "2σ"),
    (0.3, pg.QtCore.Qt.DashDotLine, "3σ"),
]


class FanChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._plot = make_plot("Trade #", "Equity ($)")
        self._plot.setMinimumHeight(480)
        self._plot.addLegend(offset=(10, 10))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._plot)
        self._median = np.array([])
        self._bands: list[tuple[str, np.ndarray, np.ndarray]] = []
        HoverTooltip(self._plot, self._hover_text)

    def set_data(self, equity_matrix: np.ndarray, account_size: float,
                 featured: dict[str, int], ruin_threshold: float | None,
                 y_max: float | None = None, band_finals: dict | None = None,
                 line_matrix: np.ndarray | None = None,
                 target: float | None = None,
                 x_label: str = "Trade #") -> None:
        plot = self._plot
        plot.clear()
        legend = plot.getPlotItem().legend
        if legend is not None:
            legend.clear()
        plot.setLabel("bottom", x_label)

        lines = line_matrix if line_matrix is not None else equity_matrix
        n_paths, n_steps = equity_matrix.shape
        x = np.arange(n_steps, dtype=float)

        # ── σ bands (outermost first so inner bands render on top) ────────────
        self._bands = []
        for (lo_pct, hi_pct), alpha, label in zip(
            reversed(BAND_PERCENTILES), reversed(BAND_ALPHAS),
            reversed(BAND_LABELS),
        ):
            lo = np.percentile(equity_matrix, lo_pct, axis=0)
            hi = np.percentile(equity_matrix, hi_pct, axis=0)
            lo_c = pg.PlotDataItem(x, lo, pen=None)
            hi_c = pg.PlotDataItem(x, hi, pen=None)
            brush = pg.mkBrush(*_BAND_RGB, int(alpha * 255))
            fill = pg.FillBetweenItem(hi_c, lo_c, brush=brush)
            plot.addItem(fill)
            if legend is not None:
                # NEVER hand the FillBetweenItem itself to the legend — its
                # ItemSample crashes the paint (access violation). A detached
                # PlotDataItem with a fill swatch renders the same legend entry.
                proxy = pg.PlotDataItem(pen=pg.mkPen(*_BAND_RGB, 160),
                                        fillLevel=0, fillBrush=brush)
                legend.addItem(proxy, label)
            self._bands.append((label, lo, hi))
        self._bands.reverse()   # tooltip reads 1σ→3σ order

        # ── faded sample paths as one NaN-joined item ─────────────────────────
        featured_indices = set(featured.values())
        sample_pool = [i for i in range(n_paths) if i not in featured_indices]
        rng = np.random.default_rng(seed=0)
        sampled = rng.choice(sample_pool,
                             size=min(SAMPLE_PATH_COUNT, len(sample_pool)),
                             replace=False)
        if len(sampled):
            xs = np.concatenate([np.append(x, np.nan) for _ in sampled])
            ys = np.concatenate([np.append(lines[i], np.nan) for i in sampled])
            plot.addItem(pg.PlotDataItem(
                xs, ys, connect="finite",
                pen=pg.mkPen(180, 180, 180, 20, width=1)))

        # ── featured paths + end labels ───────────────────────────────────────
        for label, idx in featured.items():
            color = FEATURED_COLORS.get(label, "#ffffff")
            eq = np.asarray(lines[idx], dtype=float)
            plot.plot(x, eq, pen=pg.mkPen(color, width=1.5), name=label,
                      connect="finite")
            finite = np.flatnonzero(np.isfinite(eq))
            end_i = int(finite[-1]) if len(finite) else n_steps - 1
            text = pg.TextItem(f" {label}", color=color, anchor=(0, 0.5))
            text.setPos(float(end_i), float(eq[end_i]))
            plot.addItem(text)

        # ── median ────────────────────────────────────────────────────────────
        self._median = np.median(equity_matrix, axis=0)
        plot.plot(x, self._median, pen=pg.mkPen("#ffffff", width=2.5),
                  name="Median")

        # ── band-final hlines ─────────────────────────────────────────────────
        if band_finals:
            for (lo_pct, hi_pct), (alpha, dash, label) in zip(
                BAND_PERCENTILES, _BAND_LINE_STYLES
            ):
                lo_val, hi_val = band_finals[(lo_pct, hi_pct)]
                color = pg.mkColor(*_BAND_RGB, int(alpha * 255))
                for val, side in ((hi_val, "↑"), (lo_val, "↓")):
                    if y_max is not None and val > y_max:
                        continue
                    plot.addItem(pg.InfiniteLine(
                        pos=val, angle=0,
                        pen=pg.mkPen(color, width=1, style=dash),
                        label=f"{label} {side}  ${val:,.0f}",
                        labelOpts={"color": color, "position": 0.97}))

        # ── ruin floor / target / starting equity ────────────────────────────
        if ruin_threshold is not None:
            floor = account_size * ruin_threshold
            plot.addItem(pg.InfiniteLine(
                pos=floor, angle=0,
                pen=pg.mkPen(255, 80, 80, 150, width=1,
                             style=pg.QtCore.Qt.DashLine),
                label=f"Ruin floor ${floor:,.0f}",
                labelOpts={"color": pg.mkColor(255, 80, 80, 200),
                           "position": 0.03}))
        if target is not None:
            plot.addItem(pg.InfiniteLine(
                pos=target, angle=0,
                pen=pg.mkPen(80, 220, 120, 180, width=1,
                             style=pg.QtCore.Qt.DashLine),
                label=f"Target ${target:,.0f}",
                labelOpts={"color": pg.mkColor(80, 220, 120, 230),
                           "position": 0.90}))
        plot.addItem(pg.InfiniteLine(
            pos=account_size, angle=0,
            pen=pg.mkPen(255, 255, 255, 50, width=1,
                         style=pg.QtCore.Qt.DotLine)))

        # ── ranges ────────────────────────────────────────────────────────────
        plot.setXRange(0, n_steps - 1, padding=0.02)
        if y_max is not None:
            plot.setYRange(0, y_max, padding=0)
        else:
            plot.enableAutoRange(axis="y")

    def _hover_text(self, x: float, y: float) -> str | None:
        if self._median.size == 0:
            return None
        i = int(round(x))
        if i < 0 or i >= self._median.size:
            return None
        lines = [f"<b>Trade #{i}</b>",
                 f"Median: ${self._median[i]:,.0f}"]
        for label, lo, hi in self._bands:
            lines.append(f"{label}: ${lo[i]:,.0f} – ${hi[i]:,.0f}")
        return "<br>".join(lines)
