"""
The report heatmap's data model — pure (no Streamlit, no Altair): the kit's
heatmap component renders what this computes.

Everything numeric or colour-related is the Optimizer's own code:
  cells        compact.slice_grid -> heatmap_model._build_grid_arrays
               (the Explore grid builder; zero-trade cells stay NaN)
  hatching     HeatmapChart's rule: total_trades (NaN = 0) < min_trades
  colours      the Explore tab's range rule (finite values only; a flat range
               widened by ±0.5) + heatmap_model.color_positions /
               _colorscale_rgb — the same split + two curves, so one setting
               gives the same colours as the Optimizer
  labels       heatmap_model._fmt_cell / _cell_text_color; hover values
               _fmt_metric; axis labels _fmt_axis_value

The only report-specific choice is the DEFAULT split ("bad" bound) per
metric: break-even for ticks / avg trade / profit factor (as the
Optimizer), and 1.0 for both Sharpes (the pass criterion — the Optimizer
uses 0) so warm colours mean "good enough", not "positive".
"""

from dataclasses import dataclass

import numpy as np

from modules.common.backend import palette
from modules.optimizer.backend.heatmap_model import (METRIC_PIVOTS, _build_grid_arrays,
                                                     _cell_text_color, _colorscale_rgb,
                                                     _fmt_axis_value, _fmt_cell,
                                                     _fmt_metric, color_positions)
from modules.optimizer.backend.metrics import METRIC_LABELS, METRIC_ORDER

from .compact import slice_grid

# Default split ("bad" bound) per metric — see the module docstring.
# None = the middle of the value range (the Optimizer's own default there).
REPORT_SPLIT_DEFAULTS = {**METRIC_PIVOTS, "sharpe_trade": 1.0, "sharpe_daily": 1.0}

NEUTRAL_FILL = palette.SURFACE_2       # "colours off" cells
NEUTRAL_TEXT = palette.TEXT
BLANK_TEXT = "rgba(150,150,150,0.85)"  # label colour of a non-finite cell (∞)


def hex_rgb(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


@dataclass
class HeatmapData:
    metric: str
    arrays: dict              # {metric: [ny, nx]} — the Explore grid arrays
    x_param: str
    x_values: list
    y_param: str | None
    y_values: list            # ["—"] for a 1-parameter run
    slider_values: dict
    masked: np.ndarray        # [ny, nx] bool — hatched cells
    z: np.ndarray             # the metric with NaN for every non-finite value
    zmin: float
    zmax: float
    pivot: float
    norms: np.ndarray         # colour-scale positions 0..1
    good: float
    bad: float
    min_trades: int

    @property
    def no_finite(self) -> bool:
        return not np.isfinite(self.z).any()

    @property
    def all_masked(self) -> bool:
        return bool(self.masked.all())

    @property
    def slider_desc(self) -> str:
        return " · ".join(f"{p} = {_fmt_axis_value(v)}"
                          for p, v in self.slider_values.items())


def default_split(metric: str):
    return REPORT_SPLIT_DEFAULTS.get(metric)


def compute_heatmap(summary, metric: str, x_axis: dict, y_axis: dict | None,
                    slider_values: dict, min_trades: int, split=None,
                    good: float = 0.0, bad: float = 0.0) -> HeatmapData:
    """The heatmap for one slice. `split` None = the metric's report default
    (REPORT_SPLIT_DEFAULTS; None there = the middle of the range)."""
    if metric not in METRIC_ORDER:
        raise ValueError(f"unknown metric {metric!r}")
    x_param, x_values = x_axis["param"], list(x_axis["values"])
    y_param = y_axis["param"] if y_axis else None
    y_values = list(y_axis["values"]) if y_axis else ["—"]
    grid = slice_grid(summary, x_param, y_param, slider_values)
    arrays = _build_grid_arrays(grid, x_param, x_values, y_param,
                                y_values if y_axis else None)

    # range: exactly the Explore tab / HeatmapChart rule
    z = arrays[metric].astype(float).copy()
    z[~np.isfinite(z)] = np.nan
    finite = z[np.isfinite(z)]
    zmin = float(finite.min()) if finite.size else 0.0
    zmax = float(finite.max()) if finite.size else 1.0
    if zmin == zmax:
        zmin, zmax = zmin - 0.5, zmax + 0.5
    pivot = split if split is not None else default_split(metric)
    if pivot is None:
        pivot = zmin + (zmax - zmin) / 2
    norms = color_positions(z, zmin, zmax, pivot, good, bad)

    counts = np.nan_to_num(arrays["total_trades"], nan=0.0)
    masked = counts < min_trades
    return HeatmapData(metric, arrays, x_param, x_values, y_param, y_values,
                       dict(slider_values), masked, z, zmin, zmax, float(pivot),
                       norms, float(good), float(bad), int(min_trades))


def cell_records(hd: HeatmapData, colors_on: bool = True) -> list[dict]:
    """One dict per cell for the chart: index (i, j), fill (None = blank),
    in-cell text + colour, masked flag and every hover field (all metrics +
    the cell's parameter values)."""
    rows = []
    ny, nx = len(hd.y_values), len(hd.x_values)
    for j in range(ny):
        for i in range(nx):
            raw = hd.arrays[hd.metric][j, i]
            finite = bool(np.isfinite(hd.z[j, i]))
            if colors_on:
                fill = hex_rgb(_colorscale_rgb(float(hd.norms[j, i]))) if finite else None
                text_color = _cell_text_color(float(hd.norms[j, i])) if finite \
                    else BLANK_TEXT
            else:
                fill = NEUTRAL_FILL
                text_color = NEUTRAL_TEXT if finite else BLANK_TEXT
            rec = {
                "i": i, "j": j,
                "x_label": _fmt_axis_value(hd.x_values[i]),
                "y_label": _fmt_axis_value(hd.y_values[j]) if hd.y_param else "",
                "fill": fill,
                "text": _fmt_cell(raw, hd.metric),
                "text_color": text_color,
                "masked": bool(hd.masked[j, i]),
            }
            hover = {hd.x_param: _fmt_axis_value(hd.x_values[i])}
            if hd.y_param:
                hover[hd.y_param] = _fmt_axis_value(hd.y_values[j])
            for p, v in hd.slider_values.items():
                hover[p] = _fmt_axis_value(v)
            for m in METRIC_ORDER:
                hover[METRIC_LABELS[m]] = _fmt_metric(hd.arrays[m][j, i], m)
            hover["Note"] = (f"masked (< {hd.min_trades} trades)"
                             if hd.masked[j, i] else "")
            rec["hover"] = hover
            rows.append(rec)
    return rows


def colorbar_steps(hd: HeatmapData, n: int = 96) -> list[dict]:
    """[{v0, v1, fill}] strips from zmin to zmax with the heatmap's colours."""
    edges = np.linspace(hd.zmin, hd.zmax, n + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    pos = color_positions(mids, hd.zmin, hd.zmax, hd.pivot, hd.good, hd.bad)
    return [{"v0": float(a), "v1": float(b), "fill": hex_rgb(_colorscale_rgb(float(p)))}
            for a, b, p in zip(edges[:-1], edges[1:], pos)]
