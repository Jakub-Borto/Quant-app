"""
Heatmap data model for the Optimizer's Explore tab.

Extracted verbatim from legacy_streamlit/views/optimizer.py — everything the
heatmap needs that is NOT rendering: the colorscale and its gamma curve, cell
/axis/metric formatting, the [ny, nx] grid-array builder and the per-cell
hover text. The pyqtgraph widget (modules/common/ui/charts/heatmap.py) turns
these into pixels; because it uses _colorscale_rgb / _curved_colorscale
directly, its colors are identical to the old Plotly surface.
"""

import numpy as np
import pandas as pd

from .metrics import METRIC_LABELS, METRIC_ORDER

# Assets whose day buckets are least meaningful — the FF calendar is USD-only.
NON_US_CALENDAR_ASSETS = {"6E", "6J", "6B", "6C"}

COMBO_CONFIRM_THRESHOLD = 2000
MIN_TRADES_DEFAULT = 10

# low -> high: dark blue -> blue -> yellow -> orange -> red
HEATMAP_COLORSCALE = [
    [0.00, "rgb(4,32,79)"],
    [0.25, "rgb(43,108,169)"],
    [0.50, "rgb(244,208,63)"],
    [0.75, "rgb(232,135,30)"],
    [1.00, "rgb(192,57,43)"],
]


def _fmt_metric(value, metric: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if isinstance(value, float) and np.isinf(value):
        return "∞"
    if metric == "total_trades":
        return f"{int(value)}"
    if metric == "total_ticks":
        return f"{value:.0f}"
    return f"{value:.2f}"


def _fmt_axis_value(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def _build_grid_arrays(grid: pd.DataFrame, x_param, x_values, y_param, y_values):
    """
    Metric grid (indexed by cell cols) -> {metric: [ny, nx] array}, reindexed
    onto the FULL cartesian product so zero-trade cells exist as NaN rows.
    """
    nx = len(x_values)
    ny = len(y_values) if y_param is not None else 1   # 1-D grid: single row
    if y_param is not None:
        full_idx = pd.MultiIndex.from_product([x_values, y_values],
                                              names=[x_param, y_param])
    else:
        full_idx = pd.Index(x_values, name=x_param)
    g = grid.reindex(full_idx)

    arrays = {}
    for metric in METRIC_ORDER:
        vals = g[metric].to_numpy(dtype=float)
        # from_product varies y fastest -> reshape (nx, ny), transpose to [ny, nx]
        arrays[metric] = vals.reshape(nx, ny).T if y_param is not None else vals[None, :]
    return arrays


def _colorscale_rgb(v: float) -> tuple:
    """RGB of HEATMAP_COLORSCALE at normalized position v in [0, 1]."""
    stops = [(pos, tuple(int(c) for c in color[4:-1].split(",")))
             for pos, color in HEATMAP_COLORSCALE]
    v = min(1.0, max(0.0, v))
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if v <= p1:
            f = 0.0 if p1 == p0 else (v - p0) / (p1 - p0)
            return tuple(round(a + (b - a) * f) for a, b in zip(c0, c1))
    return stops[-1][1]


def _cell_text_color(v_norm: float) -> str:
    """Black on light cells (the yellow/orange band), white on dark ones."""
    r, g, b = _colorscale_rgb(v_norm)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "rgba(20,20,20,0.9)" if luminance > 140 else "rgba(255,255,255,0.92)"


# The natural split value per metric (the default of the Explore tab's "Split at"; all eight are higher-is-better).
# None = no natural split -> the middle of the value range.
METRIC_PIVOTS = {
    "total_ticks": 0.0, "avg_trade": 0.0, "sharpe_trade": 0.0, "sharpe_daily": 0.0,
    "profit_factor": 1.0, "win_rate": 50.0, "win_rate_be": 50.0, "total_trades": None,
}
CURVE_LIMIT = 4.0          # curve exponent e in [-4, 4] -> gamma = 2**e (1/16 .. 16)


def color_positions(z, zmin: float, zmax: float, pivot, good: float = 0.0,
                    bad: float = 0.0):
    """Value(s) -> position on the colorscale [0, 1]: a SPLIT value and one
    curve per side.

    The split (`pivot`; METRIC_PIVOTS gives the natural one: 0 ticks, PF 1,
    50 % win rate; None = the range's middle) gets the scale's MIDDLE color
    (0.5, yellow). Values above it run 0.5 -> 1 (orange, red), values below it
    0.5 -> 0 (blue, dark blue): u = distance from the split as a share of the
    way to that side's extreme, shaped as u ** 2**curve (`good` above, `bad`
    below). curve 0 = linear; > 0 = exp-like, the strongest colors only at the
    most extreme cells; < 0 = log-like, strong colors reach cells close to the
    split. A split outside [zmin, zmax] leaves every cell on one side: only
    that half of the colors is used. With the split at the range's middle and
    both curves 0 this is the plain linear scale."""
    z = np.asarray(z, dtype=float)
    span = zmax - zmin
    if span <= 0:
        return np.full(z.shape, 0.5)
    if pivot is None:
        pivot = zmin + span / 2
    out = np.empty(z.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        hi = z >= pivot
        up = zmax - pivot
        u = np.clip((z - pivot) / up, 0.0, 1.0) if up > 0 else np.zeros(z.shape)
        out[hi] = 0.5 + 0.5 * u[hi] ** (2.0 ** good)
        down = pivot - zmin
        d = np.clip((pivot - z) / down, 0.0, 1.0) if down > 0 else np.zeros(z.shape)
        out[~hi] = 0.5 - 0.5 * d[~hi] ** (2.0 ** bad)
    return np.clip(out, 0.0, 1.0)


def two_curve_colorscale(zmin: float, zmax: float, pivot, good: float = 0.0,
                         bad: float = 0.0, n: int = 129) -> list:
    """Colorbar stops ([value fraction, "rgb(..)"]) for color_positions."""
    if good == 0.0 and bad == 0.0 and (pivot is None or pivot == zmin + (zmax - zmin) / 2):
        return HEATMAP_COLORSCALE                      # the plain linear scale
    fractions = np.linspace(0.0, 1.0, n)
    positions = color_positions(zmin + fractions * (zmax - zmin), zmin, zmax, pivot,
                                good, bad)
    return [[float(f), "rgb({},{},{})".format(*_colorscale_rgb(v))]
            for f, v in zip(fractions, positions)]


def _curved_colorscale(gamma: float) -> list:
    """
    HEATMAP_COLORSCALE with a gamma curve bent into the value->color mapping
    (the data and colorbar ticks stay in true units — only where the colors
    sit changes). gamma < 1 pulls the yellow->red band down to lower values
    (log-like: reds arrive sooner); gamma > 1 pushes it toward the top
    (exp-like: only the best cells go red); 1 = the base linear scale.
    """
    if gamma == 1.0:
        return HEATMAP_COLORSCALE
    positions = np.linspace(0.0, 1.0, 33)
    return [[float(v), "rgb({},{},{})".format(*_colorscale_rgb(v ** gamma))]
            for v in positions]


def _fmt_cell(value: float, metric: str) -> str:
    """Compact in-cell label."""
    if value is None or np.isnan(value):
        return ""
    if np.isinf(value):
        return "∞"
    if metric == "total_trades":
        return f"{int(value)}"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def build_hover_texts(arrays: dict, x_param, x_values, y_param, y_values,
                      slider_desc: str, masked: np.ndarray,
                      min_trades: int) -> np.ndarray:
    """
    [ny, nx] object array of per-cell hover text — the verbatim hover loop
    from the old _render_heatmap (all 8 metrics + the cell's param values +
    a masked note). HTML (<b>/<br>/<i>) — Qt tooltips render rich text.
    """
    nx, ny = len(x_values), len(y_values)
    hover = np.empty((ny, nx), dtype=object)
    for j in range(ny):
        for i in range(nx):
            lines = [f"<b>{x_param} = {_fmt_axis_value(x_values[i])}</b>"]
            if y_param is not None:
                lines.append(f"<b>{y_param} = {_fmt_axis_value(y_values[j])}</b>")
            if slider_desc:
                lines.append(f"<b>{slider_desc}</b>")
            lines.append("—")
            for m in METRIC_ORDER:
                lines.append(f"{METRIC_LABELS[m]}: "
                             f"{_fmt_metric(arrays[m][j, i], m)}")
            if masked[j, i]:
                lines.append(f"<i>masked (&lt; {min_trades} trades)</i>")
            hover[j, i] = "<br>".join(lines)
    return hover
