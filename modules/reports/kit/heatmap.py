"""
The kit's heatmap component — an Optimizer run shown inside a report exactly
as the Optimizer's Explore heatmap would show it.

    from modules.reports.kit import heatmap
    heatmap(key="es_grid",
            runs={"ES – fade/against": "heatmap1_files/optimizer_es_fade_against",
                  "ES – compact":      "data/es_compact"},
            title="ES grid", default_metric="sharpe_daily")

Paths resolve relative to the REPORT folder (not the process cwd); each is a
copied full Optimizer run (trades.parquet + meta.json) or a compact run
written by the Compress tab — detected automatically, shown identically.

Controls: run · metric · X / Y axis (X ≠ Y; picking the other axis' param
swaps them) · one value per remaining swept param · min trades (hatching) ·
colouring (split value + one curve above / below it, the Optimizer's own
colour logic) · colours on/off · lock. All values are pre-cost.

Saved views (modules.reports.kit.state): every change is saved to
`.report_state.json` under the component's `key`; a fresh session starts
from it. A saved value that is no longer valid falls back to its default
with a notice (other fields keep theirs). Lock disables every control and
freezes the saved view; while locked nothing is written, so a stale-field
reset is shown but only saved after unlocking and changing something.

Numbers: modules.reports.backend.compact (the Optimizer's
compute_metrics_by_cell) + heatmap_view (the Explore grid / colour / hatch
code). Loaded runs are cached per (path, file mtimes/sizes).
"""

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from modules.common.backend import palette
from modules.optimizer.backend.heatmap_model import (CURVE_LIMIT, MIN_TRADES_DEFAULT,
                                                     _fmt_axis_value)
from modules.optimizer.backend.metrics import METRIC_LABELS, METRIC_ORDER
from modules.reports.backend.compact import RunError, load_run, run_signature
from modules.reports.backend.heatmap_view import (cell_records, colorbar_steps,
                                                  compute_heatmap, default_split)

from . import state as state_file
from .style import THEME_NAME

CELL_MIN_PX, CELL_MAX_PX = 24, 104
GRID_TARGET_W = 1100


# ── plumbing ─────────────────────────────────────────────────────────────────

def _ctx():
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import \
            get_script_run_ctx
        return get_script_run_ctx()
    except Exception:  # noqa: BLE001
        return None


def report_dir() -> Path:
    """The running report's folder (the folder of the main script)."""
    ctx = _ctx()
    if ctx is not None and getattr(ctx, "main_script_path", None):
        return Path(ctx.main_script_path).resolve().parent
    main = sys.modules.get("__main__")
    if main is not None and getattr(main, "__file__", None):
        return Path(main.__file__).resolve().parent
    return Path.cwd()


def _resolve(base: Path, path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p)


@st.cache_resource(show_spinner="Loading run…", max_entries=32)
def _load_cached(path: str, _signature_key: str):
    return load_run(path)


def _load(path: Path):
    sig = run_signature(path)
    return _load_cached(str(path), json.dumps(sig, default=str))


def _k(key: str, field: str) -> str:
    return f"rk:{key}:{field}"


# ── saved-view callbacks ─────────────────────────────────────────────────────

def _mark_dirty(key: str) -> None:
    st.session_state[_k(key, "_dirty")] = True


def _on_metric(key: str) -> None:
    st.session_state[_k(key, "split")] = None     # a new metric: its default split
    _mark_dirty(key)


def _on_axis(key: str, which: str) -> None:
    x, y = st.session_state.get(_k(key, "x")), st.session_state.get(_k(key, "y"))
    if x is not None and x == y:                  # took the other axis' param: swap
        prev = st.session_state.get(_k(key, "_prev_y" if which == "x" else "_prev_x"))
        other = st.session_state.get(_k(key, "_prev_x" if which == "x" else "_prev_y"))
        st.session_state[_k(key, "y" if which == "x" else "x")] = other or prev
    _mark_dirty(key)


def _on_reset_colors(key: str) -> None:
    st.session_state[_k(key, "split")] = None
    st.session_state[_k(key, "good")] = 0.0
    st.session_state[_k(key, "bad")] = 0.0
    _mark_dirty(key)


def _current_view(key: str) -> dict:
    ss = st.session_state
    slices = {p: ss.get(_k(key, f"slice:{p}"))
              for p in ss.get(_k(key, "_slice_params"), [])}
    return {
        "run": ss.get(_k(key, "run")),
        "metric": ss.get(_k(key, "metric")),
        "x": ss.get(_k(key, "x")),
        "y": ss.get(_k(key, "y")),
        "slices": slices,
        "min_trades": ss.get(_k(key, "min_trades")),
        "split": ss.get(_k(key, "split")),
        "curve_good": ss.get(_k(key, "good")),
        "curve_bad": ss.get(_k(key, "bad")),
        "colors": ss.get(_k(key, "colors")),
    }


def _on_lock(key: str, folder: str) -> None:
    locked = bool(st.session_state.get(_k(key, "locked")))
    if locked:            # freeze exactly what is on screen
        entry = {**_current_view(key), "locked": True}
    else:                 # unlock: only the flag; the view saves on the next change
        entry = dict(state_file.read_components(folder).get(key, {}))
        entry["locked"] = False
    state_file.write_entry(folder, key, entry)
    st.session_state[_k(key, "_dirty")] = False


# ── field resolution (saved -> session -> default) ───────────────────────────

class _Fields:
    """Initialises each widget's session value once per session from the
    saved view (falling back to defaults with a notice), and silently repairs
    values that became invalid since (e.g. after switching run)."""

    def __init__(self, key: str, saved: dict | None, first: bool):
        self.key, self.saved, self.first = key, saved or {}, first
        self.notices: list[str] = []

    def init(self, field: str, saved_name: str, valid, default, describe=None):
        k = _k(self.key, field)
        if self.first:
            if saved_name in self.saved:
                value = self.saved[saved_name]
                if valid(value):
                    st.session_state[k] = value
                    return
                shown = describe(value) if describe else repr(value)
                self.notices.append(f"{saved_name.replace('_', ' ')} {shown} is no "
                                    "longer valid — reset to its default")
            st.session_state[k] = default
        elif k not in st.session_state or not valid(st.session_state[k]):
            st.session_state[k] = default


def _metric_key(value) -> str | None:
    if value in METRIC_ORDER:
        return value
    for m, label in METRIC_LABELS.items():
        if value == label:
            return m
    return None


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ── the component ────────────────────────────────────────────────────────────

def heatmap(key: str, runs: dict, *, title: str | None = None,
            default_metric: str | None = None) -> None:
    """Render one heatmap component (see the module docstring)."""
    if not isinstance(key, str) or not key.strip():
        st.error("heatmap(): `key` is required — a short name unique within "
                 "the report, e.g. key=\"es_grid\".")
        return
    ctx = _ctx()
    if ctx is not None and _k(key, "run") in ctx.widget_user_keys_this_run:
        st.error(f"heatmap(): duplicate key **{key}** on this page — every kit "
                 "component needs its own key.")
        return

    folder = report_dir()
    # normally the report bootstrap already set a corrupt state file aside
    # (and shows the warning at the top of the page); this covers a report
    # started some other way
    warning = state_file.check_state_file(folder)
    if warning:
        st.warning(warning)
    labels = [str(lbl) for lbl in (runs or {})]
    with st.container(border=True):
        if title:
            st.markdown(f"#### {title}")
        if not labels:
            st.error(f"heatmap “{key}”: the run mapping is empty.")
            return
        _render(key, runs, labels, folder, default_metric)


def _render(key, runs, labels, folder: Path, default_metric) -> None:
    ss = st.session_state
    first = _k(key, "_init") not in ss
    saved = state_file.read_components(folder).get(key) if first else None
    fields = _Fields(key, saved, first)

    if first:
        ss[_k(key, "locked")] = bool((saved or {}).get("locked", False))
    fields.init("run", "run", lambda v: v in labels, labels[0],
                describe=lambda v: f"“{v}”")
    locked = bool(ss[_k(key, "locked")])

    # every control lives in ONE collapsed dropdown: the page shows the title
    # and the heatmap. A constant label keeps it open/closed across reruns.
    controls = st.expander("Settings", expanded=False, icon=":material/tune:")
    with controls:
        top = st.columns([6, 1], vertical_alignment="bottom")
        top[0].selectbox("Run", labels, key=_k(key, "run"), disabled=locked,
                         on_change=_mark_dirty, args=(key,))
        top[1].toggle("Lock", key=_k(key, "locked"), on_change=_on_lock,
                      args=(key, str(folder)),
                      help="Freeze this component: every control is disabled and "
                           "its saved view cannot change")

    label = ss[_k(key, "run")]
    path = _resolve(folder, runs[label])
    try:
        run = _load(path)
    except RunError as e:
        ss[_k(key, "_init")] = True
        st.error(f"**{label}** — `{path}`\n\n{e}")
        _show_notices(key, fields)
        return

    meta = run.meta
    axes = run.axes
    params = [a["param"] for a in axes]
    by_param = {a["param"]: a for a in axes}

    metric_default = _metric_key(default_metric) or METRIC_ORDER[0]
    fields.init("metric", "metric", lambda v: v in METRIC_ORDER, metric_default,
                describe=lambda v: f"“{v}”")
    fields.init("x", "x", lambda v: v in params, params[0],
                describe=lambda v: f"“{v}”")
    x_now = ss[_k(key, "x")]
    if len(params) > 1:
        y_default = next(p for p in params if p != x_now) \
            if params[1] == x_now else params[1]
        fields.init("y", "y", lambda v: v in params and v != ss[_k(key, "x")],
                    y_default, describe=lambda v: f"“{v}”")
    else:
        ss[_k(key, "y")] = None
    y_now = ss[_k(key, "y")]
    slice_params = [p for p in params if p not in (x_now, y_now)]
    saved_slices = (fields.saved.get("slices") or {}) if first else {}
    for p in slice_params:
        values = by_param[p]["values"]
        k = _k(key, f"slice:{p}")
        if first and p in saved_slices:
            if saved_slices[p] in values:
                ss[k] = saved_slices[p]
                continue
            fields.notices.append(f"slice {p} = {saved_slices[p]!r} is not a swept "
                                  "value — reset to the first value")
            ss[k] = values[0]
        elif k not in ss or ss[k] not in values:
            ss[k] = values[0]
    ss[_k(key, "_slice_params")] = slice_params

    fields.init("min_trades", "min_trades",
                lambda v: _is_number(v) and v >= 0 and float(v).is_integer(),
                int(meta.get("min_trades_default", MIN_TRADES_DEFAULT)))
    fields.init("split", "split", lambda v: v is None or _is_number(v), None)
    for f, s in (("good", "curve_good"), ("bad", "curve_bad")):
        fields.init(f, s, lambda v: _is_number(v) and -CURVE_LIMIT <= v <= CURVE_LIMIT,
                    0.0)
    fields.init("colors", "colors", lambda v: isinstance(v, bool), True)
    ss[_k(key, "min_trades")] = int(ss[_k(key, "min_trades")])
    for f in ("good", "bad"):
        ss[_k(key, f)] = float(ss[_k(key, f)])

    # ── controls (inside the Settings dropdown) ───────────────────────────
    with controls:
        _controls(key, params, by_param, slice_params, locked)
    metric = ss[_k(key, "metric")]

    # ── the heatmap ───────────────────────────────────────────────────────
    hd = compute_heatmap(run.summary, metric, by_param[x_now],
                         by_param[y_now] if y_now else None,
                         {p: ss[_k(key, f"slice:{p}")] for p in slice_params},
                         ss[_k(key, "min_trades")], split=ss[_k(key, "split")],
                         good=ss[_k(key, "good")], bad=ss[_k(key, "bad")])
    colors_on = bool(ss[_k(key, "colors")])
    if hd.no_finite:
        st.info("No cell has a finite value for this metric in this slice.")
    if hd.all_masked:
        st.warning("All cells are below the min-trades threshold.")
    with st.container(horizontal_alignment="center"):
        st.altair_chart(_chart(hd, colors_on), theme=None, width="content",
                        key=_k(key, "_chart"))
    st.caption(_caption(run, label))

    # ── persist + remember the axes for the swap logic ────────────────────
    ss[_k(key, "_prev_x")], ss[_k(key, "_prev_y")] = x_now, y_now
    ss[_k(key, "_init")] = True
    if ss.get(_k(key, "_dirty")) and not locked:
        state_file.write_entry(folder, key, {**_current_view(key), "locked": False})
        ss[_k(key, "_dirty")] = False
        ss[_k(key, "_notice")] = []
    _show_notices(key, fields)


def _controls(key, params, by_param, slice_params, locked) -> None:
    """Metric / axes / min trades / slices / colouring widgets."""
    ss = st.session_state
    row = st.columns([3, 2, 2, 1.4], vertical_alignment="bottom")
    row[0].selectbox("Metric", METRIC_ORDER, key=_k(key, "metric"),
                     format_func=lambda m: METRIC_LABELS[m], disabled=locked,
                     on_change=_on_metric, args=(key,))
    row[1].selectbox("X axis", params, key=_k(key, "x"), disabled=locked,
                     on_change=_on_axis, args=(key, "x"))
    if len(params) > 1:
        row[2].selectbox("Y axis", params, key=_k(key, "y"), disabled=locked,
                         on_change=_on_axis, args=(key, "y"),
                         help="Picking the X param here swaps the two axes")
    else:
        row[2].caption("Only one param was swept — it is the X axis.")
    row[3].number_input("Min trades", min_value=0, step=1, key=_k(key, "min_trades"),
                        disabled=locked, on_change=_mark_dirty, args=(key,),
                        help="cells with fewer trades are hatched")

    if slice_params:
        cols = st.columns(max(len(slice_params), 2))
        for c, p in zip(cols, slice_params):
            c.selectbox(f"{p} (slice)", by_param[p]["values"],
                        key=_k(key, f"slice:{p}"), format_func=_fmt_axis_value,
                        disabled=locked, on_change=_mark_dirty, args=(key,))

    metric = ss[_k(key, "metric")]
    natural = default_split(metric)
    st.caption("Colouring")              # a plain section: expanders can't nest
    crow = st.columns([2, 3, 3, 1.2, 1.2], vertical_alignment="bottom")
    crow[0].number_input(
        "Split at", key=_k(key, "split"), format="%g", disabled=locked,
        placeholder=("auto (middle)" if natural is None else f"auto ({natural:g})"),
        on_change=_mark_dirty, args=(key,),
        help="The value where the colour scale changes sides (warm above, "
             "cool below). Empty = the metric's default: 0 for ticks / avg "
             "trade, 1 for profit factor and both Sharpes, 50 for win "
             "rates, the middle of the range for trade count.")
    crow[1].slider("Above split curve", -CURVE_LIMIT, CURVE_LIMIT, step=0.01,
                   key=_k(key, "good"), disabled=locked,
                   on_change=_mark_dirty, args=(key,),
                   help="0 = linear · > 0 = only the highest cells reach red · "
                        "< 0 = red already reaches cells just above the split")
    crow[2].slider("Below split curve", -CURVE_LIMIT, CURVE_LIMIT, step=0.01,
                   key=_k(key, "bad"), disabled=locked,
                   on_change=_mark_dirty, args=(key,),
                   help="0 = linear · > 0 = only the lowest cells reach dark "
                        "blue · < 0 = dark blue already reaches cells just "
                        "below the split")
    crow[3].toggle("Colours", key=_k(key, "colors"), disabled=locked,
                   on_change=_mark_dirty, args=(key,),
                   help="Off = neutral cells, values only (hatching stays)")
    crow[4].button("Reset", key=_k(key, "_reset"), disabled=locked,
                   on_click=_on_reset_colors, args=(key,),
                   help="Split back to the metric's default, both curves to 0")


def _show_notices(key: str, fields: _Fields) -> None:
    ss = st.session_state
    if fields.first and fields.notices:
        ss[_k(key, "_notice")] = fields.notices
    notices = ss.get(_k(key, "_notice")) or []
    if notices:
        st.caption("⚠ Saved view: " + " · ".join(notices))


def _caption(run, label: str) -> str:
    meta = run.meta
    fixed = meta.get("fixed_params") or {}
    fixed_txt = ", ".join(f"{k}={_fmt_axis_value(v)}" for k, v in fixed.items()) \
        or "none"
    bits = [f"**{label}**", f"{meta.get('strategy', '?')}",
            f"{meta.get('dataset') or meta.get('ticker', '?')}",
            f"{meta.get('start_date', '?')} → {meta.get('end_date', '?')}",
            f"{meta.get('n_combos', len(run.summary))} combinations",
            "pre-cost"]
    if run.compact:
        bits.append("compact")
    return " · ".join(bits) + f"  \nFixed parameters: {fixed_txt}"


# ── Altair rendering ─────────────────────────────────────────────────────────

def _tick_format(zmin: float, zmax: float) -> str:
    """d3 format for the colour bar: plain numbers (never 9e+2), decimals
    only where the range needs them, trailing zeros trimmed."""
    span = abs(zmax - zmin)
    if span >= 50:
        return ",.0f"
    if span >= 5:
        return ",.1~f"
    if span >= 0.5:
        return ",.2~f"
    return ",.4~f"


def _chart(hd, colors_on: bool):
    alt.theme.enable(THEME_NAME)
    nx, ny = len(hd.x_values), len(hd.y_values)
    cell = int(max(CELL_MIN_PX, min(CELL_MAX_PX, GRID_TARGET_W / max(nx, 1))))
    width, height = nx * cell, max(ny * cell, cell)

    records = cell_records(hd, colors_on)
    hover_fields = list(records[0]["hover"].keys()) if records else []
    rows = []
    for r in records:
        row = {"i": r["i"], "i1": r["i"] + 1, "j": r["j"], "j1": r["j"] + 1,
               "fill": r["fill"] or "transparent", "text": r["text"],
               "text_color": r["text_color"], "masked": r["masked"],
               "xc": r["i"] + 0.5, "yc": r["j"] + 0.5}
        for n, f in enumerate(hover_fields):
            row[f"h{n}"] = r["hover"][f]
        rows.append(row)
    df = pd.DataFrame(rows)

    x_labels = [_fmt_axis_value(v) for v in hd.x_values]
    y_labels = [_fmt_axis_value(v) for v in hd.y_values] if hd.y_param else [""]
    x_scale = alt.Scale(domain=[0, nx], nice=False, zero=False)
    y_scale = alt.Scale(domain=[0, ny], nice=False, zero=False)
    x_axis = alt.Axis(values=[i + 0.5 for i in range(nx)], grid=False, ticks=False,
                      domain=False, labelAngle=0 if nx <= 14 else -45,
                      labelExpr=f"{json.dumps(x_labels)}[floor(datum.value)]",
                      title=hd.x_param)
    y_axis = alt.Axis(values=[j + 0.5 for j in range(ny)], grid=False, ticks=False,
                      domain=False,
                      labelExpr=f"{json.dumps(y_labels)}[floor(datum.value)]",
                      title=hd.y_param or None, labels=bool(hd.y_param))
    tooltip = [alt.Tooltip(f"h{n}:N", title=f) for n, f in enumerate(hover_fields)]

    base = alt.Chart(df)
    rects = base.mark_rect(stroke=palette.CHART_BG, strokeWidth=1).encode(
        x=alt.X("i:Q", scale=x_scale, axis=x_axis),
        x2="i1:Q",
        y=alt.Y("j:Q", scale=y_scale, axis=y_axis),
        y2="j1:Q",
        color=alt.Color("fill:N", scale=None),
        tooltip=tooltip)
    layers = [rects]

    masked = df[df["masked"]]
    if not masked.empty:
        # the app's hatch: 3 diagonals per cell, 1 on heavily masked grids
        segs = [((0.0, 0.0), (1.0, 1.0)), ((0.0, 0.5), (0.5, 1.0)),
                ((0.5, 0.0), (1.0, 0.5))]
        if len(masked) * 3 > 900:
            segs = segs[:1]
        lines = [{"a": i + u0, "b": i + u1, "c": j + v0, "d": j + v1}
                 for i, j in zip(masked["i"], masked["j"])
                 for (u0, v0), (u1, v1) in segs]
        hatch = alt.Chart(pd.DataFrame(lines)).mark_rule(
            color="rgba(255,255,255,0.55)", strokeWidth=1.2).encode(
            x=alt.X("a:Q", scale=x_scale), x2="b:Q",
            y=alt.Y("c:Q", scale=y_scale), y2="d:Q")
        layers.append(hatch)

    longest = max((len(t) for t in df["text"]), default=1) or 1
    font = max(7.0, min(14.0, cell * 0.88 / (0.62 * longest), cell * 0.5))
    if font >= 7.0 and nx * ny <= 2000:
        labels = base.mark_text(font=palette.FONT_FAMILY.replace('"', ""),
                                fontSize=font).encode(
            x=alt.X("xc:Q", scale=x_scale), y=alt.Y("yc:Q", scale=y_scale),
            text="text:N", color=alt.Color("text_color:N", scale=None),
            tooltip=tooltip)
        layers.append(labels)

    title = METRIC_LABELS[hd.metric] + (f"  ·  {hd.slider_desc}" if hd.slider_desc else "")
    grid = alt.layer(*layers).properties(width=width, height=height, title=title)
    if not colors_on or hd.no_finite:
        return grid

    bar = pd.DataFrame(colorbar_steps(hd))
    colorbar = alt.Chart(bar).mark_rect().encode(
        y=alt.Y("v0:Q", scale=alt.Scale(domain=[hd.zmin, hd.zmax], nice=False,
                                        zero=False),
                axis=alt.Axis(orient="right", title=METRIC_LABELS[hd.metric],
                              grid=False, format=_tick_format(hd.zmin, hd.zmax))),
        y2="v1:Q",
        color=alt.Color("fill:N", scale=None)).properties(width=14, height=height)
    return alt.hconcat(grid, colorbar, spacing=12).resolve_scale(y="independent")
