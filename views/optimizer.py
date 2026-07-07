# views/optimizer.py
#
# Strategy Optimizer — sweep up to 3 strategy params (X axis × Y axis ×
# slider), store the full trades table of every grid cell, and explore an
# interactive metric heatmap with zero backtest re-runs. A hypothesis
# generator, deliberately WITHOUT any "pick the best config" control: read the
# surface for plateaus and cross-half stability, not the single brightest cell.

import time
from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from optimization import io as opt_io
from optimization.buckets import (
    BUCKET_ORDER, EVENT_KEYWORDS, FF_EVENTS_PATH, load_bucket_map,
)
from optimization.engine import check_param_columns, median_split_date, run_grid
from optimization.metrics import METRIC_LABELS, METRIC_ORDER, compute_metrics_by_cell
from optimization.param_space import (
    MAX_SWEPT, ROLE_LABELS, ROLES, build_range, combo_count, parse_values,
    sweep_kind,
)

ASSET_INFO = {
    # Equity Index
    "ES":  {"tick_size": 0.25, "ticks_per_point": 4,   "dollars_per_tick": 12.50},
    "NQ":  {"tick_size": 0.25, "ticks_per_point": 4,   "dollars_per_tick": 5.00},
    "RTY": {"tick_size": 0.10, "ticks_per_point": 10,  "dollars_per_tick": 5.00},
    "YM":  {"tick_size": 1.00, "ticks_per_point": 1,   "dollars_per_tick": 5.00},
    "MES": {"tick_size": 0.25, "ticks_per_point": 4,   "dollars_per_tick": 1.25},
    "MNQ": {"tick_size": 0.25, "ticks_per_point": 4,   "dollars_per_tick": 0.50},
    "M2K": {"tick_size": 0.10, "ticks_per_point": 10,  "dollars_per_tick": 0.50},
    "MYM": {"tick_size": 1.00, "ticks_per_point": 1,   "dollars_per_tick": 0.50},
    # Rates
    "ZN":  {"tick_size": 0.015625,   "ticks_per_point": 64,      "dollars_per_tick": 15.625},
    "ZB":  {"tick_size": 0.03125,    "ticks_per_point": 32,      "dollars_per_tick": 31.25},
    "ZF":  {"tick_size": 0.0078125,  "ticks_per_point": 128,     "dollars_per_tick": 7.8125},
    "ZT":  {"tick_size": 0.00390625, "ticks_per_point": 256,     "dollars_per_tick": 7.8125},
    "SR3": {"tick_size": 0.0025,     "ticks_per_point": 400,     "dollars_per_tick": 6.25},
    # Energy
    "CL":  {"tick_size": 0.01,   "ticks_per_point": 100,   "dollars_per_tick": 10.00},
    "QM":  {"tick_size": 0.025,  "ticks_per_point": 40,    "dollars_per_tick": 12.50},
    "NG":  {"tick_size": 0.001,  "ticks_per_point": 1000,  "dollars_per_tick": 10.00},
    "RB":  {"tick_size": 0.0001, "ticks_per_point": 10000, "dollars_per_tick": 4.20},
    "HO":  {"tick_size": 0.0001, "ticks_per_point": 10000, "dollars_per_tick": 4.20},
    # Metals
    "GC":  {"tick_size": 0.10,   "ticks_per_point": 10,   "dollars_per_tick": 10.00},
    "MGC": {"tick_size": 0.10,   "ticks_per_point": 10,   "dollars_per_tick": 1.00},
    "SI":  {"tick_size": 0.005,  "ticks_per_point": 200,  "dollars_per_tick": 25.00},
    "HG":  {"tick_size": 0.0005, "ticks_per_point": 2000, "dollars_per_tick": 12.50},
    # Grains
    "ZC":  {"tick_size": 0.25, "ticks_per_point": 4, "dollars_per_tick": 12.50},
    "ZS":  {"tick_size": 0.25, "ticks_per_point": 4, "dollars_per_tick": 12.50},
    "ZW":  {"tick_size": 0.25, "ticks_per_point": 4, "dollars_per_tick": 12.50},
    # FX
    "6E":  {"tick_size": 0.00005,    "ticks_per_point": 20000,   "dollars_per_tick": 6.25},
    "6J":  {"tick_size": 0.0000005,  "ticks_per_point": 2000000, "dollars_per_tick": 6.25},
    "6B":  {"tick_size": 0.0001,     "ticks_per_point": 10000,   "dollars_per_tick": 6.25},
    "6C":  {"tick_size": 0.00005,    "ticks_per_point": 20000,   "dollars_per_tick": 5.00},
    # Crypto
    "BTC": {"tick_size": 5.00, "ticks_per_point": 0.2, "dollars_per_tick": 25.00},
}

HIDDEN_PARAMS = {"tick_size"}

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


# ── Navigation ────────────────────────────────────────────────────────────────

def go_page(page: str):
    st.session_state.page = page
    st.rerun()


# ── Data / strategy scanning (mirrored from the backtester, house convention) ─

def get_parquet_structure() -> dict:
    parquet_path = Path("data/parquet")
    structure = {}
    if not parquet_path.exists():
        return structure
    for type_dir in sorted(parquet_path.iterdir()):
        if not type_dir.is_dir():
            continue
        structure[type_dir.name] = {}
        for asset_dir in sorted(type_dir.iterdir()):
            if not asset_dir.is_dir():
                continue
            datasets = sorted([f.name for f in asset_dir.iterdir() if f.is_dir()])
            if datasets:
                structure[type_dir.name][asset_dir.name] = datasets
    return structure


def get_strategies() -> list:
    strategies_path = Path("strategies")
    if not strategies_path.exists():
        return []
    results = []
    for f in strategies_path.glob("*.py"):
        if f.stem not in ["__init__", "base"]:
            results.append(f.stem)
    for f in strategies_path.iterdir():
        if f.is_dir() and (f / "__init__.py").exists():
            results.append(f.name)
    return sorted(results)


def load_strategy(name: str):
    flat_path   = Path("strategies") / f"{name}.py"
    folder_path = Path("strategies") / name / "__init__.py"

    if flat_path.exists():
        path   = flat_path
        is_pkg = False
    elif folder_path.exists():
        path   = folder_path
        is_pkg = True
    else:
        raise FileNotFoundError(f"Strategy '{name}' not found")

    if is_pkg:
        spec = importlib.util.spec_from_file_location(
            name, path,
            submodule_search_locations=[str(Path("strategies") / name)],
        )
    else:
        spec = importlib.util.spec_from_file_location(name, path)

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "run") or not callable(module.run):
        raise ValueError(f"Strategy '{name}' has no callable run()")
    return module


# ── Setup: dataset / strategy / date selectors ────────────────────────────────

def render_setup_controls(structure: dict, strategies: list):
    col1, col2 = st.columns(2)

    with col1:
        asset_type = st.selectbox("Type", list(structure.keys()), key="opt_type")

        assets = list(structure.get(asset_type, {}).keys())
        if not assets:
            st.error(f"No assets found under {asset_type}")
            return None
        asset = st.selectbox("Asset", assets, key=f"opt_asset_{asset_type}")

        datasets = structure[asset_type].get(asset, [])
        if not datasets:
            st.error(f"No datasets found under {asset_type}/{asset}")
            return None
        dataset = st.selectbox("Dataset", datasets,
                               key=f"opt_dataset_{asset_type}_{asset}")

        strategy_name = st.selectbox("Strategy", strategies, key="opt_strategy")

    with col2:
        folder_path     = Path("data/parquet") / asset_type / asset / dataset
        available_dates = sorted([
            pd.Timestamp(f.stem) for f in folder_path.glob("*.parquet")
            if f.stem[0].isdigit()
        ])
        if not available_dates:
            st.error(f"No dated parquet files in {folder_path}")
            return None

        start_date = st.date_input(
            "Start date",
            value=available_dates[0].date(),
            min_value=available_dates[0].date(),
            max_value=available_dates[-1].date(),
            key=f"opt_start_{asset_type}_{asset}_{dataset}",
        )
        end_date = st.date_input(
            "End date",
            value=available_dates[-1].date(),
            min_value=available_dates[0].date(),
            max_value=available_dates[-1].date(),
            key=f"opt_end_{asset_type}_{asset}_{dataset}",
        )

    return asset_type, asset, dataset, strategy_name, start_date, end_date


# ── Setup: parameter sweep panel ──────────────────────────────────────────────

def _render_value_widget(key: str, default, widget_key: str):
    """Single-value editor for a param held at its default."""
    if isinstance(default, bool):
        return st.checkbox(key, value=default, key=widget_key)
    elif isinstance(default, float):
        return st.number_input(key, value=default, step=0.1, format="%.2f",
                               key=widget_key, label_visibility="collapsed")
    elif isinstance(default, int):
        return st.number_input(key, value=default, step=1, key=widget_key,
                               label_visibility="collapsed")
    elif isinstance(default, str):
        return st.text_input(key, value=default, key=widget_key,
                             label_visibility="collapsed")
    else:
        st.warning(f"Unsupported param type for '{key}': {type(default).__name__}")
        return default


def _param_layout(strategy, visible: dict) -> list:
    """[(section_label, [param, ...]), ...] honoring PARAM_SECTIONS."""
    if not hasattr(strategy, "PARAM_SECTIONS"):
        return [("Parameters", list(visible.keys()))]
    sections, rendered = [], set()
    for label, keys in strategy.PARAM_SECTIONS.items():
        keys = [k for k in keys if k in visible]
        if keys:
            sections.append((label, keys))
            rendered.update(keys)
    unassigned = [k for k in visible if k not in rendered]
    if unassigned:
        sections.append(("Other", unassigned))
    return sections


def _values_preview(values: list) -> str:
    fmt = [f"{v:g}" if isinstance(v, float) else str(v) for v in values]
    if len(fmt) > 8:
        fmt = fmt[:4] + ["…"] + fmt[-2:]
    return f"{len(values)} value{'s' if len(values) != 1 else ''}: " + ", ".join(fmt)


def _render_sweep_inputs(param: str, default, kind: str, key_prefix: str):
    """
    The sweep editor for one checked param: min/max/step for numeric params,
    a comma-separated value list for string params. Returns the value list,
    or None when the input is invalid (error shown inline).
    """
    if kind == "categorical":
        text = st.text_input(
            param, value=str(default),
            key=f"{key_prefix}_vals_{param}",
            label_visibility="collapsed",
            help="comma-separated values to test",
        )
        try:
            values = parse_values(text)
        except ValueError as e:
            st.caption(f"⚠ {e}")
            return None
    else:
        c_lo, c_hi, c_step = st.columns(3)
        if kind == "int":
            lo   = c_lo.number_input("min", value=int(default), step=1,
                                     key=f"{key_prefix}_lo_{param}")
            hi   = c_hi.number_input("max", value=int(default), step=1,
                                     key=f"{key_prefix}_hi_{param}")
            step = c_step.number_input("step", value=1, min_value=1, step=1,
                                       key=f"{key_prefix}_step_{param}")
        else:
            lo   = c_lo.number_input("min", value=float(default), step=0.1,
                                     format="%g", key=f"{key_prefix}_lo_{param}")
            hi   = c_hi.number_input("max", value=float(default), step=0.1,
                                     format="%g", key=f"{key_prefix}_hi_{param}")
            step = c_step.number_input("step", value=0.1, min_value=0.0, step=0.05,
                                       format="%g", key=f"{key_prefix}_step_{param}")
        try:
            values = build_range(lo, hi, step, kind)
        except ValueError as e:
            st.caption(f"⚠ {e}")
            return None

    st.caption(_values_preview(values))
    return values


def render_param_panel(strategy, key_prefix: str):
    """
    All params of the strategy. Sweepability is inferred from each default's
    type (int/float -> min/max/step range, str -> value list, bool -> fixed
    only) — strategies declare nothing. Checked params get the sweep editor,
    unchecked ones a single-value widget. Returns (fixed_params,
    swept_values {param: list | None-on-bad-input}, sweep_order).
    """
    visible = {k: v for k, v in getattr(strategy, "PARAMS", {}).items()
               if k not in HIDDEN_PARAMS}
    sweepable = {p: sweep_kind(v) for p, v in visible.items()
                 if sweep_kind(v) is not None}

    # count from the PREVIOUS rerun's widget state — drives the 3-sweep cap
    checked_now = {p for p in sweepable
                   if st.session_state.get(f"{key_prefix}_sweep_{p}", False)}
    cap_reached = len(checked_now) >= MAX_SWEPT

    st.write("")
    st.subheader("Parameters")
    st.caption(
        f"Check up to {MAX_SWEPT} params to sweep — numeric params take "
        f"min/max/step, text params a comma-separated value list; everything "
        f"else is held at the value shown."
    )

    fixed_params, swept_values, checked = {}, {}, []

    for section_label, keys in _param_layout(strategy, visible):
        st.caption(section_label)
        for row_start in range(0, len(keys), 3):
            row = keys[row_start:row_start + 3]
            cols = st.columns(3)
            for col, param in zip(cols, row):
                with col:
                    default = visible[param]
                    if param in sweepable:
                        is_checked = st.checkbox(
                            f"**{param}**",
                            key=f"{key_prefix}_sweep_{param}",
                            disabled=(cap_reached and param not in checked_now),
                            help="sweep this param",
                        )
                        if is_checked:
                            checked.append(param)
                            swept_values[param] = _render_sweep_inputs(
                                param, default, sweepable[param], key_prefix)
                        else:
                            fixed_params[param] = _render_value_widget(
                                param, default, f"{key_prefix}_fix_{param}")
                    else:
                        st.markdown(f"{param}")
                        fixed_params[param] = _render_value_widget(
                            param, default, f"{key_prefix}_fix_{param}")

    # sweep order = selection order: survivors keep their old rank, new checks
    # append in render order
    prev = st.session_state.get(f"{key_prefix}_sweep_order", [])
    order = [p for p in prev if p in checked] + [p for p in checked if p not in prev]
    st.session_state[f"{key_prefix}_sweep_order"] = order

    return fixed_params, swept_values, order


def render_role_assignment(sweep_order: list, key_prefix: str):
    """Role selectboxes for the swept params. Returns {param: role} or None."""
    if not sweep_order:
        return {}

    available = ROLES[:max(len(sweep_order), 1)]
    st.write("")
    st.caption("Axis roles")
    roles_by_param = {}
    cols = st.columns(MAX_SWEPT)
    for i, param in enumerate(sweep_order):
        with cols[i]:
            roles_by_param[param] = st.selectbox(
                param, available,
                index=min(i, len(available) - 1),
                format_func=ROLE_LABELS.get,
                key=f"{key_prefix}_role_{param}_{len(sweep_order)}",
            )

    if len(set(roles_by_param.values())) != len(roles_by_param):
        st.error("Each swept param needs a distinct role (X / Y / Slider).")
        return None
    return roles_by_param


# ── Setup: the run itself ─────────────────────────────────────────────────────

def execute_grid_run(strategy, asset_type, asset, dataset, strategy_name,
                     start_date, end_date, fixed_params, axes,
                     run_name, be_band_ticks, min_trades_default):
    asset_info  = ASSET_INFO[asset]
    folder_path = Path("data/parquet") / asset_type / asset / dataset

    ff_found   = FF_EVENTS_PATH.exists()
    bucket_map = load_bucket_map()
    if not ff_found:
        st.warning("ff_usd_events.parquet missing — every day is bucketed 'normal'.")

    progress_bar  = st.progress(0)
    log_container = st.container(height=300)
    log_box       = log_container.empty()
    logs          = []

    def on_progress(current, total, message: str = ""):
        progress_bar.progress(current / total)
        if message:
            logs.append(message)
        log_box.code("\n".join(logs), language=None)

    t0 = time.time()
    trades = run_grid(
        strategy, folder_path, start_date, end_date,
        base_params=fixed_params, axes=axes,
        tick_size=asset_info["tick_size"],
        ticks_per_point=asset_info["ticks_per_point"],
        bucket_map=bucket_map,
        on_progress=on_progress,
    )
    elapsed = time.time() - t0

    split = median_split_date(trades)
    axes_by_role = {a["role"]: {"param": a["param"], "values": a["values"]}
                    for a in axes}
    meta = {
        "strategy":           strategy_name,
        "dataset":            f"{asset_type}/{asset}/{dataset}",
        "ticker":             asset,
        "tick_size":          asset_info["tick_size"],
        "ticks_per_point":    asset_info["ticks_per_point"],
        "start_date":         str(start_date),
        "end_date":           str(end_date),
        "axes": {role: axes_by_role.get(role) for role in ROLES},
        "fixed_params":       fixed_params,
        "min_trades_default": min_trades_default,
        "be_band_ticks":      be_band_ticks,
        "event_keywords":     EVENT_KEYWORDS,
        "ff_events_found":    ff_found,
        "split_date":         None if split is None else str(split.date()),
        "n_combos":           combo_count(axes),
        "n_trades":           len(trades),
        "created_at":         pd.Timestamp.now().isoformat(),
    }

    run_dir = opt_io.save_run(run_name, trades, meta)
    meta["run_name"] = run_dir.name

    st.session_state.opt_loaded_run = run_dir.name
    st.session_state.opt_trades     = trades
    st.session_state.opt_meta       = meta
    st.session_state.pop("opt_grid_key", None)

    # The opt_mode radio is already instantiated this run, so its state can't
    # be set here — stash a hand-off flag (+ the success message, which an
    # immediate rerun would otherwise wipe); render() consumes both before
    # the radio exists on the next run.
    minutes, seconds = int(elapsed // 60), int(elapsed % 60)
    st.session_state.opt_run_success = (
        f"Done in {f'{minutes}m {seconds}s' if minutes else f'{seconds}s'} — "
        f"{meta['n_combos']} backtests, {len(trades)} trades. Saved to {run_dir}"
    )
    st.session_state.opt_switch_to_explore = True
    st.rerun()


def render_setup():
    structure = get_parquet_structure()
    if not structure:
        st.error("No datasets found in data/parquet")
        return
    strategies = get_strategies()
    if not strategies:
        st.error("No strategies found in strategies/")
        return

    result = render_setup_controls(structure, strategies)
    if result is None:
        return
    asset_type, asset, dataset, strategy_name, start_date, end_date = result

    if asset not in ASSET_INFO:
        st.error(f"Unknown asset: {asset}. Add it to ASSET_INFO.")
        return
    if start_date > end_date:
        st.error("Start date must be before end date.")
        return
    if asset in NON_US_CALENDAR_ASSETS:
        st.warning(f"{asset}: day buckets come from the USD calendar only — "
                   f"foreign-calendar events are not tagged.")

    strategy = load_strategy(strategy_name)

    visible = {k: v for k, v in getattr(strategy, "PARAMS", {}).items()
               if k not in HIDDEN_PARAMS}
    if not any(sweep_kind(v) is not None for v in visible.values()):
        st.info(
            f"**{strategy_name} has no sweepable params** — it needs a "
            f"`PARAMS` dict with int/float/str defaults."
        )
        return

    key_prefix = f"opt_{strategy_name}"
    fixed_params, swept_values, sweep_order = render_param_panel(
        strategy, key_prefix)

    if not sweep_order:
        st.info("Check at least one param to sweep.")
        return
    if any(swept_values.get(p) is None for p in sweep_order):
        st.error("Fix the sweep inputs flagged (⚠) above.")
        return

    roles_by_param = render_role_assignment(sweep_order, key_prefix)
    if roles_by_param is None:
        return

    # axes ordered x -> y -> slider
    param_by_role = {role: p for p, role in roles_by_param.items()}
    axes = [{"param": param_by_role[role],
             "values": swept_values[param_by_role[role]],
             "role": role}
            for role in ROLES if role in param_by_role]
    try:
        check_param_columns(axes)
    except ValueError as e:
        st.error(str(e))
        return

    # live grid readout
    n_combos = combo_count(axes)
    sizes    = " × ".join(f"|{a['param']}| = {len(a['values'])}" for a in axes)
    st.info(f"{sizes}   →   **{n_combos} backtests**")

    if len(sweep_order) == 1:
        st.warning("Only 1 swept param — the heatmap degenerates to a single row.")
    if len(sweep_order) < 3:
        st.caption("No slider (fewer than 3 swept params).")
    for a in axes:
        if len(a["values"]) == 1:
            st.warning(f"Axis '{a['param']}' has a single value — degenerate axis.")

    # run settings
    st.write("")
    with st.expander("Run settings", expanded=True):
        c1, c2, c3 = st.columns([2, 1, 1])
        run_name = c1.text_input(
            "Run name",
            value=f"{asset}_{strategy_name}_{start_date}_{end_date}",
            key="opt_run_name",
        )
        be_band_ticks = c2.number_input(
            "BE band (ticks)", value=0.0, min_value=0.0, step=0.5,
            key="opt_be_band",
            help="win = pnl > band, breakeven = |pnl| <= band, loss = pnl < -band",
        )
        min_trades_default = c3.number_input(
            "Min trades default", value=MIN_TRADES_DEFAULT, min_value=0, step=5,
            key="opt_min_trades_default",
            help="default hatch threshold in the heatmap (changeable there)",
        )
    st.caption(
        "Grid speed rides on the strategy's internal day cache being "
        "param-independent (ivb_model_optimized: yes). If each combo takes as "
        "long as the first, the strategy re-reads its data every run — "
        "expect grid_size × cold_run_time."
    )

    confirmed = True
    if n_combos > COMBO_CONFIRM_THRESHOLD:
        st.warning(f"{n_combos} backtests exceeds the {COMBO_CONFIRM_THRESHOLD} "
                   f"combo guard — this can take a long time.")
        confirmed = st.checkbox(f"Run all {n_combos} backtests anyway",
                                key="opt_confirm_large")

    st.write("")
    _, _, btn_col, _, _ = st.columns(5)
    with btn_col:
        run = st.button("Run grid", type="primary", width='stretch',
                        disabled=not confirmed, key="opt_run_btn")

    if run:
        if not run_name.strip():
            st.error("Please enter a run name.")
            return
        execute_grid_run(
            strategy, asset_type, asset, dataset, strategy_name,
            start_date, end_date, fixed_params, axes,
            run_name, be_band_ticks, min_trades_default,
        )


# ── Explore: filtering + heatmap ──────────────────────────────────────────────

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


def _hatch_shapes(masked_ij: list) -> list:
    """
    Translucent diagonal hatch segments over masked cells (Plotly heatmaps
    have no native hatching). Cell (i, j) spans i±0.5 × j±0.5. Degrades to a
    single diagonal per cell on heavily masked grids to keep Plotly snappy.
    """
    full = [((0.0, 0.0), (1.0, 1.0)),
            ((0.0, 0.5), (0.5, 1.0)),
            ((0.5, 0.0), (1.0, 0.5))]
    segments = full if len(masked_ij) * 3 <= 900 else full[:1]

    shapes = []
    for i, j in masked_ij:
        for (u0, v0), (u1, v1) in segments:
            shapes.append(dict(
                type="line",
                x0=i - 0.5 + u0, y0=j - 0.5 + v0,
                x1=i - 0.5 + u1, y1=j - 0.5 + v1,
                line=dict(color="rgba(255,255,255,0.55)", width=1.2),
                xref="x", yref="y",
            ))
    return shapes


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


def _render_heatmap(arrays: dict, metric: str, x_param, x_values, y_param,
                    y_values, slider_desc: str, min_trades: int):
    nx, ny = len(x_values), len(y_values)
    counts = np.nan_to_num(arrays["total_trades"], nan=0.0)
    masked = counts < min_trades

    z = arrays[metric].copy()
    z[~np.isfinite(z)] = np.nan          # NaN AND inf sit outside the scale

    finite = z[np.isfinite(z)]
    if finite.size == 0:
        st.info("No cell has a finite value for this metric under the current "
                "filters.")
        if not masked.all():
            return
    if masked.all():
        st.warning("All cells are below the min-trades threshold under the "
                   "current filters.")

    zmin = float(finite.min()) if finite.size else 0.0
    zmax = float(finite.max()) if finite.size else 1.0
    if zmin == zmax:
        zmin, zmax = zmin - 0.5, zmax + 0.5

    # hover: all 8 metrics + trade count + the cell's param values
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

    # ── square cells: fixed pixel size drives the exact figure dimensions ────
    cell = int(max(26, min(96, 1150 / nx, 640 / ny)))
    margin = dict(l=70, r=150, t=56, b=64)
    fig_width  = cell * nx + margin["l"] + margin["r"]
    fig_height = cell * ny + margin["t"] + margin["b"]

    # in-cell value labels with luminance-based contrast (skip on tiny cells)
    annotations = []
    if cell >= 34 and nx * ny <= 600:
        font_size = int(max(9, min(15, cell * 0.26)))
        span = zmax - zmin
        for j in range(ny):
            for i in range(nx):
                raw = arrays[metric][j, i]
                text = _fmt_cell(raw, metric)
                if not text:
                    continue
                color = ("rgba(150,150,150,0.85)" if not np.isfinite(raw)
                         else _cell_text_color((z[j, i] - zmin) / span))
                annotations.append(dict(
                    x=i, y=j, text=text, showarrow=False,
                    font=dict(size=font_size, color=color),
                ))

    title = METRIC_LABELS[metric] + (f"  ·  {slider_desc}" if slider_desc else "")

    fig = go.Figure(go.Heatmap(
        z=z,
        x=list(range(nx)),
        y=list(range(ny)),
        colorscale=HEATMAP_COLORSCALE,
        zmin=zmin, zmax=zmax,
        hovertext=hover,
        hovertemplate="%{hovertext}<extra></extra>",
        colorbar=dict(
            title=dict(text=METRIC_LABELS[metric], side="right"),
            thickness=14, outlinewidth=0, ticks="outside", ticklen=3,
        ),
        xgap=2, ygap=2,
    ))
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=15)),
        annotations=annotations,
        shapes=_hatch_shapes([(i, j) for j in range(ny) for i in range(nx)
                              if masked[j, i]]),
        xaxis=dict(title=dict(text=x_param, font=dict(size=13)),
                   tickmode="array", tickvals=list(range(nx)),
                   ticktext=[_fmt_axis_value(v) for v in x_values],
                   showgrid=False, zeroline=False, ticks="",
                   tickfont=dict(size=12), constrain="domain"),
        yaxis=dict(title=dict(text=y_param or "", font=dict(size=13)),
                   tickmode="array", tickvals=list(range(ny)),
                   ticktext=[_fmt_axis_value(v) for v in y_values]
                   if y_param is not None else [""],
                   showticklabels=y_param is not None,
                   showgrid=False, zeroline=False, ticks="",
                   tickfont=dict(size=12),
                   # cells stay square even when a narrow container clamps
                   # the figure width (plotly letterboxes instead)
                   scaleanchor="x", scaleratio=1, constrain="domain"),
        width=fig_width,
        height=fig_height,
        margin=margin,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    # Center the chart: with width='content' Streamlit sizes the element
    # container itself to the figure, and the parent flex column left-aligns
    # it — so the container must center itself (align-self; margin:auto as a
    # fallback for non-flex parents).
    st.markdown(
        "<style>.st-key-opt_heatmap "
        "{ align-self: center; margin-left: auto; margin-right: auto; }"
        "</style>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, width='content', key="opt_heatmap")

    st.caption("Hatched = fewer trades than the min-trades threshold · "
               "blank = no finite value (no trades, PF ∞, undefined Sharpe)")
    with st.expander("How to read this surface", expanded=False):
        st.markdown(
            "- **Hatched cells** keep their color but sit below the min-trades "
            "threshold — read them with suspicion.\n"
            "- **Blank cells** have no finite metric value and are excluded "
            "from the color scale.\n"
            "- **Sharpe (daily)** ×√252 is approximate once day types are "
            "filtered out (fewer days) — comparative use only.\n"
            "- Look for **plateaus that survive both halves and day-type "
            "changes**, not the single brightest cell — the top cell of a "
            "large grid is selection-biased by construction."
        )


def _load_selected_run():
    """Run selectbox + one-time load of trades/meta into session_state."""
    runs = opt_io.list_runs()
    if not runs:
        st.info("No saved optimization runs yet — create one under **New Run**.")
        return None

    current = st.session_state.get("opt_loaded_run")
    index = runs.index(current) if current in runs else 0
    selected = st.selectbox("Optimization run", runs, index=index,
                            key="opt_run_select")

    if selected != st.session_state.get("opt_loaded_run") \
            or "opt_trades" not in st.session_state:
        trades, meta = opt_io.load_run(selected)
        st.session_state.opt_loaded_run = selected
        st.session_state.opt_trades     = trades
        st.session_state.opt_meta       = meta
        st.session_state.pop("opt_grid_key", None)

    return st.session_state.opt_trades, st.session_state.opt_meta


def render_explore():
    success = st.session_state.pop("opt_run_success", None)
    if success:
        st.success(success)

    loaded = _load_selected_run()
    if loaded is None:
        return
    trades, meta = loaded

    axes     = meta.get("axes", {})
    x_axis   = axes.get("x")
    y_axis   = axes.get("y")
    s_axis   = axes.get("slider")
    be_band  = meta.get("be_band_ticks", 0.0)
    split    = meta.get("split_date")

    st.caption(
        f"**{meta.get('strategy')}** on `{meta.get('dataset')}` · "
        f"{meta.get('start_date')} → {meta.get('end_date')} · "
        f"{meta.get('n_combos')} combos · {meta.get('n_trades')} trades · "
        f"BE band {be_band:g} ticks"
    )
    if not meta.get("ff_events_found", True):
        st.warning("This run was built without ff_usd_events.parquet — every "
                   "day is bucketed 'normal'.")
    with st.expander("Held parameters", expanded=False):
        st.json(meta.get("fixed_params", {}))

    if x_axis is None:
        st.error("Run has no X axis — meta.json is incomplete.")
        return

    # ── controls ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 2, 1])
    metric = c1.selectbox("Metric", METRIC_ORDER,
                          format_func=METRIC_LABELS.get, key="opt_metric")
    half = c2.radio("Data half", ["both", "1st", "2nd"], horizontal=True,
                    key="opt_half", disabled=split is None,
                    help="split at the median trading day of the full run")
    min_trades = c3.number_input(
        "Min trades", min_value=0, step=5,
        value=int(meta.get("min_trades_default", MIN_TRADES_DEFAULT)),
        key=f"opt_min_trades_{meta.get('run_name', '')}",
        help="cells with fewer trades (after filtering) are hatched",
    )

    slider_value, slider_desc = None, ""
    if s_axis is not None:
        options = s_axis["values"]
        slider_value = st.select_slider(
            f"{s_axis['param']} (slider axis)", options=options,
            key=f"opt_slider_{meta.get('run_name', '')}",
        ) if len(options) > 1 else options[0]
        slider_desc = f"{s_axis['param']} = {_fmt_axis_value(slider_value)}"

    st.caption("Day types included")
    bucket_cols = st.columns(len(BUCKET_ORDER))
    selected_buckets = []
    for i, (key, label) in enumerate(BUCKET_ORDER):
        with bucket_cols[i]:
            if st.checkbox(label, value=True, key=f"opt_bucket_{key}"):
                selected_buckets.append(key)
    if not selected_buckets:
        st.warning("No day types selected.")
        return

    # ── filter (reductions over the stored trades) + vectorized recompute ─────
    # min_trades and the metric toggle are NOT part of the key: they only
    # re-mask / recolor the cached grid, no recompute.
    grid_key = (meta.get("run_name"), slider_value, tuple(selected_buckets),
                half, float(be_band))
    if st.session_state.get("opt_grid_key") != grid_key:
        df = trades
        if s_axis is not None and slider_value is not None:
            df = df[df[s_axis["param"]] == slider_value]
        if len(selected_buckets) < len(BUCKET_ORDER):
            df = df[df["day_bucket"].isin(selected_buckets)]
        if half != "both" and split is not None:
            dates = pd.to_datetime(df["date"])
            split_ts = pd.Timestamp(split)
            df = df[dates <= split_ts] if half == "1st" else df[dates > split_ts]

        cell_cols = [x_axis["param"]] + ([y_axis["param"]] if y_axis else [])
        grid = compute_metrics_by_cell(df, cell_cols, be_band)
        st.session_state.opt_grid     = grid
        st.session_state.opt_grid_key = grid_key
    grid = st.session_state.opt_grid

    x_values = x_axis["values"]
    y_param  = y_axis["param"] if y_axis else None
    y_values = y_axis["values"] if y_axis else ["—"]

    arrays = _build_grid_arrays(grid, x_axis["param"], x_values, y_param,
                                y_axis["values"] if y_axis else None)

    _render_heatmap(arrays, metric, x_axis["param"], x_values, y_param,
                    y_values, slider_desc, int(min_trades))


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    if st.button("← Back"):
        go_page("home")
    st.title("Strategy Optimizer")
    st.caption("Sweep up to 3 strategy params, store every cell's trades, "
               "explore the metric surface — no auto-picked 'best' config.")
    st.write("")

    # hand-off from a just-finished grid run — must happen BEFORE the radio
    # widget is instantiated (Streamlit forbids setting a widget's state after)
    if st.session_state.pop("opt_switch_to_explore", False):
        st.session_state.opt_mode = "Explore"

    mode = st.radio("Mode", ["New Run", "Explore"], horizontal=True,
                    key="opt_mode", label_visibility="collapsed")
    st.write("")

    if mode == "New Run":
        render_setup()
    else:
        render_explore()
