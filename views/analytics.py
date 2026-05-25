"""
views/analytics.py
==================

Analytics page for the quant research platform.

What this page does (in plain English)
--------------------------------------
1. You pick a saved trades file (the output of a backtest).
2. You configure one or more "instances" — each instance is a combination of
   (trades file + position sizer + sizer params + label).
3. You hit Run, and the page:
     - loads each trades file fresh from disk,
     - applies the chosen sizer to get dollar PnL and an equity curve,
     - stores the sized results in session state.
4. It then shows you, per instance, an individual equity curve, a combined
   overlay chart, and a metrics comparison table.

Why it's structured this way
----------------------------
- Every section of the UI is its own function. `render()` just orchestrates.
- Pure data functions (loading, sizing, metrics) are kept separate from
  Streamlit UI functions (rendering, plotting), so they're trivial to test.
- No module imports from other views. If we ever want to split this into a
  package or a worker, nothing else needs to change.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ===========================================================================
# Constants
# ===========================================================================

# Directory layout is fixed by the project blueprint — these paths are the
# contract between the backtester (writer) and analytics (reader).
TRADES_DIR = Path("data") / "trades"
SIZING_DIR = Path("position_sizing")

# Filenames in /position_sizing/ that aren't sizers themselves.
NON_SIZER_MODULES: set[str] = {"__init__", "base"}

# Keys inside a sizer's PARAMS dict that are ALWAYS driven by the shared
# defaults at the top of the page. We filter these out of the per-instance
# UI so the user doesn't re-type the same account size for every instance.
# To allow per-instance overrides, simply empty this set.
SHARED_SIZER_KEYS: set[str] = {"account_size", "dollars_per_tick"}

# Defaults for the shared controls. Matching backtester.py conventions.
DEFAULT_ACCOUNT_SIZE = 100_000.0
DEFAULT_DOLLARS_PER_TICK = 12.50

# Trading-day count used to annualize the Sharpe ratio. Standard convention
# for US equity/futures backtests.
TRADING_DAYS_PER_YEAR = 252

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
    "ZN":  {"tick_size": 0.015625, "ticks_per_point": 64,  "dollars_per_tick": 15.625},  # 1/64
    "ZB":  {"tick_size": 0.03125,  "ticks_per_point": 32,  "dollars_per_tick": 31.25},   # 1/32
    "ZF":  {"tick_size": 0.0078125,"ticks_per_point": 128, "dollars_per_tick": 7.8125},  # 1/128
    "ZT":  {"tick_size": 0.00390625,"ticks_per_point": 256, "dollars_per_tick": 7.8125},  # 1/128 — verify, ZT is quoted in 1/256 in some venues
    "SR3": {"tick_size": 0.0025,   "ticks_per_point": 400, "dollars_per_tick": 6.25},

    # Energy
    "CL":  {"tick_size": 0.01, "ticks_per_point": 100, "dollars_per_tick": 10.00},
    "QM":  {"tick_size": 0.025,"ticks_per_point": 40,  "dollars_per_tick": 12.50},
    "NG":  {"tick_size": 0.001,"ticks_per_point": 1000,"dollars_per_tick": 10.00},
    "RB":  {"tick_size": 0.0001,"ticks_per_point": 10000,"dollars_per_tick": 4.20},  # ~4.20 at 42000 gal contract — price-dependent, verify
    "HO":  {"tick_size": 0.0001,"ticks_per_point": 10000,"dollars_per_tick": 4.20},  # same as RB

    # Metals
    "GC":  {"tick_size": 0.10, "ticks_per_point": 10,  "dollars_per_tick": 10.00},
    "MGC": {"tick_size": 0.10, "ticks_per_point": 10,  "dollars_per_tick": 1.00},
    "SI":  {"tick_size": 0.005,"ticks_per_point": 200, "dollars_per_tick": 25.00},
    "HG":  {"tick_size": 0.0005,"ticks_per_point": 2000,"dollars_per_tick": 12.50},

    # Grains
    "ZC":  {"tick_size": 0.25, "ticks_per_point": 4,   "dollars_per_tick": 12.50},
    "ZS":  {"tick_size": 0.25, "ticks_per_point": 4,   "dollars_per_tick": 12.50},
    "ZW":  {"tick_size": 0.25, "ticks_per_point": 4,   "dollars_per_tick": 12.50},

    # FX
    "6E":  {"tick_size": 0.00005,"ticks_per_point": 20000,"dollars_per_tick": 6.25},
    "6J":  {"tick_size": 0.0000005,"ticks_per_point": 2000000,"dollars_per_tick": 6.25},
    "6B":  {"tick_size": 0.0001,"ticks_per_point": 10000,"dollars_per_tick": 6.25},
    "6C":  {"tick_size": 0.00005,"ticks_per_point": 20000,"dollars_per_tick": 5.00},

    # Crypto
    "BTC": {"tick_size": 5.00, "ticks_per_point": 0.2, "dollars_per_tick": 25.00},
}


# ===========================================================================
# Navigation
# ===========================================================================

def go_page(page: str) -> None:
    """Route helper — kept local so analytics doesn't depend on other views."""
    st.session_state.page = page
    st.rerun()

def get_dollars_per_tick(trades_filename: str) -> float:
    asset = trades_filename.split("_")[0]
    if asset not in ASSET_INFO:
        raise ValueError(f"Unknown asset '{asset}' derived from filename '{trades_filename}'. Add it to ASSET_INFO.")
    return ASSET_INFO[asset]["dollars_per_tick"]

# ===========================================================================
# Filesystem discovery
# ===========================================================================

def list_trades_files() -> list[str]:
    """Return all saved trades parquet filenames, sorted alphabetically."""
    if not TRADES_DIR.exists():
        return []
    return sorted(p.name for p in TRADES_DIR.glob("*.parquet"))


def list_sizers() -> list[str]:
    """
    Return all available position sizers (module stems), sorted alphabetically.

    A "sizer" is any .py file in /position_sizing/ except the ones listed in
    NON_SIZER_MODULES. Drop a new file in that folder and it shows up here —
    no code change required.
    """
    if not SIZING_DIR.exists():
        return []
    return sorted(
        p.stem
        for p in SIZING_DIR.glob("*.py")
        if p.stem not in NON_SIZER_MODULES
    )


# ===========================================================================
# Dynamic module loading
# ===========================================================================

def load_sizer(name: str):
    """
    Dynamically load a sizer module from /position_sizing/{name}.py.

    Same importlib pattern as backtester.load_strategy — keeps analytics
    self-contained without requiring sizers to be registered anywhere.
    """
    path = SIZING_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ===========================================================================
# Trades loading
# ===========================================================================

def _coerce_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure entry_time / exit_time / date columns are typed correctly.

    Notes on the dtype check:
    - We use pd.api.types.is_datetime64_any_dtype because it correctly handles
      timezone-aware dtypes (e.g. datetime64[ns, America/New_York]). NumPy's
      np.issubdtype raises on tz-aware dtypes, which is how an earlier version
      of this code crashed on real data.
    - For the 'date' column, we only re-parse when it's stored as strings
      (object dtype). If pandas already gave us a python date object column
      (typical for `.dt.date` results), we leave it alone.
    """
    for col in ("entry_time", "exit_time"):
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])

    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        if df["date"].dtype == object:
            df["date"] = pd.to_datetime(df["date"]).dt.date

    return df


def load_trades(filename: str) -> pd.DataFrame:
    """Read a trades parquet file and normalize its datetime columns."""
    df = pd.read_parquet(TRADES_DIR / filename)
    return _coerce_datetime_columns(df)


# ===========================================================================
# Per-instance run pipeline
# ===========================================================================

def run_instance(trades_file: str, sizer_name: str, params: dict) -> pd.DataFrame:
    """
    Load trades, apply the chosen sizer, return the sized DataFrame.

    Non-mutation guarantee: the sizer is contractually required to return a
    copy (see position_sizing/base.py). We also reload from parquet every
    time, so nothing persists between Run clicks — clean slate on each run.
    """
    raw = load_trades(trades_file)
    sizer = load_sizer(sizer_name)
    return sizer.apply(raw, params)


# ===========================================================================
# Metrics
# ===========================================================================
# All metric helpers work on the dollar-denominated columns produced by the
# sizer (trade_pnl, equity). Do NOT use the `ticks` column here — that's the
# raw backtester output before sizing.

def _empty_metrics_row(label: str, skipped: int = 0) -> dict:
    """Default metrics row when a run has no trades — keeps the table shape stable."""
    return {
        "Label": label,
        "Total ($)": 0.0,
        "Final Equity ($)": 0.0,
        "Max Drawdown ($)": 0.0,
        "Max Drawdown (%)": 0.0,
        "Sharpe": 0.0,
        "Win Rate": 0.0,
        "Profit Factor": 0.0,
        "Skipped Trades": skipped,
    }


def _max_drawdown(equity: pd.Series) -> tuple[float, float]:
    """
    Return (max_drawdown_dollars, max_drawdown_percent).

    Both are expressed as negative numbers (or zero). The percentage is
    computed against the running peak at each point, not the global peak —
    this is the standard definition and is what people usually expect when
    they see "max drawdown %".
    """
    if equity.empty:
        return 0.0, 0.0

    running_peak = equity.cummax()
    drawdown_abs = equity - running_peak
    max_dd_dollars = float(drawdown_abs.min())

    # Guard against div-by-zero if starting equity somehow was non-positive.
    dd_pct_series = np.where(
        running_peak > 0,
        drawdown_abs / running_peak * 100.0,
        0.0,
    )
    max_dd_pct = float(np.min(dd_pct_series))

    return max_dd_dollars, max_dd_pct


def _annualized_sharpe(trades: pd.DataFrame) -> float:
    """
    Sharpe ratio computed on daily PnL, annualized by sqrt(252).

    Preferred input: a 'date' column (day-grain, no timezone issues).
    Fallback: resample 'entry_time' to daily. This matters because a strategy
    might enter and exit within a single day, and we want one PnL sample per
    trading day for Sharpe to make sense.
    """
    if "date" in trades.columns:
        daily_pnl = trades.groupby("date")["trade_pnl"].sum()
    else:
        daily_pnl = trades.set_index("entry_time")["trade_pnl"].resample("1D").sum()

    # Need at least 2 observations and non-zero std for Sharpe to be defined.
    if len(daily_pnl) <= 1 or daily_pnl.std(ddof=1) == 0:
        return 0.0

    return float(
        daily_pnl.mean() / daily_pnl.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    )


def _win_rate_and_profit_factor(pnl: pd.Series) -> tuple[float, float]:
    """
    Return (win_rate, profit_factor).

    Profit factor edge cases:
    - Zero losses and at least one win  -> inf (well-defined edge case)
    - No trades at all                   -> 0.0 (avoids NaN in the table)
    """
    if pnl.empty:
        return 0.0, 0.0

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    win_rate = float(len(wins) / len(pnl))

    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())  # make loss positive for the ratio

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    return win_rate, profit_factor


def compute_analytics_metrics(trades: pd.DataFrame, label: str) -> dict:
    """
    Compute the full metrics row for a single sized run.

    Every value in this dict becomes a column in the comparison table, so
    adding new metrics is just a matter of extending this function and the
    empty-row template above.
    """
    skipped = int(trades.attrs.get("skipped_trades", 0))

    # Guard clause — if the sizer produced no usable output, return the
    # empty-row template so the comparison table still has a row for this
    # instance instead of silently dropping it.
    if trades.empty or "trade_pnl" not in trades.columns or "equity" not in trades.columns:
        return _empty_metrics_row(label, skipped)

    pnl = trades["trade_pnl"]
    equity = trades["equity"]

    max_dd_dollars, max_dd_pct = _max_drawdown(equity)
    win_rate, profit_factor = _win_rate_and_profit_factor(pnl)
    sharpe = _annualized_sharpe(trades)

    return {
        "Label": label,
        "Total ($)": round(float(pnl.sum()), 2),
        "Final Equity ($)": round(float(equity.iloc[-1]), 2),
        "Max Drawdown ($)": round(max_dd_dollars, 2),
        "Max Drawdown (%)": round(max_dd_pct, 2),
        "Sharpe": round(sharpe, 2),
        "Win Rate": round(win_rate, 4),
        "Profit Factor": (
            round(profit_factor, 2) if np.isfinite(profit_factor) else float("inf")
        ),
        "Skipped Trades": skipped,
    }


# ===========================================================================
# Plotting
# ===========================================================================

def _equity_trace(trades: pd.DataFrame, label: str) -> go.Scatter:
    """One equity-curve line, reused by both individual and combined figures."""
    return go.Scatter(
        x=trades["entry_time"],
        y=trades["equity"],
        mode="lines",
        name=label,
        line=dict(width=2),
        hovertemplate=f"{label}<br>%{{x}}<br>Equity: $%{{y:,.2f}}<extra></extra>",
    )


def _apply_equity_layout(fig: go.Figure, account_size: float, show_legend: bool) -> None:
    """Shared layout + starting-equity reference line for equity figures."""
    fig.add_hline(
        y=account_size,
        line_dash="dash",
        line_color="gray",
        annotation_text="Starting equity",
        annotation_position="bottom right",
    )
    layout: dict[str, Any] = dict(
        height=400,
        hovermode="x unified",
        xaxis_title="Time",
        yaxis_title="Equity ($)",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    if show_legend:
        # Horizontal legend above the chart — only useful when multiple traces.
        layout["legend"] = dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        )
    fig.update_layout(**layout)


def equity_curve_figure(trades: pd.DataFrame, label: str, account_size: float) -> go.Figure:
    """Single-run equity curve."""
    fig = go.Figure()
    fig.add_trace(_equity_trace(trades, label))
    _apply_equity_layout(fig, account_size, show_legend=False)
    return fig


def combined_equity_figure(runs: list[dict], account_size: float) -> go.Figure:
    """All non-empty runs overlaid on one chart, one trace per run."""
    fig = go.Figure()
    for run in runs:
        if run["trades"].empty:
            continue
        fig.add_trace(_equity_trace(run["trades"], run["label"]))
    _apply_equity_layout(fig, account_size, show_legend=True)
    return fig


# ===========================================================================
# UI — Section 1: Shared defaults
# ===========================================================================

def render_shared_defaults(trades_files: list[str]) -> tuple[str, float]:
    st.subheader("Shared defaults")
    col_file, col_account = st.columns(2)

    with col_file:
        default_trades_file = st.selectbox(
            "Default trades file",
            trades_files,
            key="shared_default_trades_file",
        )
    with col_account:
        account_size = st.number_input(
            "account_size",
            value=DEFAULT_ACCOUNT_SIZE,
            step=1000.0,
            format="%.2f",
            key="shared_account_size",
        )

    return default_trades_file, account_size


# ===========================================================================
# UI — Section 2: Instance builder
# ===========================================================================

def _render_single_sizer_param(key: str, default: Any, widget_key: str) -> Any:
    """
    Render one sizer param as an st.number_input.

    Branches on type so ints keep their step=1 feel and floats get step=0.1
    with 2-decimal display — matches backtester.render_params exactly.
    """
    if isinstance(default, float):
        return st.number_input(key, value=default, step=0.1, format="%.2f", key=widget_key)
    return st.number_input(key, value=default, step=1, key=widget_key)


def render_sizer_params(
    sizer,
    account_size: float,
    key_prefix: str,
) -> dict:
    params: dict = {}

    if hasattr(sizer, "PARAMS"):
        user_keys = [k for k in sizer.PARAMS.keys() if k not in SHARED_SIZER_KEYS]

        if user_keys:
            param_cols = st.columns(len(user_keys))
            for i, key in enumerate(user_keys):
                with param_cols[i]:
                    params[key] = _render_single_sizer_param(
                        key,
                        sizer.PARAMS[key],
                        widget_key=f"{key_prefix}_{key}",
                    )

    params["account_size"] = account_size
    # dollars_per_tick intentionally NOT injected here — derived from filename
    # in _execute_instance so it's always correct for the selected asset.
    return params


def _render_instance_header(
    instance_index: int,
    trades_files: list[str],
    sizers: list[str],
    default_trades_file: str,
) -> tuple[str, str]:
    """Render the trades-file and sizer selectboxes for one instance."""
    row = st.columns(2)
    with row[0]:
        trades_file = st.selectbox(
            "trades file",
            trades_files,
            index=(
                trades_files.index(default_trades_file)
                if default_trades_file in trades_files
                else 0
            ),
            key=f"inst_{instance_index}_trades_file",
        )
    with row[1]:
        sizer_name = st.selectbox(
            "sizer",
            sizers,
            index=0,
            key=f"inst_{instance_index}_sizer",
        )
    return trades_file, sizer_name


def _render_instance_label(instance_index: int, sizer_name: str) -> str:
    """
    Render the editable label with a sensible auto-default.

    Keying on BOTH (index, sizer_name) means Streamlit treats it as a fresh
    widget whenever the sizer changes — which regenerates the auto-default
    — while preserving manual edits as long as the sizer stays the same.
    """
    auto_label = f"{sizer_name}_{instance_index + 1}"
    label = st.text_input(
        "label",
        value=auto_label,
        key=f"label_{instance_index}_{sizer_name}",
    )
    # Empty string -> fall back to auto-label so runs always have a name.
    return label or auto_label


def render_one_instance(
    instance_index: int,
    trades_files: list[str],
    sizers: list[str],
    default_trades_file: str,
    account_size: float,
) -> dict | None:
    with st.expander(f"Instance {instance_index + 1}", expanded=(instance_index == 0)):
        trades_file, sizer_name = _render_instance_header(
            instance_index, trades_files, sizers, default_trades_file
        )

        try:
            sizer = load_sizer(sizer_name)
        except Exception as e:
            st.error(f"Failed to load sizer `{sizer_name}`: {e}")
            return None

        params = render_sizer_params(
            sizer,
            account_size=account_size,
            key_prefix=f"inst_{instance_index}",
        )
        label = _render_instance_label(instance_index, sizer_name)

    return {
        "label": label,
        "trades_file": trades_file,
        "sizer": sizer_name,
        "params": params,
    }


def render_instance_builder(
    trades_files: list[str],
    sizers: list[str],
    default_trades_file: str,
    account_size: float,
) -> list[dict]:
    st.subheader("Instances")

    n_instances = st.number_input(
        "Number of instances",
        min_value=1,
        value=1,
        step=1,
        key="n_instances",
    )

    configs: list[dict] = []
    for i in range(int(n_instances)):
        cfg = render_one_instance(
            instance_index=i,
            trades_files=trades_files,
            sizers=sizers,
            default_trades_file=default_trades_file,
            account_size=account_size,
        )
        if cfg is not None:
            configs.append(cfg)

    return configs


# ===========================================================================
# UI — Section 3: Run
# ===========================================================================

def _execute_instance(cfg: dict) -> dict | None:
    try:
        dollars_per_tick = get_dollars_per_tick(cfg["trades_file"])
    except ValueError as e:
        st.error(str(e))
        return None

    params = {**cfg["params"], "dollars_per_tick": dollars_per_tick}

    try:
        sized = run_instance(
            trades_file=cfg["trades_file"],
            sizer_name=cfg["sizer"],
            params=params,
        )
    except Exception as e:
        st.error(f"Instance `{cfg['label']}` failed: {e}")
        return None

    return {
        "label": cfg["label"],
        "trades_file": cfg["trades_file"],
        "sizer": cfg["sizer"],
        "params": params,
        "trades": sized,
    }


def _warn_about_skipped_trades(runs: list[dict]) -> None:
    """
    Summarize any skipped trades (size=0) across all runs into one warning.

    Typically this happens with kelly/risk_based sizers when a tight stop
    combined with a small account gives a position size that rounds to 0.
    """
    offenders = [
        (r["label"], int(r["trades"].attrs.get("skipped_trades", 0)))
        for r in runs
        if int(r["trades"].attrs.get("skipped_trades", 0)) > 0
    ]
    if not offenders:
        return

    summary = ", ".join(f"{label}: {count}" for label, count in offenders)
    st.warning(f"Some trades were skipped (size=0) — {summary}")


def execute_all_instances(configs: list[dict]) -> list[dict]:
    """
    Run every configured instance with a progress bar and return the run list.

    Note: this function writes errors to the Streamlit UI via _execute_instance
    but returns only successful runs, so callers don't have to filter Nones.
    """
    runs: list[dict] = []
    progress = st.progress(0.0, text="Running instances…")

    total = max(len(configs), 1)
    for idx, cfg in enumerate(configs, start=1):
        result = _execute_instance(cfg)
        if result is not None:
            runs.append(result)
        progress.progress(idx / total)

    progress.empty()
    return runs


def render_run_button(instance_configs: list[dict]) -> None:
    """
    Render the centered Run button and handle the click.

    On click we clear session state first so stale runs (e.g. from a previous
    larger instance count) don't leak into the results section.
    """
    _, _, btn_col, _, _ = st.columns(5)
    with btn_col:
        clicked = st.button("Run", type="primary", width='stretch')

    if not clicked:
        return

    # Wipe first — prevents stale results when the user reduces instance count.
    st.session_state.analytics_runs = None

    runs = execute_all_instances(instance_configs)
    st.session_state.analytics_runs = runs
    _warn_about_skipped_trades(runs)


# ===========================================================================
# UI — Section 4: Results
# ===========================================================================

def render_individual_equity_curves(runs: list[dict], account_size: float) -> None:
    """One subheader + one equity chart per run."""
    st.subheader("Individual equity curves")
    for run in runs:
        st.subheader(run["label"])
        trades = run["trades"]
        if trades.empty:
            st.info("No trades to display for this instance.")
            continue
        st.plotly_chart(
            equity_curve_figure(trades, run["label"], account_size),
            width='stretch',
        )
        st.write("")


def render_combined_equity_curve(runs: list[dict], account_size: float) -> None:
    """All non-empty runs overlaid on one chart."""
    st.subheader("Combined equity curve")
    non_empty = [r for r in runs if not r["trades"].empty]
    if not non_empty:
        st.info("No non-empty runs to combine.")
        return
    st.plotly_chart(
        combined_equity_figure(non_empty, account_size),
        width='stretch',
    )


def render_metrics_table(runs: list[dict]) -> None:
    """One-row-per-run comparison table."""
    st.subheader("Metrics comparison")
    rows = [compute_analytics_metrics(r["trades"], r["label"]) for r in runs]
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


def render_results(runs: list[dict], account_size: float) -> None:
    """Render all three results sections, in order."""
    render_individual_equity_curves(runs, account_size)
    render_combined_equity_curve(runs, account_size)
    st.write("")
    render_metrics_table(runs)


# ===========================================================================
# Page entry point
# ===========================================================================

def _render_header() -> None:
    """Title, blurb, and back button — same pattern as other views."""
    st.title("Analytics")
    st.write("Apply position sizing to saved backtest trades and compare runs.")
    st.write("")
    if st.button("← Back"):
        go_page("home")


def _validate_environment(trades_files: list[str], sizers: list[str]) -> bool:
    """
    Make sure we have trades to analyze and sizers to apply.

    Returns True if we're good to render the rest of the page, False if
    we've shown a warning and the caller should bail out.
    """
    if not trades_files:
        st.warning("No trades files found in `data/trades/`. Run a backtest first.")
        return False
    if not sizers:
        st.warning("No position sizers found in `position_sizing/`.")
        return False
    return True


def render() -> None:
    _render_header()

    trades_files = list_trades_files()
    sizers = list_sizers()
    if not _validate_environment(trades_files, sizers):
        return

    default_trades_file, account_size = render_shared_defaults(trades_files)
    st.write("")

    instance_configs = render_instance_builder(
        trades_files=trades_files,
        sizers=sizers,
        default_trades_file=default_trades_file,
        account_size=account_size,
    )
    st.write("")

    render_run_button(instance_configs)
    st.write("")

    runs = st.session_state.get("analytics_runs")
    if runs:
        render_results(runs, account_size)


# Allow `streamlit run views/analytics.py` for quick isolated testing.
if __name__ == "__main__":
    render()