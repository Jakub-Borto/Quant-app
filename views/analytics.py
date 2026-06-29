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
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
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

# Human labels for the day-type keys written into trades-file metadata by the
# backtester (Part B's DAY_TYPE_ORDER). Mirrored here rather than imported —
# same no-cross-view-imports convention as the duplicated ASSET_INFO above.
DAY_TYPE_LABELS = {
    "holiday":     "Holidays",
    "fomc":        "FOMC",
    "cpi":         "CPI",
    "nfp":         "Non-Farm Employment",
    "ppi":         "PPI",
    "high_impact": "Other High Impact News",
    "normal":      "Normal Trading Days",
}

ASSET_INFO = {
    # Equity Index
    "ES":  {"tick_size": 0.25, "ticks_per_point": 4,   "dollars_per_tick": 12.50,   "commissions_per_contract": 2.88},
    "NQ":  {"tick_size": 0.25, "ticks_per_point": 4,   "dollars_per_tick": 5.00,    "commissions_per_contract": 2.88},
    "RTY": {"tick_size": 0.10, "ticks_per_point": 10,  "dollars_per_tick": 5.00,    "commissions_per_contract": 2.88},
    "YM":  {"tick_size": 1.00, "ticks_per_point": 1,   "dollars_per_tick": 5.00,    "commissions_per_contract": 2.88},
    "MES": {"tick_size": 0.25, "ticks_per_point": 4,   "dollars_per_tick": 1.25,    "commissions_per_contract": 0.95, "parent": "ES"},
    "MNQ": {"tick_size": 0.25, "ticks_per_point": 4,   "dollars_per_tick": 0.50,    "commissions_per_contract": 0.95, "parent": "NQ"},
    "M2K": {"tick_size": 0.10, "ticks_per_point": 10,  "dollars_per_tick": 0.50,    "commissions_per_contract": 0.95, "parent": "RTY"},
    "MYM": {"tick_size": 1.00, "ticks_per_point": 1,   "dollars_per_tick": 0.50,    "commissions_per_contract": 0.95, "parent": "YM"},

    # Rates
    "ZN":  {"tick_size": 0.015625, "ticks_per_point": 64,  "dollars_per_tick": 15.625,   "commissions_per_contract": 2.30},  # 1/64
    "ZB":  {"tick_size": 0.03125,  "ticks_per_point": 32,  "dollars_per_tick": 31.25,   "commissions_per_contract": 2.37},   # 1/32
    "ZF":  {"tick_size": 0.0078125,"ticks_per_point": 128, "dollars_per_tick": 7.8125,  "commissions_per_contract": 2.15},  # 1/128
    "ZT":  {"tick_size": 0.00390625,"ticks_per_point": 256, "dollars_per_tick": 7.8125, "commissions_per_contract": 2.15},  # 1/128 — verify, ZT is quoted in 1/256 in some venues
    "SR3": {"tick_size": 0.0025,   "ticks_per_point": 400, "dollars_per_tick": 6.25,    "commissions_per_contract": 2.10},   # commision

    # Energy
    "CL":  {"tick_size": 0.01, "ticks_per_point": 100, "dollars_per_tick": 10.00,   "commissions_per_contract": 3.00},
    "QM":  {"tick_size": 0.025,"ticks_per_point": 40,  "dollars_per_tick": 12.50,   "commissions_per_contract": 2.70},
    "NG":  {"tick_size": 0.001,"ticks_per_point": 1000,"dollars_per_tick": 10.00,   "commissions_per_contract": 3.10},
    "RB":  {"tick_size": 0.0001,"ticks_per_point": 10000,"dollars_per_tick": 4.20,  "commissions_per_contract": 3.00},  # ~4.20 at 42000 gal contract — price-dependent, verify
    "HO":  {"tick_size": 0.0001,"ticks_per_point": 10000,"dollars_per_tick": 4.20,  "commissions_per_contract": 3.00},  # same as RB

    # Metals
    "GC":  {"tick_size": 0.10, "ticks_per_point": 10,  "dollars_per_tick": 10.00,   "commissions_per_contract": 3.10},
    "MGC": {"tick_size": 0.10, "ticks_per_point": 10,  "dollars_per_tick": 1.00,    "commissions_per_contract": 1.20, "parent": "GC"},
    "SI":  {"tick_size": 0.005,"ticks_per_point": 200, "dollars_per_tick": 25.00,   "commissions_per_contract": 3.10},
    "HG":  {"tick_size": 0.0005,"ticks_per_point": 2000,"dollars_per_tick": 12.50,  "commissions_per_contract": 3.10},

    # Grains
    "ZC":  {"tick_size": 0.25, "ticks_per_point": 4,   "dollars_per_tick": 12.50,   "commissions_per_contract": 3.60},
    "ZS":  {"tick_size": 0.25, "ticks_per_point": 4,   "dollars_per_tick": 12.50,   "commissions_per_contract": 3.60},
    "ZW":  {"tick_size": 0.25, "ticks_per_point": 4,   "dollars_per_tick": 12.50,   "commissions_per_contract": 3.60},

    # FX
    "6E":  {"tick_size": 0.00005,"ticks_per_point": 20000,"dollars_per_tick": 6.25, "commissions_per_contract": 3.10},
    "6J":  {"tick_size": 0.0000005,"ticks_per_point": 2000000,"dollars_per_tick": 6.25, "commissions_per_contract": 3.10},
    "6B":  {"tick_size": 0.0001,"ticks_per_point": 10000,"dollars_per_tick": 6.25,  "commissions_per_contract": 3.10},
    "6C":  {"tick_size": 0.00005,"ticks_per_point": 20000,"dollars_per_tick": 5.00, "commissions_per_contract": 3.10},

    # Crypto
    "BTC": {"tick_size": 5.00, "ticks_per_point": 0.2, "dollars_per_tick": 25.00,   "commissions_per_contract": 8.00},
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


def _micro_child(asset: str) -> str | None:
    """Return the micro ticker whose `parent` is `asset`, or None. An asset is
    'microable' iff such a child exists — that's the only decomposition flag."""
    for ticker, info in ASSET_INFO.items():
        if info.get("parent") == asset:
            return ticker
    return None


def get_commission_info(trades_filename: str) -> tuple[float | None, float | None]:
    """
    (full_commission, micro_commission) for the file's asset, mirroring
    get_dollars_per_tick. `full` is None if the asset has no
    commissions_per_contract key (graceful degradation → caller bills 0 +
    warns). `micro` is the child's commission when a micro child exists, else
    None (non-microable).
    """
    asset = trades_filename.split("_")[0]
    info  = ASSET_INFO.get(asset, {})
    full  = info.get("commissions_per_contract")

    child = _micro_child(asset)
    micro = ASSET_INFO[child].get("commissions_per_contract") if child else None
    return full, micro

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


def read_filter_metadata(filename: str) -> dict | None:
    """
    Read the filter kv-metadata the backtester stamps onto saved trades
    (Part C). Returns None for unfiltered or legacy (pre-metadata) files; else
    {"day_types": [...keys], "trade_types": "all" | [...values]}.
    """
    schema_meta = pq.read_schema(TRADES_DIR / filename).metadata or {}
    if schema_meta.get(b"filtered", b"false").decode() != "true":
        return None

    day_types = json.loads(schema_meta.get(b"selected_day_types", b"[]").decode())
    tt_raw    = schema_meta.get(b"selected_trade_types", b"all").decode()
    trade_types = "all" if tt_raw == "all" else json.loads(tt_raw)
    return {"day_types": day_types, "trade_types": trade_types}


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

def _taken_subset(trades: pd.DataFrame) -> pd.DataFrame:
    """
    The "taken" trades — rows the sizer actually allocated to (contracts > 0).

    Skipped trades are contracts-0 rows the sizer leaves in the frame; counting
    them as non-wins distorts win rate. Fallback: if there's no `contracts`
    column, treat every row as taken.
    """
    if "contracts" in trades.columns:
        return trades[trades["contracts"] > 0]
    return trades


def _sharpe_trade(trades: pd.DataFrame) -> float:
    """
    Per-trade Sharpe annualized by actual trading days — mirrors the
    backtester's trade-Sharpe: mean/std(ddof=1) * sqrt(n_trading_days).
    """
    pnl = trades["trade_pnl"]
    std = pnl.std(ddof=1)
    if len(pnl) <= 1 or not np.isfinite(std) or std == 0:
        return 0.0

    n_trading_days = trades["date"].nunique() if "date" in trades.columns else len(pnl)
    return float(pnl.mean() / std * np.sqrt(n_trading_days))


def _calmar(total_pnl: float, max_dd_dollars: float) -> float:
    """
    Cheap Calmar (not time-annualized): total P&L / |max drawdown $|.
    ∞ when there's no drawdown (max DD >= 0). Consistent with the backtester.
    """
    if max_dd_dollars < 0:
        return total_pnl / abs(max_dd_dollars)
    return float("inf")


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


# ===========================================================================
# Cost model — commission + slippage, deducted in dollar space AFTER the sizer
# ===========================================================================
# Costs are post-hoc dollar deductions on the sizer's gross `trade_pnl`. They
# never run inside a sizer. The `contracts` column (1.3 = 1 full + 3 micro)
# drives both: commission decomposes full vs micro (commission is per-contract,
# not proportional); slippage is proportional so it scales with `contracts`
# directly. Slippage is classified on the GROSS sign, before any deduction.

def compute_cost_series(trades: pd.DataFrame, filename: str, n: int,
                        label: str = "") -> dict:
    """
    Return {"commission": Series, "slippage": Series, "warnings": [str]} aligned
    to `trades.index`. Skipped trades (contracts == 0) cost 0 automatically.
    """
    warnings: list[str] = []
    zero = pd.Series(0.0, index=trades.index)

    if trades.empty or "contracts" not in trades.columns or "trade_pnl" not in trades.columns:
        return {"commission": zero, "slippage": zero.copy(), "warnings": warnings}

    asset = filename.split("_")[0]
    tag   = f"{label} ({asset})" if label else asset

    try:
        dpt = get_dollars_per_tick(filename)
    except ValueError as e:
        warnings.append(str(e))
        return {"commission": zero, "slippage": zero.copy(), "warnings": warnings}

    contracts = trades["contracts"].astype(float)
    gross     = trades["trade_pnl"]
    full_comm, micro_comm = get_commission_info(filename)

    # ── Commission (full/micro decomposition; ×2 = entry + exit) ──────────────
    if full_comm is None:
        commission = zero.copy()
        warnings.append(f"{tag}: no commissions_per_contract — commission billed as $0.")
    else:
        full_count = np.floor(contracts)
        if micro_comm is not None:
            micro_count = np.round((contracts - full_count) * 10)   # round(), never int() — float dust
            commission  = (full_count * full_comm + micro_count * micro_comm) * 2
        else:
            if ((contracts - full_count).abs() > 1e-9).any():
                warnings.append(f"{tag}: fractional contracts on a non-microable asset — likely a sizer bug; using round().")
            commission = np.round(contracts) * full_comm * 2
        commission = pd.Series(np.asarray(commission, dtype=float), index=trades.index)

    # ── Slippage (proportional; classified on GROSS sign) ─────────────────────
    slip_ticks = pd.Series(
        np.where(gross > 0, n, np.where(gross < 0, 2 * n, n)),
        index=trades.index,
    ).astype(float)
    slippage = pd.Series(np.asarray(slip_ticks * dpt * contracts, dtype=float), index=trades.index)

    return {"commission": commission, "slippage": slippage, "warnings": warnings}


def enrich_run(run: dict, account_size: float, n: int) -> dict:
    """
    Augment a run with cost artifacts. Adds:
      - curves: ordered {label: (pnl_series, equity_series)} for the four curves
      - net_trades: copy of the sized frame with trade_pnl/equity set to NET
      - gross_total, cost_drag (= gross_total − net_total), warnings
    Computed at render time so the slippage slider / account size update the
    curves without re-running the sizer.
    """
    trades = run["trades"]
    costs  = compute_cost_series(trades, run["trades_file"], n, run["label"])

    has_pnl = (not trades.empty) and ("trade_pnl" in trades.columns)
    gross   = trades["trade_pnl"] if has_pnl else pd.Series(dtype=float)
    comm, slip = costs["commission"], costs["slippage"]

    pnl_net = gross - comm - slip

    def _equity(series: pd.Series) -> pd.Series:
        return account_size + series.cumsum()

    curves = {
        "Gross":         (gross,         _equity(gross)),
        "+ Commissions": (gross - comm,  _equity(gross - comm)),
        "+ Slippage":    (gross - slip,  _equity(gross - slip)),
        "+ Both (net)":  (pnl_net,       _equity(pnl_net)),
    }

    net_trades = trades.copy()
    if has_pnl:
        net_trades["trade_pnl"] = pnl_net
        net_trades["equity"]    = _equity(pnl_net)

    gross_total = float(gross.sum())   if len(gross)   else 0.0
    net_total   = float(pnl_net.sum()) if len(pnl_net) else 0.0

    return {
        **run,
        "net_trades":  net_trades,
        "curves":      curves,
        "gross_total": gross_total,
        "cost_drag":   gross_total - net_total,
        "warnings":    costs["warnings"],
    }


# ── Metric registry — single source of truth for selector, blocks, table ──────
# Each entry: (key, label, formatter). The formatter turns a raw value into the
# string shown in per-instance st.metric tiles; the comparison table stores the
# rounded raw value (so it stays sortable). Add a metric here and it appears in
# the multiselect, the per-instance grid, and the table automatically.

def _fmt_money(x):  return f"${x:,.2f}"
def _fmt_ratio(x):  return "∞" if x == float("inf") else f"{x:.2f}"
def _fmt_pct(x):    return f"{x:.1%}"          # win rate is a 0–1 fraction
def _fmt_pct_pts(x):return f"{x:.2f}%"         # max-dd % is already in percent
def _fmt_int(x):    return f"{int(x)}"

METRIC_REGISTRY = [
    ("final_equity",  "Final Equity ($)", _fmt_money),
    ("total",         "Total ($)",        _fmt_money),
    ("avg_trade",     "Avg Trade ($)",    _fmt_money),
    ("avg_win",       "Avg Win ($)",      _fmt_money),
    ("avg_loss",      "Avg Loss ($)",     _fmt_money),
    ("largest_win",   "Largest Win ($)",  _fmt_money),
    ("largest_loss",  "Largest Loss ($)", _fmt_money),
    ("sharpe_daily",  "Sharpe (daily)",   _fmt_ratio),
    ("sharpe_trade",  "Sharpe (trade)",   _fmt_ratio),
    ("profit_factor", "Profit Factor",    _fmt_ratio),
    ("calmar",        "Calmar",           _fmt_ratio),
    ("max_dd_dollars","Max Drawdown ($)", _fmt_money),
    ("max_dd_pct",    "Max Drawdown (%)", _fmt_pct_pts),
    ("max_peak",      "Max Peak ($)",     _fmt_money),
    ("total_trades",  "Total Trades",     _fmt_int),
    ("win_rate",      "Win Rate",         _fmt_pct),
    ("skipped",       "Skipped Trades",   _fmt_int),
    ("cost_drag",     "Cost Drag ($)",    _fmt_money),
]

# Convenience lookups.
_KEY_TO_LABEL     = {key: label for key, label, _f in METRIC_REGISTRY}
_KEY_TO_FORMATTER = {key: fmt for key, _label, fmt in METRIC_REGISTRY}

# The four equity curves the statistics can be computed on. The user picks one;
# the chart always shows all four. DEFAULT_CURVE = net (the cost-adjusted truth).
CURVE_LABELS  = ["Gross", "+ Commissions", "+ Slippage", "+ Both (net)"]
DEFAULT_CURVE = "+ Both (net)"


def _empty_metric_values(skipped: int = 0) -> dict:
    """Zero-filled values for every registry key — empty/all-skipped instances."""
    vals = {key: 0.0 for key, _l, _f in METRIC_REGISTRY}
    vals["total_trades"] = 0
    vals["skipped"]      = skipped
    return vals


def compute_metric_values(view: pd.DataFrame, gross_total: float) -> dict:
    """
    Raw value for every registry key, computed on a single curve's view frame
    (`trade_pnl`/`equity` set to the chosen curve — gross / +comm / +slip / net).
    `cost_drag = gross_total − curve_total` (0 for the gross curve, full Σcosts
    for net). Taken set (contracts > 0) drives total_trades / win_rate /
    profit_factor. Empty/zero-safe.
    """
    skipped = int(view.attrs.get("skipped_trades", 0))

    if view.empty or "trade_pnl" not in view.columns or "equity" not in view.columns:
        vals = _empty_metric_values(skipped)
        vals["cost_drag"] = round(float(gross_total), 2)   # no curve total → full drag
        return vals

    pnl    = view["trade_pnl"]
    equity = view["equity"]
    taken  = _taken_subset(view)

    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    max_dd_dollars, max_dd_pct = _max_drawdown(equity)
    win_rate, profit_factor    = _win_rate_and_profit_factor(taken["trade_pnl"])
    curve_total = float(pnl.sum())

    return {
        "final_equity":   round(float(equity.iloc[-1]), 2),
        "total":          round(curve_total, 2),
        "avg_trade":      round(float(pnl.mean()), 2),
        "avg_win":        round(float(wins.mean()), 2)   if len(wins)   else 0.0,
        "avg_loss":       round(float(losses.mean()), 2) if len(losses) else 0.0,
        "largest_win":    round(float(pnl.max()), 2),
        "largest_loss":   round(float(pnl.min()), 2),
        "sharpe_daily":   round(_annualized_sharpe(view), 2),
        "sharpe_trade":   round(_sharpe_trade(view), 2),
        "profit_factor":  (round(profit_factor, 2) if np.isfinite(profit_factor) else float("inf")),
        "calmar":         (round(_calmar(curve_total, max_dd_dollars), 2)
                           if np.isfinite(_calmar(curve_total, max_dd_dollars)) else float("inf")),
        "max_dd_dollars": round(max_dd_dollars, 2),
        "max_dd_pct":     round(max_dd_pct, 2),
        "max_peak":       round(float(equity.cummax().max()), 2),
        "total_trades":   int(len(taken)),
        "win_rate":       round(win_rate, 4),
        "skipped":        skipped,
        "cost_drag":      round(float(gross_total) - curve_total, 2),
    }


def render_instance_metrics_block(values: dict, cols_per_row: int = 5) -> None:
    """Grid of ALL registry metrics (registry order), chunked into rows of
    `cols_per_row`. Values come from whichever curve the user selected."""
    keys = [k for k, _l, _f in METRIC_REGISTRY]
    for i in range(0, len(keys), cols_per_row):
        chunk = keys[i:i + cols_per_row]
        cols  = st.columns(cols_per_row)
        for col, key in zip(cols, chunk):
            col.metric(_KEY_TO_LABEL[key], _KEY_TO_FORMATTER[key](values[key]))


# ===========================================================================
# Plotting
# ===========================================================================

def _equity_trace(trades: pd.DataFrame, label: str) -> go.Scatter:
    """One equity-curve line, reused by both individual and combined figures."""
    has_contracts = "contracts" in trades.columns
    hover = f"{label}<br>%{{x}}<br>Equity: $%{{y:,.2f}}"
    if has_contracts:
        # e.g. 1.3 = 1 full contract + 3 mini contracts
        hover += "<br>Contracts: %{customdata[0]:.1f}"
    return go.Scatter(
        x=trades["entry_time"],
        y=trades["equity"],
        mode="lines",
        name=label,
        line=dict(width=2),
        customdata=trades[["contracts"]] if has_contracts else None,
        hovertemplate=hover + "<extra></extra>",
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


# Distinct styling so the gross-vs-net cost drag reads at a glance.
_CURVE_STYLE = {
    "Gross":         dict(width=2,   color="#1f77b4"),
    "+ Commissions": dict(width=1.5, color="#ff7f0e", dash="dot"),
    "+ Slippage":    dict(width=1.5, color="#9467bd", dash="dash"),
    "+ Both (net)":  dict(width=2.5, color="#2ca02c"),
}


def instance_cost_figure(run: dict, account_size: float) -> go.Figure:
    """Four labelled equity curves (gross / +comm / +slip / net) for one run."""
    trades        = run["trades"]
    x             = trades["entry_time"] if "entry_time" in trades.columns else trades.index
    has_contracts = "contracts" in trades.columns

    fig = go.Figure()
    for label, (_pnl, equity) in run["curves"].items():
        hover      = f"{label}<br>%{{x}}<br>Equity: $%{{y:,.2f}}"
        customdata = None
        if has_contracts and label == "Gross":   # contracts identical across curves; show once
            hover     += "<br>Contracts: %{customdata[0]:.1f}"
            customdata = trades[["contracts"]]
        fig.add_trace(go.Scatter(
            x=x, y=equity, mode="lines", name=label,
            line=_CURVE_STYLE.get(label, dict(width=2)),
            customdata=customdata,
            hovertemplate=hover + "<extra></extra>",
        ))
    _apply_equity_layout(fig, account_size, show_legend=True)
    return fig


def combined_equity_figure(enriched_runs: list[dict], account_size: float) -> go.Figure:
    """All non-empty runs overlaid on one chart, one NET trace per run."""
    fig = go.Figure()
    for run in enriched_runs:
        if run["net_trades"].empty:
            continue
        fig.add_trace(_equity_trace(run["net_trades"], run["label"]))
    _apply_equity_layout(fig, account_size, show_legend=True)
    return fig


# ===========================================================================
# UI — Section 1: Shared defaults
# ===========================================================================

def render_shared_defaults(trades_files: list[str]) -> tuple[str, float, int]:
    st.subheader("Shared defaults")
    col_file, col_account, col_slip = st.columns(3)

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
    with col_slip:
        slippage_n = st.slider(
            "Slippage (ticks/side)",
            min_value=1, max_value=5, value=1, step=1,
            key="shared_slippage_n",
            help="Entry-side ticks slipped per trade; market exits (losers) slip 2×. "
                 "Drives every instance.",
        )

    return default_trades_file, account_size, int(slippage_n)


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
    """
    Render the trades-file and sizer selectboxes for one instance.

    Part F (Option A): an "Override trades file" checkbox decides whether the
    instance pins its own file or follows the shared default. When off, NO
    trades-file selectbox is rendered — so no per-instance widget key exists to
    go stale, and changing the shared default flows here on the next run.
    """
    row = st.columns(2)
    with row[0]:
        override = st.checkbox(
            "Override trades file",
            value=False,
            key=f"inst_{instance_index}_override",
        )
        if override:
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
        else:
            trades_file = default_trades_file
            st.caption(f"Using default: `{default_trades_file}`")
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

def _render_filter_caption(trades_file: str) -> None:
    """Warn when an instance's trades file is a filtered sub-strategy slice (Part G)."""
    try:
        meta = read_filter_metadata(trades_file)
    except Exception:
        return
    if meta is None:
        return

    day_labels = ", ".join(DAY_TYPE_LABELS.get(k, k) for k in meta["day_types"]) or "—"
    if meta["trade_types"] == "all":
        tt_labels = "all"
    else:
        tt_labels = ", ".join(str(t) for t in meta["trade_types"]) or "—"

    st.warning(
        f"⚠ Filtered file: day types = {day_labels}; trade types = {tt_labels}. "
        "This is a sub-strategy slice — equity is stitched across excluded trades, "
        "so drawdown durations are compressed; do not read it as the live timeline."
    )


_SLIPPAGE_CAPTION = (
    "Slippage is a first-order post-hoc deduction on the recorded trades: it "
    "cannot model that a worse entry might have prevented a TP from filling at "
    "all. Read it as a cost overlay / lower bound on damage, not a "
    "re-simulation — true path-dependent slippage lives in the backtester."
)


def _curve_frame(run: dict, curve_label: str) -> pd.DataFrame:
    """View frame for one curve: a copy of the sized trades with trade_pnl /
    equity replaced by the selected curve's series. Stats are computed on this."""
    trades = run["trades"]
    frame  = trades.copy()
    if not trades.empty and "trade_pnl" in trades.columns:
        pnl, equity = run["curves"][curve_label]
        frame["trade_pnl"] = pnl
        frame["equity"]    = equity
    return frame


def render_individual_equity_curves(enriched_runs: list[dict], account_size: float,
                                    stats_curve: str) -> None:
    """Per instance: caption, four-curve chart, slippage note, metrics block
    computed on the user-selected `stats_curve`."""
    st.subheader("Individual equity curves")
    for run in enriched_runs:
        st.subheader(run["label"])
        _render_filter_caption(run["trades_file"])
        values = compute_metric_values(_curve_frame(run, stats_curve), run["gross_total"])
        if run["trades"].empty:
            st.info("No trades to display for this instance.")
            render_instance_metrics_block(values)   # safe zeros, keeps shape
            st.write("")
            continue
        st.plotly_chart(instance_cost_figure(run, account_size), width='stretch')
        st.caption(_SLIPPAGE_CAPTION)
        st.caption(f"Statistics below are computed on the **{stats_curve}** curve.")
        render_instance_metrics_block(values)
        st.write("")


def render_combined_equity_curve(enriched_runs: list[dict], account_size: float,
                                 stats_curve: str) -> None:
    """All non-empty runs overlaid on one chart (selected curve, one trace per run)."""
    st.subheader(f"Combined equity curve ({stats_curve})")
    non_empty = [r for r in enriched_runs if not r["trades"].empty]
    if not non_empty:
        st.info("No non-empty runs to combine.")
        return
    views = [{"label": r["label"], "net_trades": _curve_frame(r, stats_curve)}
             for r in non_empty]
    st.plotly_chart(
        combined_equity_figure(views, account_size),
        width='stretch',
    )


def render_metrics_table(enriched_runs: list[dict], stats_curve: str) -> None:
    """One-row-per-run comparison table, all registry metrics on the selected curve."""
    st.subheader("Metrics comparison")
    st.caption(f"Computed on the **{stats_curve}** curve.")
    rows = []
    for run in enriched_runs:
        vals = compute_metric_values(_curve_frame(run, stats_curve), run["gross_total"])
        row  = {"Label": run["label"]}
        for key, label, _f in METRIC_REGISTRY:
            row[label] = vals[key]
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


def render_results(runs: list[dict], account_size: float, slippage_n: int,
                   stats_curve: str) -> None:
    """Enrich runs with costs once, then render all sections."""
    enriched = [enrich_run(r, account_size, slippage_n) for r in runs]

    # Surface any cost-model warnings (missing commission, fractional non-micro).
    for run in enriched:
        for w in run["warnings"]:
            st.warning(w)

    render_individual_equity_curves(enriched, account_size, stats_curve)
    render_combined_equity_curve(enriched, account_size, stats_curve)
    st.write("")
    render_metrics_table(enriched, stats_curve)


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


def render_curve_selector() -> str:
    """Global selector for WHICH equity curve the statistics are computed on.
    The chart always shows all four; this only drives the metrics block + table."""
    return st.radio(
        "Statistics based on",
        options=CURVE_LABELS,
        index=CURVE_LABELS.index(DEFAULT_CURVE),
        horizontal=True,
        key="shared_stats_curve",
        help="Pick which curve the metrics below are calculated on — raw (gross) "
             "or after commissions, slippage, or both.",
    )


def render() -> None:
    _render_header()

    trades_files = list_trades_files()
    sizers = list_sizers()
    if not _validate_environment(trades_files, sizers):
        return

    default_trades_file, account_size, slippage_n = render_shared_defaults(trades_files)
    stats_curve = render_curve_selector()
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
        render_results(runs, account_size, slippage_n, stats_curve)


# Allow `streamlit run views/analytics.py` for quick isolated testing.
if __name__ == "__main__":
    render()