# views/backtester.py
import json
import math
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import importlib.util
import sys

from optimization.buckets import load_bucket_map

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


# ── News / holiday classification ─────────────────────────────────────────────
#
# The classification logic lives in optimization/buckets.py (shared with the
# Strategy Optimizer — same priority rules, same EVENT_KEYWORDS config), so
# both views classify a given date identically. This view keeps only its
# historical names: day_type column, 'high_impact' instead of the shared
# 'other_high_impact'. DAY_TYPE_ORDER drives the filter UI and the
# news/holiday breakdown table.

DAY_TYPE_ORDER = [
    ("holiday",     "Holidays"),
    ("fomc",        "FOMC"),
    ("cpi",         "CPI"),
    ("nfp",         "Non-Farm Employment"),
    ("ppi",         "PPI"),
    ("high_impact", "Other High Impact News"),
    ("normal",      "Normal Trading Days"),
]


def load_day_classifications() -> dict[str, str]:
    """
    {date_iso: day_type} from the FF events parquet ({} when missing —
    every date then resolves to 'normal' in tag_trades()).
    """
    bucket_map = load_bucket_map()
    return {
        date: ("high_impact" if bucket == "other_high_impact" else bucket)
        for date, bucket in bucket_map.items()
    }


def tag_trades(trades: pd.DataFrame, day_classifications: dict) -> pd.DataFrame:
    """Adds a single 'day_type' column ('normal' for unlisted dates)."""
    trades = trades.copy()
    trades["day_type"] = trades["date"].apply(
        lambda d: day_classifications.get(pd.Timestamp(d).date().isoformat(), "normal")
    )
    return trades


# ── Persistence ───────────────────────────────────────────────────────────────

def _build_filter_metadata(filtered: bool, selected_day_types: list,
                           selected_trade_types) -> dict:
    """
    Build the parquet key-value metadata (bytes->bytes) recording the active
    filter state. selected_trade_types is either the string "all" or a list.
    """
    if selected_trade_types == "all":
        tt_value = b"all"
    else:
        tt_value = json.dumps(list(selected_trade_types)).encode()

    return {
        b"filtered":             b"true" if filtered else b"false",
        b"selected_day_types":   json.dumps(list(selected_day_types)).encode(),
        b"selected_trade_types": tt_value,
    }


def _read_filter_metadata(path: Path) -> dict:
    """Return the filter kv-metadata subset of an existing trades parquet."""
    schema_meta = pq.read_schema(path).metadata or {}
    keys = (b"filtered", b"selected_day_types", b"selected_trade_types")
    return {k: schema_meta.get(k) for k in keys}


def save_trades(trades: pd.DataFrame, dataset: str, strategy: str,
                start_date, end_date, filtered: bool,
                selected_day_types: list, selected_trade_types) -> str:
    trades_path = Path("data/trades")
    trades_path.mkdir(parents=True, exist_ok=True)

    base_name = f"{dataset}_{strategy}_{start_date}_{end_date}"
    stem      = base_name + ("_filtered" if filtered else "")

    new_meta = _build_filter_metadata(filtered, selected_day_types, selected_trade_types)

    # Filter-aware dedup: a re-save is a duplicate only when BOTH the row
    # content and the filter metadata match an existing file.
    for f in sorted(trades_path.glob(f"{stem}*.parquet")):
        if pd.read_parquet(f).equals(trades) and _read_filter_metadata(f) == new_meta:
            return None

    output_path = trades_path / f"{stem}.parquet"
    n = 2
    while output_path.exists():
        output_path = trades_path / f"{stem}_{n}.parquet"
        n += 1

    # Write via pyarrow so we can attach kv metadata; from_pandas keeps the
    # b'pandas' schema metadata so pd.read_parquet reconstructs the frame.
    table = pa.Table.from_pandas(trades)
    meta  = dict(table.schema.metadata or {})
    meta.update(new_meta)
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, output_path)

    return str(output_path)


# ── Navigation ────────────────────────────────────────────────────────────────

def go_page(page: str):
    st.session_state.page = page
    st.rerun()


# ── Data scanning ─────────────────────────────────────────────────────────────

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

    # flat .py files
    for f in strategies_path.glob("*.py"):
        if f.stem not in ["__init__", "base"]:
            results.append(f.stem)

    # folders with __init__.py
    for f in strategies_path.iterdir():
        if f.is_dir() and (f / "__init__.py").exists():
            results.append(f.name)

    return sorted(results)


def load_strategy(name: str):
    flat_path   = Path("strategies") / f"{name}.py"
    folder_path = Path("strategies") / name / "__init__.py"

    if flat_path.exists():
        path     = flat_path
        is_pkg   = False
    elif folder_path.exists():
        path     = folder_path
        is_pkg   = True
    else:
        raise FileNotFoundError(f"Strategy '{name}' not found")

    if is_pkg:
        spec = importlib.util.spec_from_file_location(
            name,
            path,
            submodule_search_locations=[str(Path("strategies") / name)],
        )
    else:
        spec = importlib.util.spec_from_file_location(name, path)

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # register so relative imports resolve
    spec.loader.exec_module(module)

    if not hasattr(module, "run"):
        raise ValueError(f"Strategy '{name}' has no run() function")
    if not callable(module.run):
        raise ValueError(f"Strategy '{name}'.run is not callable")

    return module


def check_for_data_errors(structure: dict) -> bool:
    if not structure:
        st.error("No datasets found in data/parquet")
        return True
    if not get_strategies():
        st.error("No strategies found in strategies/")
        return True
    return False


# ── UI controls ───────────────────────────────────────────────────────────────

def render_controls(structure: dict, strategies: list):
    col1, col2 = st.columns(2)

    with col1:
        asset_types = list(structure.keys())
        asset_type  = st.selectbox("Type", asset_types)

        assets = list(structure.get(asset_type, {}).keys())
        if not assets:
            st.error(f"No assets found under {asset_type}")
            return None, None, None, None, None, None
        asset = st.selectbox("Asset", assets, key=f"bt_asset_{asset_type}")

        datasets = structure[asset_type].get(asset, [])
        if not datasets:
            st.error(f"No datasets found under {asset_type}/{asset}")
            return None, None, None, None, None, None
        dataset = st.selectbox("Dataset", datasets, key=f"bt_dataset_{asset_type}_{asset}")

        strategy_name = st.selectbox("Strategy", strategies)

    with col2:
        folder_path     = Path("data/parquet") / asset_type / asset / dataset
        available_dates = sorted([
            pd.Timestamp(f.stem) for f in folder_path.glob("*.parquet")
            if f.stem[0].isdigit()
        ])

        if available_dates:
            start_date = st.date_input(
                "Start date",
                value=available_dates[0].date(),
                min_value=available_dates[0].date(),
                max_value=available_dates[-1].date(),
                key=f"start_date_{asset_type}_{asset}_{dataset}",
            )
            end_date = st.date_input(
                "End date",
                value=available_dates[-1].date(),
                min_value=available_dates[0].date(),
                max_value=available_dates[-1].date(),
                key=f"end_date_{asset_type}_{asset}_{dataset}",
            )
        else:
            start_date = end_date = None

    return asset_type, asset, dataset, strategy_name, start_date, end_date


def _render_param_widget(key: str, default):
    if isinstance(default, bool):
        return st.checkbox(key, value=default)
    elif isinstance(default, float):
        return st.number_input(key, value=default, step=0.1, format="%.2f")
    elif isinstance(default, int):
        return st.number_input(key, value=default, step=1)
    elif isinstance(default, str):
        return st.text_input(key, value=default)
    else:
        st.warning(f"Unsupported param type for '{key}': {type(default).__name__}")
        return default

def render_params(strategy) -> dict:
    if not hasattr(strategy, "PARAMS"):
        return {}

    visible = {k: v for k, v in strategy.PARAMS.items() if k not in HIDDEN_PARAMS}
    if not visible:
        return {}

    st.write("")
    st.subheader("Parameters")
    params = {}

    if hasattr(strategy, "PARAM_SECTIONS"):
        rendered = set()

        for section_label, keys in strategy.PARAM_SECTIONS.items():
            section_keys = [k for k in keys if k in visible]
            if not section_keys:
                continue
            st.caption(section_label)
            cols = st.columns(len(section_keys))
            for i, key in enumerate(section_keys):
                with cols[i]:
                    params[key] = _render_param_widget(key, visible[key])
                rendered.add(key)

        unassigned = [k for k in visible if k not in rendered]
        if unassigned:
            st.caption("Other")
            cols = st.columns(len(unassigned))
            for i, key in enumerate(unassigned):
                with cols[i]:
                    params[key] = _render_param_widget(key, visible[key])

    else:
        items  = list(visible.items())
        chunks = [items[i:i + 10] for i in range(0, len(items), 10)]
        for chunk in chunks:
            cols = st.columns(len(chunk))
            for i, (key, default) in enumerate(chunk):
                with cols[i]:
                    params[key] = _render_param_widget(key, default)

    return params


# ── Strategy execution ────────────────────────────────────────────────────────

def execute_run(strategy, asset_type, asset, dataset,
                start_date, end_date, params) -> bool:
    st.session_state.trades      = None
    st.session_state.folder_path = None

    if start_date > end_date:
        st.error("Start date must be before end date.")
        return False

    if asset not in ASSET_INFO:
        st.error(f"Unknown asset: {asset}. Add it to ASSET_INFO.")
        return False

    asset_info      = ASSET_INFO[asset]
    ticks_per_point = asset_info["ticks_per_point"]
    folder_path     = Path("data/parquet") / asset_type / asset / dataset

    params["tick_size"] = asset_info["tick_size"]

    with st.spinner("Running strategy..."):
        trades = strategy.run(
            folder_path=folder_path,
            start_date=pd.Timestamp(start_date),
            end_date=pd.Timestamp(end_date),
            params=params,
        )

    if trades.empty:
        st.warning("Strategy produced no trades.")
        return False

    trades["ticks"]            = trades["pnl_points"] * ticks_per_point
    trades["cumulative_ticks"] = trades["ticks"].cumsum()
    st.session_state.trades      = trades
    st.session_state.folder_path = folder_path
    return True


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(trades: pd.DataFrame) -> dict:
    winning   = trades[trades["ticks"] > 0]
    losing    = trades[trades["ticks"] < 0]
    breakeven = trades[trades["ticks"] == 0]

    avg_win   = winning["ticks"].mean() if len(winning) > 0 else 0.0
    avg_loss  = losing["ticks"].mean()  if len(losing)  > 0 else 0.0
    win_rate  = len(winning) / len(trades)
    loss_rate = len(losing)  / len(trades)

    gross_wins   = winning["ticks"].sum()
    gross_losses = abs(losing["ticks"].sum())
    if gross_losses > 0:
        profit_factor = gross_wins / gross_losses
    elif gross_wins > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    # Both Sharpes are daily-aggregated and annualized ×sqrt(252). Dates are
    # normalized to Timestamps first: strategies return strings (orb) or date
    # objects (ivb), and reindexing a string-keyed daily series against a
    # DatetimeIndex silently matches nothing — all-zero series, Sharpe 0.
    dates        = pd.to_datetime(trades["date"]).dt.normalize()
    traded_daily = trades.groupby(dates)["ticks"].sum()

    # Daily Sharpe — zero-fill every business day between first & last trade
    # (union keeps any weekend/holiday traded day in the spine too)
    spine     = pd.bdate_range(dates.min(), dates.max()).union(traded_daily.index)
    daily_pnl = traded_daily.reindex(spine, fill_value=0)
    daily_std = daily_pnl.std(ddof=1)
    sharpe_daily = (daily_pnl.mean() / daily_std) * (252 ** 0.5) if daily_std > 0 else 0.0

    # Trade Sharpe — daily P&L over TRADED days only (no zero-fill)
    traded_std   = traded_daily.std(ddof=1)
    sharpe_trade = (traded_daily.mean() / traded_std) * (252 ** 0.5) \
        if len(traded_daily) >= 2 and traded_std > 0 else 0.0

    # Equity curve / drawdown
    cumulative   = trades["cumulative_ticks"]
    rolling_max  = cumulative.cummax()
    drawdown     = cumulative - rolling_max
    max_drawdown = drawdown.min()

    total       = trades["ticks"].sum()
    global_peak = rolling_max.max()
    calmar      = total / abs(max_drawdown) if max_drawdown < 0 else float("inf")

    # ── Risk/reward ───────────────────────────────────────────────────────────
    # Planned RR (target geometry): abs(tp - entry) / abs(sl - entry).
    # Realised RR (signed R-multiple): pnl_points / abs(entry - sl) — a full
    # target reads +planned_RR, a full stop reads -1.0, scratches scale between.
    # Both use the same valid rows: stop distance > 0.
    if all(c in trades.columns for c in ["entry_price", "sl"]):
        sl_dist = (trades["entry_price"] - trades["sl"]).abs()
        valid   = sl_dist > 0
    else:
        sl_dist = None
        valid   = None

    if sl_dist is not None and "tp" in trades.columns:
        tp_dist           = (trades["tp"] - trades["entry_price"]).abs()
        rr_planned        = (tp_dist / sl_dist)[valid]
        avg_rr_planned    = rr_planned.mean()    if len(rr_planned) > 0 else None
        median_rr_planned = rr_planned.median()  if len(rr_planned) > 0 else None
    else:
        avg_rr_planned = median_rr_planned = None

    if sl_dist is not None and "pnl_points" in trades.columns:
        rr_realised        = (trades["pnl_points"] / sl_dist)[valid]
        avg_rr_realised    = rr_realised.mean()   if len(rr_realised) > 0 else None
        median_rr_realised = rr_realised.median() if len(rr_realised) > 0 else None
    else:
        avg_rr_realised = median_rr_realised = None

    # Consecutive wins / losses
    results = (trades["ticks"] > 0).astype(int).tolist()
    max_consec_wins = max_consec_losses = cur_wins = cur_losses = 0
    for r in results:
        if r == 1:
            cur_wins  += 1
            cur_losses = 0
        else:
            cur_losses += 1
            cur_wins   = 0
        max_consec_wins   = max(max_consec_wins,   cur_wins)
        max_consec_losses = max(max_consec_losses, cur_losses)

    # Trade duration
    if "entry_time" in trades.columns and "exit_time" in trades.columns:
        durations           = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 60
        avg_duration_min    = durations.mean()
        median_duration_min = durations.median()
    else:
        avg_duration_min = median_duration_min = None

    long_trades  = trades[trades["direction"] == "long"]
    short_trades = trades[trades["direction"] == "short"]

    return {
        "total_ticks":          total,
        "avg_trade":            trades["ticks"].mean(),
        "avg_win":              avg_win,
        "avg_loss":             avg_loss,
        "largest_win":          winning["ticks"].max() if len(winning) > 0 else 0,
        "largest_loss":         losing["ticks"].min()  if len(losing)  > 0 else 0,
        "avg_rr_planned":       avg_rr_planned,
        "median_rr_planned":    median_rr_planned,
        "avg_rr_realised":      avg_rr_realised,
        "median_rr_realised":   median_rr_realised,
        "total_trades":         len(trades),
        "win_rate":             win_rate,
        "loss_rate":            loss_rate,
        "breakeven_rate":       len(breakeven) / len(trades),
        "sharpe_daily":         sharpe_daily,
        "sharpe_trade":         sharpe_trade,
        "profit_factor":        profit_factor,
        "calmar":               calmar,
        "max_drawdown":         max_drawdown,
        "max_peak":             global_peak,
        "max_consec_wins":      max_consec_wins,
        "max_consec_losses":    max_consec_losses,
        "avg_duration_min":     avg_duration_min,
        "median_duration_min":  median_duration_min,
        "long_trades":          len(long_trades),
        "short_trades":         len(short_trades),
        "long_winrate":         (long_trades["ticks"] > 0).mean()  if len(long_trades)  > 0 else 0.0,
        "short_winrate":        (short_trades["ticks"] > 0).mean() if len(short_trades) > 0 else 0.0,
    }


def render_metrics(trades: pd.DataFrame):
    st.write("")
    st.subheader("Performance")

    m = compute_metrics(trades)

    pf_display         = "∞" if m["profit_factor"] == float("inf") else f"{m['profit_factor']:.2f}"
    calmar_display     = "∞" if m["calmar"]        == float("inf") else f"{m['calmar']:.2f}"
    avg_dur_display    = f"{m['avg_duration_min']:.0f}m"    if m["avg_duration_min"]    is not None else "N/A"
    median_dur_display = f"{m['median_duration_min']:.0f}m" if m["median_duration_min"] is not None else "N/A"

    def _rr(key):
        return f"{m[key]:.2f}" if m[key] is not None else "N/A"

    # Row 1 — Core P&L
    st.write("")
    r1c1, r1c2, r1c3, r1c4, r1c5, r1c6, r1c7, r1c8 = st.columns(8)
    r1c1.metric("Total Ticks",          f"{m['total_ticks']:.0f}")
    r1c2.metric("Total Trades",         m["total_trades"])
    r1c3.metric("Avg Trade/Expectancy", f"{m['avg_trade']:.2f}")
    r1c4.metric("Avg Win",              f"{m['avg_win']:.1f}")
    r1c5.metric("Avg Loss",             f"{m['avg_loss']:.1f}")
    r1c6.metric("Largest Win",          f"{m['largest_win']:.0f}")
    r1c7.metric("Largest Loss",         f"{m['largest_loss']:.0f}")
    r1c8.metric("Profit Factor",        pf_display)

    # Row 2 — Rates & risk-adjusted
    st.write("")
    r2c1, r2c2, r2c3, r2c4, r2c5, r2c6, _, _ = st.columns(8)
    r2c1.metric("Win Rate",         f"{m['win_rate']:.1%}")
    r2c2.metric("Loss Rate",        f"{m['loss_rate']:.1%}")
    r2c3.metric("Breakeven Rate",   f"{m['breakeven_rate']:.1%}")
    r2c4.metric("Sharpe (daily)",   f"{m['sharpe_daily']:.2f}",
                help="daily P&L over every business day between first and "
                     "last trade — days without trades count as 0; ×√252")
    r2c5.metric("Sharpe (traded days)", f"{m['sharpe_trade']:.2f}",
                help="daily P&L over days with at least one trade; ×√252")
    r2c6.metric("Calmar",           calmar_display)

    # Row 3 — Risk/reward
    st.write("")
    r3c1, r3c2, r3c3, r3c4, _, _, _, _ = st.columns(8)
    r3c1.metric("Planned Avg RR",    _rr("avg_rr_planned"))
    r3c2.metric("Planned Median RR", _rr("median_rr_planned"))
    r3c3.metric("Realised Avg RR",   _rr("avg_rr_realised"))
    r3c4.metric("Realised Median RR", _rr("median_rr_realised"))

    # Row 4 — Drawdown + streaks + duration
    st.write("")
    r4c1, r4c2, r4c3, r4c4, r4c5, r4c6 = st.columns(6)
    r4c1.metric("Max Drawdown",    f"{m['max_drawdown']:.0f} ticks")
    r4c2.metric("Max Peak",        f"{m['max_peak']:.0f} ticks")
    r4c3.metric("Consec. Wins",    m["max_consec_wins"])
    r4c4.metric("Consec. Losses",  m["max_consec_losses"])
    r4c5.metric("Avg Duration",    avg_dur_display)
    r4c6.metric("Median Duration", median_dur_display)

    # Row 5 — Directional
    st.write("")
    r5c1, r5c2, r5c3, r5c4, _, _, _ = st.columns(7)
    r5c1.metric("Long Win Rate",  f"{m['long_winrate']:.1%}")
    r5c2.metric("Short Win Rate", f"{m['short_winrate']:.1%}")
    r5c3.metric("Long Trades",    m["long_trades"])
    r5c4.metric("Short Trades",   m["short_trades"])

    # Exit breakdown
    st.write("")
    st.subheader("Exit Breakdown")
    exit_stats = trades.groupby("exit_reason")["ticks"].agg(
        count="count", avg="mean", total="sum"
    ).reset_index()
    exit_stats.columns        = ["Exit Reason", "Count", "Avg Ticks", "Total Ticks"]
    exit_stats["Avg Ticks"]   = exit_stats["Avg Ticks"].round(1)
    exit_stats["Total Ticks"] = exit_stats["Total Ticks"].round(0).astype(int)
    st.dataframe(exit_stats, width='stretch', hide_index=True)


def render_news_holiday_breakdown(trades: pd.DataFrame):
    """
    Always-visible section — computed from the trade_type-filtered trades
    but unaffected by the day_type filter. Rows are driven by DAY_TYPE_ORDER
    so every day-type category (incl. the carved-out event categories) appears.
    """
    if "day_type" not in trades.columns:
        return

    st.write("")
    st.subheader("News & Holiday Exposure")

    rows = []
    for tag, label in DAY_TYPE_ORDER:
        subset = trades[trades["day_type"] == tag]
        n      = len(subset)
        if n == 0:
            rows.append({
                "Day Type":    label,
                "Trades":      0,
                "Wins":        0,
                "Losses":      0,
                "Win Rate":    "N/A",
                "Total Ticks": 0,
                "Avg Ticks":   "N/A",
            })
        else:
            wins   = (subset["ticks"] > 0).sum()
            losses = (subset["ticks"] < 0).sum()
            rows.append({
                "Day Type":    label,
                "Trades":      n,
                "Wins":        wins,
                "Losses":      losses,
                "Win Rate":    f"{wins / n:.1%}",
                "Total Ticks": int(round(subset["ticks"].sum())),
                "Avg Ticks":   f"{subset['ticks'].mean():.1f}",
            })

    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')


# ── Market exposure (α/β regression) ─────────────────────────────────────────
#
# Regresses the strategy's daily tick P&L against the traded asset's own daily
# move in ticks — real alpha vs disguised beta. Sizing-agnostic, tick space.
# Four cells: {days traded, all days} × {settlement move, RTH move}.

_ALPHA_BETA_TOOLTIP = """\
**Alpha (α):** the strategy's return that is NOT explained by the benchmark — the part
of the edge independent of just being long/short the asset. Positive α = the strategy
generates tick-edge beyond market exposure. Reported in ticks/day (and annualized ×252).

**Beta (β):** how much the strategy's daily P&L moves with the asset's daily move.
β≈0 = no linear market exposure (market-neutral in direction); β≈1 = moves like
buy-and-hold; β<0 = moves opposite.

**Limitations:** in-sample — if the strategy was tuned on this data, α is optimistic by
construction and does not predict out-of-sample performance. Single-factor — tests
market-*direction* exposure only, not volatility, tail, or liquidity risk. A strategy
can show β≈0 here and still blow up on a vol spike.
"""

_TSTAT_TOOLTIP = """\
**t-stat = estimate ÷ its standard error** (signal-to-noise); |t| > ~2 ≈ "unlikely to
be noise."

**Days-traded (A) vs all-days (B):** going to B usually *shrinks* α and β (flat
zero-return days pull the line toward flat) but *balloons* the day count. More days
shrink the standard error, which can inflate the t-stat — and plain OLS treats the
zero-variance flat days as if each were a real independent observation, badly
overstating B's significance. That's why these use **HAC (Newey-West) standard
errors**, which down-weight the empty/autocorrelated days. Trust only the HAC t-stat
shown here; a naive-OLS t-stat (especially for B) is flattering you with empty days.

**Settlement vs RTH:** settlement spans the overnight gap you didn't hold; RTH is only
your in-market hours. Comparing the two shows whether overnight movement is distorting
the exposure estimate.
"""


def load_asset_statistics(asset: str) -> pd.DataFrame | None:
    """Per-asset daily statistics file, or None when unavailable."""
    path = Path("data/parquet/Futures") / asset / f"{asset}_statistics" / "statistics.parquet"
    if not path.exists():
        return None
    stats = pd.read_parquet(path)
    stats["date"] = pd.to_datetime(stats["date"]).dt.normalize()
    return stats.sort_values("date").reset_index(drop=True)


def build_benchmark_series(stats: pd.DataFrame, tick_size: float) -> dict:
    """
    Build the two benchmark daily-move series (in ticks) with exclusions applied.

    - settle: settle_px.diff() over the previous ROW (previous available trading
      day, not a calendar shift), where settle_px = settlement when
      settle_is_actual else globex_close (never rth_close — different mark).
    - rth: rth_close − rth_open, same-day only.
    - Roll days are excluded from BOTH series: prices jump old→new contract,
      a discontinuity, not a market move. One uncaught roll can dominate the fit.
      (Symbol changes always fall on flagged rows; a symbol-change guard is
      included as insurance.)

    Returns {"settle": Series, "rth": Series (both date-indexed),
             "n_roll", "n_missing_settle", "n_missing_rth"}.
    """
    s = stats
    is_roll = s["is_roll_day"].fillna(False).astype(bool)
    # Insurance: any cross-contract diff is invalid even if somehow unflagged.
    symbol_switch = (s["symbol"] != s["symbol"].shift(1)).fillna(False)
    symbol_switch.iloc[0] = False

    actual    = s["settle_is_actual"].fillna(False).astype(bool) & s["settlement"].notna()
    settle_px = s["settlement"].where(actual, s["globex_close"])
    settle_move = settle_px.diff() / tick_size
    settle_move[symbol_switch] = float("nan")

    rth_move = (s["rth_close"] - s["rth_open"]) / tick_size

    settle_valid = settle_move.notna() & ~is_roll
    rth_valid    = rth_move.notna()    & ~is_roll

    return {
        "settle": pd.Series(settle_move[settle_valid].values,
                            index=s.loc[settle_valid, "date"]),
        "rth":    pd.Series(rth_move[rth_valid].values,
                            index=s.loc[rth_valid, "date"]),
        "n_roll":           int(is_roll.sum()),
        "n_missing_settle": int((settle_move.isna() & ~is_roll).sum()),
        "n_missing_rth":    int((rth_move.isna() & ~is_roll).sum()),
    }


def _fit_alpha_beta(y: pd.Series, x: pd.Series) -> dict | None:
    """
    OLS of strategy_ticks ~ const + benchmark_ticks with HAC (Newey-West)
    standard errors, maxlags=5 (~one trading week). Returns None when the fit
    is undefined (< 3 rows or zero benchmark variance).
    """
    if len(y) < 3 or len(x) < 3 or float(x.std(ddof=0)) == 0.0:
        return None

    import statsmodels.api as sm
    X   = sm.add_constant(x.to_numpy(dtype=float))
    res = sm.OLS(y.to_numpy(dtype=float), X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    return {
        "alpha":     float(res.params[0]),
        "alpha_ann": float(res.params[0]) * 252,
        "beta":      float(res.params[1]),
        "t_alpha":   float(res.tvalues[0]),
        "t_beta":    float(res.tvalues[1]),
        "r2":        float(res.rsquared),
        "n":         int(res.nobs),
    }


def _regression_cell_md(res: dict | None) -> str:
    """Compact per-cell markdown table (or an insufficient-data note)."""
    if res is None:
        return "*insufficient data*"
    return (
        "| | |\n|---|---|\n"
        f"| α (ticks/day) | {res['alpha']:.2f} |\n"
        f"| α annualized | {res['alpha_ann']:.0f} |\n"
        f"| β | {res['beta']:.3f} |\n"
        f"| t(α) HAC | {res['t_alpha']:.2f} |\n"
        f"| t(β) HAC | {res['t_beta']:.2f} |\n"
        f"| R² | {res['r2']:.3f} |\n"
        f"| n | {res['n']} |"
    )


def render_market_exposure(trades: pd.DataFrame, asset: str):
    """
    Collapsed 'Market Exposure' section: 2×2 regression grid on the FILTERED
    trades (respects the active day-type / trade-type filters).
    """
    st.write("")
    with st.expander("Market Exposure (α/β regression)", expanded=False):
        stats = load_asset_statistics(asset)
        if stats is None:
            st.info(f"No statistics file for {asset}; regression unavailable.")
            return

        try:
            import statsmodels.api as sm  # noqa: F401 — availability check only
        except ImportError:
            st.info("`statsmodels` is not installed; regression unavailable.")
            return

        tick_size = ASSET_INFO[asset]["tick_size"]

        # Strategy daily tick P&L — same aggregation as the daily Sharpe.
        daily = trades.groupby("date")["ticks"].sum()
        daily.index = pd.to_datetime(daily.index).normalize()
        daily = daily.sort_index()

        # Restrict the stats spine to the backtest window (Sharpe convention).
        lo, hi = daily.index.min(), daily.index.max()
        stats  = stats[(stats["date"] >= lo) & (stats["date"] <= hi)]
        if stats.empty:
            st.info("Statistics file has no overlap with the backtest window.")
            return

        bench = build_benchmark_series(stats, tick_size)

        # Traded dates entirely absent from the stats file (never regressed
        # against a phantom benchmark).
        stats_dates = set(stats["date"])
        n_absent    = int(sum(d not in stats_dates for d in daily.index))

        st.caption(
            f"Excluded: {bench['n_roll']} roll days · "
            f"{bench['n_missing_settle']} missing settlement move · "
            f"{bench['n_missing_rth']} missing RTH · "
            f"{n_absent} traded days absent from stats file"
        )

        hdr1, hdr2 = st.columns(2)
        hdr1.markdown("**α / β — what they mean**", help=_ALPHA_BETA_TOOLTIP)
        hdr2.markdown("**t-stats & the 2×2 grid**", help=_TSTAT_TOOLTIP)

        for sample_label, all_days in [("Days traded", False), ("All days", True)]:
            cols = st.columns(2)
            for col, (bench_label, series) in zip(
                cols, [("Settlement", bench["settle"]), ("RTH", bench["rth"])]
            ):
                with col:
                    st.markdown(f"**{sample_label} × {bench_label}**")
                    if all_days:
                        # Stats-file business-day spine, strategy zero-filled —
                        # matches the daily-Sharpe zero-fill convention.
                        x = series
                        y = daily.reindex(series.index, fill_value=0.0)
                    else:
                        common = daily.index.intersection(series.index)
                        y = daily.loc[common]
                        x = series.loc[common]
                    st.markdown(_regression_cell_md(_fit_alpha_beta(y, x)))
                    st.write("")


# ── Equity curve ──────────────────────────────────────────────────────────────

def render_equity_curve(trades: pd.DataFrame):
    st.write("")
    st.subheader("Equity Curve")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trades["entry_time"],
        y=trades["cumulative_ticks"],
        mode="lines+markers",
        name="Cumulative Ticks",
        line=dict(width=2),
        marker=dict(size=6, opacity=0.6),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Ticks",
        hovermode="x unified",
        height=400,
    )

    return st.plotly_chart(fig, width='stretch', on_select="rerun", key="equity_curve")


# ── RR distribution ───────────────────────────────────────────────────────────

def render_rr_distribution(trades: pd.DataFrame):
    """
    Overlaid histogram of planned vs. realised RR (R-multiple) per trade.
    x = RR, y = number of trades. Same formulas as compute_metrics:
      planned  = |tp - entry| / |entry - sl|
      realised = pnl_points   / |entry - sl|
    over rows with stop distance > 0.
    """
    if not all(c in trades.columns for c in ["entry_price", "sl"]):
        st.write("")
        st.subheader("RR Distribution")
        st.info("RR distribution needs entry_price + sl (+ tp / pnl_points).")
        return

    sl_dist = (trades["entry_price"] - trades["sl"]).abs()
    valid   = sl_dist > 0

    planned  = ((trades["tp"] - trades["entry_price"]).abs() / sl_dist)[valid] \
        if "tp" in trades.columns else None
    realised = (trades["pnl_points"] / sl_dist)[valid] \
        if "pnl_points" in trades.columns else None

    # Realised RR of WINNERS only — trades that went to profit (realised > 0);
    # a planned-3 trade that only banked +1 R shows here as 1; SL hits excluded.
    won = realised[realised > 0]  if realised is not None else None
    # Break-even trades — realised exactly 0 R (their own column at x=0).
    be  = realised[realised == 0] if realised is not None else None

    have_planned  = planned  is not None and len(planned)  > 0
    have_realised = realised is not None and len(realised) > 0
    have_won      = won      is not None and len(won)      > 0
    have_be       = be       is not None and len(be)       > 0
    if not have_planned and not have_realised:
        st.write("")
        st.subheader("RR Distribution")
        st.info("RR distribution needs entry_price + sl (+ tp / pnl_points).")
        return

    st.write("")
    st.subheader("RR Distribution")

    w = st.number_input(
        "RR bin width", value=0.5, min_value=0.1, step=0.1, format="%.2f",
        key="rr_bin_width",
    )

    # Shared bins so the two series line up. Widen range slightly past the data.
    values = pd.concat([s for s in (planned, realised) if s is not None])
    lo, hi = float(values.min()), float(values.max())
    start  = math.floor(lo / w) * w
    end    = math.ceil(hi / w) * w + w   # +1 bin of headroom on the right edge
    xbins  = dict(start=start, end=end, size=w)

    fig = go.Figure()
    if have_planned:
        fig.add_trace(go.Histogram(
            x=planned, name="Planned RR", xbins=xbins,
            marker_color="#1f77b4", opacity=0.6,
            hovertemplate="%{fullData.name}: %{y}<extra></extra>",
        ))
    if have_realised:
        fig.add_trace(go.Histogram(
            x=realised, name="Realised RR", xbins=xbins,
            marker_color="#ff7f0e", opacity=0.6,
            hovertemplate="%{fullData.name}: %{y}<extra></extra>",
        ))
    if have_won:
        fig.add_trace(go.Histogram(
            x=won, name="Realised RR (wins)", xbins=xbins,
            marker_color="#2ca02c", opacity=0.6,
            hovertemplate="%{fullData.name}: %{y}<extra></extra>",
        ))
    if have_be:
        fig.add_trace(go.Histogram(
            x=be, name="Break even", xbins=xbins,
            marker_color="#9467bd", opacity=0.6,
            hovertemplate="%{fullData.name}: %{y}<extra></extra>",
        ))

    fig.add_vline(x=0,  line_dash="dash", line_color="gray",  opacity=0.6)
    fig.add_vline(x=-1, line_dash="dot",  line_color="red",   opacity=0.5,
                  annotation_text="full stop", annotation_position="top")

    fig.update_layout(
        barmode="overlay",
        xaxis_title="RR (R-multiple)",
        yaxis_title="Number of trades",
        hovermode="x unified",
        height=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width='stretch', key="rr_distribution")


# ── Chart view ────────────────────────────────────────────────────────────────

def render_chart_view_controls() -> dict:
    st.write("")
    st.subheader("Chart View Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        view_mode = st.radio(
            "Chart start from",
            options=["Fixed session time", "Candles before entry"],
            key="chart_view_mode",
            horizontal=True,
        )

    with col2:
        if view_mode == "Candles before entry":
            candles_before     = st.number_input("Candles before entry", value=30, min_value=1, max_value=390, step=1, key="chart_candles_before")
            session_start_time = None
        else:
            candles_before     = None
            session_start_time = st.time_input("Session start time (NY)", value=pd.Timestamp("09:30").time(), key="chart_session_start", step=60)

    with col3:
        candles_after = st.number_input("Candles after exit", value=10, min_value=0, max_value=390, step=1, key="chart_candles_after")

    return {
        "view_mode":          view_mode,
        "candles_before":     candles_before,
        "session_start_time": session_start_time,
        "candles_after":      candles_after,
    }


def resolve_chart_window(session: pd.DataFrame, entry_ts: pd.Timestamp,
                         exit_ts: pd.Timestamp, chart_settings: dict) -> pd.DataFrame:
    exit_loc = session.index.searchsorted(exit_ts, side="right") - 1
    exit_loc = max(0, min(exit_loc, len(session) - 1))
    end_loc  = min(exit_loc + chart_settings["candles_after"], len(session) - 1)

    if chart_settings["view_mode"] == "Candles before entry":
        entry_loc = session.index.searchsorted(entry_ts, side="left")
        entry_loc = max(0, min(entry_loc, len(session) - 1))
        start_loc = max(0, entry_loc - chart_settings["candles_before"])
    else:
        time_mask = session.index.time >= chart_settings["session_start_time"]
        start_loc = int(time_mask.argmax()) if time_mask.any() else 0

    return session.iloc[start_loc: end_loc + 1]


def build_trade_figure(trade, chart_candles: pd.DataFrame,
                       entry_ts: str, exit_ts: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=chart_candles.index,
        open=chart_candles["open"], high=chart_candles["high"],
        low=chart_candles["low"],  close=chart_candles["close"],
        name="Price",
    ))
    fig.add_shape(type="line", x0=entry_ts, x1=exit_ts,
                  y0=trade["entry_price"], y1=trade["entry_price"],
                  line=dict(color="blue", width=1, dash="solid"))
    fig.add_shape(type="line", x0=entry_ts, x1=exit_ts,
                  y0=trade["sl"], y1=trade["sl"],
                  line=dict(color="red", width=1, dash="dash"))
    fig.add_shape(type="rect", x0=entry_ts, x1=exit_ts,
                  y0=trade["sl"], y1=trade["entry_price"],
                  fillcolor="red", opacity=0.05, line_width=0)
    fig.add_shape(type="line", x0=entry_ts, x1=exit_ts,
                  y0=trade["tp"], y1=trade["tp"],
                  line=dict(color="green", width=1, dash="dash"))
    fig.add_shape(type="rect", x0=entry_ts, x1=exit_ts,
                  y0=trade["entry_price"], y1=trade["tp"],
                  fillcolor="green", opacity=0.05, line_width=0)
    fig.add_trace(go.Scatter(
        x=[entry_ts], y=[trade["entry_price"]], mode="markers",
        marker=dict(symbol="triangle-up" if trade["direction"] == "long" else "triangle-down",
                    size=14, color="blue"),
        name="Entry",
    ))
    fig.add_trace(go.Scatter(
        x=[exit_ts], y=[trade["exit_price"]], mode="markers",
        marker=dict(symbol="x", size=14, color="orange"),
        name="Exit",
    ))
    fig.update_layout(
        height=700, xaxis_title="Time", yaxis_title="Price",
        xaxis_rangeslider_visible=False, hovermode="x unified",
        yaxis=dict(tickformat=",.2f"),
    )
    return fig


def _is_timestamp(val) -> bool:
    try:
        pd.Timestamp(val)
        return isinstance(val, str) and (":" in val or "-" in val)
    except Exception:
        return False


@st.fragment
def render_trade_detail(selected, trades: pd.DataFrame, chart_settings: dict):
    if not selected or not selected.selection.points:
        return

    idx   = selected.selection.points[0]["point_index"]
    trade = trades.iloc[idx]

    folder_path     = st.session_state.folder_path
    asset           = folder_path.parts[-2]
    ticks_per_point = ASSET_INFO[asset]["ticks_per_point"]

    st.write("")
    st.divider()
    st.subheader(f"Trade Detail — {trade['date']}")

    duration  = trade["exit_time"] - trade["entry_time"]
    hours     = int(duration.total_seconds() // 3600)
    minutes   = int((duration.total_seconds() % 3600) // 60)
    sl_ticks  = abs(trade["entry_price"] - trade["sl"]) * ticks_per_point
    tp_ticks  = abs(trade["entry_price"] - trade["tp"]) * ticks_per_point
    actual_rr = tp_ticks / sl_ticks if sl_ticks > 0 else 0

    metric_items = [
        ("Direction",   trade["direction"].upper()),
        ("Tick PnL",    f"{trade['ticks']:.0f}"),
        ("SL Ticks",    f"{sl_ticks:.0f}"),
        ("TP Ticks",    f"{tp_ticks:.0f}"),
        ("RR",          f"{actual_rr:.2f}"),
        ("Duration",    f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"),
        ("Exit Reason", str(trade["exit_reason"])),
    ]
    if "trade_type" in trade.index and pd.notna(trade["trade_type"]):
        metric_items.append(("Trade Type", str(trade["trade_type"])))
    if "day_type" in trade.index and pd.notna(trade["day_type"]):
        metric_items.append(("Day Type", str(trade["day_type"])))

    per_row = 5
    for start in range(0, len(metric_items), per_row):
        cols = st.columns(per_row)
        for col, (label, value) in zip(cols, metric_items[start:start + per_row]):
            col.metric(label, value)

    if "notes" in trade.index and pd.notna(trade["notes"]):
        try:
            notes = json.loads(trade["notes"])
            items = list(notes.items())
            if items:
                st.write("")
                st.markdown("**Trade notes**")
                # Render in rows of up to 4 tiles. We use caption + markdown
                # instead of st.metric because a metric value is single-line and
                # clips long values (e.g. a joined timestamp list) with an
                # ellipsis; markdown wraps so the full note is always visible.
                per_row = 4
                for start in range(0, len(items), per_row):
                    chunk = items[start:start + per_row]
                    cols  = st.columns(per_row)
                    for col, (key, val) in zip(cols, chunk):
                        if isinstance(val, list):
                            display = ", ".join(
                                pd.Timestamp(v).strftime("%H:%M") if _is_timestamp(v) else str(v)
                                for v in val
                            )
                        elif _is_timestamp(val):
                            display = pd.Timestamp(val).strftime("%H:%M")
                        else:
                            display = str(val)
                        col.caption(key)
                        col.markdown(f"**{display}**")
        except Exception as e:
            st.warning(str(e))

    trade_date    = pd.Timestamp(trade["date"])
    session       = pd.read_parquet(folder_path / f"{trade_date.date().isoformat()}.parquet")
    session       = session[session.index.date == trade_date.date()]
    chart_candles = resolve_chart_window(session, trade["entry_time"], trade["exit_time"], chart_settings)

    st.plotly_chart(
        build_trade_figure(trade, chart_candles, str(trade["entry_time"]), str(trade["exit_time"])),
        width='stretch',
    )


# ── Trades table ──────────────────────────────────────────────────────────────

def render_trades_table(trades: pd.DataFrame, dataset: str, strategy_name: str,
                        start_date, end_date, filtered: bool,
                        selected_day_types: list, selected_trade_types):
    st.write("")
    st.subheader("Trades")
    display_cols = ["date", "direction", "entry_time", "exit_time",
                    "entry_price", "exit_price", "exit_reason", "ticks"]
    if "trade_type" in trades.columns:
        display_cols.append("trade_type")
    if "day_type" in trades.columns:
        display_cols.append("day_type")
    st.dataframe(trades[display_cols], width='stretch')

    st.write("")
    _, _, save_col, _, _ = st.columns(5)
    with save_col:
        if st.button("Save Trades", width='stretch'):
            # Strip day_type before saving — it's derived, not strategy output
            save_cols = [c for c in trades.columns if c != "day_type"]
            result    = save_trades(
                trades[save_cols], dataset, strategy_name, start_date, end_date,
                filtered, selected_day_types, selected_trade_types,
            )
            if result is None:
                st.info("Identical trades file already exists — not saved.")
            else:
                st.success(f"Saved to {result}")


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    if "trades" not in st.session_state:
        st.session_state.trades      = None
    if "folder_path" not in st.session_state:
        st.session_state.folder_path = None

    if st.button("← Back"):
        go_page("home")
    st.title("Backtester")
    st.caption("Run a strategy on a dataset and inspect the results.")
    st.write("")

    structure = get_parquet_structure()
    if check_for_data_errors(structure):
        return

    strategies = get_strategies()
    result     = render_controls(structure, strategies)
    asset_type, asset, dataset, strategy_name, start_date, end_date = result

    if asset_type is None:
        return

    strategy = load_strategy(strategy_name)
    params   = render_params(strategy)

    st.write("")
    _, _, btn_col, _, _ = st.columns(5)
    with btn_col:
        run = st.button("Run", type="primary", width='stretch')

    if run:
        execute_run(strategy, asset_type, asset, dataset, start_date, end_date, params)

    if st.session_state.trades is None:
        return

    # ── Base trades from session state ────────────────────────────────────────
    trades = st.session_state.trades

    # ── Tag every trade with day_type (always, before any filter) ─────────────
    day_classifications = load_day_classifications()
    trades = tag_trades(trades, day_classifications)

    # ── Trade type filter ──────────────────────────────────────────────────────
    # selected_trade_types_meta captures the filter for parquet metadata:
    # "all" when there's no trade_type column or every type is selected, else
    # the list of selected values.
    selected_trade_types_meta = "all"
    trade_type_filtered       = False
    if "trade_type" in trades.columns:
        unique_types = sorted(trades["trade_type"].dropna().unique().tolist())
        if unique_types:
            st.write("")
            st.caption("Filter by trade type")
            cols           = st.columns(min(len(unique_types), 6))
            selected_types = []
            for i, tt in enumerate(unique_types):
                with cols[i % 6]:
                    if st.checkbox(tt, value=True, key=f"filter_tt_{tt}"):
                        selected_types.append(tt)
            if not selected_types:
                st.warning("No trade types selected.")
                return
            trades = trades[trades["trade_type"].isin(selected_types)].copy()
            trades["cumulative_ticks"] = trades["ticks"].cumsum()

            trade_type_filtered = len(selected_types) < len(unique_types)
            if trade_type_filtered:
                selected_trade_types_meta = selected_types

    # ── News & holiday breakdown — computed HERE, before day_type filter ───────
    render_news_holiday_breakdown(trades)

    # ── Day type filter — checkboxes driven by the shared DAY_TYPE_ORDER ───────
    st.write("")
    st.caption("Filter by day type")
    day_filter_cols    = st.columns(len(DAY_TYPE_ORDER))
    selected_day_types = []
    for i, (tag, label) in enumerate(DAY_TYPE_ORDER):
        with day_filter_cols[i]:
            if st.checkbox(label, value=True, key=f"filter_dt_{tag}"):
                selected_day_types.append(tag)

    if not selected_day_types:
        st.warning("No day types selected.")
        return

    trades = trades[trades["day_type"].isin(selected_day_types)].copy()
    trades["cumulative_ticks"] = trades["ticks"].cumsum()

    if trades.empty:
        st.warning("No trades match the selected filters.")
        return

    # ── Filter state for save metadata ─────────────────────────────────────────
    day_type_filtered = len(selected_day_types) < len(DAY_TYPE_ORDER)
    filtered          = day_type_filtered or trade_type_filtered

    # ── Rest of the page uses filtered trades ─────────────────────────────────
    render_metrics(trades)
    # Asset of the RUN in session state (not the selectbox, which may have moved)
    run_asset = st.session_state.folder_path.parts[-2]
    render_market_exposure(trades, run_asset)
    selected       = render_equity_curve(trades)
    chart_settings = render_chart_view_controls()
    render_trade_detail(selected, trades, chart_settings)
    render_rr_distribution(trades)
    render_trades_table(
        trades, dataset, strategy_name, start_date, end_date,
        filtered, selected_day_types, selected_trade_types_meta,
    )


'''
Known limitations:
- direction must be lowercase 'long' or 'short'.
- day_type is derived at render time and stripped before saving trades.
- Holiday takes priority over high_impact when both tags exist on the same date.

Essential columns for analytics:
  ticks          ← required by all sizers
  pnl_points     ← required by risk_based sizer
  entry_price    ← required by risk_based sizer + charts
  sl             ← required by risk_based sizer + RR
  tp             ← required by RR
  entry_time     ← required for charts
  date           ← required for Sharpe + day classification
'''