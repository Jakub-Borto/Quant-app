# views/backtester.py
import json
import streamlit as st
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import importlib.util
import sys

from optimization.buckets import load_bucket_map
from views.trade_report import (
    DAY_TYPE_ORDER, compute_metrics, render_chart_view_controls,
    render_equity_curve, render_market_exposure, render_metrics,
    render_news_holiday_breakdown, render_rr_distribution,
    render_trade_detail,
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


# ── News / holiday classification ─────────────────────────────────────────────
#
# The classification logic lives in optimization/buckets.py (shared with the
# Strategy Optimizer — same priority rules, same EVENT_KEYWORDS config), so
# both views classify a given date identically. This view keeps only its
# historical names: day_type column, 'high_impact' instead of the shared
# 'other_high_impact'. DAY_TYPE_ORDER (imported from views/trade_report.py,
# also shared) drives the filter UI and the news/holiday breakdown table.

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
    render_market_exposure(trades, run_asset, ASSET_INFO[run_asset]["tick_size"])
    selected       = render_equity_curve(trades)
    chart_settings = render_chart_view_controls()
    render_trade_detail(selected, trades, chart_settings,
                        st.session_state.folder_path,
                        ASSET_INFO[run_asset]["ticks_per_point"])
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