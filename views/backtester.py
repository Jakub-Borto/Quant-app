# views/backtester.py
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import importlib.util

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

HIDDEN_PARAMS = {"tick_size"}

def save_trades(trades: pd.DataFrame, dataset: str, strategy: str, 
                start_date, end_date) -> str:
    
    trades_path = Path("data/trades")
    trades_path.mkdir(parents=True, exist_ok=True)
    
    base_name = f"{dataset}_{strategy}_{start_date}_{end_date}"
    
    # check existing files with same base name
    existing = sorted(trades_path.glob(f"{base_name}*.parquet"))
    
    for f in existing:
        existing_trades = pd.read_parquet(f)
        if existing_trades.equals(trades):
            return None  # identical file exists, skip
    
    # find next available name
    if not existing:
        output_path = trades_path / f"{base_name}.parquet"
    else:
        output_path = trades_path / f"{base_name}_{len(existing) + 1}.parquet"
    
    trades.to_parquet(output_path)
    return str(output_path)

def go_page(page: str):
    st.session_state.page = page
    st.rerun()

def get_parquet_structure() -> dict:
    """
    Scans data/parquet and returns nested structure:
    { type: { asset: [dataset_folder, ...] } }
    """
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

def get_strategies():
    strategies_path = Path("strategies")
    if not strategies_path.exists():
        return []
    return sorted([
        f.stem for f in strategies_path.glob("*.py")
        if f.stem not in ["__init__", "base"]
    ])

def load_strategy(name: str):
    path = Path("strategies") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
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
        folder_path = Path("data/parquet") / asset_type / asset / dataset
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
            start_date = None
            end_date   = None

    return asset_type, asset, dataset, strategy_name, start_date, end_date

def render_params(strategy) -> dict:
    if not hasattr(strategy, "PARAMS"):
        return {}

    visible = {k: v for k, v in strategy.PARAMS.items() if k not in HIDDEN_PARAMS}

    if not visible:
        return {}

    st.write("")
    st.subheader("Parameters")
    params = {}
    items = list(visible.items())
    chunk_size = 10
    chunks = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]

    for chunk in chunks:
        param_cols = st.columns(len(chunk))
        for i, (key, default) in enumerate(chunk):
            with param_cols[i]:
                if isinstance(default, float):
                    params[key] = st.number_input(key, value=default, step=0.1, format="%.2f")
                else:
                    params[key] = st.number_input(key, value=default, step=1)

    return params

def execute_run(strategy, asset_type, asset, dataset,
                start_date, end_date, params) -> bool:
    st.session_state.trades       = None
    st.session_state.folder_path  = None

    if start_date > end_date:
        st.error("Start date must be before end date.")
        return False

    if asset not in ASSET_INFO:
        st.error(f"Unknown asset: {asset}. Add it to ASSET_INFO.")
        return False

    asset_info      = ASSET_INFO[asset]
    ticks_per_point = asset_info["ticks_per_point"]
    folder_path     = Path("data/parquet") / asset_type / asset / dataset

    # Inject hidden params before passing to strategy
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

def compute_metrics(trades: pd.DataFrame) -> dict:
    winning   = trades[trades["ticks"] > 0]
    losing    = trades[trades["ticks"] < 0]
    breakeven = trades[trades["ticks"] == 0]

    avg_win  = winning["ticks"].mean() if len(winning) > 0 else 0.0
    avg_loss = losing["ticks"].mean()  if len(losing)  > 0 else 0.0
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

    # Daily Sharpe — zero-fill business days (institutional standard)
    all_dates = pd.bdate_range(trades["date"].min(), trades["date"].max())
    daily_pnl = (
        trades.groupby("date")["ticks"]
        .sum()
        .reindex(all_dates, fill_value=0)
    )
    daily_std    = daily_pnl.std(ddof=1)
    sharpe_daily = (daily_pnl.mean() / daily_std) * (252 ** 0.5) if daily_std > 0 else 0.0

    # Trade Sharpe — per-trade consistency, annualized by actual trading days
    trade_std      = trades["ticks"].std(ddof=1)
    n_trading_days = trades["date"].nunique()
    sharpe_trade   = (trades["ticks"].mean() / trade_std) * (n_trading_days ** 0.5) if trade_std > 0 else 0.0

    # Equity curve / drawdown
    cumulative   = trades["cumulative_ticks"]
    rolling_max  = cumulative.cummax()
    drawdown     = cumulative - rolling_max
    max_drawdown = drawdown.min()

    max_dd_idx     = drawdown.idxmin()
    peak_at_max_dd = rolling_max.loc[max_dd_idx]
    if peak_at_max_dd > 0:
        max_drawdown_pct = (max_drawdown / peak_at_max_dd) * 100
    else:
        max_drawdown_pct = None

    total       = trades["ticks"].sum()
    global_peak = rolling_max.max()

    calmar = total / abs(max_drawdown) if max_drawdown < 0 else float("inf")

    # Planned RR — abs(tp - entry) / abs(sl - entry), all trades
    if "entry_price" in trades.columns and "sl" in trades.columns and "tp" in trades.columns:
        sl_dist    = (trades["entry_price"] - trades["sl"]).abs()
        tp_dist    = (trades["tp"] - trades["entry_price"]).abs()
        rr_series  = (tp_dist / sl_dist)[sl_dist > 0]
        avg_rr     = rr_series.mean()
        median_rr  = rr_series.median()
    else:
        avg_rr    = None
        median_rr = None

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
        avg_duration_min    = None
        median_duration_min = None

    # Per-direction stats
    long_trades  = trades[trades["direction"] == "long"]
    short_trades = trades[trades["direction"] == "short"]
    long_winrate  = (long_trades["ticks"] > 0).mean()  if len(long_trades)  > 0 else 0.0
    short_winrate = (short_trades["ticks"] > 0).mean() if len(short_trades) > 0 else 0.0

    return {
        "total_ticks":          total,
        "avg_trade":            trades["ticks"].mean(),
        "avg_win":              avg_win,
        "avg_loss":             avg_loss,
        "largest_win":          winning["ticks"].max() if len(winning) > 0 else 0,
        "largest_loss":         losing["ticks"].min()  if len(losing)  > 0 else 0,
        "avg_rr":               avg_rr,
        "median_rr":            median_rr,
        "total_trades":         len(trades),
        "win_rate":             win_rate,
        "loss_rate":            loss_rate,
        "breakeven_rate":       len(breakeven) / len(trades),
        "sharpe_daily":         sharpe_daily,
        "sharpe_trade":         sharpe_trade,
        "profit_factor":        profit_factor,
        "calmar":               calmar,
        "max_drawdown":         max_drawdown,
        "max_drawdown_pct":     max_drawdown_pct,
        "max_peak":             global_peak,
        "max_consec_wins":      max_consec_wins,
        "max_consec_losses":    max_consec_losses,
        "avg_duration_min":     avg_duration_min,
        "median_duration_min":  median_duration_min,
        "long_winrate":         long_winrate,
        "short_winrate":        short_winrate,
    }


def render_metrics(trades: pd.DataFrame):
    st.write("")
    st.subheader("Performance")

    m = compute_metrics(trades)

    pf_display         = "∞" if m["profit_factor"] == float("inf") else f"{m['profit_factor']:.2f}"
    calmar_display     = "∞" if m["calmar"]        == float("inf") else f"{m['calmar']:.2f}"
    mdd_pct_display    = f"{m['max_drawdown_pct']:.1f}%" if m["max_drawdown_pct"] is not None else "N/A"
    avg_dur_display    = f"{m['avg_duration_min']:.0f}m"    if m["avg_duration_min"]    is not None else "N/A"
    median_dur_display = f"{m['median_duration_min']:.0f}m" if m["median_duration_min"] is not None else "N/A"
    avg_rr_display     = f"{m['avg_rr']:.2f}"    if m["avg_rr"]    is not None else "N/A"
    median_rr_display  = f"{m['median_rr']:.2f}" if m["median_rr"] is not None else "N/A"
    
    '''
    
    
    
    '''

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

    # Row 2 — Risk-adjusted
    st.write("")
    r2c1, r2c2, r2c3, r2c4, r2c5, r2c6, r2c7, r2c8, r2c9 = st.columns(9)
    r2c1.metric("Win Rate",         f"{m['win_rate']:.1%}")
    r2c2.metric("Avg RR",           avg_rr_display)
    r2c3.metric("Median RR",            median_rr_display)
    r2c4.metric("Loss Rate",        f"{m['loss_rate']:.1%}")
    r2c5.metric("Breakeven Rate",   f"{m['breakeven_rate']:.1%}")
    r2c6.metric("Sharpe (daily)",   f"{m['sharpe_daily']:.2f}")
    r2c7.metric("Sharpe (trade)",   f"{m['sharpe_trade']:.2f}")
    r2c8.metric("Profit Factor",    pf_display)
    r2c9.metric("Calmar",           calmar_display)

    # Row 3 — Drawdown + streaks + duration
    st.write("")
    r3c1, r3c2, r3c3, r3c4, r3c5, r3c6, r3c7 = st.columns(7)
    r3c1.metric("Max Drawdown",    f"{m['max_drawdown']:.0f} ticks")
    r3c2.metric("Max Drawdown %",  mdd_pct_display)
    r3c3.metric("Max Peak",        f"{m['max_peak']:.0f} ticks")
    r3c4.metric("Consec. Wins",    m["max_consec_wins"])
    r3c5.metric("Consec. Losses",  m["max_consec_losses"])
    r3c6.metric("Avg Duration",    avg_dur_display)
    r3c7.metric("Median Duration", median_dur_display)

    # Row 4 — Directional breakdown
    st.write("")
    r4c1, r4c2, r4c3, r4c4, _, _, _ = st.columns(7)
    r4c1.metric("Long Win Rate",  f"{m['long_winrate']:.1%}")
    r4c2.metric("Short Win Rate", f"{m['short_winrate']:.1%}")
    r4c3.metric("Long Trades",    len(trades[trades["direction"] == "long"]))
    r4c4.metric("Short Trades",   len(trades[trades["direction"] == "short"]))

    # Exit breakdown
    st.write("")
    st.subheader("Exit Breakdown")
    exit_stats = trades.groupby("exit_reason")["ticks"].agg(
        count="count",
        avg="mean",
        total="sum"
    ).reset_index()
    exit_stats.columns        = ["Exit Reason", "Count", "Avg Ticks", "Total Ticks"]
    exit_stats["Avg Ticks"]   = exit_stats["Avg Ticks"].round(1)
    exit_stats["Total Ticks"] = exit_stats["Total Ticks"].round(0).astype(int)
    st.dataframe(exit_stats, width='stretch', hide_index=True)

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

    selected = st.plotly_chart(
        fig,
        width='stretch',
        on_select="rerun",
        key="equity_curve"
    )

    return selected

def render_chart_view_controls() -> dict:
    """
    Renders chart view configuration controls.
    Returns a dict with the user's chosen settings.
    """
    st.write("")
    st.subheader("Chart View Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        view_mode = st.radio(
            "Chart start from",
            options=["Candles before entry", "Fixed session time"],
            key="chart_view_mode",
            horizontal=True,
        )

    with col2:
        if view_mode == "Candles before entry":
            candles_before = st.number_input(
                "Candles before entry",
                value=30,
                min_value=1,
                max_value=390,
                step=1,
                key="chart_candles_before",
            )
            session_start_time = None
        else:
            candles_before = None
            session_start_time = st.time_input(
                "Session start time (NY)",
                value=pd.Timestamp("09:30").time(),
                key="chart_session_start",
                step=60,
            )

    with col3:
        candles_after = st.number_input(
            "Candles after exit",
            value=10,
            min_value=0,
            max_value=390,
            step=1,
            key="chart_candles_after",
        )

    return {
        "view_mode":          view_mode,
        "candles_before":     candles_before,
        "session_start_time": session_start_time,
        "candles_after":      candles_after,
    }

def resolve_chart_window(
    session:        pd.DataFrame,
    entry_ts:       pd.Timestamp,
    exit_ts:        pd.Timestamp,
    chart_settings: dict,
) -> pd.DataFrame:
    """
    Slices the session DataFrame to the configured chart window.

    Two modes:
      - "Candles before entry": start N bars before entry_ts
      - "Fixed session time":   start at a fixed time (e.g. 09:30)

    Always extends candles_after bars past the exit bar.
    """
    # Find exit position
    exit_loc = session.index.searchsorted(exit_ts, side="right") - 1
    exit_loc = max(0, min(exit_loc, len(session) - 1))
    end_loc  = min(exit_loc + chart_settings["candles_after"], len(session) - 1)

    if chart_settings["view_mode"] == "Candles before entry":
        entry_loc  = session.index.searchsorted(entry_ts, side="left")
        entry_loc  = max(0, min(entry_loc, len(session) - 1))
        start_loc  = max(0, entry_loc - chart_settings["candles_before"])

    else:
        # Fixed session time — find first bar at or after the chosen time
        target_time = chart_settings["session_start_time"]
        time_mask   = session.index.time >= target_time
        if time_mask.any():
            start_loc = int(time_mask.argmax())
        else:
            # Fallback: start of session
            start_loc = 0

    return session.iloc[start_loc : end_loc + 1]

def build_trade_figure(trade, chart_candles: pd.DataFrame,
                       entry_ts: str, exit_ts: str) -> go.Figure:
    fig = go.Figure()

    # candlesticks
    fig.add_trace(go.Candlestick(
        x=chart_candles.index,
        open=chart_candles["open"],
        high=chart_candles["high"],
        low=chart_candles["low"],
        close=chart_candles["close"],
        name="Price",
    ))

    # entry line
    fig.add_shape(
        type="line",
        x0=entry_ts, x1=exit_ts,
        y0=trade["entry_price"], y1=trade["entry_price"],
        line=dict(color="blue", width=1, dash="solid"),
    )

    # sl line + shaded risk zone
    fig.add_shape(
        type="line",
        x0=entry_ts, x1=exit_ts,
        y0=trade["sl"], y1=trade["sl"],
        line=dict(color="red", width=1, dash="dash"),
    )
    fig.add_shape(
        type="rect",
        x0=entry_ts, x1=exit_ts,
        y0=trade["sl"], y1=trade["entry_price"],
        fillcolor="red", opacity=0.05, line_width=0,
    )

    # tp line + shaded reward zone
    fig.add_shape(
        type="line",
        x0=entry_ts, x1=exit_ts,
        y0=trade["tp"], y1=trade["tp"],
        line=dict(color="green", width=1, dash="dash"),
    )
    fig.add_shape(
        type="rect",
        x0=entry_ts, x1=exit_ts,
        y0=trade["entry_price"], y1=trade["tp"],
        fillcolor="green", opacity=0.05, line_width=0,
    )

    # entry marker
    fig.add_trace(go.Scatter(
        x=[entry_ts],
        y=[trade["entry_price"]],
        mode="markers",
        marker=dict(
            symbol="triangle-up" if trade["direction"] == "long" else "triangle-down",
            size=14,
            color="blue",
        ),
        name="Entry",
    ))

    # exit marker
    fig.add_trace(go.Scatter(
        x=[exit_ts],
        y=[trade["exit_price"]],
        mode="markers",
        marker=dict(symbol="x", size=14, color="orange"),
        name="Exit",
    ))

    fig.update_layout(
        height=700,
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
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
    asset           = folder_path.parts[-2]  # data/parquet/{type}/{asset}/{dataset}
    ticks_per_point = ASSET_INFO[asset]["ticks_per_point"]

    st.write("")
    st.divider()
    st.subheader(f"Trade Detail — {trade['date']}")

    duration = trade["exit_time"] - trade["entry_time"]
    hours    = int(duration.total_seconds() // 3600)
    minutes  = int((duration.total_seconds() % 3600) // 60)
    sl_ticks = abs(trade["entry_price"] - trade["sl"]) * ticks_per_point
    tp_ticks = abs(trade["entry_price"] - trade["tp"]) * ticks_per_point
    actual_rr = tp_ticks / sl_ticks if sl_ticks > 0 else 0

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Direction",   trade["direction"].upper())
    c2.metric("Tick PnL",    f"{trade['ticks']:.0f}")
    c3.metric("SL Ticks",    f"{sl_ticks:.0f}")
    c4.metric("TP Ticks",    f"{tp_ticks:.0f}")
    c5.metric("RR",          f"{actual_rr:.2f}")
    c6.metric("Duration",    f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m")
    c7.metric("Exit Reason", trade["exit_reason"])

    if "notes" in trade.index and pd.notna(trade["notes"]):
        try:
            notes = json.loads(trade["notes"])
            cols = st.columns(len(notes))
            for col, (key, val) in zip(cols, notes.items()):
                if isinstance(val, list):
                    display = ", ".join(
                        pd.Timestamp(v).strftime("%H:%M")
                        if _is_timestamp(v) else str(v)
                        for v in val
                    )
                elif _is_timestamp(val):
                    display = pd.Timestamp(val).strftime("%H:%M")
                else:
                    display = str(val)
                col.metric(key, display)
        except Exception as e:
            st.warning(str(e))

    trade_date    = pd.Timestamp(trade["date"])
    session       = pd.read_parquet(folder_path / f"{trade_date.date().isoformat()}.parquet")
    session       = session[session.index.date == trade_date.date()]
    chart_candles = resolve_chart_window(
        session        = session,
        entry_ts       = trade["entry_time"],
        exit_ts        = trade["exit_time"],
        chart_settings = chart_settings,
    )

    entry_ts = str(trade["entry_time"])
    exit_ts  = str(trade["exit_time"])

    st.plotly_chart(
        build_trade_figure(trade, chart_candles, entry_ts, exit_ts),
        width='stretch',
    )

def render_trades_table(trades: pd.DataFrame, dataset: str, strategy_name: str,
                        start_date, end_date):
    st.write("")
    st.subheader("Trades")
    display_cols = ["date", "direction", "entry_time", "exit_time",
                    "entry_price", "exit_price", "exit_reason", "ticks"]
    if "trade_type" in trades.columns:
        display_cols.append("trade_type")
    st.dataframe(trades[display_cols], width='stretch')

    st.write("")
    _, _, save_col, _, _ = st.columns(5)
    with save_col:
        if st.button("Save Trades", width='stretch'):
            result = save_trades(
                trades, dataset, strategy_name,
                start_date, end_date
            )
            if result is None:
                st.info("Identical trades file already exists — not saved.")
            else:
                st.success(f"Saved to {result}")


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

    result = render_controls(structure, strategies)
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
        execute_run(strategy, asset_type, asset, dataset,
                    start_date, end_date, params)

    if st.session_state.trades is not None:
        trades = st.session_state.trades

        if "trade_type" in trades.columns:
            unique_types = sorted(trades["trade_type"].dropna().unique().tolist())
            if unique_types:
                st.write("")
                st.caption("Filter by trade type")
                cols = st.columns(min(len(unique_types), 6))
                selected_types = []
                for i, tt in enumerate(unique_types):
                    with cols[i % 6]:
                        if st.checkbox(tt, value=True, key=f"filter_tt_{tt}"):
                            selected_types.append(tt)
                if not selected_types:
                    st.warning("No trade types selected.")
                    return
                trades = trades[trades["trade_type"].isin(selected_types)]
                trades = trades.copy()
                trades["cumulative_ticks"] = trades["ticks"].cumsum()

        render_metrics(trades)
        selected       = render_equity_curve(trades)
        chart_settings = render_chart_view_controls()
        render_trade_detail(selected, trades, chart_settings)
        render_trades_table(trades, dataset, strategy_name, start_date, end_date)


'''
Known limitations:
- 252 annualization factor assumes daily trading. Fine for intraday on RTH days;
  revisit if the strategy holds positions for weeks or trades on a non-daily cadence.
- "direction must be lowercase 'long' or 'short'."

Essential columns for analytics:
  ticks          ← required by all three sizers
  pnl_points     ← required by risk_based only
  entry_price    ← required by risk_based only
  sl             ← required by risk_based only
  entry_time     ← required for charts
  date           ← required for Sharpe
'''

