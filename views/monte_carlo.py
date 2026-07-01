"""
views/monte_carlo.py

Monte Carlo simulation page.
Loads trades, applies position sizing, runs a selected MC simulation,
and renders a fan chart + metrics table.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# NOTE: this dict is duplicated across views (analytics, monte_carlo) — known
# tech debt. The commissions_per_contract / parent keys MUST be kept in lockstep
# with views/analytics.py or commissions silently diverge per page.
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

def _get_dollars_per_tick(trade_filename: str) -> float:
    asset = trade_filename.split("_")[0]
    if asset not in ASSET_INFO:
        raise ValueError(f"Unknown asset '{asset}' from filename '{trade_filename}'. Add it to ASSET_INFO.")
    return ASSET_INFO[asset]["dollars_per_tick"]


def _micro_child(asset: str) -> str | None:
    """Return the micro ticker whose `parent` is `asset`, or None. An asset is
    'microable' iff such a child exists — that's the only decomposition flag."""
    for ticker, info in ASSET_INFO.items():
        if info.get("parent") == asset:
            return ticker
    return None


def _get_commission_info(trades_filename: str) -> tuple[float | None, float | None]:
    """
    (full_commission, micro_commission) for the file's asset, mirroring
    analytics.get_commission_info. `full` is None if the asset has no
    commissions_per_contract key (graceful degradation → caller bills 0 + warns).
    `micro` is the child's commission when a micro child exists, else None
    (non-microable).
    """
    asset = trades_filename.split("_")[0]
    info  = ASSET_INFO.get(asset, {})
    full  = info.get("commissions_per_contract")

    child = _micro_child(asset)
    micro = ASSET_INFO[child].get("commissions_per_contract") if child else None
    return full, micro

# ---------------------------------------------------------------------------
# How many faded sample (grey) paths to render on each fan chart. A FIXED COUNT,
# not a fraction — 50k paths still shows this many lines (intended, for
# readability). Applies to every fan chart, including the three prop-firm ones.
SAMPLE_PATH_COUNT = 200
# ---------------------------------------------------------------------------


def go_page(page: str):
    st.session_state.page = page
    st.rerun()


# ---------------------------------------------------------------------------
# Dynamic module loading — same pattern as backtester / analytics
# ---------------------------------------------------------------------------

def _load_module(path: Path):
    spec   = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_plugins(folder: str, exclude: list[str] = None) -> dict[str, Path]:
    """Return {display_name: Path} for all .py files in folder."""
    exclude = exclude or ["__init__", "base"]
    return {
        p.stem: p
        for p in sorted(Path(folder).glob("*.py"))
        if p.stem not in exclude
    }


def _param_widgets(params: dict, key_prefix: str) -> dict:
    """Auto-generate number_input widgets from a PARAMS dict."""
    result = {}
    for name, default in params.items():
        if isinstance(default, int):
            result[name] = st.number_input(
                name, value=default, step=1,
                key=f"{key_prefix}_{name}"
            )
        elif isinstance(default, float):
            result[name] = st.number_input(
                name, value=default, step=0.1, format="%.2f",
                key=f"{key_prefix}_{name}"
            )
    return result


# ---------------------------------------------------------------------------
# Trades loader
# ---------------------------------------------------------------------------

def _get_trade_files() -> list[str]:
    trades_path = Path("data/trades")
    if not trades_path.exists():
        return []
    return sorted([f.stem for f in trades_path.glob("*.parquet")])


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _max_drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown as a positive dollar value."""
    peak    = np.maximum.accumulate(equity)
    dd      = peak - equity
    return float(np.max(dd))


def _sharpe(equity: np.ndarray, account_size: float) -> float:
    """Annualised Sharpe on daily P&L, ddof=1."""
    pnl = np.diff(equity)
    if len(pnl) < 2 or np.std(pnl, ddof=1) == 0:
        return np.nan
    return float(np.mean(pnl) / np.std(pnl, ddof=1) * np.sqrt(252))


def _compute_metrics(
    equity_matrix: np.ndarray,   # shape (n_paths, n_trades+1)
    account_size:  float,
    ruin_threshold: float | None,
) -> dict:
    """
    Compute summary metrics across all paths.
    ruin_threshold: None = disabled, otherwise fraction of account_size
                    e.g. 0.0 → ruin = equity <= 0
                         0.5 → ruin = equity <= 0.5 * account_size
    """
    final_equities = equity_matrix[:, -1]
    sharpes        = np.array([_sharpe(eq, account_size) for eq in equity_matrix])

    # Deepest drop below starting capital per path.
    # = how far below account_size the path ever went.
    # Paths that never dipped below starting capital contribute 0.
    below_start = np.maximum(0.0, account_size - equity_matrix.min(axis=1))

    # Band final equities — what each Gaussian percentile ends at
    band_finals = {}
    for lo_pct, hi_pct in BAND_PERCENTILES:
        lo_val = float(np.percentile(final_equities, lo_pct))
        hi_val = float(np.percentile(final_equities, hi_pct))
        band_finals[(lo_pct, hi_pct)] = (lo_val, hi_val)

    metrics = {
        "Median final equity":        float(np.median(final_equities)),
        "Mean final equity":          float(np.mean(final_equities)),
        "band_finals":                band_finals,
        # Worst drop below starting capital across all paths (3σ tail)
        "Worst drop below start":     float(np.percentile(below_start, 99.85)),
        # Median drop below starting capital (typical bad path)
        "Median drop below start":    float(np.median(below_start)),
        # % of paths that ever went below starting capital at all
        "P(ever below start)":        float(np.mean(below_start > 0) * 100),
        "Median Sharpe":              float(np.nanmedian(sharpes)),
    }

    if ruin_threshold is not None:
        floor         = account_size * ruin_threshold
        ruin_mask     = np.any(equity_matrix <= floor, axis=1)
        metrics["P(ruin)"]        = float(np.mean(ruin_mask) * 100)
        metrics["Ruin threshold"] = floor
    else:
        metrics["P(ruin)"]        = None
        metrics["Ruin threshold"] = None

    return metrics


# ---------------------------------------------------------------------------
# Featured path selection
# ---------------------------------------------------------------------------

def _select_featured_paths(equity_matrix: np.ndarray) -> dict[str, int]:
    """
    Return {label: path_index} for the most notable paths.
    Deduplicates — if two labels point to the same path, only the first kept.
    """
    final_eq  = equity_matrix[:, -1]
    peak_eq   = equity_matrix.max(axis=1)
    # Lowest trough ever reached — path that came closest to ruin
    trough_eq = equity_matrix.min(axis=1)

    # Priority order matters for deduplication:
    # Best paths first — if a moonshot path also has the worst drawdown dollar-wise,
    # it shows as "Best final equity", not "Max drawdown".
    # Max drawdown (lowest trough) is last priority.
    candidates = {
        "Best final equity":  int(np.argmax(final_eq)),
        "Best peak equity":   int(np.argmax(peak_eq)),
        "Worst final equity": int(np.argmin(final_eq)),
        "Lowest trough":      int(np.argmin(trough_eq)),
    }

    seen   = set()
    unique = {}
    for label, idx in candidates.items():
        if idx not in seen:
            unique[label] = idx
            seen.add(idx)
    return unique


# ---------------------------------------------------------------------------
# Fan chart
# ---------------------------------------------------------------------------

FEATURED_COLORS = {
    "Best final equity":  "#44ff44",
    "Best peak equity":   "#aaffaa",
    "Worst final equity": "#aa2222",
    "Lowest trough":      "#ff4444",
}

# 1σ/2σ/3σ band fill opacities
BAND_ALPHAS = [0.25, 0.15, 0.08]

# Gaussian percentile pairs for 1σ/2σ/3σ
BAND_PERCENTILES = [
    (16.0,  84.0),   # 1σ — 68%
    (2.5,   97.5),   # 2σ — 95%
    (0.15,  99.85),  # 3σ — 99.7%
]

BAND_LABELS = ["1σ (68%)", "2σ (95%)", "3σ (99.7%)"]
BAND_COLOR  = "100, 160, 255"   # RGB for the band fills


def _build_fan_chart(
    equity_matrix:  np.ndarray,
    account_size:   float,
    featured:       dict[str, int],
    ruin_threshold: float | None,
    y_max:          float | None = None,
    band_finals:    dict | None  = None,
) -> go.Figure:

    n_paths, n_steps = equity_matrix.shape
    x = list(range(n_steps))

    fig = go.Figure()

    # -- 3σ bands (outermost first so inner bands render on top) -------------
    for (lo_pct, hi_pct), alpha, label in zip(
        reversed(BAND_PERCENTILES),
        reversed(BAND_ALPHAS),
        reversed(BAND_LABELS),
    ):
        lo = np.percentile(equity_matrix, lo_pct, axis=0)
        hi = np.percentile(equity_matrix, hi_pct, axis=0)

        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=np.concatenate([hi, lo[::-1]]).tolist(),
            fill="toself",
            fillcolor=f"rgba({BAND_COLOR},{alpha})",
            line=dict(color="rgba(0,0,0,0)"),
            name=label,
            hoverinfo="skip",
        ))

    # -- faded sample paths --------------------------------------------------
    featured_indices = set(featured.values())
    all_indices      = list(range(n_paths))
    sample_pool      = [i for i in all_indices if i not in featured_indices]

    rng      = np.random.default_rng(seed=0)
    sampled  = rng.choice(sample_pool, size=min(SAMPLE_PATH_COUNT, len(sample_pool)), replace=False)

    for idx in sampled:
        fig.add_trace(go.Scatter(
            x=x, y=equity_matrix[idx].tolist(),
            mode="lines",
            line=dict(color="rgba(180,180,180,0.08)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

    # -- featured paths ------------------------------------------------------
    for label, idx in featured.items():
        color = FEATURED_COLORS.get(label, "#ffffff")
        eq    = equity_matrix[idx]

        fig.add_trace(go.Scatter(
            x=x, y=eq.tolist(),
            mode="lines",
            line=dict(color=color, width=1.5),
            name=label,
            hovertemplate=f"{label}<br>Trade: %{{x}}<br>Equity: $%{{y:,.0f}}<extra></extra>",
        ))

        # end-of-line annotation
        fig.add_annotation(
            x=n_steps - 1,
            y=float(eq[-1]),
            text=f"  {label}",
            showarrow=False,
            xanchor="left",
            font=dict(color=color, size=10),
        )

    # -- median path ---------------------------------------------------------
    median_eq = np.median(equity_matrix, axis=0)
    fig.add_trace(go.Scatter(
        x=x, y=median_eq.tolist(),
        mode="lines",
        line=dict(color="#ffffff", width=2.5),
        name="Median",
        hovertemplate="Median<br>Trade: %{x}<br>Equity: $%{y:,.0f}<extra></extra>",
    ))

    # -- band final equity lines ---------------------------------------------
    # Dotted horizontal lines at each σ percentile pair's final equity value.
    # Drawn at the right edge of the chart so you can read where each band ends.
    if band_finals:
        band_line_styles = [
            ("rgba(100,160,255,0.7)", "dot",   "1σ"),   # 68%
            ("rgba(100,160,255,0.5)", "dash",  "2σ"),   # 95%
            ("rgba(100,160,255,0.3)", "dashdot","3σ"),  # 99.7%
        ]
        for (lo_pct, hi_pct), (color, dash, label) in zip(
            BAND_PERCENTILES, band_line_styles
        ):
            lo_val, hi_val = band_finals[(lo_pct, hi_pct)]

            # Skip lines that are above the capped y_max — they'd be invisible anyway
            for val, side in ((hi_val, "↑"), (lo_val, "↓")):
                if y_max is not None and val > y_max:
                    continue
                fig.add_hline(
                    y=val,
                    line=dict(color=color, width=1, dash=dash),
                    annotation_text=f"{label} {side}  ${val:,.0f}",
                    annotation_position="right",
                    annotation_font=dict(color=color, size=9),
                )

    # -- ruin threshold line -------------------------------------------------
    if ruin_threshold is not None:
        floor = account_size * ruin_threshold
        fig.add_hline(
            y=floor,
            line=dict(color="rgba(255,80,80,0.6)", width=1, dash="dash"),
            annotation_text=f"Ruin floor ${floor:,.0f}",
            annotation_font_color="rgba(255,80,80,0.8)",
        )

    # -- starting equity reference line --------------------------------------
    fig.add_hline(
        y=account_size,
        line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        height=600,
        margin=dict(l=60, r=120, t=40, b=40),
        xaxis=dict(title="Trade #", showgrid=True, gridcolor="#222"),
        yaxis=dict(
            title="Equity ($)",
            showgrid=True,
            gridcolor="#222",
            tickformat="$,.0f",
            range=[0, y_max] if y_max is not None else None,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#444",
            borderwidth=1,
            font=dict(size=11),
        ),
        hovermode="x",
    )

    return fig


# ---------------------------------------------------------------------------
# Metrics table renderer
# ---------------------------------------------------------------------------

def _render_metrics(metrics: dict, account_size: float):
    st.subheader("Metrics")

    def fmt_dollar(v):     return f"${v:,.0f}"   if v is not None else "—"
    def fmt_pct(v):        return f"{v:.1f}%"    if v is not None else "—"
    def fmt_num(v):        return f"{v:.3f}"     if v is not None and not np.isnan(v) else "—"
    def pct_of_start(v):   return fmt_pct((v / account_size - 1) * 100)
    def loss_pct(v):       return fmt_pct(-(v / account_size) * 100)   # drop as % of start

    rows = []

    # --- Final equity -------------------------------------------------------
    final_median = metrics["Median final equity"]
    final_mean   = metrics["Mean final equity"]
    rows.append(("Median final equity", fmt_dollar(final_median), pct_of_start(final_median)))
    rows.append(("Mean final equity",   fmt_dollar(final_mean),   pct_of_start(final_mean)))

    # --- Band final equities ------------------------------------------------
    band_labels = ["1σ — 68%", "2σ — 95%", "3σ — 99.7%"]
    for (lo_pct, hi_pct), label in zip(BAND_PERCENTILES, band_labels):
        lo_val, hi_val = metrics["band_finals"][(lo_pct, hi_pct)]
        rows.append((
            f"Final equity {label}",
            f"{fmt_dollar(lo_val)} – {fmt_dollar(hi_val)}",
            f"{pct_of_start(lo_val)} – {pct_of_start(hi_val)}",
        ))

    rows.append(("", "", ""))   # spacer

    # --- Drawdown below starting capital ------------------------------------
    worst  = metrics["Worst drop below start"]
    median = metrics["Median drop below start"]
    p_ever = metrics["P(ever below start)"]

    rows.append((
        "Worst drop below starting capital (3σ)",
        fmt_dollar(worst),
        loss_pct(worst),
    ))
    rows.append((
        "Median drop below starting capital",
        fmt_dollar(median) if median > 0 else "Never",
        loss_pct(median)   if median > 0 else "—",
    ))
    rows.append((
        "% of paths that ever went below start",
        fmt_pct(p_ever),
        "",
    ))

    rows.append(("", "", ""))   # spacer

    # --- Sharpe & ruin ------------------------------------------------------
    rows.append(("Median Sharpe", fmt_num(metrics["Median Sharpe"]), ""))

    if metrics["P(ruin)"] is not None:
        floor     = metrics["Ruin threshold"]
        floor_pct = (1 - floor / account_size) * 100
        rows.append((
            f"P(ruin) — floor {fmt_dollar(floor)} ({floor_pct:.0f}% loss)",
            fmt_pct(metrics["P(ruin)"]),
            "",
        ))

    df = pd.DataFrame(rows, columns=["Metric", "Value", "vs Starting Capital"])
    st.dataframe(df, width='stretch', hide_index=True, height=500)


# ===========================================================================
# Prop-firm UI (dedicated flow — per-rule toggles + three charts/stats)
# ===========================================================================

_PROP_OPTIMISM_CAPTION = (
    "Breach is checked on **closed-trade** equity — an intra-trade dip below the "
    "floor that recovers to a green close is not counted. Every P(pass)/P(payout) "
    "here is therefore an **upper bound**."
)


def _rule_widget(label: str, default: dict, key_prefix: str,
                 *, step: float = 100.0, pct: bool = False, help: str = None) -> dict:
    """Render a checkbox + value input for an {enabled, value} rule. Value greys
    out when the checkbox is off. Returns the resolved {enabled, value}."""
    c1, c2 = st.columns([1.5, 1])
    with c1:
        enabled = st.checkbox(label, value=bool(default["enabled"]),
                              key=f"{key_prefix}_en", help=help)
    with c2:
        if pct:
            v = st.number_input("max % of profit from one day",
                                value=float(default["value"]) * 100.0,
                                min_value=0.0, max_value=100.0, step=5.0,
                                key=f"{key_prefix}_val", disabled=not enabled,
                                label_visibility="collapsed")
            value = v / 100.0
        else:
            value = st.number_input("value", value=float(default["value"]), step=step,
                                    format="%.2f", key=f"{key_prefix}_val",
                                    disabled=not enabled, label_visibility="collapsed")
    return {"enabled": bool(enabled), "value": float(value)}


def _fmt_pct01(v):    return f"{v*100:.1f}%" if v is not None else "—"
def _fmt_dollar(v):   return f"${v:,.0f}"    if v is not None else "—"


def _fmt_pctiles(p: dict | None) -> str:
    if p is None:
        return "— (none)"
    return f"{p['median']:.0f}  (25–75: {p['p25']:.0f}–{p['p75']:.0f}, 95th: {p['p95']:.0f})"


def _prop_fan(sim: dict, account_size: float, costs_on: bool):
    """Fan chart for Sim 1 / Sim 2, reusing the shared fan-chart machinery."""
    eqm      = sim["equity_matrix"]
    featured = _select_featured_paths(eqm)
    metrics  = _compute_metrics(eqm, account_size, None)
    fig = _build_fan_chart(
        equity_matrix  = eqm,
        account_size   = account_size,
        featured       = featured,
        ruin_threshold = None,
        y_max          = None,
        band_finals    = metrics["band_finals"],
    )
    # Target line (pass / payout threshold).
    if sim.get("target"):
        tgt = account_size + sim["target"]
        fig.add_hline(y=tgt, line=dict(color="rgba(80,220,120,0.7)", width=1, dash="dash"),
                      annotation_text=f"Target ${tgt:,.0f}",
                      annotation_font_color="rgba(80,220,120,0.9)")
    fig.update_layout(xaxis_title="Trade # (path stops at pass/fail, then holds flat)")
    st.plotly_chart(fig, width="stretch")


def _render_prop_sim(sim: dict, account_size: float, costs_on: bool):
    st.write("")
    st.subheader(sim["title"])
    caption = _PROP_OPTIMISM_CAPTION
    if costs_on:
        caption += "  Equity is net of commissions & slippage."
    st.caption(caption)

    _prop_fan(sim, account_size, costs_on)

    s = sim["stats"]
    rows = []
    if "p_pass" in s:        # challenge
        rows.append(("P(pass)", _fmt_pct01(s["p_pass"])))
        rows.append(("Trades to pass", _fmt_pctiles(s["trades_to_pass"])))
        fb = s["failure_breakdown"]
        rows.append(("Failure — max-loss breach", _fmt_pct01(fb["max_loss"])))
        rows.append(("Failure — unresolved at horizon", _fmt_pct01(fb["unresolved"])))
        rows.append(("Consistency hold rate", _fmt_pct01(s["consistency_hold_rate"])))
        rows.append(("Median final equity (passers)", _fmt_dollar(s["median_final_equity_passers"])))
        rows.append(("Worst peak-to-trough (passers)", _fmt_dollar(s["worst_peak_to_trough_passers"])))
    else:                    # payout
        rows.append(("P(payout | funded)", _fmt_pct01(s["p_payout"])))
        rows.append(("Trades to payout", _fmt_pctiles(s["trades_to_payout"])))
        rows.append(("Funded breach rate (max-loss)", _fmt_pct01(s["breach_rate"])))
        rows.append(("Unresolved at horizon", _fmt_pct01(s["breach_breakdown"]["unresolved"])))
        rows.append(("Held-then-diluted-and-paid", _fmt_pct01(s["held_then_paid"])))
        rows.append(("Held-then-breached-while-grinding", _fmt_pct01(s["held_then_breached"])))
        rows.append(("Consistency hold rate", _fmt_pct01(s["consistency_hold_rate"])))

    df = pd.DataFrame(rows, columns=["Statistic", "Value"])
    st.dataframe(df, width="stretch", hide_index=True)


def _render_combined_sim(sim: dict, account_size: float, costs_on: bool):
    st.write("")
    st.subheader(sim["title"])
    st.caption(
        "End-to-end: challenge phase, then a fresh funded phase concatenated at "
        "the reset (yellow dot). Funded accounts reset to the starting balance — "
        "challenge profit is not carried. Green lines went on to get paid."
    )

    mat     = sim["equity_matrix"]
    reset_x = sim["reset_x"]
    paid    = sim["paid_mask"]

    if mat.shape[0] == 0:
        st.info("No paths passed the challenge — nothing to chart.")
    else:
        fig = go.Figure()
        x = list(range(mat.shape[1]))
        rng_ = np.random.default_rng(0)
        sample = rng_.choice(mat.shape[0], size=min(SAMPLE_PATH_COUNT, mat.shape[0]), replace=False)
        for r in sample:
            color = "rgba(80,200,120,0.30)" if paid[r] else "rgba(180,180,180,0.18)"
            fig.add_trace(go.Scatter(x=x, y=mat[r].tolist(), mode="lines",
                          line=dict(color=color, width=1), showlegend=False, hoverinfo="skip"))
            rx = int(reset_x[r])
            fig.add_trace(go.Scatter(x=[rx], y=[float(mat[r, rx])], mode="markers",
                          marker=dict(color="#ffcc00", size=4),
                          showlegend=False, hoverinfo="skip"))
        fig.add_hline(y=account_size, line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"),
                      annotation_text=f"Start ${account_size:,.0f}")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#111111", plot_bgcolor="#111111",
            height=560, margin=dict(l=60, r=120, t=40, b=40),
            xaxis=dict(title="Trade # (challenge → reset → funded)", showgrid=True, gridcolor="#222"),
            yaxis=dict(title="Equity ($)", showgrid=True, gridcolor="#222", tickformat="$,.0f"),
        )
        st.plotly_chart(fig, width="stretch")

    s = sim["stats"]
    p_pass, p_po, p_paid = s["funnel"]
    c1, c2, c3 = st.columns(3)
    c1.metric("P(pass)", f"{p_pass*100:.1f}%")
    c2.metric("P(payout | passed)", f"{p_po*100:.1f}%")
    c3.metric("P(paid) end-to-end", f"{p_paid*100:.2f}%")

    rows = [
        ("Total trades to payout (challenge + funded)", _fmt_pctiles(s["total_trades_to_payout"])),
        ("Realized payout per paid account", _fmt_dollar(s["realized_payout_per_paid"])),
        ("Expected payout value  =  P(paid) × realized", _fmt_dollar(s["expected_payout_value"])),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Statistic", "Value"]),
                 width="stretch", hide_index=True)


def _render_prop_firm(mc_module, mc_name, trades_path, trade_filename,
                      sizer_module, sizer_params, account_size):
    defaults = getattr(mc_module, "PARAMS", {})

    st.markdown("**General**")
    g1, g2, g3 = st.columns(3)
    with g1:
        n_paths = st.number_input("Paths", value=int(defaults.get("n_paths", 5000)),
                                  min_value=100, step=500, key="pf_n_paths")
    with g2:
        max_trades = st.number_input("Max trades (horizon)",
                                     value=int(defaults.get("max_trades", 500)),
                                     min_value=10, step=50, key="pf_max_trades",
                                     help="Trades before an unresolved path is stopped "
                                          "and counted as 'unresolved' (~trading days at 1 trade/day).")
    with g3:
        seed = st.number_input("Seed", value=int(defaults.get("seed", 42)),
                               step=1, key="pf_seed")

    st.markdown("**Challenge — passing ruleset**")
    profit_target = _rule_widget("Profit Target ($)", defaults["profit_target"],
                                 "pf_ch_target", step=500.0)
    ch_eod = _rule_widget("Max Loss Limit (EOD, trailing $)", defaults["challenge_max_loss_eod"],
                          "pf_ch_eod", step=500.0,
                          help="Trailing from the highest end-of-day balance; ratchets up only.")
    ch_daily = _rule_widget("Daily Loss Limit ($)", defaults["challenge_daily_loss"],
                            "pf_ch_daily", step=250.0,
                            help="Risk cap, not a breach — caps size so one day can't lose more than this.")
    ch_cons = _rule_widget("Consistency Rule", defaults["challenge_consistency"],
                           "pf_ch_cons", pct=True,
                           help="Max % of total profit allowed from a single day. A pass gate, not a breach.")
    ch_climit = _rule_widget("Contract Limit", defaults["challenge_contract_limit"],
                             "pf_ch_climit", step=1.0,
                             help="Hard size cap, full-contract units (3.0 = 3 minis = 30 micros).")

    st.markdown("**Payout — funded ruleset**")
    targeted_payout = _rule_widget("Targeted Payout ($)", defaults["targeted_payout"],
                                   "pf_po_target", step=500.0)
    po_eod = _rule_widget("Max Loss Limit (EOD, trailing $)", defaults["payout_max_loss_eod"],
                          "pf_po_eod", step=500.0)
    po_daily = _rule_widget("Daily Loss Limit ($)", defaults["payout_daily_loss"],
                            "pf_po_daily", step=250.0)
    po_cons = _rule_widget("Consistency Rule", defaults["payout_consistency"],
                           "pf_po_cons", pct=True)
    po_climit = _rule_widget("Contract Limit", defaults["payout_contract_limit"],
                             "pf_po_climit", step=1.0)
    max_withdrawal = _rule_widget("Maximum Withdrawal ($/payout)", defaults["maximum_withdrawal"],
                                  "pf_po_mw", step=500.0,
                                  help="Caps the dollar amount per payout: realized = min(profit, this).")

    apply_costs = st.checkbox("Apply commissions & slippage", value=True, key="pf_apply_costs")
    if apply_costs:
        slippage_n = st.slider("Slippage (ticks/side)", min_value=1, max_value=5,
                               value=1, step=1, key="pf_slip",
                               help="Entry-side ticks slipped per trade; market exits (losers) slip 2×.")
    else:
        slippage_n = 1

    st.write("")
    if st.button("Run Prop-Firm Simulation", type="primary"):
        try:
            dollars_per_tick = _get_dollars_per_tick(trade_filename)
        except ValueError as e:
            st.error(str(e)); return
        try:
            trades = pd.read_parquet(trades_path)
        except Exception as e:
            st.error(f"Could not load trades: {e}"); return

        full_comm, micro_comm = _get_commission_info(trade_filename)
        if apply_costs and full_comm is None:
            st.warning(
                f"No commission rate for asset '{trade_filename.split('_')[0]}' "
                "— commissions billed at 0; slippage still applies."
            )
        cost_ctx = {
            "enabled":    apply_costs,
            "n":          slippage_n,
            "full_comm":  full_comm,
            "micro_comm": micro_comm,
            "microable":  micro_comm is not None,
        }
        final_sizer_params = {**sizer_params, "dollars_per_tick": dollars_per_tick}
        params = {
            "n_paths": int(n_paths), "max_trades": int(max_trades), "seed": int(seed),
            "account_size": account_size,
            "increment": sizer_params.get("contract_increment", 1.0),
            "cost_ctx": cost_ctx,
            "profit_target": profit_target,
            "challenge_max_loss_eod": ch_eod, "challenge_daily_loss": ch_daily,
            "challenge_consistency": ch_cons, "challenge_contract_limit": ch_climit,
            "targeted_payout": targeted_payout,
            "payout_max_loss_eod": po_eod, "payout_daily_loss": po_daily,
            "payout_consistency": po_cons, "payout_contract_limit": po_climit,
            "maximum_withdrawal": max_withdrawal,
        }
        with st.spinner(f"Running prop-firm MC — {int(n_paths):,} paths × 2 sims..."):
            try:
                results = mc_module.run(trades=trades, sizer_module=sizer_module,
                                        sizer_params=final_sizer_params, params=params)
            except Exception as e:
                st.error(f"Simulation error: {e}"); return

        st.session_state.pf_results = results
        st.session_state.pf_costs   = apply_costs

    if st.session_state.get("pf_results") is None:
        return

    results  = st.session_state.pf_results
    costs_on = st.session_state.get("pf_costs", False)
    for w in results.get("warnings", []):
        st.warning(w)

    _render_prop_sim(results["sim1"], results["account_size"], costs_on)
    _render_prop_sim(results["sim2"], results["account_size"], costs_on)
    _render_combined_sim(results["sim3"], results["account_size"], costs_on)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    # Initialize session state keys on first load
    for key in ("mc_results", "mc_result_account", "mc_result_ruin"):
        if key not in st.session_state:
            st.session_state[key] = None

    if st.button("← Back"):
        go_page("home")

    st.title("Monte Carlo Simulation")
    st.write("")

    # -------------------------------------------------------------------------
    # 1. Load trades
    # -------------------------------------------------------------------------
    st.subheader("Trades")
    trade_files = _get_trade_files()
    if not trade_files:
        st.error("No trade files found in data/trades/")
        return

    selected_trade_file = st.selectbox("Trade file", trade_files, key="mc_trade_file")
    trades_path         = Path("data/trades") / f"{selected_trade_file}.parquet"

    # -------------------------------------------------------------------------
    # 2. Position sizing
    # -------------------------------------------------------------------------
    st.subheader("Position Sizing")

    sizer_plugins = _get_plugins("position_sizing")
    if not sizer_plugins:
        st.error("No sizer scripts found in position_sizing/")
        return

    col1, col2 = st.columns(2)
    with col1:
        account_size = st.number_input("Account size ($)", value=100_000.0,
                                       step=1000.0, format="%.2f", key="mc_account_size")
    with col2:
        sizer_name   = st.selectbox("Sizer", list(sizer_plugins.keys()), key="mc_sizer")
        sizer_module = _load_module(sizer_plugins[sizer_name])

    sizer_params_raw = getattr(sizer_module, "PARAMS", {})
    sizer_ui_params  = {k: v for k, v in sizer_params_raw.items()
                        if k not in ("account_size", "dollars_per_tick")}

    if sizer_ui_params:
        with st.expander("Sizer parameters"):
            sizer_specific = _param_widgets(sizer_ui_params, key_prefix="mc_sizer_param")
    else:
        sizer_specific = {}

    sizer_params = {
        **sizer_params_raw,
        **sizer_specific,
        "account_size": account_size,
        # dollars_per_tick injected at run time from filename
    }

    # -------------------------------------------------------------------------
    # 3. Monte Carlo type
    # -------------------------------------------------------------------------
    st.subheader("Simulation")

    mc_plugins = _get_plugins("monte_carlo")
    if not mc_plugins:
        st.error("No Monte Carlo scripts found in monte_carlo/")
        return

    mc_name   = st.selectbox("Method", list(mc_plugins.keys()), key="mc_method")
    mc_module = _load_module(mc_plugins[mc_name])

    # Prop-firm methods get a dedicated UI (per-rule toggles) + three-chart output.
    if getattr(mc_module, "PROP_FIRM", False):
        _render_prop_firm(mc_module, mc_name, trades_path, selected_trade_file,
                          sizer_module, sizer_params, account_size)
        return

    col4, _ = st.columns(2)
    with col4:
        ruin_options = {
            "No threshold": None,
            "Ruin at 0% (account wiped)": 0.0,
            "Ruin at 50% loss":           0.5,
        }
        ruin_label     = st.selectbox("Ruin definition", list(ruin_options.keys()), key="mc_ruin")
        ruin_threshold = ruin_options[ruin_label]

    mc_params_raw = getattr(mc_module, "PARAMS", {})
    if mc_params_raw:
        with st.expander("Simulation parameters"):
            mc_specific = _param_widgets(mc_params_raw, key_prefix="mc_sim_param")
    else:
        mc_specific = {}

    mc_params = {**mc_params_raw, **mc_specific}

    # Costs — commissions + slippage applied to the simulated paths, net of which
    # equity (and thus ruin) is reported. Costs feed back into per-step sizing.
    apply_costs = st.checkbox("Apply commissions & slippage", value=True, key="mc_apply_costs")
    if apply_costs:
        slippage_n = st.slider(
            "Slippage (ticks/side)",
            min_value=1, max_value=5, value=1, step=1,
            key="mc_slippage_n",
            help="Entry-side ticks slipped per trade; market exits (losers) slip 2×.",
        )
    else:
        slippage_n = 1

    # -------------------------------------------------------------------------
    # 4. Run
    # -------------------------------------------------------------------------
    st.write("")
    if st.button("Run Simulation", type="primary"):
        try:
            dollars_per_tick = _get_dollars_per_tick(selected_trade_file)
        except ValueError as e:
            st.error(str(e))
            return

        try:
            trades = pd.read_parquet(trades_path)
        except Exception as e:
            st.error(f"Could not load trades: {e}")
            return

        final_sizer_params = {**sizer_params, "dollars_per_tick": dollars_per_tick}

        # Resolve cost constants from the filename's asset (page owns ASSET_INFO;
        # base/bootstrap stay asset-agnostic). cost_ctx rides into the engine via params.
        full_comm, micro_comm = _get_commission_info(selected_trade_file)
        if apply_costs and full_comm is None:
            st.warning(
                f"No commission rate for asset '{selected_trade_file.split('_')[0]}' "
                "— commissions billed at 0; slippage still applies."
            )
        cost_ctx = {
            "enabled":    apply_costs,
            "n":          slippage_n,
            "full_comm":  full_comm,
            "micro_comm": micro_comm,
            "microable":  micro_comm is not None,
        }
        run_params = {**mc_params, "cost_ctx": cost_ctx}

        with st.spinner(f"Running {mc_name} — {mc_params.get('n_paths', '?')} paths..."):
            try:
                results = mc_module.run(
                    trades       = trades,
                    sizer_module = sizer_module,
                    sizer_params = final_sizer_params,
                    params       = run_params,
                )
            except Exception as e:
                st.error(f"Simulation error: {e}")
                return

        for w in results.get("warnings", []):
            st.warning(w)

        st.session_state.mc_results        = results
        st.session_state.mc_result_account = account_size
        st.session_state.mc_result_ruin    = ruin_threshold
        st.session_state.mc_result_costs   = apply_costs

    # -------------------------------------------------------------------------
    # 5. Results
    # -------------------------------------------------------------------------
    if st.session_state.get("mc_results") is None:
        return

    results      = st.session_state.mc_results
    account_size = st.session_state.mc_result_account
    ruin_thresh  = st.session_state.mc_result_ruin

    equity_matrix = results["equity_matrix"]   # (n_paths, n_trades+1)

    st.write("")
    st.subheader("Equity Fan Chart")

    if st.session_state.get("mc_result_costs"):
        st.caption(
            "Equity is net of commissions & slippage. Note: slippage is applied "
            "post-hoc to recorded trades — a worse entry that would have prevented "
            "a take-profit fill is not modelled."
        )

    # Scale toggle — capped view avoids Kelly-style outliers dominating the axis
    if "mc_chart_capped" not in st.session_state:
        st.session_state.mc_chart_capped = True

    cap_label = "Show full equity curve" if st.session_state.mc_chart_capped else "Cap at 3× account"
    if st.button(cap_label, key="mc_cap_toggle"):
        st.session_state.mc_chart_capped = not st.session_state.mc_chart_capped
        st.rerun()

    y_max = account_size * 3 if st.session_state.mc_chart_capped else None

    featured = _select_featured_paths(equity_matrix)
    metrics  = _compute_metrics(equity_matrix, account_size, ruin_thresh)
    fig      = _build_fan_chart(
        equity_matrix  = equity_matrix,
        account_size   = account_size,
        featured       = featured,
        ruin_threshold = ruin_thresh,
        y_max          = y_max,
        band_finals    = metrics["band_finals"],
    )
    st.plotly_chart(fig, width='stretch')

    st.write("")
    _render_metrics(metrics, account_size)