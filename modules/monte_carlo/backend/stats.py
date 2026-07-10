"""
Monte-Carlo result statistics for the Monte Carlo module.

Extracted verbatim from legacy_streamlit/views/monte_carlo.py: the fan-chart
band constants, per-path drawdown/Sharpe, the cross-path summary metrics, the
featured-path selection and the metrics-table row builder. The fan chart
itself is a shared pyqtgraph widget (modules/common/ui/charts/fan_chart.py)
fed by these numbers.
"""

import numpy as np
import pandas as pd

# Max individual sample paths drawn on a fan chart.
SAMPLE_PATH_COUNT = 200

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


# ── Metrics computation ──────────────────────────────────────────────────────

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


# ── Featured path selection ──────────────────────────────────────────────────

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


# ── Metrics table rows (from the old _render_metrics) ───────────────────────

def metrics_table_rows(metrics: dict, account_size: float) -> pd.DataFrame:
    """The generic-MC metrics table as a DataFrame (formatting verbatim)."""

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

    return pd.DataFrame(rows, columns=["Metric", "Value", "vs Starting Capital"])
