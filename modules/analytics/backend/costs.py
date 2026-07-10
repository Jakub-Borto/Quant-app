"""
Cost model for the Analytics module — commission + slippage, deducted in
dollar space AFTER the sizer.

Extracted verbatim from legacy_streamlit/views/analytics.py. Costs are
post-hoc dollar deductions on the sizer's gross `trade_pnl`. They never run
inside a sizer. The `contracts` column (1.3 = 1 full + 3 micro) drives both:
commission decomposes full vs micro (commission is per-contract, not
proportional); slippage is proportional so it scales with `contracts`
directly. Slippage is classified on the GROSS sign, before any deduction.

Asset lookups (dollars-per-tick, commissions) come from the consolidated
modules.common.backend.asset_info.
"""

import numpy as np
import pandas as pd

from modules.common.backend.asset_info import get_commission_info, get_dollars_per_tick

# Keys inside a sizer's PARAMS dict that are ALWAYS driven by the shared
# defaults at the top of the window. We filter these out of the per-instance
# UI so the user doesn't re-type the same account size for every instance.
# To allow per-instance overrides, simply empty this set.
SHARED_SIZER_KEYS: set[str] = {"account_size", "dollars_per_tick"}

# Defaults for the shared controls. Matching backtester conventions.
DEFAULT_ACCOUNT_SIZE = 100_000.0
DEFAULT_DOLLARS_PER_TICK = 12.50


def compute_cost_series(trades: pd.DataFrame, filename: str, n: int,
                        label: str = "") -> dict:
    """
    Return {"commission": Series, "slippage": Series, "warnings": [str]} aligned
    to `trades.index`. Skipped trades (contracts == 0) cost 0 automatically.
    """
    warnings: list[str] = []
    zero = pd.Series(0.0, index=trades.index)

    # Empty frame: legitimately zero cost — no warning.
    if trades.empty:
        return {"commission": zero, "slippage": zero.copy(), "warnings": warnings}

    asset = filename.split("_")[0]
    tag   = f"{label} ({asset})" if label else asset

    # Resolve the position-size column defensively: live sizers emit `contracts`,
    # the documented sizer contract says `size`. Accept either (prefer contracts).
    if "contracts" in trades.columns:
        size_col = "contracts"
    elif "size" in trades.columns:
        size_col = "size"
    else:
        size_col = None

    # A non-empty frame missing a required column is a real problem — and this is
    # the one cost failure that fails in the flattering direction (costs vanish,
    # net == gross). Warn loudly and name the columns present so it's diagnosable.
    missing = []
    if size_col is None:
        missing.append("contracts/size")
    if "trade_pnl" not in trades.columns:
        missing.append("trade_pnl")
    if missing:
        warnings.append(
            f"{tag}: cost path bailed out — missing required column(s) "
            f"{', '.join(missing)}; costs billed as $0. Columns present: "
            f"{list(trades.columns)}."
        )
        return {"commission": zero, "slippage": zero.copy(), "warnings": warnings}

    try:
        dpt = get_dollars_per_tick(filename)
    except ValueError as e:
        warnings.append(str(e))
        return {"commission": zero, "slippage": zero.copy(), "warnings": warnings}

    contracts = trades[size_col].astype(float)
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
