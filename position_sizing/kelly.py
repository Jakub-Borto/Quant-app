# position_sizing/kelly.py
import math
import numpy as np
import pandas as pd

PARAMS = {
    "fraction": 0.25,
    "contract_increment": 0.1,  # 1.0 = whole contracts; set 0.1 for mini contracts
    "account_size": 100000.0,
    "dollars_per_tick": 12.50,
}

def apply(trades: pd.DataFrame, params: dict) -> pd.DataFrame:
    trades = trades.copy()

    fraction = params["fraction"]
    increment = params["contract_increment"]
    account_size = params["account_size"]
    dollars_per_tick = params["dollars_per_tick"]

    # Kelly computed from full trades file — in-sample by definition.
    # Represents optimal sizing if historical stats hold going forward.
    winning = trades[trades["ticks"] > 0]
    losing = trades[trades["ticks"] < 0]

    win_rate = len(winning) / len(trades)
    loss_rate = 1 - win_rate

    avg_win = winning["ticks"].mean() if len(winning) > 0 else 0
    avg_loss = abs(losing["ticks"].mean()) if len(losing) > 0 else 0

    if avg_loss == 0 or avg_win == 0:
        # degenerate case: all wins or all losses, sizing undefined
        trades["trade_pnl"] = 0.0
        trades["equity"] = account_size
        trades["contracts"] = 0.0
        trades.attrs["skipped_trades"] = len(trades)
        return trades

    win_loss_ratio = avg_win / avg_loss
    kelly_pct = win_rate - (loss_rate / win_loss_ratio)
    kelly_pct = max(kelly_pct, 0.0)  # never short the strategy
    kelly_pct *= fraction             # fractional Kelly

    risk_per_contract = avg_loss * dollars_per_tick

    trade_pnl_list = []
    size_list = []
    skipped = 0
    equity = account_size

    for _, trade in trades.iterrows():
        if risk_per_contract > 0:
            raw = (equity * kelly_pct) / risk_per_contract
            size = round(math.floor(raw / increment) * increment, 1)
        else:
            size = 0

        if size == 0:
            skipped += 1

        pnl = trade["ticks"] * dollars_per_tick * size
        trade_pnl_list.append(pnl)
        size_list.append(size)
        equity += pnl

    trades["trade_pnl"] = trade_pnl_list
    trades["equity"] = account_size + pd.Series(trade_pnl_list).cumsum().values
    trades["contracts"] = size_list  # per-trade size; shown on hover
    trades.attrs["skipped_trades"] = skipped

    return trades


# ---------------------------------------------------------------------------
# Vectorized Monte Carlo hooks (Option B).
#
# IMPORTANT — prepare-once semantics: win_rate / avg_win / avg_loss (and thus
# kelly_pct) are computed ONCE here from the ORIGINAL trades, not re-estimated
# per resampled path the way apply() implicitly does on each bootstrap sample.
# The observed full-file edge is the best estimate; the bootstrap should test
# draw/sequence luck, not let each path re-estimate its own edge. This makes MC
# results differ modestly from the per-path apply() path (more in the tails) —
# intended, not a bug.
# ---------------------------------------------------------------------------

def mc_prepare(trades: pd.DataFrame, params: dict) -> dict:
    """Path-invariant Kelly state from the original trades. See base.run_paths_vectorized."""
    fraction         = params["fraction"]
    increment        = params["contract_increment"]
    dollars_per_tick = params["dollars_per_tick"]

    winning = trades[trades["ticks"] > 0]
    losing  = trades[trades["ticks"] < 0]

    win_rate  = len(winning) / len(trades)
    loss_rate = 1 - win_rate

    avg_win  = winning["ticks"].mean() if len(winning) > 0 else 0
    avg_loss = abs(losing["ticks"].mean()) if len(losing) > 0 else 0

    if avg_loss == 0 or avg_win == 0:
        # degenerate: all wins or all losses, sizing undefined -> all-zero sizes
        return {"degenerate": True, "kelly_pct": 0.0, "risk_per_contract": 0.0,
                "increment": increment, "per_trade": {}}

    win_loss_ratio = avg_win / avg_loss
    kelly_pct = max(win_rate - (loss_rate / win_loss_ratio), 0.0) * fraction
    risk_per_contract = avg_loss * dollars_per_tick

    return {
        "degenerate": False,
        "kelly_pct": kelly_pct,
        "risk_per_contract": risk_per_contract,
        "increment": increment,
        "per_trade": {},
    }


def mc_size(equity: np.ndarray, step: dict, state: dict, params: dict) -> np.ndarray:
    """Kelly size per path from current equity; vectorized, clamped >= 0."""
    if state["degenerate"] or state["risk_per_contract"] <= 0:
        return np.zeros(equity.shape, dtype=float)

    inc = state["increment"]
    raw = (equity * state["kelly_pct"]) / state["risk_per_contract"]
    size = np.round(np.floor(raw / inc) * inc, 1)
    return np.maximum(size, 0.0)