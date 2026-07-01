# position_sizing/risk_based.py
import math
import numpy as np
import pandas as pd

PARAMS = {
    "risk_pct": 0.01,
    "contract_increment": 0.1,  # 1.0 = whole contracts; set 0.1 for mini contracts
    "account_size": 100000.0,
    "dollars_per_tick": 12.50,
}

def apply(trades: pd.DataFrame, params: dict) -> pd.DataFrame:
    trades = trades.copy()

    risk_pct = params["risk_pct"]
    increment = params["contract_increment"]
    account_size = params["account_size"]
    dollars_per_tick = params["dollars_per_tick"]

    # ticks_per_point is an instrument constant — derive it once from the file
    # rather than requiring it as a param, since it's already encoded in ticks/pnl_points
    nonzero = trades[trades["pnl_points"] != 0]
    if nonzero.empty:
        trades["trade_pnl"] = 0.0
        trades["equity"] = account_size
        trades["contracts"] = 0.0
        return trades
    ticks_per_point = (nonzero["ticks"] / nonzero["pnl_points"]).iloc[0]

    trade_pnl_list = []
    size_list = []
    skipped = 0
    equity = account_size

    for _, trade in trades.iterrows():
        sl_ticks = abs(trade["entry_price"] - trade["sl"]) * ticks_per_point
        sl_dollars = sl_ticks * dollars_per_tick

        if sl_dollars > 0:
            raw = (equity * risk_pct) / sl_dollars
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
    trades["skipped"] = trades["trade_pnl"] == 0  # flag for UI warning

    # attach skipped count as metadata for the analytics UI to surface
    trades.attrs["skipped_trades"] = skipped

    return trades


# ---------------------------------------------------------------------------
# Vectorized Monte Carlo hooks (Option B).
# Per-trade stop distance (sl_ticks) is path-invariant — it depends only on the
# original trade's entry_price/sl — so it's precomputed once and gathered per
# step by the engine. Size still depends on per-path equity, computed live.
# ---------------------------------------------------------------------------

def mc_prepare(trades: pd.DataFrame, params: dict) -> dict:
    """Path-invariant risk-based state from the original trades. See base.run_paths_vectorized."""
    increment        = params["contract_increment"]
    dollars_per_tick = params["dollars_per_tick"]

    nonzero = trades[trades["pnl_points"] != 0]
    if nonzero.empty:
        # no usable trades -> all-zero sizes (sl_dollars <= 0 everywhere)
        sl_ticks_arr = np.zeros(len(trades), dtype=float)
    else:
        ticks_per_point = (nonzero["ticks"] / nonzero["pnl_points"]).iloc[0]
        sl_ticks_arr = (
            (trades["entry_price"] - trades["sl"]).abs() * ticks_per_point
        ).to_numpy(dtype=float)

    return {
        "risk_pct": params["risk_pct"],
        "increment": increment,
        "dollars_per_tick": dollars_per_tick,
        "per_trade": {"sl_ticks": sl_ticks_arr},
    }


def mc_size(equity: np.ndarray, step: dict, state: dict, params: dict) -> np.ndarray:
    """Risk-% size per path from current equity and drawn stop distance; clamped >= 0."""
    inc = state["increment"]
    sl_dollars = step["sl_ticks"] * state["dollars_per_tick"]

    # Guard divide-by-zero: where sl_dollars <= 0 the size is 0.
    safe = sl_dollars > 0
    raw = np.where(safe, (equity * state["risk_pct"]) / np.where(safe, sl_dollars, 1.0), 0.0)
    size = np.round(np.floor(raw / inc) * inc, 1)
    size = np.where(safe, size, 0.0)
    return np.maximum(size, 0.0)