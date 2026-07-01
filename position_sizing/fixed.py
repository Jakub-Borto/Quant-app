import numpy as np
import pandas as pd

PARAMS = {
    "contracts": 1.0,  # float => decimal input; allows mini contracts (e.g. 1.5)
    "account_size": 100000.0,
    "dollars_per_tick": 12.50,
}

def apply(trades: pd.DataFrame, params: dict) -> pd.DataFrame:
    trades = trades.copy()

    size = round(params["contracts"], 1)  # carry one decimal (mini contracts)
    account_size = params["account_size"]
    dollars_per_tick = params["dollars_per_tick"]

    trades["trade_pnl"] = trades["ticks"] * dollars_per_tick * size
    trades["equity"] = account_size + trades["trade_pnl"].cumsum()
    trades["contracts"] = size  # per-trade size (broadcast); shown on hover

    return trades


# ---------------------------------------------------------------------------
# Vectorized Monte Carlo hooks (Option B).
# When both mc_prepare and mc_size are present, the MC engine sizes all paths
# at once per trade step instead of calling apply() once per path. Fixed
# sizing is position-independent, so size is a constant for every path/step.
# ---------------------------------------------------------------------------

def mc_prepare(trades: pd.DataFrame, params: dict) -> dict:
    """Path-invariant state for the vectorized MC engine. See base.run_paths_vectorized."""
    return {
        "size_const": round(params["contracts"], 1),  # carry one decimal (mini contracts)
        "per_trade": {},
    }


def mc_size(equity: np.ndarray, step: dict, state: dict, params: dict) -> np.ndarray:
    """Constant size for every path (ignores equity); clamped >= 0."""
    size = np.full(equity.shape, state["size_const"], dtype=float)
    return np.maximum(size, 0.0)