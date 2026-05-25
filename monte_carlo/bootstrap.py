"""
monte_carlo/bootstrap.py

Bootstrap Monte Carlo — resamples trades WITH replacement.

Each path draws n_trades samples from the original trade set,
allowing the same trade to appear multiple times and some trades
to not appear at all. This models uncertainty in the edge estimate
rather than just testing sequence sensitivity.

Compared to reshuffling (which only reorders), bootstrap asks:
"What if my true edge is slightly different from what I observed?"
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Allow importing from monte_carlo/base.py regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))
from base import run_paths


PARAMS = {
    "n_paths":  1000,
    "n_trades": 0,     # 0 = use actual trade count from file
    "seed":     42,
}


def run(
    trades:       pd.DataFrame,
    sizer_module,
    sizer_params: dict,
    params:       dict,
) -> dict:
    """
    Run bootstrap Monte Carlo simulation.

    Parameters
    ----------
    trades       : raw trades DataFrame as saved by the backtester
    sizer_module : loaded position sizing module
    sizer_params : full sizer params dict including account_size, dollars_per_tick
    params       : simulation params (n_paths, n_trades, seed)

    Returns
    -------
    dict with:
        equity_matrix : np.ndarray of shape (n_paths, n_trades + 1)
        n_trades      : int
        method        : str
    """
    merged   = {**PARAMS, **params}
    n_paths  = int(merged["n_paths"])
    seed     = int(merged["seed"])
    # 0 means "match the file" — default behaviour unchanged
    n_sample = int(merged["n_trades"]) or len(trades)

    if len(trades) == 0:
        raise ValueError("Trade file is empty.")
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1.")
    if n_sample < 1:
        raise ValueError("n_trades must be at least 1.")

    def _bootstrap_resample(
        trades: pd.DataFrame,
        rng:    np.random.Generator,
        _path:  int,
    ) -> pd.DataFrame:
        """
        Sample n_sample rows WITH replacement.
        n_sample can be larger or smaller than len(trades).
        Index is reset so the sizer sees a clean sequential index.
        """
        indices   = rng.integers(0, len(trades), size=n_sample)
        resampled = trades.iloc[indices].reset_index(drop=True)
        return resampled

    equity_matrix = run_paths(
        trades       = trades,
        sizer_module = sizer_module,
        sizer_params = sizer_params,
        resample_fn  = _bootstrap_resample,
        n_paths      = n_paths,
        seed         = seed,
    )

    return {
        "equity_matrix": equity_matrix,
        "n_trades":      n_sample,
        "method":        "bootstrap",
    }