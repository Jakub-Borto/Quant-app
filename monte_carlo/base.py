"""
monte_carlo/base.py

Shared utilities for Monte Carlo simulation scripts.
Import from here to avoid duplicating logic across mc types.
"""

import numpy as np
import pandas as pd


def build_equity_curve(
    trades:       pd.DataFrame,
    sizer_module,
    sizer_params: dict,
) -> np.ndarray:
    """
    Apply sizer to a trades DataFrame and return the equity curve
    as a numpy array starting from account_size.

    Returns shape: (n_trades + 1,)  — index 0 is the starting balance.
    """
    sized       = sizer_module.apply(trades, sizer_params)
    account     = sizer_params["account_size"]
    equity      = sized["equity"].to_numpy(dtype=np.float64)

    # Prepend the starting balance so the curve starts at account_size
    return np.concatenate([[account], equity])


def run_paths(
    trades:       pd.DataFrame,
    sizer_module,
    sizer_params: dict,
    resample_fn,           # callable(trades, rng, path_index) -> pd.DataFrame
    n_paths:      int,
    seed:         int,
) -> np.ndarray:
    """
    Run n_paths simulations.

    resample_fn receives (trades, rng, path_index) and must return a
    resampled/reshuffled trades DataFrame with the same columns.

    Returns equity_matrix of shape (n_paths, n_trades + 1).
    """
    rng    = np.random.default_rng(seed)
    curves = []

    for i in range(n_paths):
        resampled = resample_fn(trades, rng, i)
        curve     = build_equity_curve(resampled, sizer_module, sizer_params)
        curves.append(curve)

    # Stack — all curves must be the same length (n_trades + 1)
    return np.vstack(curves)