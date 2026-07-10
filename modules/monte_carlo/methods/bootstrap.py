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
from base import run_paths, run_paths_vectorized, _step_commission


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

    cost_ctx = params.get("cost_ctx")
    warnings = []

    # Draw matrix once: (n_paths, n_sample), sampled WITH replacement.
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(trades), size=(n_paths, n_sample))

    has_hook = hasattr(sizer_module, "mc_prepare") and hasattr(sizer_module, "mc_size")

    if has_hook:
        # Fast vectorized path — costs (if any) feed back into per-step sizing.
        dollars_per_tick = sizer_params["dollars_per_tick"]
        equity_matrix = run_paths_vectorized(
            trades           = trades,
            sizer_module     = sizer_module,
            sizer_params     = sizer_params,
            idx              = idx,
            dollars_per_tick = dollars_per_tick,
            cost_ctx         = cost_ctx,
        )
    else:
        # Slow fallback — sizer lacks the vectorized hook.
        costs_on = bool(cost_ctx and cost_ctx.get("enabled"))
        if costs_on:
            # Best-effort: deduct costs post-hoc per path. Costs CANNOT feed back
            # into sizing here, so ruin probability is understated vs the fast path.
            warnings.append(
                "This sizer lacks the vectorized MC hook; costs are deducted "
                "post-hoc and do not feed back into sizing (ruin may be understated)."
            )
            equity_matrix = _run_fallback_with_costs(
                trades, sizer_module, sizer_params, idx,
                sizer_params["dollars_per_tick"], cost_ctx,
            )
        else:
            def _bootstrap_resample(trades, rng, _path):
                indices = rng.integers(0, len(trades), size=n_sample)
                return trades.iloc[indices].reset_index(drop=True)

            equity_matrix = run_paths(
                trades       = trades,
                sizer_module = sizer_module,
                sizer_params = sizer_params,
                resample_fn  = _bootstrap_resample,
                n_paths      = n_paths,
                seed         = seed,
            )

    result = {
        "equity_matrix": equity_matrix,
        "n_trades":      n_sample,
        "method":        "bootstrap",
    }
    if warnings:
        result["warnings"] = warnings
    return result


def _run_fallback_with_costs(
    trades, sizer_module, sizer_params, idx, dollars_per_tick, cost_ctx,
) -> np.ndarray:
    """
    Per-path fallback for hookless sizers when costs are on. Applies the sizer
    per path (slow), then deducts commissions + slippage POST-HOC from each
    trade's gross P&L. Costs do not feed back into sizing — best-effort only.
    """
    n_paths, n_sample = idx.shape
    account_size = sizer_params["account_size"]
    n = cost_ctx["n"]
    curves = []

    for i in range(n_paths):
        resampled = trades.iloc[idx[i]].reset_index(drop=True)
        sized     = sizer_module.apply(resampled, sizer_params)
        contracts = sized["contracts"].to_numpy(dtype=float)
        t         = resampled["ticks"].to_numpy(dtype=float)

        gross      = t * dollars_per_tick * contracts
        commission = _step_commission(contracts, cost_ctx)
        slip_ticks = np.where(t > 0, n, np.where(t < 0, 2 * n, n)).astype(float)
        slippage   = slip_ticks * dollars_per_tick * contracts
        net        = gross - commission - slippage

        equity = account_size + np.cumsum(net)
        curves.append(np.concatenate([[account_size], equity]))

    return np.vstack(curves)