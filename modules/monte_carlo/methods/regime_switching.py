"""
monte_carlo/regime_switching.py

Regime-switching Monte Carlo — simulates equity by moving the market between
regimes through a transition matrix and drawing trade outcomes conditional on
the current regime.

Not "MCMC": Markov Chain Monte Carlo is a Bayesian sampling technique solving a
different problem. This is a Markov regime-switching simulation.

Why it beats the bootstrap: a bootstrap draws trades independently, so the runs
of losers that volatility clustering produces get scattered apart and drawdown
is systematically understated. Here the chain stays in high-vol for a realistic
stretch and keeps drawing high-vol trades while it does.

REGIME_PANEL = True routes this method to modules/monte_carlo/regime_panel.py
(the prop-firm precedent) — the regime source pickers and the transition-matrix
preview need more than a params form.

Unlike the bootstrap, this method needs the regime files as well as the trades.
It does NOT load them itself: the panel already loaded and estimated them to
render the pre-run preview, and passes that estimated model in via
params["model"] so the model displayed is exactly the model simulated. All the
real work lives in the pure, Qt-free, tested engine.
"""

from modules.monte_carlo.backend import regime_switching as engine

PARAMS = {
    "n_paths": 1000,
    "seed":    42,
}

# Routes to the dedicated panel instead of the generic params form.
REGIME_PANEL = True


def run(trades, sizer_module, sizer_params: dict, params: dict) -> dict:
    """
    Parameters
    ----------
    trades       : raw trades DataFrame as saved by the backtester
    sizer_module : loaded position sizing module (needs mc_prepare/mc_size)
    sizer_params : full sizer params incl. account_size, dollars_per_tick
    params       : n_paths, seed, cost_ctx, horizon, and `model` — the dict
                   from engine.estimate()

    Returns
    -------
    dict with equity_matrix (n_paths, horizon+1), n_trades (the DAY count —
    this matrix is indexed by trading day, not by trade), method, model and
    warnings.
    """
    merged = {**PARAMS, **params}
    model = merged.get("model")
    if not model:
        raise ValueError(
            "Regime-switching MC needs an estimated model. Load a regime run "
            "in the panel first — the transition matrix is built there so you "
            "can inspect it before simulating.")

    horizon = int(merged.get("horizon") or model["pool_days"])
    equity_matrix = engine.simulate(
        trades, sizer_module, sizer_params, model,
        horizon       = horizon,
        n_paths       = int(merged["n_paths"]),
        seed          = int(merged["seed"]),
        cost_ctx      = merged.get("cost_ctx"),
        start_state   = merged.get("start_state"),
    )
    return {
        "equity_matrix": equity_matrix,
        "n_trades":      horizon,
        "method":        "regime_switching",
        "model":         model,
        "warnings":      list(model.get("warnings", [])),
    }
