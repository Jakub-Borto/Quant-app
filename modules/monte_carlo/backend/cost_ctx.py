"""
Cost-context assembly for the Monte Carlo module.

Extracted verbatim from the run-button handlers in legacy_streamlit/views/
monte_carlo.py. The MC window owns asset resolution (via the consolidated
asset_info); the simulation engines (methods/base.py, bootstrap, prop_firm)
stay asset-agnostic — cost_ctx rides into the engine via params.
"""

from modules.common.backend.asset_info import get_commission_info


def build_cost_ctx(trades_filename: str, apply_costs: bool, slippage_n: int) -> tuple[dict, bool]:
    """
    Returns (cost_ctx, missing_commission_warning_needed).

    cost_ctx keys: enabled / n / full_comm / micro_comm / microable —
    exactly the dict the old view assembled before calling mc_module.run().
    The warning flag is True when costs are on but the asset has no
    commission rate (the old view showed a st.warning; the window shows a
    banner).
    """
    full_comm, micro_comm = get_commission_info(trades_filename)
    warn_missing = apply_costs and full_comm is None
    cost_ctx = {
        "enabled":    apply_costs,
        "n":          slippage_n,
        "full_comm":  full_comm,
        "micro_comm": micro_comm,
        "microable":  micro_comm is not None,
    }
    return cost_ctx, warn_missing
