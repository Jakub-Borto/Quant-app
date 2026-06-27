"""Basic risk script: VAL/VAH (or swing) stop + fixed RR target.

risk_script: 1. Thin wrapper over the shared compute_sl_tp / run_trade in sl_tp.py — the level
data now arrives via the `levels` dict. Behaviour is identical to the original direct call.
"""

from .sl_tp import compute_sl_tp, run_trade


def run(post_retest, post_entry, entry_ts, entry_price, direction, levels, params):
    """Returns the standard trade dict, or None if risk is non-positive."""
    sl, tp = compute_sl_tp(
        post_retest = post_retest,
        entry_ts    = entry_ts,
        entry_price = entry_price,
        direction   = direction,
        val         = levels["val"],
        vah         = levels["vah"],
        params      = params,
    )

    if sl is None:
        return None

    return run_trade(
        post_entry  = post_entry,
        entry_ts    = entry_ts,
        entry_price = entry_price,
        direction   = direction,
        sl          = sl,
        tp          = tp,
        params      = params,
    )
