"""VWAP-target risk script: switchable SL + VWAP deviation-band TP (now or trailing).

risk_script: 3. The SL is chosen by `sl_placement` (VAL/VAH or the zone_sl_risk logic) and stays
fixed for the whole trade. The TP is a tick-vwap deviation band (±2σ/±3σ, globex or rth) read from
the indicators parquet, used either frozen at entry ("now") or trailed bar-by-bar ("trailing").

INDICATORS REQUIRED: if the VWAP bands are unavailable for the day (no indicators / missing
columns / NaN at entry), this script returns None (no trade) — the whole trade is skipped because
the TP cannot be computed.

exit_reason stays tp / sl / eod (+ run_trade's tp_timeout / sl_timeout). Which TP was chosen is
recorded in the trade notes via risk_notes: tp_type = "tp_vwap_2" | "tp_vwap_3" | "1:1".
"""

import pandas as pd

from .sl_tp        import run_trade, run_trade_trailing
from .zone_sl_risk import _zone_sl


def run(post_retest, post_entry, entry_ts, entry_price, direction, levels, params):
    """Returns the standard trade dict (with risk_notes), or None."""
    # --- indicators required: no VWAP bands => no trade ---
    vwap_bands = levels.get("vwap_bands")
    if vwap_bands is None:
        return None

    # --- SL placement (fixed for the whole trade) ---
    if params["sl_placement"] == 1:
        sl = levels["val"] if direction == "long" else levels["vah"]
    else:
        sl = _zone_sl(post_retest, entry_ts, direction, levels)

    risk = abs(entry_price - sl)
    if risk <= 0:
        return None

    # --- band selection: pick the σ2 and σ3 columns for this session + direction ---
    session = params["vwap_session"]
    ud      = "up" if direction == "long" else "dn"
    col2    = f"vwap_tick_{session}_std2_{ud}"
    col3    = f"vwap_tick_{session}_std3_{ud}"

    if col2 not in vwap_bands.columns or col3 not in vwap_bands.columns:
        return None

    band2_e = vwap_bands[col2].get(entry_ts, float("nan"))
    band3_e = vwap_bands[col3].get(entry_ts, float("nan"))
    if pd.isna(band2_e) or pd.isna(band3_e):
        return None

    # --- entry-time escalation: if price already sits past the target band ---
    if direction == "long":
        past2 = entry_price >= band2_e
        past3 = entry_price >= band3_e
    else:
        past2 = entry_price <= band2_e
        past3 = entry_price <= band3_e

    eff_std  = params["vwap_std"]
    fallback = False
    if eff_std == 2:
        if past3:
            fallback = True       # past 3σ too => plain 1:1
        elif past2:
            eff_std = 3           # past 2σ only => bump target to 3σ
    else:  # eff_std == 3
        if past3:
            fallback = True

    escalated = fallback or (eff_std != params["vwap_std"])

    # --- TP + execution ---
    if fallback:
        tp = entry_price + risk if direction == "long" else entry_price - risk
        trade   = run_trade(post_entry, entry_ts, entry_price, direction, sl, tp, params)
        tp_type = "1:1"
    else:
        col_eff = col3 if eff_std == 3 else col2
        tp_type = f"tp_vwap_{eff_std}"

        if params["vwap_tp_mode"] == "trailing":
            trade = run_trade_trailing(post_entry, entry_ts, entry_price, direction,
                                       sl, vwap_bands[col_eff], params)
        else:  # "now": freeze the band value at entry
            tp    = float(vwap_bands[col_eff].get(entry_ts, float("nan")))
            trade = run_trade(post_entry, entry_ts, entry_price, direction, sl, tp, params)

    if trade is None:
        return None

    trade["risk_notes"] = {
        "tp_type":   tp_type,
        "escalated": bool(escalated),
    }

    return trade


'''
- if the poc is to close to vah/val then set the sl somewhere else
- if we are at the other side of vwap maybe target the 2nd std 2025-04-29
'''
