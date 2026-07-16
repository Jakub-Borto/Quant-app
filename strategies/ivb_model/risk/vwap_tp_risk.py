"""VWAP-target risk script: switchable SL + VWAP deviation-band TP (now or trailing).

Self-contained. The SL is chosen by `sl_placement` ("VAL/VAH", the zone-pullback logic
"zone_logic" — owned here as _zone_sl — or the pullback swing "swing_low" — owned here as
_swing_sl) and stays fixed for the whole trade. The TP is a tick-vwap
deviation band (±2σ/±3σ, globex or rth) read from the day context's band arrays, used either
frozen at entry ("now") or trailed bar-by-bar ("trailing"). Both fill simulators (_run_trade and
the trailing variant _run_trade_trailing) are owned here too — no shared module, no cross-script
imports.

INDICATORS REQUIRED: if the VWAP bands are unavailable for the day (no indicators / missing
columns / NaN at entry), this script returns None (no trade) — the whole trade is skipped because
the TP cannot be computed.

exit_reason stays tp / sl / eod (+ tp_timeout / sl_timeout). Which TP was chosen is recorded in
the trade notes via risk_notes: tp_type = "tp_vwap_2" | "tp_vwap_3" | "1:1".
"""

import numpy as np


def _zone_sl(entry_win, entry_pos, direction, levels):
    """Pick the stop from the pullback window's extremes vs the VAL/POC/VAH zones."""
    poc = levels["poc"]
    vah = levels["vah"]
    val = levels["val"]

    # --- SL window: drop the breakout bar (index 0), keep retest .. bar before entry ---
    # entry is taken at the entry bar's OPEN, so that bar's low/close are future data => excluded.
    m = entry_win.pos[1:] < entry_pos

    # degenerate: nothing to measure => fall back to the basic VAL/VAH stop
    if not m.any():
        return val if direction == "long" else vah

    if direction == "long":
        lowest_close = float(entry_win.c[1:][m].min())   # where the pullback bottomed (by close)
        lowest_low   = float(entry_win.l[1:][m].min())   # how far the wick reached

        if poc <= lowest_close <= vah:                 # bottomed in the UPPER zone
            return poc if lowest_low >= poc else lowest_low
        elif val <= lowest_close < poc:                # bottomed in the LOWER zone
            return val if lowest_low >= val else lowest_low
        else:                                          # shouldn't occur (pullback re-enters VA)
            return val

    else:
        highest_close = float(entry_win.c[1:][m].max())  # where the pullback topped (by close)
        highest_high  = float(entry_win.h[1:][m].max())  # how far the wick reached

        if val <= highest_close <= poc:                # topped in the LOWER zone
            return poc if highest_high <= poc else highest_high
        elif poc < highest_close <= vah:               # topped in the UPPER zone
            return vah if highest_high <= vah else highest_high
        else:                                          # shouldn't occur (pullback re-enters VA)
            return vah


def _swing_sl(entry_win, entry_pos, direction, levels):
    """Swing stop over the post_retest bars up to and including the entry bar
    (same rule as basic_risk's "swing_low"); VAL/VAH fallback when empty."""
    m = entry_win.pos <= entry_pos
    if not m.any():
        return levels["val"] if direction == "long" else levels["vah"]
    return float(entry_win.l[m].min()) if direction == "long" \
      else float(entry_win.h[m].max())


def _run_trade(trade_win, entry_ts, entry_price, direction, sl, tp, params) -> dict:
    """Simulate the trade from entry to exit. Returns the trade dict (no trade_type/notes)."""
    timeout = params["trade_timeout"]
    n_all   = trade_win.n
    t_end   = min(timeout, n_all)              # len(pre_timeout)
    low     = trade_win.l
    high    = trade_win.h
    index   = trade_win.index

    if direction == "long":
        sl_hit = low[:t_end]  <= sl
        tp_hit = high[:t_end] >= tp
    else:
        sl_hit = high[:t_end] >= sl
        tp_hit = low[:t_end]  <= tp

    sl_pos = int(sl_hit.argmax()) if sl_hit.any() else None
    tp_pos = int(tp_hit.argmax()) if tp_hit.any() else None

    def make_trade(exit_ts, exit_price, exit_reason, used_sl, used_tp):
        pnl = (exit_price - entry_price) if direction == "long" \
         else (entry_price - exit_price)
        return {
            "direction":   direction,
            "entry_time":  entry_ts,
            "exit_time":   exit_ts,
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "sl":          used_sl,
            "tp":          used_tp,
            "exit_reason": exit_reason,
            "pnl_points":  pnl,
        }

    if sl_pos is not None or tp_pos is not None:
        if sl_pos is None:
            return make_trade(index[tp_pos], tp, "tp", sl, tp)
        if tp_pos is None:
            return make_trade(index[sl_pos], sl, "sl", sl, tp)
        if tp_pos <= sl_pos:
            return make_trade(index[tp_pos], tp, "tp", sl, tp)
        else:
            return make_trade(index[sl_pos], sl, "sl", sl, tp)

    if n_all < timeout:
        return make_trade(index[-1], float(trade_win.c[-1]), "eod", sl, tp)

    timeout_close = float(trade_win.c[t_end - 1])
    in_profit     = (timeout_close > entry_price) if direction == "long" \
               else (timeout_close < entry_price)

    if in_profit:
        return make_trade(index[t_end - 1], timeout_close, "tp_timeout", sl, tp)

    new_tp = entry_price
    new_sl = float(low[:t_end].min())  if direction == "long" \
        else float(high[:t_end].max())

    if n_all == timeout:                       # post_timeout empty
        return make_trade(index[-1], float(trade_win.c[-1]), "eod", new_sl, new_tp)

    if direction == "long":
        sl_hit2 = low[timeout:]  <= new_sl
        tp_hit2 = high[timeout:] >= new_tp
    else:
        sl_hit2 = high[timeout:] >= new_sl
        tp_hit2 = low[timeout:]  <= new_tp

    sl_pos2 = int(sl_hit2.argmax()) if sl_hit2.any() else None
    tp_pos2 = int(tp_hit2.argmax()) if tp_hit2.any() else None

    if sl_pos2 is None and tp_pos2 is None:
        return make_trade(index[-1], float(trade_win.c[-1]), "eod", new_sl, new_tp)

    if sl_pos2 is None:
        return make_trade(index[timeout + tp_pos2], new_tp, "tp_timeout", new_sl, new_tp)
    if tp_pos2 is None:
        return make_trade(index[timeout + sl_pos2], new_sl, "sl_timeout", new_sl, new_tp)

    if tp_pos2 <= sl_pos2:
        return make_trade(index[timeout + tp_pos2], new_tp, "tp_timeout", new_sl, new_tp)
    else:
        return make_trade(index[timeout + sl_pos2], new_sl, "sl_timeout", new_sl, new_tp)


def _run_trade_trailing(trade_win, entry_ts, entry_price, direction, sl, band, params) -> dict:
    """Like _run_trade, but the TP trails a per-bar band array instead of a fixed price.

    SL is fixed. Same-bar race is PESSIMISTIC: if SL and the trailing TP both trigger on one bar,
    the SL wins (loss). Bars with a NaN band value can only trigger the SL. The timeout / EOD tail
    mirrors _run_trade exactly (breakeven TP + swing SL after timeout).
    """
    timeout = params["trade_timeout"]
    n_all   = trade_win.n
    t_end   = min(timeout, n_all)              # len(pre_timeout)
    low     = trade_win.l
    high    = trade_win.h
    index   = trade_win.index

    b     = band[:t_end]
    valid = ~np.isnan(b)

    if direction == "long":
        sl_hit = low[:t_end]  <= sl
        tp_hit = valid & (high[:t_end] >= b)
    else:
        sl_hit = high[:t_end] >= sl
        tp_hit = valid & (low[:t_end]  <= b)

    sl_pos = int(sl_hit.argmax()) if sl_hit.any() else None
    tp_pos = int(tp_hit.argmax()) if tp_hit.any() else None

    def make_trade(exit_ts, exit_price, exit_reason, used_sl, used_tp):
        pnl = (exit_price - entry_price) if direction == "long" \
         else (entry_price - exit_price)
        return {
            "direction":   direction,
            "entry_time":  entry_ts,
            "exit_time":   exit_ts,
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "sl":          used_sl,
            "tp":          used_tp,
            "exit_reason": exit_reason,
            "pnl_points":  pnl,
        }

    if sl_pos is not None or tp_pos is not None:
        # pessimistic: on a same-bar tie (sl_pos == tp_pos) the SL wins
        sl_first = sl_pos is not None and (tp_pos is None or sl_pos <= tp_pos)
        if sl_first:
            # the live trail level when stopped (informational; may be NaN)
            trail_at_sl = float(b[sl_pos])
            return make_trade(index[sl_pos], sl, "sl", sl, trail_at_sl)
        else:
            band_exit = float(b[tp_pos])
            return make_trade(index[tp_pos], band_exit, "tp", sl, band_exit)

    if n_all < timeout:
        # no clean band TP / SL by end of data: exit at last close. Record tp = the trailing band
        # value at that bar (where the vwap target sat), falling back to the exit price if NaN.
        exit_price = float(trade_win.c[-1])
        tp_at_exit = float(b[-1])
        if np.isnan(tp_at_exit):
            tp_at_exit = exit_price
        return make_trade(index[-1], exit_price, "eod", sl, tp_at_exit)

    # --- timeout tail: mirror _run_trade (breakeven TP + swing SL) ---
    timeout_close = float(trade_win.c[t_end - 1])
    in_profit     = (timeout_close > entry_price) if direction == "long" \
               else (timeout_close < entry_price)

    if in_profit:
        # trailing TP timed out in profit: record tp = the trailing band value at the timeout bar
        # (where the vwap target sat), falling back to the exit price if NaN.
        tp_at_exit = float(b[-1])
        if np.isnan(tp_at_exit):
            tp_at_exit = timeout_close
        return make_trade(index[t_end - 1], timeout_close, "tp_timeout", sl, tp_at_exit)

    new_tp = entry_price
    new_sl = float(low[:t_end].min())  if direction == "long" \
        else float(high[:t_end].max())

    if n_all == timeout:                       # post_timeout empty
        return make_trade(index[-1], float(trade_win.c[-1]), "eod", new_sl, new_tp)

    if direction == "long":
        sl_hit2 = low[timeout:]  <= new_sl
        tp_hit2 = high[timeout:] >= new_tp
    else:
        sl_hit2 = high[timeout:] >= new_sl
        tp_hit2 = low[timeout:]  <= new_tp

    sl_pos2 = int(sl_hit2.argmax()) if sl_hit2.any() else None
    tp_pos2 = int(tp_hit2.argmax()) if tp_hit2.any() else None

    if sl_pos2 is None and tp_pos2 is None:
        return make_trade(index[-1], float(trade_win.c[-1]), "eod", new_sl, new_tp)

    if sl_pos2 is None:
        return make_trade(index[timeout + tp_pos2], new_tp, "tp_timeout", new_sl, new_tp)
    if tp_pos2 is None:
        return make_trade(index[timeout + sl_pos2], new_sl, "sl_timeout", new_sl, new_tp)

    if tp_pos2 <= sl_pos2:
        return make_trade(index[timeout + tp_pos2], new_tp, "tp_timeout", new_sl, new_tp)
    else:
        return make_trade(index[timeout + sl_pos2], new_sl, "sl_timeout", new_sl, new_tp)


def run(entry_win, trade_win, entry_pos, entry_price, direction, levels, params):
    """Returns the standard trade dict (with risk_notes), or None."""
    # --- indicators required: no VWAP bands => no trade ---
    vwap_bands = trade_win.day.vwap_bands
    if vwap_bands is None:
        return None

    # --- SL placement (fixed for the whole trade) ---
    placement = params["sl_placement"]
    if placement == "VAL/VAH":
        sl = levels["val"] if direction == "long" else levels["vah"]
    elif placement == "swing_low":
        sl = _swing_sl(entry_win, entry_pos, direction, levels)
    else:   # "zone_logic" — the default; unknown / legacy values fall back here
        sl = _zone_sl(entry_win, entry_pos, direction, levels)

    risk = abs(entry_price - sl)
    if risk <= 0:
        return None

    # --- band selection: pick the σ2 and σ3 columns for this session + direction ---
    session = params["vwap_session"]
    ud      = "up" if direction == "long" else "dn"
    col2    = f"vwap_tick_{session}_std2_{ud}"
    col3    = f"vwap_tick_{session}_std3_{ud}"

    if col2 not in vwap_bands or col3 not in vwap_bands:
        return None

    band2_e = vwap_bands[col2][entry_pos]
    band3_e = vwap_bands[col3][entry_pos]
    if np.isnan(band2_e) or np.isnan(band3_e):
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

    entry_ts = trade_win.day.index[entry_pos]

    # --- TP + execution ---
    if fallback:
        tp = entry_price + risk if direction == "long" else entry_price - risk
        trade   = _run_trade(trade_win, entry_ts, entry_price, direction, sl, tp, params)
        tp_type = "1:1"
    else:
        col_eff = col3 if eff_std == 3 else col2
        tp_type = f"tp_vwap_{eff_std}"

        if params["vwap_tp_mode"] == "trailing":
            band  = vwap_bands[col_eff][entry_pos:]
            trade = _run_trade_trailing(trade_win, entry_ts, entry_price, direction,
                                        sl, band, params)
        else:  # "now": freeze the band value at entry
            tp    = float(vwap_bands[col_eff][entry_pos])
            trade = _run_trade(trade_win, entry_ts, entry_price, direction, sl, tp, params)

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
