"""VWAP-target risk script: switchable SL + VWAP deviation-band TP (now or trailing).

risk_script: 3. Self-contained. The SL is chosen by `sl_placement` (VAL/VAH or the zone-pullback
logic, owned here as _zone_sl) and stays fixed for the whole trade. The TP is a tick-vwap
deviation band (±2σ/±3σ, globex or rth) read from the indicators parquet, used either frozen at
entry ("now") or trailed bar-by-bar ("trailing"). Both fill simulators (_run_trade and the
trailing variant _run_trade_trailing) are owned here too — no shared module, no cross-script imports.

INDICATORS REQUIRED: if the VWAP bands are unavailable for the day (no indicators / missing
columns / NaN at entry), this script returns None (no trade) — the whole trade is skipped because
the TP cannot be computed.

exit_reason stays tp / sl / eod (+ tp_timeout / sl_timeout). Which TP was chosen is recorded in
the trade notes via risk_notes: tp_type = "tp_vwap_2" | "tp_vwap_3" | "1:1".
"""

import pandas as pd


def _zone_sl(post_retest, entry_ts, direction, levels):
    """Pick the stop from the pullback window's extremes vs the VAL/POC/VAH zones."""
    poc = levels["poc"]
    vah = levels["vah"]
    val = levels["val"]

    # --- SL window: drop the breakout bar (index 0), keep retest .. bar before entry ---
    # entry is taken at the entry bar's OPEN, so that bar's low/close are future data => excluded.
    window = post_retest.iloc[1:]
    window = window[window.index < entry_ts]

    # degenerate: nothing to measure => fall back to the basic VAL/VAH stop
    if window.empty:
        return val if direction == "long" else vah

    if direction == "long":
        lowest_close = float(window["close"].min())   # where the pullback bottomed (by close)
        lowest_low   = float(window["low"].min())      # how far the wick reached

        if poc <= lowest_close <= vah:                 # bottomed in the UPPER zone
            return poc if lowest_low >= poc else lowest_low
        elif val <= lowest_close < poc:                # bottomed in the LOWER zone
            return val if lowest_low >= val else lowest_low
        else:                                          # shouldn't occur (pullback re-enters VA)
            return val

    else:
        highest_close = float(window["close"].max())   # where the pullback topped (by close)
        highest_high  = float(window["high"].max())     # how far the wick reached

        if val <= highest_close <= poc:                # topped in the LOWER zone
            return poc if highest_high <= poc else highest_high
        elif poc < highest_close <= vah:               # topped in the UPPER zone
            return vah if highest_high <= vah else highest_high
        else:                                          # shouldn't occur (pullback re-enters VA)
            return vah


def _run_trade(
    post_entry:   pd.DataFrame,
    entry_ts:     pd.Timestamp,
    entry_price:  float,
    direction:    str,
    sl:           float,
    tp:           float,
    params:       dict,
) -> dict:
    """Simulate the trade from entry to exit. Returns the trade dict (no trade_type/notes)."""
    timeout     = params["trade_timeout"]
    pre_timeout = post_entry.iloc[:timeout]
    n           = len(pre_timeout)

    if direction == "long":
        sl_hit = pre_timeout["low"]  <= sl
        tp_hit = pre_timeout["high"] >= tp
    else:
        sl_hit = pre_timeout["high"] >= sl
        tp_hit = pre_timeout["low"]  <= tp

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
            return make_trade(pre_timeout.index[tp_pos], tp, "tp", sl, tp)
        if tp_pos is None:
            return make_trade(pre_timeout.index[sl_pos], sl, "sl", sl, tp)
        if tp_pos <= sl_pos:
            return make_trade(pre_timeout.index[tp_pos], tp, "tp", sl, tp)
        else:
            return make_trade(pre_timeout.index[sl_pos], sl, "sl", sl, tp)

    if n < timeout:
        exit_ts    = post_entry.index[-1]
        exit_price = float(post_entry.iloc[-1]["close"])
        return make_trade(exit_ts, exit_price, "eod", sl, tp)

    timeout_close = float(pre_timeout.iloc[-1]["close"])
    in_profit     = (timeout_close > entry_price) if direction == "long" \
               else (timeout_close < entry_price)

    if in_profit:
        return make_trade(pre_timeout.index[-1], timeout_close, "tp_timeout", sl, tp)

    new_tp = entry_price
    new_sl = float(pre_timeout["low"].min())  if direction == "long" \
        else float(pre_timeout["high"].max())

    post_timeout = post_entry.iloc[timeout:]

    if post_timeout.empty:
        exit_ts    = post_entry.index[-1]
        exit_price = float(post_entry.iloc[-1]["close"])
        return make_trade(exit_ts, exit_price, "eod", new_sl, new_tp)

    if direction == "long":
        sl_hit2 = post_timeout["low"]  <= new_sl
        tp_hit2 = post_timeout["high"] >= new_tp
    else:
        sl_hit2 = post_timeout["high"] >= new_sl
        tp_hit2 = post_timeout["low"]  <= new_tp

    sl_pos2 = int(sl_hit2.argmax()) if sl_hit2.any() else None
    tp_pos2 = int(tp_hit2.argmax()) if tp_hit2.any() else None

    if sl_pos2 is None and tp_pos2 is None:
        exit_ts    = post_entry.index[-1]
        exit_price = float(post_entry.iloc[-1]["close"])
        return make_trade(exit_ts, exit_price, "eod", new_sl, new_tp)

    if sl_pos2 is None:
        return make_trade(post_timeout.index[tp_pos2], new_tp, "tp_timeout", new_sl, new_tp)
    if tp_pos2 is None:
        return make_trade(post_timeout.index[sl_pos2], new_sl, "sl_timeout", new_sl, new_tp)

    if tp_pos2 <= sl_pos2:
        return make_trade(post_timeout.index[tp_pos2], new_tp, "tp_timeout", new_sl, new_tp)
    else:
        return make_trade(post_timeout.index[sl_pos2], new_sl, "sl_timeout", new_sl, new_tp)


def _run_trade_trailing(
    post_entry:   pd.DataFrame,
    entry_ts:     pd.Timestamp,
    entry_price:  float,
    direction:    str,
    sl:           float,
    band_series:  pd.Series,
    params:       dict,
) -> dict:
    """Like _run_trade, but the TP trails a per-bar band (band_series) instead of a fixed price.

    SL is fixed. Same-bar race is PESSIMISTIC: if SL and the trailing TP both trigger on one bar,
    the SL wins (loss). Bars with a NaN band value can only trigger the SL. The timeout / EOD tail
    mirrors _run_trade exactly (breakeven TP + swing SL after timeout).
    """
    timeout     = params["trade_timeout"]
    pre_timeout = post_entry.iloc[:timeout]
    n           = len(pre_timeout)

    band = band_series.reindex(pre_timeout.index)

    if direction == "long":
        sl_hit = pre_timeout["low"]  <= sl
        tp_hit = band.notna() & (pre_timeout["high"] >= band)
    else:
        sl_hit = pre_timeout["high"] >= sl
        tp_hit = band.notna() & (pre_timeout["low"]  <= band)

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
            trail_at_sl = float(band.iloc[sl_pos])
            return make_trade(pre_timeout.index[sl_pos], sl, "sl", sl, trail_at_sl)
        else:
            band_exit = float(band.iloc[tp_pos])
            return make_trade(pre_timeout.index[tp_pos], band_exit, "tp", sl, band_exit)

    if n < timeout:
        # no clean band TP / SL by end of data: exit at last close. Record tp = the trailing band
        # value at that bar (where the vwap target sat), falling back to the exit price if NaN.
        exit_ts    = post_entry.index[-1]
        exit_price = float(post_entry.iloc[-1]["close"])
        tp_at_exit = float(band.iloc[-1])
        if pd.isna(tp_at_exit):
            tp_at_exit = exit_price
        return make_trade(exit_ts, exit_price, "eod", sl, tp_at_exit)

    # --- timeout tail: mirror _run_trade (breakeven TP + swing SL) ---
    timeout_close = float(pre_timeout.iloc[-1]["close"])
    in_profit     = (timeout_close > entry_price) if direction == "long" \
               else (timeout_close < entry_price)

    if in_profit:
        # trailing TP timed out in profit: record tp = the trailing band value at the timeout bar
        # (where the vwap target sat), falling back to the exit price if NaN.
        tp_at_exit = float(band.iloc[-1])
        if pd.isna(tp_at_exit):
            tp_at_exit = timeout_close
        return make_trade(pre_timeout.index[-1], timeout_close, "tp_timeout", sl, tp_at_exit)

    new_tp = entry_price
    new_sl = float(pre_timeout["low"].min())  if direction == "long" \
        else float(pre_timeout["high"].max())

    post_timeout = post_entry.iloc[timeout:]

    if post_timeout.empty:
        exit_ts    = post_entry.index[-1]
        exit_price = float(post_entry.iloc[-1]["close"])
        return make_trade(exit_ts, exit_price, "eod", new_sl, new_tp)

    if direction == "long":
        sl_hit2 = post_timeout["low"]  <= new_sl
        tp_hit2 = post_timeout["high"] >= new_tp
    else:
        sl_hit2 = post_timeout["high"] >= new_sl
        tp_hit2 = post_timeout["low"]  <= new_tp

    sl_pos2 = int(sl_hit2.argmax()) if sl_hit2.any() else None
    tp_pos2 = int(tp_hit2.argmax()) if tp_hit2.any() else None

    if sl_pos2 is None and tp_pos2 is None:
        exit_ts    = post_entry.index[-1]
        exit_price = float(post_entry.iloc[-1]["close"])
        return make_trade(exit_ts, exit_price, "eod", new_sl, new_tp)

    if sl_pos2 is None:
        return make_trade(post_timeout.index[tp_pos2], new_tp, "tp_timeout", new_sl, new_tp)
    if tp_pos2 is None:
        return make_trade(post_timeout.index[sl_pos2], new_sl, "sl_timeout", new_sl, new_tp)

    if tp_pos2 <= sl_pos2:
        return make_trade(post_timeout.index[tp_pos2], new_tp, "tp_timeout", new_sl, new_tp)
    else:
        return make_trade(post_timeout.index[sl_pos2], new_sl, "sl_timeout", new_sl, new_tp)


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
        trade   = _run_trade(post_entry, entry_ts, entry_price, direction, sl, tp, params)
        tp_type = "1:1"
    else:
        col_eff = col3 if eff_std == 3 else col2
        tp_type = f"tp_vwap_{eff_std}"

        if params["vwap_tp_mode"] == "trailing":
            trade = _run_trade_trailing(post_entry, entry_ts, entry_price, direction,
                                        sl, vwap_bands[col_eff], params)
        else:  # "now": freeze the band value at entry
            tp    = float(vwap_bands[col_eff].get(entry_ts, float("nan")))
            trade = _run_trade(post_entry, entry_ts, entry_price, direction, sl, tp, params)

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
