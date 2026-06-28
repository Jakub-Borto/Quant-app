"""Zone-based stop script: SL from where the pullback bottomed/topped vs the value-area zones.

risk_script: 2. Self-contained. The SL is derived from the PULLBACK candles only (retest candle
.. bar before entry — no lookahead). Then a fixed RR (`zone_rr`) sets the target. The trade fill
simulation (_run_trade) is owned here too, so the ONLY difference from basic_risk is HOW the stop
is chosen.

Zones (value area): VAL = bottom, POC = middle, VAH = top.
  Upper zone = POC..VAH,  Lower zone = VAL..POC.
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


def run(post_retest, post_entry, entry_ts, entry_price, direction, levels, params):
    """Returns the standard trade dict, or None if risk is non-positive."""
    sl = _zone_sl(post_retest, entry_ts, direction, levels)

    risk = abs(entry_price - sl)
    if risk <= 0:
        return None

    rr = params["zone_rr"]
    tp = entry_price + risk * rr if direction == "long" else entry_price - risk * rr

    return _run_trade(
        post_entry  = post_entry,
        entry_ts    = entry_ts,
        entry_price = entry_price,
        direction   = direction,
        sl          = sl,
        tp          = tp,
        params      = params,
    )
