"""Stop-loss / take-profit placement and the trade runner (fill simulation)."""

import pandas as pd


def compute_sl_tp(
    post_retest:  pd.DataFrame,
    entry_ts:     pd.Timestamp,
    entry_price:  float,
    direction:    str,
    val:          float,
    vah:          float,
    params:       dict,
) -> tuple:
    """Returns (sl, tp). (None, None) if risk is non-positive."""
    if params["sl_type"] == 0:
        sl = val if direction == "long" else vah
    else:
        bars_to_entry = post_retest.loc[:entry_ts]
        if bars_to_entry.empty:
            sl = val if direction == "long" else vah
        else:
            sl = float(bars_to_entry["low"].min())  if direction == "long" \
            else float(bars_to_entry["high"].max())

    risk = abs(entry_price - sl)
    if risk <= 0:
        return None, None

    tp = entry_price + risk * params["rr"] if direction == "long" \
    else entry_price - risk * params["rr"]

    return sl, tp


def run_trade(
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


def run_trade_trailing(
    post_entry:   pd.DataFrame,
    entry_ts:     pd.Timestamp,
    entry_price:  float,
    direction:    str,
    sl:           float,
    band_series:  pd.Series,
    params:       dict,
) -> dict:
    """Like run_trade, but the TP trails a per-bar band (band_series) instead of a fixed price.

    SL is fixed. Same-bar race is PESSIMISTIC: if SL and the trailing TP both trigger on one bar,
    the SL wins (loss). Bars with a NaN band value can only trigger the SL. The timeout / EOD tail
    mirrors run_trade exactly (breakeven TP + swing SL after timeout).
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

    # --- timeout tail: mirror run_trade (breakeven TP + swing SL) ---
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
