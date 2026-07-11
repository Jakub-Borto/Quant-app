"""Basic risk script: VAL/VAH (or swing) stop + fixed RR target.

risk_script: 1. Self-contained — owns its stop/target placement (_compute_sl_tp) and its trade
fill simulation (_run_trade). The level data arrives via the `levels` dict; the bar data via
the positional TradeWindow / EntryWindow contexts (numpy arrays, see _daydata).
"""


def _compute_sl_tp(entry_win, entry_pos, entry_price, direction, val, vah, params) -> tuple:
    """Returns (sl, tp). (None, None) if risk is non-positive."""
    if params["sl_type"] == 0:
        sl = val if direction == "long" else vah
    else:
        # swing stop from the post_retest bars up to and including the entry bar
        # (the original label-slice .loc[:entry_ts] was inclusive)
        m = entry_win.pos <= entry_pos
        if not m.any():
            sl = val if direction == "long" else vah
        else:
            sl = float(entry_win.l[m].min()) if direction == "long" \
            else float(entry_win.h[m].max())

    risk = abs(entry_price - sl)
    if risk <= 0:
        return None, None

    tp = entry_price + risk * params["rr"] if direction == "long" \
    else entry_price - risk * params["rr"]

    return sl, tp


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


def run(entry_win, trade_win, entry_pos, entry_price, direction, levels, params):
    """Returns the standard trade dict, or None if risk is non-positive."""
    sl, tp = _compute_sl_tp(
        entry_win   = entry_win,
        entry_pos   = entry_pos,
        entry_price = entry_price,
        direction   = direction,
        val         = levels["val"],
        vah         = levels["vah"],
        params      = params,
    )

    if sl is None:
        return None

    return _run_trade(
        trade_win   = trade_win,
        entry_ts    = trade_win.day.index[entry_pos],
        entry_price = entry_price,
        direction   = direction,
        sl          = sl,
        tp          = tp,
        params      = params,
    )
