"""Stop / target / timeout / breakeven exit simulation -> Trade."""

import pandas as pd

from modules.engine import Trade

from .signals import first_true


def simulate(core, entry_pos: int, direction: str, sl: float, tp: float,
             timeout_minutes: int) -> Trade:
    """Walk the bars from the entry bar to the end of the day:

    - before the timeout bar: SL / TP (SL is checked first, so it wins a
      same-bar tie);
    - at the first bar at/after entry + timeout: exit at its close if in profit
      ("timeout_profit"); otherwise move TP to breakeven and keep running (the
      timeout bar itself is re-checked against the new TP);
    - still open at the last bar: exit at its close ("eod").
    """
    entry_price = core.open[entry_pos]
    entry_time  = core.index[entry_pos]
    th = core.high[entry_pos:]
    tl = core.low[entry_pos:]
    tc = core.close[entry_pos:]
    ti8  = core.i8[entry_pos:]
    tidx = core.index[entry_pos:]
    m = len(core.i8) - entry_pos

    if direction == "long":
        sl_hit = tl <= sl
        tp_hit = th >= tp
    else:
        sl_hit = th >= sl
        tp_hit = tl <= tp

    timeout_ns = ti8[0] + pd.Timedelta(minutes=timeout_minutes).value
    tpos = int(ti8.searchsorted(timeout_ns, side="left"))

    f_sl = first_true(sl_hit[:tpos])
    f_tp = first_true(tp_hit[:tpos])
    if f_sl != -1 and (f_tp == -1 or f_sl <= f_tp):
        exit_price, exit_time, exit_reason = sl, tidx[f_sl], "sl"
    elif f_tp != -1:
        exit_price, exit_time, exit_reason = tp, tidx[f_tp], "tp"
    elif tpos < m:
        in_profit = tc[tpos] > entry_price if direction == "long" else tc[tpos] < entry_price
        if in_profit:
            exit_price, exit_time, exit_reason = tc[tpos], tidx[tpos], "timeout_profit"
        else:
            tp = entry_price
            if direction == "long":
                be_hit = th[tpos:] >= tp
            else:
                be_hit = tl[tpos:] <= tp
            f_sl2 = first_true(sl_hit[tpos:])
            f_tp2 = first_true(be_hit)
            if f_sl2 != -1 and (f_tp2 == -1 or f_sl2 <= f_tp2):
                exit_price, exit_time, exit_reason = sl, tidx[tpos + f_sl2], "sl"
            elif f_tp2 != -1:
                exit_price, exit_time, exit_reason = tp, tidx[tpos + f_tp2], "tp"
            else:
                exit_price, exit_time, exit_reason = tc[m - 1], tidx[m - 1], "eod"
    else:
        exit_price, exit_time, exit_reason = tc[m - 1], tidx[m - 1], "eod"

    return Trade(direction=direction, entry_time=entry_time, exit_time=exit_time,
                 entry_price=entry_price, exit_price=exit_price,
                 exit_reason=exit_reason, sl=sl, tp=tp)
