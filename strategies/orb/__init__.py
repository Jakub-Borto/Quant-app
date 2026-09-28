"""
ORB — Opening Range Breakout.

Each RTH day: the opening range is the high/low of the first `range_minutes`
from 09:30 NY. The first later bar that CLOSES beyond the range (with a body in
the breakout direction) is the signal; the entry is the NEXT bar's open. Stop =
range height x `sl_factor` from the entry, target = stop distance x `rr`. After
`timeout_minutes` a profitable trade exits at that bar's close and a losing one
moves its target to breakeven. Anything still open exits at the last RTH bar's
close. At most one trade per day.

Files:
  params.py    PARAMS + PARAM_SECTIONS
  day.py       prepare_day's DayCore (the RTH slice as arrays, cached by the engine)
  signals.py   opening range + first breakout
  exits.py     SL / TP / timeout / breakeven simulation -> Trade
"""

from .day import CANDLE_COLUMNS, build_day_core
from .exits import simulate
from .params import PARAM_SECTIONS, PARAMS
from .signals import first_breakout, opening_range

DATA = {"main": CANDLE_COLUMNS}


def prepare_day(date, data: dict, params: dict):
    """RTH slice of the day's candles as arrays (None = empty file)."""
    return build_day_core(data["main"], date)


def process_day(day, params: dict):
    core = day.prepared
    if len(core.i8) < 2:
        return None
    rng = opening_range(core, params["range_minutes"])
    if rng is None:
        return None
    split, orb_high, orb_low = rng
    sig = first_breakout(core, split, orb_high, orb_low)
    if sig is None:
        return None
    direction, rel = sig

    entry_pos   = split + rel + 1
    entry_price = core.open[entry_pos]
    sl_distance = (orb_high - orb_low) * params["sl_factor"]
    if direction == "long":
        sl = entry_price - sl_distance
        tp = entry_price + sl_distance * params["rr"]
    else:
        sl = entry_price + sl_distance
        tp = entry_price - sl_distance * params["rr"]
    return simulate(core, entry_pos, direction, sl, tp, params["timeout_minutes"])


__all__ = ["DATA", "PARAMS", "PARAM_SECTIONS", "prepare_day", "process_day"]
