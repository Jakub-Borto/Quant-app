"""
VWAP Trend Trading — stop-and-reverse, always-in-market intraday trend follower.

After Zarattini & Aziz, "Volume Weighted Average Price (VWAP): The Holy Grail
for Day Trading Systems" (SSRN 4631351): hold long while 1-minute closes are
above VWAP, short while below, flip on the first close on the opposite side,
flatten at the end of the trading window. Generalized with a configurable
VWAP anchor (rth/globex, decoupled from the trading window), a neutral band
around VWAP (`vwap_band_ticks` + `band_rule`), an optional midday exclusion
window, and a zero-volume-bar signal filter. `vwap_band_ticks = 0` reproduces
the paper's exact rules.

Signals are evaluated on bar CLOSES; fills happen at the NEXT bar's open.
`trade_start_time` is the first bar eligible to OPEN a position — the first
signal comes from the close of the bar before it (the paper's "wait for the
9:30 candle to close, enter at 9:31" with the default 09:31).

⚠ COST CAVEAT (read before trusting results): this system trades ~15x/day.
No transaction costs are modelled (platform convention — pnl_points is a pure
price difference). On ES, commission + one tick of slippage is ≈ 1.3 ticks
per round-turn ≈ 20 ticks/day of drag, which is the same order as the edge
the paper reports. Raw results are materially optimistic and must not be
compared like-for-like against low-frequency strategies.

Engine contract (modules.engine; full reference in STRATEGY_GUIDE.md):
  DATA           main candles (open/close/volume) + the "indicators" slot
                 (vwap_bar_rth / vwap_bar_globex), joined per day
  prepare_day    -> data.DayCore (param-independent, cached by the engine)
  process_day    -> the day's Trades (engine.run_day on the trading window)
  PARAMS, PARAM_SECTIONS, PARAMS_OPTIONS
"""

import numpy as np
import pandas as pd

from modules.engine import timed

from .data import CANDLE_COLUMNS, INDICATOR_COLUMNS, build_day_core
from .engine import run_day
from .params import PARAM_SECTIONS, PARAMS, PARAMS_OPTIONS, validate

RTH_VWAP_ANCHOR_MIN = 9 * 60 + 30    # vwap_bar_rth is NaN before 09:30 NY

DATA = {
    "main":       CANDLE_COLUMNS,
    "indicators": INDICATOR_COLUMNS,
}

# validate() once per distinct params dict (process_day runs once per day)
_CFG_MEMO: dict = {}


def _config(params: dict) -> dict:
    key = tuple(sorted(params.items()))
    cfg = _CFG_MEMO.get(key)
    if cfg is None:
        cfg = validate(params)
        if cfg["anchor"] == "rth" and cfg["start_min"] < RTH_VWAP_ANCHOR_MIN:
            print(f"[vwap_trend] WARNING: vwap_anchor='rth' with trade_start_time "
                  f"{cfg['trade_start']} — vwap_bar_rth is NaN before 09:30 NY, so no "
                  f"signals (flat) until 09:30", flush=True)
        _CFG_MEMO[key] = cfg
    return cfg


def prepare_day(date, data: dict, params: dict):
    """Candles joined with both vwap columns -> DayCore (None = empty day)."""
    return build_day_core(data["main"], data["indicators"])


def process_day(day, params: dict) -> list:
    cfg = _config(params)
    core = day.prepared
    vwap = core.vwap[cfg["anchor_col"]]

    with timed("vwap:window"):
        rth_date = day.date
        tz = core.index.tz
        start_ns = pd.Timestamp(f"{rth_date} {cfg['trade_start']}", tz=tz).value
        end_ns   = pd.Timestamp(f"{rth_date} {cfg['trade_end']}",   tz=tz).value

        fill0 = int(core.i8.searchsorted(start_ns, side="left"))
        i1    = int(core.i8.searchsorted(end_ns,   side="right"))
        if fill0 >= i1:
            return []                       # no fill-eligible bars in the window
        sig0 = max(fill0 - 1, 0)            # the bar whose close is the first signal

        i8_w = core.i8[sig0:i1]
        excl = cfg["exclusion"]
        if excl is not None:
            e0 = pd.Timestamp(f"{rth_date} {excl[0]}", tz=tz).value
            e1 = pd.Timestamp(f"{rth_date} {excl[1]}", tz=tz).value
            excl_mask = (i8_w >= e0) & (i8_w < e1)
        else:
            excl_mask = np.zeros(i1 - sig0, dtype=bool)

    return run_day(core.index[sig0:i1], core.open[sig0:i1], core.close[sig0:i1],
                   core.volume[sig0:i1], vwap[sig0:i1], excl_mask, cfg)


__all__ = ["DATA", "PARAMS", "PARAM_SECTIONS", "PARAMS_OPTIONS",
           "prepare_day", "process_day"]
