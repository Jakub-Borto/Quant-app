"""
IVB Model — Initial Balance breakout with order-flow confirmation (package).

Engine contract (modules.engine; the full reference is STRATEGY_GUIDE.md):
  DATA            main candles incl. the enriched tick_volume / passive_orders JSON
                  columns + the "indicators" slot (CVD + the 8 tick-vwap bands)
  PREPARE_PARAMS  ["session_start"] — the day core is anchored at the session start
  prepare_day     -> core.build_day_core(...) : DayData (param-independent, cached)
  process_day     -> core.process_day(...) as a Trade (or None)
  PARAMS, PARAM_SECTIONS, PARAMS_OPTIONS (assembled in params.py from core +
  every finder + every risk script)

Internal layout:
  params.py      core params + assembly of the finders' / risk scripts' own params
  _daydata.py    per-day numpy context: parsed JSON + positional windows/masks
  profile.py     compute_ivb_profile
  baselines.py   rolling / passive / cvd-change day-level baselines
  absorption.py  shared absorption level scan on pre-parsed tick_volume
  entries/       one module per entry type (7): find_entry + its own params
  risk/          self-contained risk scripts (run + their own params), RISK_SCRIPTS
  core.py        breakout/retest detection, entry dispatcher, process_day

The cached DayData is reused across runs and optimizer combos; core.process_day
resets its per-run state first (day.reset_run_state()), so runs never leak into
each other.
"""

from modules.engine import Trade

from . import core
from .core import VWAP_BAND_COLUMNS, build_day_core, session_start_minutes
from .params import PARAM_SECTIONS, PARAMS, PARAMS_OPTIONS

# only these candle columns are consumed (`volume` / `volume_delta` are not)
CANDLE_COLUMNS = [
    "open", "high", "low", "close",
    "buy_volume", "sell_volume", "volume_delta_pct",
    "tick_volume", "passive_orders",
]
# indicators: CVD + the 8 tick-vwap ±2σ/±3σ band columns (of ~32 in the file)
INDICATOR_COLUMNS = ["cumulative_delta"] + VWAP_BAND_COLUMNS

DATA = {
    "main":       CANDLE_COLUMNS,
    "indicators": INDICATOR_COLUMNS,
}

# the RTH slice (and with it the whole day core) is anchored at session_start
PREPARE_PARAMS = ["session_start"]


def prepare_day(date, data: dict, params: dict):
    """One day's candles + indicators -> DayData (None = unusable day: empty
    or tz-naive — the engine then skips it)."""
    return build_day_core(data["main"], data["indicators"],
                          session_start_minutes(params))


def process_day(day, params: dict):
    trade = core.process_day(day.prepared, params)
    if trade is None:
        return None
    return Trade(
        direction=trade["direction"],
        entry_time=trade["entry_time"],
        exit_time=trade["exit_time"],
        entry_price=trade["entry_price"],
        exit_price=trade["exit_price"],
        exit_reason=trade["exit_reason"],
        sl=trade["sl"],
        tp=trade["tp"],
        trade_type=trade["trade_type"],
        notes=trade["notes"],
    )


__all__ = ["DATA", "PREPARE_PARAMS", "PARAMS", "PARAM_SECTIONS", "PARAMS_OPTIONS",
           "prepare_day", "process_day"]
