"""
delta_pct_signal — does one-minute aggressive order-flow imbalance
(`volume_delta_pct`) predict the next few minutes, as momentum or reversal?
(spec: delta_pct_signal_spec.md v1). A measurement tool, not a finished
system: time exits only, no stop or target, one position at a time.

Rules, evaluated at the close of bar t (all counting by position in the file):
  1. Signal: volume_delta_pct >= +theta (buy side) or <= -theta (sell side),
     with bar t inside [entry_start, entry_end], volume >= 1 and, when
     min_volume_mult > 0, volume >= mult x the median volume of the
     `volume_lookback` bars before t (no signal during that warm-up), the
     price filter (any / with / against the delta), and a bar t+1 at or before
     the flat bar.
  2. Direction: follow (buy side -> long) or fade (buy side -> short).
  3. Entry at the open of t+1; exit at the close of e + hold_bars - 1, or at
     the close of the flat bar (last bar at/before flat_time) if that is earlier.
  4. Signals during a trade: ignore / close_and_reopen (exit + re-enter at the
     next open, "restart" or "reverse") / extend_or_reverse (same direction
     pushes the exit out; opposite reverses).
  HH:MM params >= 18:00 mean the evening of the previous calendar day.

Files:
  params.py   PARAMS / PARAM_SECTIONS / PARAMS_OPTIONS
  day.py      prepare_day: arrays + rolling volume median (cached)
  logic.py    validation, session anchoring, signals, the position walk
"""

from .day import build
from .logic import config, signals, walk
from .params import PARAM_SECTIONS, PARAMS, PARAMS_OPTIONS

DATA = {"main": ["open", "high", "low", "close", "volume", "volume_delta_pct"]}

PREPARE_PARAMS = ["volume_lookback"]


def prepare_day(date, data, params):
    return build(date, data["main"], params["volume_lookback"])


def process_day(day, params):
    cfg = config(params)                     # validates (memoized per param set)
    core = day.prepared
    pos, dirs, flat_bar = signals(core, cfg)
    if not len(pos):
        return None
    return walk(core, cfg, pos, dirs, flat_bar)


__all__ = ["DATA", "PREPARE_PARAMS", "PARAMS", "PARAM_SECTIONS", "PARAMS_OPTIONS",
           "prepare_day", "process_day"]
