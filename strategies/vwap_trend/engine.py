"""Pure execution engine: side decisions + bars -> trades. No I/O.

Position semantics: pos-during-bar-u = decision made on the close of bar u-1,
filled at open[u] (never at the signal bar's own close — that would be
look-ahead). Stop-and-reverse: a flip closes and opens at the SAME fill, so
consecutive trades share exit_time/exit_price with the next entry_time/
entry_price by construction. A bar inside the exclusion window forces the
position flat at that bar's open. Whatever is still open after the last bar
of the window is flattened at that bar's close ('eod'); a side change decided
on the last bar has no next open and is discarded.
"""

import json

import numpy as np

from .signals import classify, resolve

TRADE_TYPE = "vwap_trend"


def _jval(x):
    """numpy scalar -> JSON-safe python value (NaN -> None)."""
    x = float(x)
    return x if np.isfinite(x) else None


def _make_trade(index, open_, close, volume, vwap, date, cfg,
                direction_code: int, u_entry: int, u_exit: int, is_eod: bool) -> dict:
    direction = "long" if direction_code > 0 else "short"
    entry_price = open_[u_entry]
    if is_eod:
        exit_price, exit_reason = close[u_exit], "eod"
        bars_held = u_exit - u_entry + 1          # entry bar .. last bar inclusive
    else:
        exit_price = open_[u_exit]
        exit_reason = None                        # caller fills in flip/flat/exclusion
        bars_held = u_exit - u_entry

    pnl_points = exit_price - entry_price if direction_code > 0 else entry_price - exit_price

    # sl / tp per cfg["sl_convention"] — this system has no planned stop or
    # target; see the package docstring for what each convention means for
    # Analytics' RR metrics
    conv = cfg["sl_convention"]
    sl = tp = np.nan
    if conv == "vwap_at_entry":
        # the level that would have flipped the position at entry time
        v = vwap[u_entry]
        sl = v - cfg["band_points"] if direction_code > 0 else v + cfg["band_points"]
    elif conv == "realized_exit":
        if pnl_points < 0:
            sl = exit_price
        elif pnl_points > 0:
            tp = exit_price

    notes = {
        "vwap_at_entry": _jval(vwap[u_entry]),
        "vwap_at_exit":  _jval(vwap[u_exit]),
        "bars_held":     int(bars_held),
        "band_ticks":    cfg["band_ticks"],
        "filled_on_zero_volume_bar": bool(volume[u_entry] == 0 or volume[u_exit] == 0),
    }

    return {
        "date":        date,
        "direction":   direction,
        "trade_type":  TRADE_TYPE,
        "entry_time":  index[u_entry],
        "exit_time":   index[u_exit],
        "entry_price": entry_price,
        "exit_price":  exit_price,
        "sl":          sl,
        "tp":          tp,
        "exit_reason": exit_reason,
        "pnl_points":  pnl_points,
        "notes":       json.dumps(notes),
    }


def run_day(index, open_, close, volume, vwap, excl, date, cfg) -> list:
    """One session window -> list of trade dicts.

    All arrays are already restricted to [signal_start .. trade_end]: bar 0 is
    the bar BEFORE the first fill-eligible bar when one exists (its close is
    the first signal), so positions can only ever start at bar 1 — which is the
    first bar at/after trade_start_time.
    """
    n = len(close)
    if n < 2:
        return []       # a lone bar has no next open to fill on

    codes     = classify(close, vwap, volume, cfg["band_points"], cfg["skip_zero_volume"])
    decisions = resolve(codes, excl, cfg["carry_forward"])

    trades = []
    dec_l  = decisions.tolist()
    excl_l = excl.tolist()
    current = 0
    u_entry = -1
    for u in range(1, n):
        new = 0 if excl_l[u] else dec_l[u - 1]
        if new == current:
            continue
        if current != 0:
            trade = _make_trade(index, open_, close, volume, vwap, date, cfg,
                                current, u_entry, u, is_eod=False)
            trade["exit_reason"] = ("exclusion" if excl_l[u]
                                    else "vwap_flip" if new != 0
                                    else "band_flat")
            trades.append(trade)
        if new != 0:
            u_entry = u
        current = new

    if current != 0:    # forced flat at the close of the window's last bar
        trades.append(_make_trade(index, open_, close, volume, vwap, date, cfg,
                                  current, u_entry, n - 1, is_eod=True))
    return trades
