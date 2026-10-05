"""Validation, session anchoring, signal detection and the position walk.

Spec sections referenced as §n (delta_pct_signal_spec.md v1).
"""

import datetime
import math

import numpy as np
import pandas as pd

from modules.engine import Trade

SESSION_START_MIN = 18 * 60          # 18:00 NY opens the Globex session
NO_SESSION = (17 * 60, 18 * 60)      # 17:00-17:59: no bars
RTH_FIRST, RTH_LAST = 9 * 60 + 30, 15 * 60 + 59

DIRECTION_MODES = ("follow", "fade")
PRICE_FILTERS = ("any", "with", "against")
IN_TRADE_MODES = ("ignore", "close_and_reopen", "extend_or_reverse")


# ══ validation (memoized per param set, STRATEGY_GUIDE §8.5) ═══════════════════
_CFG: dict = {}


def _hhmm(params, name) -> int:
    raw = params[name]
    try:
        hh, mm = (int(x) for x in str(raw).split(":"))
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError
    except ValueError:
        raise ValueError(f"{name} must be \"HH:MM\" (got {raw!r})") from None
    minute = hh * 60 + mm
    if NO_SESSION[0] <= minute < NO_SESSION[1]:
        raise ValueError(f"{name} {raw!r} lies in 17:00-17:59, when there is no session")
    return minute


def _session_order(minute: int) -> int:
    """Minutes since the session's 18:00 open (18:00 -> 0, 16:59 -> 1379)."""
    return (minute - SESSION_START_MIN) % (24 * 60)


def _whole(params, name, lo):
    v = params[name]
    if isinstance(v, bool) or not float(v).is_integer() or v < lo:
        raise ValueError(f"{name} must be a whole number >= {lo} (got {v!r})")
    return int(v)


def config(params) -> dict:
    key = tuple(sorted(params.items()))
    cfg = _CFG.get(key)
    if cfg is not None:
        return cfg
    theta = float(params["delta_threshold"])
    if not 0 < theta <= 100:
        raise ValueError(f"delta_threshold must be > 0 and <= 100 (got {params['delta_threshold']!r})")
    hold = _whole(params, "hold_bars", 1)
    _whole(params, "volume_lookback", 1)
    mult = float(params["min_volume_mult"])
    if mult < 0:
        raise ValueError(f"min_volume_mult must be >= 0 (got {params['min_volume_mult']!r})")
    for name, allowed in (("direction_mode", DIRECTION_MODES), ("price_filter", PRICE_FILTERS),
                          ("on_signal_in_trade", IN_TRADE_MODES)):
        if params[name] not in allowed:
            raise ValueError(f"{name} must be one of {list(allowed)} (got {params[name]!r})")
    start, end, flat = (_hhmm(params, n) for n in ("entry_start", "entry_end", "flat_time"))
    if _session_order(start) > _session_order(end):
        raise ValueError(f"entry_start {params['entry_start']!r} is after entry_end "
                         f"{params['entry_end']!r} in the session (18:00 -> 17:00)")
    if _session_order(end) >= _session_order(flat):
        raise ValueError(f"entry_end {params['entry_end']!r} must be before flat_time "
                         f"{params['flat_time']!r} in the session (18:00 -> 17:00)")
    cfg = {"theta": theta, "hold": hold, "mult": mult,
           "fade": params["direction_mode"] == "fade",
           "price_filter": params["price_filter"], "mode": params["on_signal_in_trade"],
           "start": start, "end": end, "flat": flat,
           "trade_type": f"{params['direction_mode']}_{params['price_filter']}"}
    _CFG[key] = cfg
    return cfg


# ══ session anchoring (§6) ═════════════════════════════════════════════════════
def anchor(date: datetime.date, minute: int, tz) -> pd.Timestamp:
    """HH:MM -> timestamp: >= 18:00 is the evening of D-1, else D."""
    day = date - datetime.timedelta(days=1) if minute >= SESSION_START_MIN else date
    naive = datetime.datetime.combine(day, datetime.time(minute // 60, minute % 60))
    return pd.Timestamp(naive).tz_localize(tz, ambiguous=True, nonexistent="shift_forward")


# ══ signals (§4) ═══════════════════════════════════════════════════════════════
def signals(core, cfg):
    """(positions of valid signals, their direction +1 long / -1 short, flat bar F).
    F is -1 when no bar lies at or before flat_time."""
    idx, tz = core.index, core.index.tz
    start_ts, end_ts, flat_ts = (anchor(core.date, cfg[k], tz) for k in ("start", "end", "flat"))
    first = int(idx.searchsorted(start_ts, side="left"))
    last = int(idx.searchsorted(end_ts, side="right")) - 1
    flat_bar = int(idx.searchsorted(flat_ts, side="right")) - 1
    if flat_bar < 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), -1
    # 4.2.1 window and 4.2.4 next bar exists at/before the flat bar: t <= F - 1
    last = min(last, flat_bar - 1)
    if last < first:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), flat_bar
    sl = slice(first, last + 1)
    d, v = core.delta[sl], core.volume[sl]
    o, c = core.open[sl], core.close[sl]
    buy = d >= cfg["theta"]                                    # 4.1
    sell = d <= -cfg["theta"]
    ok = (buy | sell) & (v >= 1)                               # 4.2.2 volume >= 1 always
    if cfg["mult"] > 0:                                        # 4.2.2 median filter (+ warm-up)
        med = core.median[sl]
        ok &= ~np.isnan(med)
        with np.errstate(invalid="ignore"):
            ok &= v >= cfg["mult"] * med
    pf = cfg["price_filter"]                                   # 4.2.3
    if pf == "with":
        ok &= np.where(buy, c > o, c < o)
    elif pf == "against":
        ok &= np.where(buy, c <= o, c >= o)
    pos = np.flatnonzero(ok) + first
    side = np.where(buy[pos - first], 1, -1)
    return pos, (-side if cfg["fade"] else side), flat_bar


# ══ the position walk (§5) ═════════════════════════════════════════════════════
def _notes(core, cfg, t, e, bars_held, extensions):
    vol = float(core.volume[t])
    med = core.median[t]
    ratio = None
    if cfg["mult"] > 0 and not math.isnan(med) and med != 0:
        ratio = round(vol / med, 2)
    entry_ts = core.index[e]
    minute = entry_ts.hour * 60 + entry_ts.minute
    rth = entry_ts.date() == core.date and RTH_FIRST <= minute <= RTH_LAST
    return {"signal_time": core.index[t].strftime("%H:%M"),
            "signal_delta_pct": float(core.delta[t]),
            "signal_volume": int(vol),
            "volume_ratio": ratio,
            "bars_held": int(bars_held),
            "extensions": int(extensions),
            "session": "rth" if rth else "eth"}


def walk(core, cfg, pos, dirs, flat_bar) -> list:
    hold, mode, idx = cfg["hold"], cfg["mode"], core.index
    trades = []
    cur = None          # [t, e, direction(+1/-1), x, x_uncapped, extensions]

    def _open(t, d):
        e = t + 1
        x_raw = e + hold - 1
        return [t, e, d, min(x_raw, flat_bar), x_raw, 0]

    def _close_scheduled(tr):
        t, e, d, x, x_raw, ext = tr
        trades.append(Trade("long" if d > 0 else "short", idx[e], idx[x],
                            float(core.open[e]), float(core.close[x]),
                            "flat" if x_raw > flat_bar else "time",
                            trade_type=cfg["trade_type"],
                            notes=_notes(core, cfg, t, e, x - e + 1, ext)))

    def _close_at_open(tr, s, reason):
        t, e, d, _x, _xr, ext = tr
        trades.append(Trade("long" if d > 0 else "short", idx[e], idx[s + 1],
                            float(core.open[e]), float(core.open[s + 1]), reason,
                            trade_type=cfg["trade_type"],
                            notes=_notes(core, cfg, t, e, s - e + 1, ext)))

    for s, d in zip(pos.tolist(), dirs.tolist()):
        if cur is not None and s >= cur[3]:          # the trade closed at x's close
            _close_scheduled(cur)
            cur = None
        if cur is None:
            cur = _open(s, d)
            continue
        # an in-trade signal: e <= s < x (§5.3)
        if mode == "ignore":
            continue
        if mode == "extend_or_reverse" and d == cur[2]:
            x_raw = s + 1 + hold - 1
            cur[3], cur[4] = min(x_raw, flat_bar), x_raw
            cur[5] += 1
            continue
        _close_at_open(cur, s, "restart" if d == cur[2] else "reverse")
        cur = _open(s, d)
    if cur is not None:
        _close_scheduled(cur)
    return trades
