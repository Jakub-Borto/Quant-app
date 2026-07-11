"""Core orchestration: breakout/retest detection, entry dispatcher, day processor.

Rewritten on positional numpy windows (see _daydata.DayData): a day is materialized once as
arrays + pre-parsed JSON, breakout/retest/flip bookkeeping is plain integer positions, and the
finders / risk scripts receive EntryWindow / TradeWindow contexts instead of DataFrame slices.
Every stage is wrapped in an accumulating timer (_timing) reported once per backtest run.
"""

import json
import numpy as np
from datetime import time

from ._timing   import timed
from ._daydata  import DayData, EntryWindow, TradeWindow
from .profile   import compute_ivb_profile
from .baselines import (
    build_rolling_baseline, build_passive_baseline, build_cvd_change_baseline,
    BASELINE_WARMUP_MINUTES,
)
from .entries   import FINDER_REGISTRY, FINDER_NAMES
from .risk      import RISK_REGISTRY, RISK_NAMES


# The session START is the `session_start` param ("HH:MM" NY wall time, default "09:30");
# the session END stays fixed. Slicing uses integer minutes-since-midnight
# (t >= time(H, M) <=> 60h+m >= 60H+M for any second/microsecond value, and strictly-before
# likewise) — avoids materializing 1380 datetime.time objects per day.
RTH_END      = time(16, 0)
_RTH_END_MIN = RTH_END.hour * 60 + RTH_END.minute


def session_start_minutes(params: dict) -> int:
    """Resolve the session_start param to minutes since midnight (validated)."""
    raw = str(params.get("session_start", "09:30")).strip()
    try:
        hh, mm = raw.split(":")
        start_min = int(hh) * 60 + int(mm)
    except Exception:
        raise ValueError(f'session_start must be "HH:MM" (got {raw!r})') from None
    if not 0 <= start_min < _RTH_END_MIN:
        raise ValueError(
            f"session_start {raw!r} must lie at/after 00:00 and before the session end "
            f"{RTH_END.strftime('%H:%M')}"
        )
    return start_min


# ---------------------------------------------------------------------------
# Day-core construction (cacheable across runs; start_min is part of the cache key)
# ---------------------------------------------------------------------------

def build_day_core(session, ind_df=None, start_min: int = 570):
    """One day's files -> a cacheable DayData (arrays + lazy parsed JSON + raw indicator
    arrays). Returns None for unusable days (empty / tz-naive index). Everything here is a
    pure function of the files + the resolved session start; per-run state is derived from
    it in process_day."""
    if session.empty:
        return None
    if session.index.tz is None:
        return None

    with timed("day:rth_filter"):
        idx      = session.index
        minutes  = idx.hour.to_numpy() * 60 + idx.minute.to_numpy()
        rth_mask = (minutes >= start_min) & (minutes < _RTH_END_MIN)

    with timed("day:daydata_build"):
        day = DayData(session, rth_mask, minutes[rth_mask],
                      warmup_min=start_min + BASELINE_WARMUP_MINUTES)

    # raw indicator arrays, positionally aligned. Alignment fast path: indicator files share
    # the candle index, so the RTH mask applies directly; reindex stays as the fallback.
    if ind_df is not None:
        with timed("day:indicators_align"):
            same_index = ind_df.index.equals(idx)

            if "cumulative_delta" in ind_df.columns:
                col = ind_df["cumulative_delta"]
                day.cvd_raw = (col.to_numpy(dtype=np.float64)[rth_mask] if same_index
                               else col.reindex(day.index).to_numpy(dtype=np.float64))

            present = [c for c in VWAP_BAND_COLUMNS if c in ind_df.columns]
            if present:
                if same_index:
                    day.bands_raw = {
                        c: ind_df[c].to_numpy(dtype=np.float64)[rth_mask] for c in present
                    }
                else:
                    aligned = ind_df[present].reindex(day.index)
                    day.bands_raw = {
                        c: aligned[c].to_numpy(dtype=np.float64) for c in present
                    }

    return day


# ---------------------------------------------------------------------------
# Breakout / retest detection (absolute day positions)
# ---------------------------------------------------------------------------

def detect_breakout(day: DayData, start: int, ivb_high: float, ivb_low: float) -> tuple:
    closes = day.close[start:]

    long_breakout  = closes > ivb_high
    short_breakout = closes < ivb_low

    long_pos  = int(long_breakout.argmax())  if long_breakout.any()  else None
    short_pos = int(short_breakout.argmax()) if short_breakout.any() else None

    if long_pos is None and short_pos is None:
        return None, None

    if long_pos is not None and short_pos is not None:
        direction = "long" if long_pos <= short_pos else "short"
    elif long_pos is not None:
        direction = "long"
    else:
        direction = "short"

    breakout_pos = long_pos if direction == "long" else short_pos
    return direction, start + breakout_pos


def detect_retest(
    day:           DayData,
    breakout_pos:  int,
    direction:     str,
    vah:           float,
    val:           float,
    retest_window: int,
) -> int:
    scan_start = breakout_pos + 1
    scan_end   = min(scan_start + retest_window, day.n)

    if scan_start >= scan_end:
        return None

    if direction == "long":
        retest_mask = day.low[scan_start:scan_end] <= vah
    else:
        retest_mask = day.high[scan_start:scan_end] >= val

    if not retest_mask.any():
        return None

    return scan_start + int(retest_mask.argmax())


# ---------------------------------------------------------------------------
# Entry dispatcher
# ---------------------------------------------------------------------------

def find_entry(win: EntryWindow, params: dict) -> tuple:
    """
    Calls all enabled entry finders on the shared window context and returns the one with
    the earliest entry; if no finder finds an entry, the earliest invalidation.

    Returns: (entry_rel, entry_price, invalidation_rel, entry_notes, trade_type) with
    window-relative bar indices (ascending index == ascending timestamp).
    """
    valid_entries = params.get("valid_entries", "1" * len(FINDER_REGISTRY))
    flags         = valid_entries.ljust(len(FINDER_REGISTRY), "0")

    candidates = []
    for fn, name, flag in zip(FINDER_REGISTRY, FINDER_NAMES, flags):
        if flag == "1":
            with timed(f"entry:{name}"):
                candidates.append(fn(win, params))

    entries       = [c for c in candidates if c[0] is not None]
    invalidations = [c for c in candidates if c[0] is None and c[2] is not None]

    if entries:
        return min(entries, key=lambda c: c[0])

    if invalidations:
        return min(invalidations, key=lambda c: c[2])

    return None, None, None, None, None


# ---------------------------------------------------------------------------
# Day processor
# ---------------------------------------------------------------------------

# tick-vwap deviation band columns consumed by vwap_tp_risk (built per day when available)
VWAP_BAND_COLUMNS = [
    "vwap_tick_globex_std2_up", "vwap_tick_globex_std2_dn",
    "vwap_tick_globex_std3_up", "vwap_tick_globex_std3_dn",
    "vwap_tick_rth_std2_up",    "vwap_tick_rth_std2_dn",
    "vwap_tick_rth_std3_up",    "vwap_tick_rth_std3_dn",
]


def process_day(day: DayData, params: dict):
    if day is None:
        return None
    day.reset_run_state()                   # cached days carry a previous run's baselines

    ib_n = params["ib_minutes"]
    if day.n < ib_n:
        return None

    ivb_high  = float(day.high[:ib_n].max())
    ivb_low   = float(day.low[:ib_n].min())
    ivb_range = ivb_high - ivb_low

    if ivb_range <= 0:
        return None

    if ib_n >= day.n:                       # post-IB window empty
        return None

    # breakout first — it needs nothing but closes, so profile/baseline work is skipped
    # entirely on no-breakout days (pure reordering, no observable difference)
    with timed("day:breakout_retest"):
        direction, breakout_pos = detect_breakout(day, ib_n, ivb_high, ivb_low)
    if direction is None:
        return None

    with timed("day:profile"):
        poc, vah, val = compute_ivb_profile(day, ib_n)
    if poc is None:
        return None

    # --- gated per-run state: build only what the enabled finders/detectors consume ---
    # A baseline left None is never read: window contexts skip gathering it and every
    # consumer is switched off by the same bit strings used here (find_entry ljust for
    # entries, _build_trailing_sl str+ljust for trails — replicated exactly).
    n_finders   = len(FINDER_REGISTRY)
    entry_flags = params.get("valid_entries", "1" * n_finders).ljust(n_finders, "0")
    risk_idx    = params["risk_script"] - 1
    if not (0 <= risk_idx < len(RISK_REGISTRY)):
        risk_idx = 0
    if risk_idx == 3:                       # vwap_trailing_risk re-detects signals in-trade
        trail_flags = str(params.get("trailing_entries", "0" * n_finders)).ljust(n_finders, "0")
    else:
        trail_flags = "0" * n_finders
    active = [e == "1" or t == "1" for e, t in zip(entry_flags, trail_flags)]

    need_rolling = active[0] or active[1] or active[3]   # absorption_delta / consec / passive_size
    need_passive = active[3] or active[4]                # passive_size / passive_wall
    need_cvd     = active[5] or active[6]                # the two cvd_divergence flavours

    with timed("day:baselines"):
        valid_pos = day.warmup_pos
        if need_rolling:
            day.sell_baseline, day.buy_baseline = build_rolling_baseline(day, valid_pos, params)
        if need_passive:
            day.passive_baseline_long  = build_passive_baseline(day, valid_pos, "long",  params)
            day.passive_baseline_short = build_passive_baseline(day, valid_pos, "short", params)

    # CVD (cumulative_delta) + its bar-to-bar change std — None when no indicators were
    # loaded (or no cvd bit is on) => the cvd_divergence finders disable themselves.
    if need_cvd and day.cvd_raw is not None:
        with timed("day:cvd"):
            day.cvd     = day.cvd_raw
            day.cvd_std = build_cvd_change_baseline(day, valid_pos, params)

    # VWAP deviation bands: only the two vwap risk scripts read them
    if risk_idx in (2, 3):
        day.vwap_bands = day.bands_raw

    max_flips  = params["max_flips"]
    flip_count = 0

    while True:
        breakout_ts = day.index[breakout_pos]

        with timed("day:breakout_retest"):
            retest_pos = detect_retest(
                day           = day,
                breakout_pos  = breakout_pos,
                direction     = direction,
                vah           = vah,
                val           = val,
                retest_window = params["retest_window"],
            )

        if retest_pos is None:
            return None

        retest_ts = day.index[retest_pos]

        # post_retest window = [breakout bar] + [retest .. retest+entry_window)
        pos = np.concatenate((
            [breakout_pos],
            np.arange(retest_pos, min(retest_pos + params["entry_window"], day.n)),
        ))

        with timed("day:entry_window_build"):
            win = EntryWindow(day, pos, direction, vah, val, params)

        entry_rel, entry_price, invalid_rel, entry_notes, trade_type = find_entry(win, params)

        if entry_rel is not None:
            entry_pos = int(pos[entry_rel])
            break

        if invalid_rel is None:
            return None

        if flip_count >= max_flips:
            return None

        flip_count += 1
        direction = "short" if direction == "long" else "long"

        # resume AT the invalidation bar (the original searchsorted resume point)
        resume_pos = int(pos[invalid_rel])

        with timed("day:breakout_retest"):
            direction_found, breakout_pos = detect_breakout(day, resume_pos, ivb_high, ivb_low)

        if direction_found != direction:
            return None

    # --- risk script dispatch (1-based risk_script -> RISK_REGISTRY) ---
    # the day context carries the baselines + CVD series so risk scripts can re-detect
    # entry-style signals on the live trade bars (vwap_trailing_risk); others ignore them.
    levels = {"val": val, "vah": vah, "poc": poc}

    with timed("day:trade_window_build"):
        trade_win = TradeWindow(day, entry_pos, direction, params)

    risk_fn = RISK_REGISTRY[risk_idx]

    with timed(f"risk:{RISK_NAMES[risk_idx]}"):
        trade = risk_fn(
            entry_win   = win,
            trade_win   = trade_win,
            entry_pos   = entry_pos,
            entry_price = entry_price,
            direction   = direction,
            levels      = levels,
            params      = params,
        )

    if trade is None:
        return None

    trade["trade_type"] = trade_type

    # --- build notes: process_day context + entry-specific notes ---
    with timed("day:notes"):
        process_day_notes = {
            "breakout_time": breakout_ts.strftime("%H:%M"),
            "retest_time":   retest_ts.strftime("%H:%M"),
            "flip_count":    flip_count,
            "ivb_high":      ivb_high,
            "ivb_low":       ivb_low,
            "poc":           poc,
            "vah":           vah,
            "val":           val,
        }

        # a risk script may attach its own note fields via trade["risk_notes"] (popped here so
        # it never becomes a stray column); scripts that don't set it leave notes byte-identical.
        risk_notes = trade.pop("risk_notes", None)

        trade["notes"] = json.dumps({
            **process_day_notes,
            **(entry_notes or {}),
            **(risk_notes or {}),
        })

    return trade
