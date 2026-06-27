"""Core orchestration: breakout/retest detection, entry dispatcher, day processor."""

import json
import pandas as pd
from datetime import time

from .params     import PARAMS
from .profile    import compute_ivb_profile
from .baselines  import build_rolling_baseline, build_passive_baseline, build_cvd_change_baseline
from .entries    import FINDER_REGISTRY
from .risk       import RISK_REGISTRY


# ---------------------------------------------------------------------------
# Breakout / retest detection
# ---------------------------------------------------------------------------

def detect_breakout(post_ib: pd.DataFrame, ivb_high: float, ivb_low: float) -> tuple:
    long_breakout  = post_ib["close"] > ivb_high
    short_breakout = post_ib["close"] < ivb_low

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
    return direction, breakout_pos


def detect_retest(
    post_ib:       pd.DataFrame,
    breakout_pos:  int,
    direction:     str,
    vah:           float,
    val:           float,
    poc:           float,
    retest_window: int,
) -> int:
    scan_start = breakout_pos + 1
    scan_end   = scan_start + retest_window
    scan       = post_ib.iloc[scan_start:scan_end]

    if scan.empty:
        return None

    if direction == "long":
        retest_mask = scan["low"] <= vah
    else:
        retest_mask = scan["high"] >= val

    if not retest_mask.any():
        return None

    return scan_start + int(retest_mask.argmax())


# ---------------------------------------------------------------------------
# Entry dispatcher
# ---------------------------------------------------------------------------

def find_entry(
    post_retest:            pd.DataFrame,
    direction:              str,
    ivb_high:               float,
    ivb_low:                float,
    poc:                    float,
    vah:                    float,
    val:                    float,
    sell_baseline:          pd.Series,
    buy_baseline:           pd.Series,
    passive_baseline_long:  pd.Series,
    passive_baseline_short: pd.Series,
    cvd_series:             pd.Series,
    cvd_change_std:         pd.Series,
    params:                 dict,
) -> tuple:
    """
    Calls all enabled entry finders and returns the one with the earliest entry_ts.
    If no finder finds an entry, returns the earliest invalidation_ts across all.

    Returns: (entry_ts, entry_price, invalidation_ts, entry_notes, trade_type)
    """
    shared = dict(
        post_retest            = post_retest,
        direction              = direction,
        ivb_high               = ivb_high,
        ivb_low                = ivb_low,
        poc                    = poc,
        vah                    = vah,
        val                    = val,
        sell_baseline          = sell_baseline,
        buy_baseline           = buy_baseline,
        passive_baseline_long  = passive_baseline_long,
        passive_baseline_short = passive_baseline_short,
        cvd_series             = cvd_series,
        cvd_change_std         = cvd_change_std,
        params                 = params,
    )

    valid_entries = params.get("valid_entries", "1" * len(FINDER_REGISTRY))
    flags         = valid_entries.ljust(len(FINDER_REGISTRY), "0")

    candidates = [
        fn(**shared)
        for fn, flag in zip(FINDER_REGISTRY, flags)
        if flag == "1"
    ]

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

def process_day(session: pd.DataFrame, params: dict, cvd_raw: pd.Series = None):
    rth_session = session[
        (session.index.time >= time(9, 30)) &
        (session.index.time <  time(16, 0))
    ]

    if len(rth_session) < params["ib_minutes"]:
        return None

    ib_bars   = rth_session.iloc[:params["ib_minutes"]]
    ivb_high  = float(ib_bars["high"].max())
    ivb_low   = float(ib_bars["low"].min())
    ivb_range = ivb_high - ivb_low

    if ivb_range <= 0:
        return None

    poc, vah, val = compute_ivb_profile(ib_bars)
    if poc is None:
        return None

    sell_baseline, buy_baseline = build_rolling_baseline(rth_session, params)
    passive_baseline_long       = build_passive_baseline(rth_session, "long",  params)
    passive_baseline_short      = build_passive_baseline(rth_session, "short", params)

    # CVD (cumulative_delta) + its bar-to-bar change std — day-level, like the baselines.
    # None when no indicators were loaded => the cvd_divergence finder disables itself.
    if cvd_raw is not None:
        cvd_series     = cvd_raw.reindex(rth_session.index)
        cvd_change_std = build_cvd_change_baseline(cvd_series, params)
    else:
        cvd_series = cvd_change_std = None

    post_ib = rth_session.iloc[params["ib_minutes"]:]
    if post_ib.empty:
        return None

    direction, breakout_pos = detect_breakout(post_ib, ivb_high, ivb_low)
    if direction is None:
        return None

    max_flips  = params["max_flips"]
    flip_count = 0

    while True:
        breakout_ts = post_ib.index[breakout_pos]

        retest_pos = detect_retest(
            post_ib       = post_ib,
            breakout_pos  = breakout_pos,
            direction     = direction,
            vah           = vah,
            val           = val,
            poc           = poc,
            retest_window = params["retest_window"],
        )

        if retest_pos is None:
            return None

        retest_ts = post_ib.index[retest_pos]

        post_retest = pd.concat([
            post_ib.iloc[[breakout_pos]],
            post_ib.iloc[retest_pos : retest_pos + params["entry_window"]]
        ])

        entry_ts, entry_price, invalidation_ts, entry_notes, trade_type = find_entry(
            post_retest            = post_retest,
            direction              = direction,
            ivb_high               = ivb_high,
            ivb_low                = ivb_low,
            poc                    = poc,
            vah                    = vah,
            val                    = val,
            sell_baseline          = sell_baseline,
            buy_baseline           = buy_baseline,
            passive_baseline_long  = passive_baseline_long,
            passive_baseline_short = passive_baseline_short,
            cvd_series             = cvd_series,
            cvd_change_std         = cvd_change_std,
            params                 = params,
        )

        if entry_ts is not None:
            break

        if invalidation_ts is None:
            return None

        if flip_count >= max_flips:
            return None

        flip_count += 1
        direction = "short" if direction == "long" else "long"

        resume_pos = post_ib.index.searchsorted(invalidation_ts)
        post_ib    = post_ib.iloc[resume_pos:]

        if post_ib.empty:
            return None

        direction_found, breakout_pos = detect_breakout(post_ib, ivb_high, ivb_low)

        if direction_found != direction:
            return None

    # --- risk script dispatch (1-based risk_script -> RISK_REGISTRY) ---
    levels     = {"val": val, "vah": vah, "poc": poc}
    post_entry = post_ib.loc[entry_ts:]

    idx     = params["risk_script"] - 1
    risk_fn = RISK_REGISTRY[idx] if 0 <= idx < len(RISK_REGISTRY) else RISK_REGISTRY[0]

    trade = risk_fn(
        post_retest = post_retest,
        post_entry  = post_entry,
        entry_ts    = entry_ts,
        entry_price = entry_price,
        direction   = direction,
        levels      = levels,
        params      = params,
    )

    if trade is None:
        return None

    trade["trade_type"] = trade_type

    # --- build notes: process_day context + entry-specific notes ---
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

    trade["notes"] = json.dumps({
        **process_day_notes,
        **(entry_notes or {}),
    })

    return trade
