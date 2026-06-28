"""Consecutive absorption entry: N absorption candles at the same level + body midpoint."""

import pandas as pd

from ..absorption import is_absorption_candle, find_absorption_trigger


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
    Scans post_retest bar by bar. For each absorption candle found, checks whether
    n-1 prior absorption candles exist within ±consec_abs_ticks of its absorption
    level and body midpoint. If so, enters on the open of the next bar.
    No confirmation candle required.

    Returns (entry_ts, entry_price, invalidation_ts, entry_notes, trade_type).
    """
    if post_retest.empty:
        return None, None, None, None, None

    if direction == "long":
        invalid_whole = post_retest["close"] < val
    else:
        invalid_whole = post_retest["close"] > vah

    n          = len(post_retest)
    tick_size  = params["tick_size"]
    level_tol  = params["consec_abs_ticks"] * tick_size
    required_n = params["consec_abs_n"]

    consec_params = {
        **params,
        "absorption_mult":  params["consec_abs_mult"],
        "wick_threshold":   params["consec_wick_threshold"],
    }

    # (abs_level, body_mid, ts)
    seen: list[tuple[float, float, pd.Timestamp]] = []

    # breakout bar (index 0): invalidation only
    if invalid_whole.iloc[0]:
        return None, None, post_retest.index[0], None, None

    for i in range(1, n):
        bar = post_retest.iloc[i]
        ts  = post_retest.index[i]

        if invalid_whole.iloc[i]:
            return None, None, ts, None, None

        baseline = (
            sell_baseline.get(ts, float("nan"))
            if direction == "long"
            else buy_baseline.get(ts, float("nan"))
        )

        if not is_absorption_candle(bar, baseline, direction, consec_params):
            continue

        abs_level = float(bar["low"]) if direction == "long" else float(bar["high"])
        body_mid  = (
            min(float(bar["open"]), float(bar["close"])) +
            max(float(bar["open"]), float(bar["close"]))
        ) / 2

        seen.append((abs_level, body_mid, ts))

        nearby = [
            (lvl, t) for lvl, bm, t in seen
            if abs(lvl - abs_level) <= level_tol
            and abs(bm - body_mid) <= level_tol
        ]

        if len(nearby) < required_n:
            continue

        entry_bar_idx = i + 1
        if entry_bar_idx >= n:
            return None, None, None, None, None

        if invalid_whole.iloc[entry_bar_idx]:
            return None, None, post_retest.index[entry_bar_idx], None, None

        # trigger reflects the nth (triggering) candle
        trigger_price, trigger_volume = find_absorption_trigger(bar, baseline, direction, consec_params)

        entry_ts    = post_retest.index[entry_bar_idx]
        entry_price = float(post_retest.iloc[entry_bar_idx]["open"])

        entry_notes = {
            "absorption_time": [t.strftime("%H:%M") for _, t in nearby],
            "abs_baseline":    round(baseline, 2),
            "trigger_price":   trigger_price,
            "trigger_volume":  trigger_volume,
        }

        return entry_ts, entry_price, None, entry_notes, "consecutive_absorption"

    return None, None, None, None, None
