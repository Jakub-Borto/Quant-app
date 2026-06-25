"""Absorption + delta entry: absorption candle followed by a confirming entry candle."""

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
    params:                 dict,
) -> tuple:
    """Returns (entry_ts, entry_price, invalidation_ts, entry_notes, trade_type)."""
    if post_retest.empty:
        return None, None, None, None, None

    if direction == "long":
        invalid_whole = post_retest["close"] < val
    else:
        invalid_whole = post_retest["close"] > vah

    n = len(post_retest)

    # breakout bar (index 0): invalidation only, not an absorption candidate
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

        if not is_absorption_candle(bar, baseline, direction, params):
            continue

        trigger_price, trigger_volume = find_absorption_trigger(bar, baseline, direction, params)

        absorption_ts    = ts
        absorption_level = float(bar["low"]) if direction == "long" else float(bar["high"])

        abs_scan_end = min(i + 1 + params["entry_after_absorption"], n)
        for j in range(i + 1, abs_scan_end):
            next_bar = post_retest.iloc[j]

            if invalid_whole.iloc[j]:
                return None, None, post_retest.index[j], None, None

            if direction == "long":
                if float(next_bar["close"]) < absorption_level:
                    break
            else:
                if float(next_bar["close"]) > absorption_level:
                    break

            bar_range = float(next_bar["high"]) - float(next_bar["low"])
            if bar_range <= 0:
                continue

            body    = abs(float(next_bar["close"]) - float(next_bar["open"]))
            body_ok = (body / bar_range) >= params["body_threshold"]

            if direction == "long":
                delta_ok = float(next_bar["volume_delta_pct"]) >= params["delta_threshold"]
            else:
                delta_ok = float(next_bar["volume_delta_pct"]) <= -params["delta_threshold"]

            if not (body_ok and delta_ok):
                continue

            entry_bar_idx = j + 1
            if entry_bar_idx >= n:
                return None, None, None, None, None

            if invalid_whole.iloc[entry_bar_idx]:
                return None, None, post_retest.index[entry_bar_idx], None, None

            entry_ts    = post_retest.index[entry_bar_idx]
            entry_price = float(post_retest.iloc[entry_bar_idx]["open"])

            entry_notes = {
                "absorption_time": absorption_ts.strftime("%H:%M"),
                "abs_baseline":    round(baseline, 2),
                "trigger_price":   trigger_price,
                "trigger_volume":  trigger_volume,
            }

            return entry_ts, entry_price, None, entry_notes, "absorption_delta"

    return None, None, None, None, None
