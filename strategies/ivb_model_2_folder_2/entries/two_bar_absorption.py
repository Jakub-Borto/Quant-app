"""Two bar absorption entry: reversal pair merged into a synthetic candle + confirmation."""

import json
import pandas as pd

from ..baselines import merge_tick_volume, build_two_bar_baseline


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
    """
    Pattern (long bias):
      - Bar i:   bearish, small bottom wick (<= two_bar_wick_ticks)
      - Bar i+1: bullish, small bottom wick (<= two_bar_wick_ticks), just positive delta

    Pattern (short bias):
      - Bar i:   bullish, small upper wick (<= two_bar_wick_ticks)
      - Bar i+1: bearish, small upper wick (<= two_bar_wick_ticks), just negative delta

    Merges both bars, checks absorption in defended wick against 2-bar paired baseline.
    Then scans from bar i+1 (inclusive) for a confirmation candle:
    correct direction + body_threshold + delta_threshold.
    Enters on open of the bar after the confirmation candle.

    Returns (entry_ts, entry_price, invalidation_ts, entry_notes, trade_type).
    """
    if post_retest.empty:
        return None, None, None, None, None

    if direction == "long":
        invalid_whole = post_retest["close"] < val
    else:
        invalid_whole = post_retest["close"] > vah

    n             = len(post_retest)
    tick_size     = params["tick_size"]
    max_wick      = params["two_bar_wick_ticks"] * tick_size
    required_mult = params["two_bar_abs_mult"]

    # breakout bar (index 0): invalidation only
    if invalid_whole.iloc[0]:
        return None, None, post_retest.index[0], None, None

    for i in range(1, n - 1):
        bar1 = post_retest.iloc[i]
        bar2 = post_retest.iloc[i + 1]
        ts1  = post_retest.index[i]
        ts2  = post_retest.index[i + 1]

        if invalid_whole.iloc[i] or invalid_whole.iloc[i + 1]:
            return None, None, post_retest.index[i], None, None

        # --- bar1: direction + small wick ---
        open1  = float(bar1["open"])
        close1 = float(bar1["close"])
        high1  = float(bar1["high"])
        low1   = float(bar1["low"])

        if direction == "long":
            if close1 >= open1:
                continue
            wick1 = min(open1, close1) - low1
        else:
            if close1 <= open1:
                continue
            wick1 = high1 - max(open1, close1)

        if wick1 > max_wick:
            continue

        # --- bar2: direction + small wick + just positive/negative delta ---
        open2  = float(bar2["open"])
        close2 = float(bar2["close"])
        high2  = float(bar2["high"])
        low2   = float(bar2["low"])

        if direction == "long":
            if close2 <= open2:
                continue
            wick2    = min(open2, close2) - low2
            delta_ok = float(bar2["volume_delta_pct"]) > 0
        else:
            if close2 >= open2:
                continue
            wick2    = high2 - max(open2, close2)
            delta_ok = float(bar2["volume_delta_pct"]) < 0

        if wick2 > max_wick:
            continue
        if not delta_ok:
            continue

        # --- build merged synthetic candle ---
        merged_open  = open1
        merged_high  = max(high1, high2)
        merged_low   = min(low1,  low2)
        merged_close = close2
        merged_tv    = merge_tick_volume(
            bar1.get("tick_volume", "{}"),
            bar2.get("tick_volume", "{}"),
        )

        # --- 2-bar baseline from bars before bar i (excludes breakout bar at index 0) ---
        pre_bars = post_retest.iloc[1:i]
        baseline = build_two_bar_baseline(pre_bars, direction, params)

        if pd.isna(baseline) or baseline <= 0:
            continue

        required = baseline * required_mult

        # --- absorption in defended wick of merged candle ---
        if direction == "long":
            body_bottom = min(merged_open, merged_close)
            wick_low    = merged_low
            wick_high   = body_bottom
        else:
            body_top  = max(merged_open, merged_close)
            wick_low  = body_top
            wick_high = merged_high

        if wick_high <= wick_low:
            continue

        try:
            raw_tv = json.loads(merged_tv)
        except Exception:
            continue

        absorbed       = False
        trigger_price  = None
        trigger_volume = None

        for price_str, (buy_qty, sell_qty) in raw_tv.items():
            price = float(price_str)
            if not (wick_low <= price <= wick_high):
                continue
            volume_at_level = sell_qty if direction == "long" else buy_qty
            if volume_at_level >= required:
                absorbed       = True
                trigger_price  = price
                trigger_volume = volume_at_level
                break

        if not absorbed:
            continue

        # --- confirmation candle scan starting from bar2 (inclusive) ---
        conf_scan_end = min(i + 1 + params["entry_after_absorption"], n)
        for j in range(i + 1, conf_scan_end):
            conf_bar   = post_retest.iloc[j]
            conf_open  = float(conf_bar["open"])
            conf_close = float(conf_bar["close"])
            conf_high  = float(conf_bar["high"])
            conf_low   = float(conf_bar["low"])
            bar_range  = conf_high - conf_low

            if invalid_whole.iloc[j]:
                return None, None, post_retest.index[j], None, None

            if bar_range <= 0:
                continue

            if direction == "long":
                if conf_close <= conf_open:
                    continue
                delta_ok = float(conf_bar["volume_delta_pct"]) >= params["delta_threshold"]
            else:
                if conf_close >= conf_open:
                    continue
                delta_ok = float(conf_bar["volume_delta_pct"]) <= -params["delta_threshold"]

            body    = abs(conf_close - conf_open)
            body_ok = (body / bar_range) >= params["body_threshold"]

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
                "absorption_time":  [ts1.strftime("%H:%M"), ts2.strftime("%H:%M")],
                "two_bar_baseline": round(baseline, 2),
                "trigger_price":    trigger_price,
                "trigger_volume":   trigger_volume,
            }

            return entry_ts, entry_price, None, entry_notes, "two_bar_absorption"

    return None, None, None, None, None
