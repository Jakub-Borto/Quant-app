"""Passive absorption entry: big resting order + absorption on the same candle + confirmation."""

import json
import pandas as pd

from ..absorption import is_absorption_candle


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
    Looks for a candle with both:
      - A big passive order on the defended side:
          Long: resting bid below candle open, size/count >= passive_baseline * passive_order_mult
          Short: resting ask above candle open, size/count >= passive_baseline * passive_order_mult
      - Absorption in the defended wick (using passive_absorption_mult)

    Both conditions must be on the same candle.
    Then scans up to entry_after_absorption bars for a confirmation candle:
    correct direction + body_threshold + delta_threshold.
    Enters on open of the bar after the confirmation candle.

    Returns (entry_ts, entry_price, invalidation_ts, entry_notes, trade_type).
    """
    if post_retest.empty:
        return None, None, None, None, None

    if direction == "long":
        invalid_whole    = post_retest["close"] < val
        passive_baseline = passive_baseline_long
    else:
        invalid_whole    = post_retest["close"] > vah
        passive_baseline = passive_baseline_short

    n = len(post_retest)

    passive_params = {**params, "absorption_mult": params["passive_absorption_mult"]}

    # breakout bar (index 0): invalidation only
    if invalid_whole.iloc[0]:
        return None, None, post_retest.index[0], None, None

    for i in range(1, n):
        bar = post_retest.iloc[i]
        ts  = post_retest.index[i]

        if invalid_whole.iloc[i]:
            return None, None, ts, None, None

        # --- passive order check ---
        p_baseline = passive_baseline.get(ts, float("nan"))
        if pd.isna(p_baseline) or p_baseline <= 0:
            continue

        po = bar.get("passive_orders", None)
        if not po or po == "{}":
            continue
        try:
            raw_po = json.loads(po)
        except Exception:
            continue

        bar_open              = float(bar["open"])
        required_po           = p_baseline * params["passive_order_mult"]
        has_big_passive       = False
        passive_trigger_price = None
        passive_trigger_size  = None
        passive_trigger_count = None

        for price_str, (size, count) in raw_po.items():
            price = float(price_str)
            if count <= 0:
                continue
            if direction == "long" and price >= bar_open:
                continue
            if direction == "short" and price <= bar_open:
                continue
            if (size / count) >= required_po:
                has_big_passive       = True
                passive_trigger_price = price
                passive_trigger_size  = size
                passive_trigger_count = count
                break

        if not has_big_passive:
            continue

        # --- absorption check on the same candle ---
        abs_baseline = (
            sell_baseline.get(ts, float("nan"))
            if direction == "long"
            else buy_baseline.get(ts, float("nan"))
        )

        if not is_absorption_candle(bar, abs_baseline, direction, passive_params):
            continue

        absorption_ts = ts

        # --- confirmation candle scan ---
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
                "absorption_time": absorption_ts.strftime("%H:%M"),
                "passive_price":   passive_trigger_price,
                "passive_size":    passive_trigger_size,
                "passive_count":   passive_trigger_count,
            }

            return entry_ts, entry_price, None, entry_notes, "passive_absorption"

    return None, None, None, None, None
