"""Shared absorption grading used by all absorption-based entry finders."""

import json
import pandas as pd


def is_absorption_candle(bar: pd.Series, baseline: float, direction: str, params: dict) -> bool:
    """
    True if the bar has a qualifying wick on the defended side AND a price level
    inside that wick with aggressive volume against the direction >= baseline * absorption_mult.
    """
    if pd.isna(baseline) or baseline <= 0:
        return False

    high  = float(bar["high"])
    low   = float(bar["low"])
    op    = float(bar["open"])
    close = float(bar["close"])

    bar_range = high - low
    if bar_range <= 0:
        return False

    threshold = params["wick_threshold"]
    required  = baseline * params["absorption_mult"]

    if direction == "long":
        body_bottom = min(op, close)
        wick_size   = body_bottom - low
        if wick_size / bar_range < threshold:
            return False
        wick_low  = low
        wick_high = body_bottom
    else:
        body_top  = max(op, close)
        wick_size = high - body_top
        if wick_size / bar_range < threshold:
            return False
        wick_low  = body_top
        wick_high = high

    tv = bar.get("tick_volume", None)
    if not tv or tv == "{}":
        return False

    try:
        raw = json.loads(tv)
    except Exception:
        return False

    for price_str, (buy_qty, sell_qty) in raw.items():
        price = float(price_str)
        if not (wick_low <= price <= wick_high):
            continue
        volume_at_level = sell_qty if direction == "long" else buy_qty
        if volume_at_level >= required:
            return True

    return False


def find_absorption_trigger(bar: pd.Series, baseline: float, direction: str, params: dict) -> tuple:
    """
    Returns (trigger_price, trigger_volume) for the first wick price level that
    crosses baseline * absorption_mult. Returns (None, None) if none found or on error.
    Use after is_absorption_candle has already confirmed absorption.
    """
    tv = bar.get("tick_volume", None)
    if not tv or tv == "{}":
        return None, None

    try:
        raw = json.loads(tv)
    except Exception:
        return None, None

    required = baseline * params["absorption_mult"]

    if direction == "long":
        wick_low  = float(bar["low"])
        wick_high = min(float(bar["open"]), float(bar["close"]))
    else:
        wick_low  = max(float(bar["open"]), float(bar["close"]))
        wick_high = float(bar["high"])

    for price_str, (buy_qty, sell_qty) in raw.items():
        price = float(price_str)
        if not (wick_low <= price <= wick_high):
            continue
        volume_at_level = sell_qty if direction == "long" else buy_qty
        if volume_at_level >= required:
            return price, volume_at_level

    return None, None
