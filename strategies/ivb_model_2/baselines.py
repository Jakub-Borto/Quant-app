"""Rolling baselines for absorption detection and merged-candle helpers."""

import json
import pandas as pd
from datetime import time


def build_rolling_baseline(rth_session: pd.DataFrame, params: dict):
    """Volume-per-tick rolling baseline for absorption detection.

    Returns (sell_baseline, buy_baseline), each a Series reindexed to rth_session.
    """
    tick_size = params["tick_size"]
    window    = params["absorption_baseline_window"]

    valid = rth_session[rth_session.index.time >= time(9, 35)].copy()

    range_ticks = ((valid["high"] - valid["low"]) / tick_size).replace(0, float("nan"))

    valid["_sell_per_tick"] = valid["sell_volume"] / range_ticks
    valid["_buy_per_tick"]  = valid["buy_volume"]  / range_ticks

    sell_baseline = valid["_sell_per_tick"].rolling(window, min_periods=window).mean()
    buy_baseline  = valid["_buy_per_tick"].rolling(window, min_periods=window).mean()

    sell_baseline = sell_baseline.reindex(rth_session.index)
    buy_baseline  = buy_baseline.reindex(rth_session.index)

    return sell_baseline, buy_baseline


def build_passive_baseline(rth_session: pd.DataFrame, direction: str, params: dict) -> pd.Series:
    """
    Rolling baseline of the max raw resting size on the defended side per bar.
    Long: passive buy orders below candle open.
    Short: passive sell orders above candle open.
    Only bars with valid passive data on the defended side contribute to the window.
    """
    window = params["absorption_baseline_window"]

    valid = rth_session[rth_session.index.time >= time(9, 35)].copy()

    def max_order_size(row):
        po = row.get("passive_orders", None)
        if not po or po == "{}":
            return float("nan")
        try:
            raw = json.loads(po)
        except Exception:
            return float("nan")

        bar_open = float(row["open"])
        best     = float("nan")

        for price_str, (size, count) in raw.items():
            if count <= 0:
                continue
            price = float(price_str)
            if direction == "long" and price >= bar_open:
                continue
            if direction == "short" and price <= bar_open:
                continue
            if pd.isna(best) or size > best:
                best = size

        return best

    per_bar = valid.apply(max_order_size, axis=1)
    sparse  = per_bar.dropna()
    rolling = sparse.rolling(window, min_periods=window).mean()
    return rolling.reindex(rth_session.index)


def merge_tick_volume(tv1: str, tv2: str) -> str:
    """Merge two tick_volume JSON strings by summing volumes at each price level."""
    merged = {}
    for tv in (tv1, tv2):
        if not tv or tv == "{}":
            continue
        try:
            raw = json.loads(tv)
        except Exception:
            continue
        for price_str, (buy_qty, sell_qty) in raw.items():
            if price_str not in merged:
                merged[price_str] = [0, 0]
            merged[price_str][0] += buy_qty
            merged[price_str][1] += sell_qty
    return json.dumps({k: v for k, v in merged.items()})


def build_two_bar_baseline(pre_bars: pd.DataFrame, direction: str, params: dict) -> float:
    """
    Build baseline from merged 2-bar pairs ending at the last bar of pre_bars.
    Window = absorption_window // 2 pairs. Anchored backwards from the pattern bars.
    """
    tick_size = params["tick_size"]
    n_pairs   = params["absorption_baseline_window"] // 2

    bars = pre_bars.iloc[::-1].reset_index(drop=True)
    n    = len(bars)
    densities = []

    for k in range(n_pairs):
        i1 = k * 2
        i2 = k * 2 + 1
        if i2 >= n:
            break

        bar1 = bars.iloc[i1]
        bar2 = bars.iloc[i2]

        merged_high  = max(float(bar1["high"]),  float(bar2["high"]))
        merged_low   = min(float(bar1["low"]),   float(bar2["low"]))
        merged_range = merged_high - merged_low
        range_ticks  = merged_range / tick_size

        if range_ticks <= 0:
            continue

        if direction == "long":
            volume = float(bar1["sell_volume"]) + float(bar2["sell_volume"])
        else:
            volume = float(bar1["buy_volume"])  + float(bar2["buy_volume"])

        densities.append(volume / range_ticks)

    if not densities:
        return float("nan")

    return sum(densities) / len(densities)
