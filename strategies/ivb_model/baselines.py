"""Day-level rolling baselines for absorption / passive / CVD-change grading."""

import json
import pandas as pd
from datetime import time

# Skip the first 5 minutes of RTH before baselining — the 09:30 open bar(s) are abnormally heavy
# and would distort the rolling volume-per-tick averages. Shared by all baselines below.
BASELINE_WARMUP_START = time(9, 35)


def build_rolling_baseline(rth_session: pd.DataFrame, params: dict):
    """Volume-per-tick rolling baseline for absorption detection.

    Returns (sell_baseline, buy_baseline), each a Series reindexed to rth_session.
    """
    tick_size = params["tick_size"]
    window    = params["absorption_baseline_window"]

    valid = rth_session[rth_session.index.time >= BASELINE_WARMUP_START].copy()

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

    valid = rth_session[rth_session.index.time >= BASELINE_WARMUP_START].copy()

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


def build_cvd_change_baseline(cumulative_delta: pd.Series, params: dict) -> pd.Series:
    """Rolling std of bar-to-bar CVD changes, reindexed to the session index.

    `cumulative_delta` must already be aligned to rth_session.index. Mirrors the other
    baselines: filter to >= BASELINE_WARMUP_START, rolling(window, min_periods=window),
    NaN during warmup.
    """
    window = params["absorption_baseline_window"]

    valid = cumulative_delta[cumulative_delta.index.time >= BASELINE_WARMUP_START]
    std   = valid.diff().rolling(window, min_periods=window).std()

    return std.reindex(cumulative_delta.index)
