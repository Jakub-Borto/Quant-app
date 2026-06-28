"""Passive wall entry: a cluster of big resting passive orders on the defended side.

No absorption candle and no delta confirmation — the wall of stacked liquidity is the signal.
Mirrors consecutive_absorption's price-cluster logic, but counts big passive orders (by raw
resting size) instead of absorption candles. The wall builds across bars: a running `seen` list
of qualifying levels is never cleared.
"""

import json
import pandas as pd


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
    Scans post_retest bar by bar, accumulating big passive orders on the defended side
    (long: resting bids below candle open; short: resting asks above candle open) whose raw
    size >= passive_baseline * passive_wall_mult. When passive_wall_n of them cluster within
    passive_wall_ticks of one level, a wall is confirmed and we enter on the next bar's open.

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

    n          = len(post_retest)
    tick_size  = params["tick_size"]
    level_tol  = params["passive_wall_ticks"] * tick_size
    required_n = params["passive_wall_n"]

    # (price_level, ts) of every big passive order seen so far — spans all bars, never cleared
    seen: list[tuple[float, pd.Timestamp]] = []

    # breakout bar (index 0): invalidation only
    if invalid_whole.iloc[0]:
        return None, None, post_retest.index[0], None, None

    for i in range(1, n):
        bar = post_retest.iloc[i]
        ts  = post_retest.index[i]

        if invalid_whole.iloc[i]:
            return None, None, ts, None, None

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

        bar_open = float(bar["open"])
        required = p_baseline * params["passive_wall_mult"]

        # collect every qualifying defended-side level on this bar (raw size, not size/count)
        new_levels = []
        for price_str, (size, count) in raw_po.items():
            price = float(price_str)
            if count <= 0:
                continue
            if direction == "long" and price >= bar_open:
                continue
            if direction == "short" and price <= bar_open:
                continue
            if size >= required:
                new_levels.append(price)

        # append each, re-clustering by price; trigger at the earliest level that completes a wall
        for lvl in new_levels:
            seen.append((lvl, ts))

            nearby = [(p, t) for p, t in seen if abs(p - lvl) <= level_tol]
            if len(nearby) < required_n:
                continue

            entry_bar_idx = i + 1
            if entry_bar_idx >= n:
                return None, None, None, None, None

            if invalid_whole.iloc[entry_bar_idx]:
                return None, None, post_retest.index[entry_bar_idx], None, None

            entry_ts    = post_retest.index[entry_bar_idx]
            entry_price = float(post_retest.iloc[entry_bar_idx]["open"])

            entry_notes = {
                "wall_levels": [round(p, 2) for p, _ in nearby],
                "wall_times":  [t.strftime("%H:%M") for _, t in nearby],
                "wall_count":  len(nearby),
            }

            return entry_ts, entry_price, None, entry_notes, "passive_wall"

    return None, None, None, None, None
