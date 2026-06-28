"""CVD divergence (absorption) entry.

A cumulative-volume-delta divergence at a price extreme, read as absorption:

  - SHORT bias (absorption at highs): a 2nd pivot high that is lower-or-equal to the 1st,
    yet CVD at the 2nd high is *higher* — more net buying went in but price could not make a
    higher high => buyers absorbed => short.
  - LONG bias (absorption at lows): a 2nd pivot low that is higher-or-equal to the 1st, yet
    CVD at the 2nd low is *lower* — more net selling but price held => sellers absorbed => long.

Pivots use a left-side k-bar fractal, confirmed fast by a single reversal candle (no waiting k
bars on the right). The divergence between the two most-recent confirmed pivots is graded with a
z-score that normalises the CVD difference by the session's own CVD-change volatility. A confirmed
3rd pivot supersedes a pending setup (rolls the sliding pair forward).

CVD is supplied (already reindexed to the session) via `cvd_series`; its rolling change-std via
`cvd_change_std`. Both are None when no indicators were loaded for the day => this finder disables
itself and returns all-None, never blocking the other finders.

Returns (entry_ts, entry_price, invalidation_ts, entry_notes, trade_type).
"""

import pandas as pd


def _test_divergence(p1: dict, p2: dict, direction: str, params: dict, cvd_change_std: pd.Series):
    """Grade the (P1 older, P2 newer) pivot pair. Returns a setup dict or None."""
    # separation: too close => not a pair yet; too far => P1 is stale, drop it
    sep = p2["idx"] - p1["idx"]
    if sep < params["cvd_min_separation"] or sep > params["cvd_max_separation"]:
        return None

    # price condition (lower/equal high, or higher/equal low) with a small wick tolerance
    tol = params["cvd_wick_tolerance_ticks"] * params["tick_size"]
    if direction == "short":
        if not (p2["price"] <= p1["price"] + tol):
            return None
    else:
        if not (p2["price"] >= p1["price"] - tol):
            return None

    # z-score: normalise the CVD difference by the session's CVD-change volatility at P2
    std = cvd_change_std.get(p2["ts"], float("nan"))
    if pd.isna(std) or std <= 0:
        return None

    score = (p2["cvd"] - p1["cvd"]) / std
    if direction == "short":
        if not (score >= params["cvd_min_score"]):       # CVD rose into a lower/equal high
            return None
    else:
        if not (score <= -params["cvd_min_score"]):      # CVD fell into a higher/equal low
            return None

    return {"p1": p1, "p2": p2, "score": score, "std": std}


def _is_entry_candle(bar: pd.Series, direction: str, params: dict) -> bool:
    """Confirmation/entry candle: correct direction + body_threshold + delta_threshold."""
    bar_range = float(bar["high"]) - float(bar["low"])
    if bar_range <= 0:
        return False

    body = abs(float(bar["close"]) - float(bar["open"]))
    if (body / bar_range) < params["body_threshold"]:
        return False

    if direction == "long":
        if float(bar["close"]) <= float(bar["open"]):
            return False
        return float(bar["volume_delta_pct"]) >= params["delta_threshold"]
    else:
        if float(bar["close"]) >= float(bar["open"]):
            return False
        return float(bar["volume_delta_pct"]) <= -params["delta_threshold"]


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
    # fail-safe: no CVD for this day (or empty window) => finder disabled
    if cvd_series is None or cvd_change_std is None or post_retest.empty:
        return None, None, None, None, None

    if direction == "long":
        invalid_whole = post_retest["close"] < val
    else:
        invalid_whole = post_retest["close"] > vah

    n = len(post_retest)

    # breakout bar (index 0): invalidation only, never a pivot
    if invalid_whole.iloc[0]:
        return None, None, post_retest.index[0], None, None

    # --- stage 1: vectorized candidate pivots (left-side k-bar fractal) ---
    k = params["cvd_pivot_k"]
    prev_high_max = post_retest["high"].rolling(k).max().shift(1)
    prev_low_min  = post_retest["low"].rolling(k).min().shift(1)
    cand_high = (post_retest["high"] > prev_high_max).values   # strictly > each of prev k highs
    cand_low  = (post_retest["low"]  < prev_low_min).values    # strictly < each of prev k lows
    bearish   = (post_retest["close"] < post_retest["open"]).values
    bullish   = (post_retest["close"] > post_retest["open"]).values

    # --- stage 2: forward scan maintaining the last-two confirmed pivots + active setup ---
    pivots: list[dict] = []
    active: dict | None = None

    for i in range(1, n):
        ts = post_retest.index[i]

        # invalidation first
        if invalid_whole.iloc[i]:
            return None, None, ts, None, None

        # new confirmed pivot? candidate bar p = i-1, confirmed by the reversal candle at i
        if direction == "short":
            new_pivot = cand_high[i - 1] and bearish[i]
        else:
            new_pivot = cand_low[i - 1] and bullish[i]

        if new_pivot:
            # the true swing extreme is the more-extreme of the candidate bar (i-1) and its
            # confirming candle (i): the higher high (short) / lower low (long) — the confirming
            # candle can wick past the candidate before reversing. Ties go to the candidate.
            if direction == "short":
                pivot_idx = i - 1 if float(post_retest.iloc[i - 1]["high"]) >= float(post_retest.iloc[i]["high"]) else i
                pivot_price = float(post_retest.iloc[pivot_idx]["high"])
            else:
                pivot_idx = i - 1 if float(post_retest.iloc[i - 1]["low"]) <= float(post_retest.iloc[i]["low"]) else i
                pivot_price = float(post_retest.iloc[pivot_idx]["low"])

            pivot_ts  = post_retest.index[pivot_idx]
            pivot_cvd = cvd_series.get(pivot_ts, float("nan"))
            if not pd.isna(pivot_cvd):
                pivots.append({"idx": pivot_idx, "ts": pivot_ts, "price": pivot_price, "cvd": pivot_cvd})

                # a new confirmed pivot supersedes any pending setup; roll the pair forward
                active = None
                if len(pivots) >= 2:
                    setup = _test_divergence(pivots[-2], pivots[-1], direction, params, cvd_change_std)
                    if setup is not None:
                        # entry scan starts at this confirmation candle (inclusive)
                        setup["conf_idx"] = i
                        active = setup

        # entry-candle scan for the active setup (this bar = candidate entry candle)
        if active is not None:
            if i >= active["conf_idx"] + params["entry_after_absorption"]:
                active = None                                # window elapsed, abandon setup
            elif _is_entry_candle(post_retest.iloc[i], direction, params):
                entry_bar_idx = i + 1
                if entry_bar_idx >= n:
                    return None, None, None, None, None
                if invalid_whole.iloc[entry_bar_idx]:
                    return None, None, post_retest.index[entry_bar_idx], None, None

                entry_ts    = post_retest.index[entry_bar_idx]
                entry_price = float(post_retest.iloc[entry_bar_idx]["open"])

                p1, p2 = active["p1"], active["p2"]
                entry_notes = {
                    "cvd_pivot1_time":  p1["ts"].strftime("%H:%M"),
                    "cvd_pivot1_price": round(p1["price"], 2),
                    "cvd_pivot1_cvd":   round(float(p1["cvd"]), 2),
                    "cvd_pivot2_time":  p2["ts"].strftime("%H:%M"),
                    "cvd_pivot2_price": round(p2["price"], 2),
                    "cvd_pivot2_cvd":   round(float(p2["cvd"]), 2),
                    "cvd_score":        round(float(active["score"]), 2),
                    "cvd_change_std":   round(float(active["std"]), 2),
                }

                return entry_ts, entry_price, None, entry_notes, "cvd_divergence_absorption"

    return None, None, None, None, None
