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

CVD arrives positionally aligned on the day context (day.cvd / day.cvd_std); both are None when
no indicators were loaded for the day => this finder disables itself and returns all-None,
never blocking the other finders.

Returns (entry_rel, entry_price, invalidation_rel, entry_notes, trade_type).
"""

import numpy as np

from .._daydata import prev_rolling_max, prev_rolling_min

_NO_ENTRY = (None, None, None, None, None)


def find_entry(win, params: dict) -> tuple:
    day = win.day
    # fail-safe: no CVD for this day (or empty window) => finder disabled
    if day.cvd is None or day.cvd_std is None or win.n == 0:
        return _NO_ENTRY

    n = win.n
    first_inv = win.first_inv
    if first_inv == 0:                   # breakout bar: invalidation only, never a pivot
        return None, None, 0, None, None

    o, h, l, c = win.o, win.h, win.l, win.c
    long_ = win.direction == "long"

    # --- stage 1: vectorized candidate pivots (left-side k-bar fractal) ---
    k = params["cvd_pivot_k"]
    if long_:
        cand_piv    = l < prev_rolling_min(l, k)     # strictly < each of prev k lows
        conf_candle = c > o                          # bullish reversal confirms a low pivot
    else:
        cand_piv    = h > prev_rolling_max(h, k)     # strictly > each of prev k highs
        conf_candle = c < o                          # bearish reversal confirms a high pivot

    cvd     = day.cvd
    cvd_std = day.cvd_std
    pos     = win.pos
    confirm = win.confirm_dir
    K        = params["entry_after_absorption"]
    tol      = params["cvd_wick_tolerance_ticks"] * params["tick_size"]
    min_sep  = params["cvd_min_separation"]
    max_sep  = params["cvd_max_separation"]
    min_score = params["cvd_min_score"]

    # --- stage 2: forward scan maintaining the last-two confirmed pivots + active setup ---
    pivots: list[tuple] = []             # (idx, price, cvd)
    active = None                        # (p1, p2, score, std, conf_idx)

    for i in range(1, n):
        # invalidation first
        if i == first_inv:
            return None, None, i, None, None

        # new confirmed pivot? candidate bar = i-1, confirmed by the reversal candle at i
        if cand_piv[i - 1] and conf_candle[i]:
            # true swing extreme = the more-extreme of candidate bar (i-1) and confirming
            # bar (i); ties go to the candidate
            if long_:
                pivot_idx   = i - 1 if l[i - 1] <= l[i] else i
                pivot_price = float(l[pivot_idx])
            else:
                pivot_idx   = i - 1 if h[i - 1] >= h[i] else i
                pivot_price = float(h[pivot_idx])

            pivot_cvd = cvd[pos[pivot_idx]]
            if not np.isnan(pivot_cvd):
                pivots.append((pivot_idx, pivot_price, pivot_cvd))

                # a new confirmed pivot supersedes any pending setup; roll the pair forward
                active = None
                if len(pivots) >= 2:
                    p1, p2 = pivots[-2], pivots[-1]
                    sep = p2[0] - p1[0]
                    if min_sep <= sep <= max_sep:
                        # price condition (lower/equal high, or higher/equal low) + tolerance
                        price_ok = (p2[1] >= p1[1] - tol) if long_ else (p2[1] <= p1[1] + tol)
                        if price_ok:
                            std = cvd_std[pos[p2[0]]]
                            if not np.isnan(std) and std > 0:
                                score = (p2[2] - p1[2]) / std
                                # CVD rose into a lower/equal high (short) /
                                # fell into a higher/equal low (long)
                                score_ok = (score <= -min_score) if long_ else (score >= min_score)
                                if score_ok:
                                    active = (p1, p2, score, std, i)

        # entry-candle scan for the active setup (this bar = candidate entry candle)
        if active is not None:
            if i >= active[4] + K:
                active = None                        # window elapsed, abandon setup
            elif confirm[i]:
                entry_rel = i + 1
                if entry_rel >= n:
                    return _NO_ENTRY
                if entry_rel >= first_inv:
                    return None, None, entry_rel, None, None

                p1, p2, score, std, _ = active
                entry_notes = {
                    "cvd_pivot1_time":  win.ts(p1[0]).strftime("%H:%M"),
                    "cvd_pivot1_price": round(p1[1], 2),
                    "cvd_pivot1_cvd":   round(float(p1[2]), 2),
                    "cvd_pivot2_time":  win.ts(p2[0]).strftime("%H:%M"),
                    "cvd_pivot2_price": round(p2[1], 2),
                    "cvd_pivot2_cvd":   round(float(p2[2]), 2),
                    "cvd_score":        round(float(score), 2),
                    "cvd_change_std":   round(float(std), 2),
                }
                return (entry_rel, float(o[entry_rel]), None, entry_notes,
                        "cvd_divergence_absorption")

    return _NO_ENTRY
