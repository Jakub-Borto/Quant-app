"""VWAP-target risk script with a signal-driven trailing stop.

Same parameters and TP behaviour as vwap_tp_risk (sl_placement / vwap_std /
vwap_session / vwap_tp_mode, incl. the entry-time 2σ->3σ escalation and the 1:1 fallback), plus:
while the trade is open, the seven entry-style signals are re-detected on the live bars
(gated by the `trailing_entries` bit string, same order as `valid_entries`). When a signal is
confirmed by a candle meeting BOTH `body_threshold` and `delta_threshold` (in the trade
direction), the stop ratchets to the signal candle's extreme (low for longs / high for shorts)
starting from the bar AFTER the confirming candle. The stop only ever tightens.

Two switches shape the signal log:
  trailing_in_profit (default True) — True: a signal whose candidate stop is still in loss
    (below entry for longs / above for shorts) is not even logged; False: every signal is
    logged and may trail, loss or not.
  late_trailing (default False) — True: each logged signal trails the stop to the PREVIOUS
    logged signal's level (the first logged signal only arms the log); False: a signal trails
    to its own level immediately.

Unlike the entry finders, EVERY trailing signal requires the confirming candle (also
consecutive_absorption and passive_wall, which enter without one), and there is no VAL/VAH
invalidation — the stop manages the exit. A stop hit at a trailed level reports
exit_reason = "trailing_sl" with the trailed level as exit_price; the `sl` column always keeps
the originally placed stop.

Self-contained per the risk-script convention: own copies of _zone_sl / _swing_sl, both fill simulators and
the 2-bar/CVD helpers; the only package imports are the shared absorption grader the entry
finders use themselves and the timing/context plumbing. The detectors run on the positional
TradeWindow (pre-parsed JSON, shared confirm-candle mask, O(n) strided two-bar baseline) —
the per-bar DataFrame row access and repeated json.loads of the original are gone.

INDICATORS REQUIRED: if the VWAP bands are unavailable for the day (no indicators / missing
columns / NaN at entry), this script returns None (no trade), exactly like vwap_tp_risk. The two
CVD trail detectors additionally need day.cvd / day.cvd_std and disable themselves when those
are absent.

exit_reason: tp / sl / eod (+ tp_timeout / sl_timeout) + trailing_sl. risk_notes records
tp_type / escalated / trail_count plus flat trailN_* keys per applied trail (trailN_time /
trailN_type / trailN_stop and the same fields that finder's entry_notes would carry —
absorption/trigger/passive/wall/CVD-pivot details — plus trailN_trigger_type/_trigger_time for
late trails), so every value renders as a plain note tile.
"""

import numpy as np

from .._timing    import timed
from .._daydata   import prev_rolling_max, prev_rolling_min
from ..absorption import absorption_scan, wick_bounds


def _zone_sl(entry_win, entry_pos, direction, levels):
    """Pick the stop from the pullback window's extremes vs the VAL/POC/VAH zones."""
    poc = levels["poc"]
    vah = levels["vah"]
    val = levels["val"]

    # --- SL window: drop the breakout bar (index 0), keep retest .. bar before entry ---
    # entry is taken at the entry bar's OPEN, so that bar's low/close are future data => excluded.
    m = entry_win.pos[1:] < entry_pos

    # degenerate: nothing to measure => fall back to the basic VAL/VAH stop
    if not m.any():
        return val if direction == "long" else vah

    if direction == "long":
        lowest_close = float(entry_win.c[1:][m].min())   # where the pullback bottomed (by close)
        lowest_low   = float(entry_win.l[1:][m].min())   # how far the wick reached

        if poc <= lowest_close <= vah:                 # bottomed in the UPPER zone
            return poc if lowest_low >= poc else lowest_low
        elif val <= lowest_close < poc:                # bottomed in the LOWER zone
            return val if lowest_low >= val else lowest_low
        else:                                          # shouldn't occur (pullback re-enters VA)
            return val

    else:
        highest_close = float(entry_win.c[1:][m].max())  # where the pullback topped (by close)
        highest_high  = float(entry_win.h[1:][m].max())  # how far the wick reached

        if val <= highest_close <= poc:                # topped in the LOWER zone
            return poc if highest_high <= poc else highest_high
        elif poc < highest_close <= vah:               # topped in the UPPER zone
            return vah if highest_high <= vah else highest_high
        else:                                          # shouldn't occur (pullback re-enters VA)
            return vah


def _swing_sl(entry_win, entry_pos, direction, levels):
    """Swing stop over the post_retest bars up to and including the entry bar
    (same rule as basic_risk's "swing_low"); VAL/VAH fallback when empty."""
    m = entry_win.pos <= entry_pos
    if not m.any():
        return levels["val"] if direction == "long" else levels["vah"]
    return float(entry_win.l[m].min()) if direction == "long" \
      else float(entry_win.h[m].max())


# ---------------------------------------------------------------------------
# 2-bar helpers (own copies per the risk-script convention)
# ---------------------------------------------------------------------------

def _merge_tv(tv1, tv2) -> dict:
    """Merge two pre-parsed tick_volume tuples into {price: [buy, sell]}, summing volumes.
    Insertion order (bar1's levels, then bar2's unseen levels) matches the original
    JSON-string merge, so the first-qualifying-level scan picks the same level."""
    merged: dict = {}
    for tv in (tv1, tv2):
        if tv is None:
            continue
        prices, buys, sells = tv
        pl, bl, sl_ = prices.tolist(), buys.tolist(), sells.tolist()
        for k in range(len(pl)):
            p = pl[k]
            e = merged.get(p)
            if e is None:
                merged[p] = [bl[k], sl_[k]]
            else:
                e[0] += bl[k]
                e[1] += sl_[k]
    return merged


def _pair_densities(h, l, vol, tick_size: float) -> np.ndarray:
    """density[j] of the merged pair (bars j, j+1): summed counter-volume / merged range in
    ticks; NaN where the merged range is 0 (the original skipped those pairs)."""
    vol_pair = vol[:-1] + vol[1:]
    rng = (np.maximum(h[:-1], h[1:]) - np.minimum(l[:-1], l[1:])) / tick_size
    with np.errstate(divide="ignore", invalid="ignore"):
        d = vol_pair / rng
    d[rng <= 0] = np.nan
    return d


def _baseline_from_densities(dens: np.ndarray, i: int, n_pairs: int) -> float:
    """Mean pair density anchored backwards from pattern bar i over the post-entry bars
    before it: pairs (i-1,i-2), (i-3,i-4), ... — dens[i-2], dens[i-4], ... down to j >= 0,
    first n_pairs of them, NaN pairs skipped. Summation order matches the original loop."""
    js = np.arange(i - 2, -1, -2)[:n_pairs]
    if js.size == 0:
        return float("nan")
    vals = dens[js]
    vals = vals[~np.isnan(vals)].tolist()
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Trail-event detection (mirrors the entry finders, minus invalidation,
# collecting EVERY event instead of returning at the first one)
# ---------------------------------------------------------------------------

def _trail_absorption_delta(tw, params):
    """Absorption candle + confirming candle. Stop = the absorption candle's extreme."""
    n = tw.n
    o, h, l, c = tw.o, tw.h, tw.l, tw.c
    long_   = tw.direction == "long"
    confirm = tw.confirm_dir
    K       = params["entry_after_absorption"]
    mult    = params["absorption_mult"]

    cand = (tw.wick_frac >= params["wick_threshold"]) & (tw.abs_base > 0)

    events = []
    for i in map(int, np.flatnonzero(cand)):
        baseline = float(tw.abs_base[i])
        wl, wh   = wick_bounds(o[i], h[i], l[i], c[i], tw.direction)
        found, trigger_price, trigger_volume = absorption_scan(
            tw.tv[i], wl, wh, baseline * mult, tw.direction
        )
        if not found:
            continue

        absorption_level = float(l[i]) if long_ else float(h[i])

        scan_end = min(i + 1 + K, n)
        for j in range(i + 1, scan_end):
            # close back through the absorption level kills this signal (as in the entry finder)
            if long_:
                if c[j] < absorption_level:
                    break
            else:
                if c[j] > absorption_level:
                    break

            if confirm[j]:
                events.append({"confirm_idx": j, "stop": absorption_level,
                               "name": "absorption_delta",
                               "notes": {
                                   "absorption_time": tw.ts(i).strftime("%H:%M"),
                                   "abs_baseline":    round(baseline, 2),
                                   "trigger_price":   trigger_price,
                                   "trigger_volume":  trigger_volume,
                               }})
                break

    return events


def _trail_consecutive_absorption(tw, params):
    """N clustered absorption candles + confirming candle (confirmation is trailing-only —
    the entry finder needs none). Stop = the nth (triggering) candle's extreme."""
    n = tw.n
    o, h, l, c = tw.o, tw.h, tw.l, tw.c
    long_      = tw.direction == "long"
    confirm    = tw.confirm_dir
    K          = params["entry_after_absorption"]
    tick_size  = params["tick_size"]
    level_tol  = params["consec_abs_ticks"] * tick_size
    required_n = params["consec_abs_n"]
    mult       = params["consec_abs_mult"]

    cand = (tw.wick_frac >= params["consec_wick_threshold"]) & (tw.abs_base > 0)

    # an absorption level price has since CLOSED through is dead (mirrors the entry finder):
    # long => any later close strictly below the level runs it out; closing AT it keeps it.
    # The original pruned bar-by-bar; the min/max close over the interval since the last
    # visited candidate kills exactly the same levels.
    seen: list[tuple] = []               # (abs_level, body_mid, bar index)
    last = -1
    events = []

    for i in map(int, np.flatnonzero(cand)):
        if seen:
            if long_:
                worst = c[last + 1 : i + 1].min()
                seen = [e for e in seen if worst >= e[0]]
            else:
                worst = c[last + 1 : i + 1].max()
                seen = [e for e in seen if worst <= e[0]]
        last = i

        baseline = float(tw.abs_base[i])
        wl, wh   = wick_bounds(o[i], h[i], l[i], c[i], tw.direction)
        found, trigger_price, trigger_volume = absorption_scan(
            tw.tv[i], wl, wh, baseline * mult, tw.direction
        )
        if not found:
            continue

        abs_level = float(l[i]) if long_ else float(h[i])
        body_mid  = (
            min(float(o[i]), float(c[i])) +
            max(float(o[i]), float(c[i]))
        ) / 2

        seen.append((abs_level, body_mid, i))

        nearby = [
            (lvl, t) for lvl, bm, t in seen
            if abs(lvl - abs_level) <= level_tol
            and abs(bm - body_mid) <= level_tol
        ]

        if len(nearby) < required_n:
            continue

        # trigger reflects the nth (triggering) candle, as in the entry finder
        scan_end = min(i + 1 + K, n)
        for j in range(i + 1, scan_end):
            if confirm[j]:
                events.append({"confirm_idx": j, "stop": abs_level,
                               "name": "consecutive_absorption",
                               "notes": {
                                   "absorption_time": [tw.ts(t).strftime("%H:%M") for _, t in nearby],
                                   "abs_baseline":    round(baseline, 2),
                                   "trigger_price":   trigger_price,
                                   "trigger_volume":  trigger_volume,
                               }})
                break

    return events


def _trail_two_bar_absorption(tw, params):
    """Reversal pair merged into a synthetic candle + confirming candle (baseline pairs come
    from the post-entry bars before the pattern). Stop = the merged candle's extreme."""
    n = tw.n
    o, h, l, c = tw.o, tw.h, tw.l, tw.c
    vdp        = tw.vdp
    long_      = tw.direction == "long"
    confirm    = tw.confirm_dir
    K          = params["entry_after_absorption"]
    tick_size  = params["tick_size"]
    max_wick   = params["two_bar_wick_ticks"] * tick_size
    req_mult   = params["two_bar_abs_mult"]
    n_pairs    = params["absorption_baseline_window"] // 2

    if long_:
        wick = np.minimum(o, c) - l
        bar1_ok = (c < o) & (wick <= max_wick)
        bar2_ok = (c > o) & (wick <= max_wick) & (vdp > 0)
    else:
        wick = h - np.maximum(o, c)
        bar1_ok = (c > o) & (wick <= max_wick)
        bar2_ok = (c < o) & (wick <= max_wick) & (vdp < 0)

    if n < 2:
        return []
    pair = bar1_ok[:-1] & bar2_ok[1:]

    dens = None                          # lazy: only built when a pattern candidate exists
    events = []

    for i in map(int, np.flatnonzero(pair)):
        # --- 2-bar baseline from the post-entry bars before the pair (bar 0 included) ---
        if dens is None:
            vol  = tw.day.sell_vol[tw.start:] if long_ else tw.day.buy_vol[tw.start:]
            dens = _pair_densities(h, l, vol, tick_size)
        baseline = _baseline_from_densities(dens, i, n_pairs)

        if np.isnan(baseline) or baseline <= 0:
            continue

        required = baseline * req_mult

        # --- build merged synthetic candle ---
        merged_open  = float(o[i])
        merged_high  = float(max(h[i], h[i + 1]))
        merged_low   = float(min(l[i], l[i + 1]))
        merged_close = float(c[i + 1])

        # --- absorption in defended wick of merged candle ---
        if long_:
            wick_low  = merged_low
            wick_high = min(merged_open, merged_close)
        else:
            wick_low  = max(merged_open, merged_close)
            wick_high = merged_high

        if wick_high <= wick_low:
            continue

        # absorption must register in the defended half of the full merged candle
        mid_price = (merged_low + merged_high) / 2.0

        merged = _merge_tv(tw.tv[i], tw.tv[i + 1])

        absorbed       = False
        trigger_price  = None
        trigger_volume = None

        for price, (buy_qty, sell_qty) in merged.items():
            if not (wick_low <= price <= wick_high):
                continue
            if long_ and price > mid_price:
                continue
            if (not long_) and price < mid_price:
                continue
            volume_at_level = sell_qty if long_ else buy_qty
            if volume_at_level >= required:
                absorbed       = True
                trigger_price  = price
                trigger_volume = volume_at_level
                break

        if not absorbed:
            continue

        stop = merged_low if long_ else merged_high

        # --- confirming candle scan starting from bar2 (inclusive), as in the entry finder ---
        scan_end = min(i + 1 + K, n)
        for j in range(i + 1, scan_end):
            if confirm[j]:
                events.append({"confirm_idx": j, "stop": stop,
                               "name": "two_bar_absorption",
                               "notes": {
                                   "absorption_time":  [tw.ts(i).strftime("%H:%M"),
                                                        tw.ts(i + 1).strftime("%H:%M")],
                                   "two_bar_baseline": round(baseline, 2),
                                   "trigger_price":    trigger_price,
                                   "trigger_volume":   trigger_volume,
                               }})
                break

    return events


def _trail_passive_size(tw, params):
    """Big resting order (raw size) + absorption on the same candle + confirming candle.
    Stop = the signal candle's extreme."""
    n = tw.n
    o, h, l, c = tw.o, tw.h, tw.l, tw.c
    long_      = tw.direction == "long"
    confirm    = tw.confirm_dir
    K          = params["entry_after_absorption"]
    order_mult = params["passive_size_order_mult"]
    abs_mult   = params["passive_size_absorption_mult"]

    with np.errstate(invalid="ignore"):
        cand = (
            (tw.p_base > 0)
            & (tw.best_passive >= tw.p_base * order_mult)
            & (tw.wick_frac >= params["passive_size_wick_threshold"])
            & (tw.abs_base > 0)
        )

    events = []
    for i in map(int, np.flatnonzero(cand)):
        # --- passive order check (raw size only): first qualifying defended-side level ---
        prices, sizes, counts = tw.po[i]
        required_po = float(tw.p_base[i]) * order_mult
        if long_:
            m = (counts > 0) & (prices < o[i]) & (sizes >= required_po)
        else:
            m = (counts > 0) & (prices > o[i]) & (sizes >= required_po)
        if not m.any():
            continue
        k = int(m.argmax())
        passive_trigger_price = float(prices[k])
        passive_trigger_size  = sizes[k].item()
        passive_trigger_count = counts[k].item()

        # --- absorption check on the same candle ---
        baseline = float(tw.abs_base[i])
        wl, wh   = wick_bounds(o[i], h[i], l[i], c[i], tw.direction)
        found, _, _ = absorption_scan(tw.tv[i], wl, wh, baseline * abs_mult, tw.direction)
        if not found:
            continue

        stop = float(l[i]) if long_ else float(h[i])

        scan_end = min(i + 1 + K, n)
        for j in range(i + 1, scan_end):
            if confirm[j]:
                events.append({"confirm_idx": j, "stop": stop,
                               "name": "passive_absorption_size_only",
                               "notes": {
                                   "absorption_time": tw.ts(i).strftime("%H:%M"),
                                   "passive_price":   passive_trigger_price,
                                   "passive_size":    passive_trigger_size,
                                   "passive_count":   passive_trigger_count,
                               }})
                break

    return events


def _trail_passive_wall(tw, params):
    """Cluster of big defended-side passive orders + confirming candle (confirmation is
    trailing-only — the entry finder needs none). Stop = the wall-completing bar's extreme."""
    n = tw.n
    o, h, l   = tw.o, tw.h, tw.l
    long_      = tw.direction == "long"
    confirm    = tw.confirm_dir
    K          = params["entry_after_absorption"]
    tick_size  = params["tick_size"]
    level_tol  = params["passive_wall_ticks"] * tick_size
    required_n = params["passive_wall_n"]
    wall_mult  = params["passive_wall_mult"]

    with np.errstate(invalid="ignore"):
        cand = (tw.p_base > 0) & (tw.best_passive >= tw.p_base * wall_mult)

    # every big passive level (price, bar index) — spans all bars, never cleared
    seen: list[tuple] = []
    events = []

    for i in map(int, np.flatnonzero(cand)):
        prices, sizes, counts = tw.po[i]
        required = float(tw.p_base[i]) * wall_mult
        if long_:
            m = (counts > 0) & (prices < o[i]) & (sizes >= required)
        else:
            m = (counts > 0) & (prices > o[i]) & (sizes >= required)
        if not m.any():
            continue

        # append each level; at most one trail event per bar (same stop either way)
        completed = False
        for lvl in prices[m].tolist():
            seen.append((lvl, i))
            if completed:
                continue

            nearby = [(p, t) for p, t in seen if abs(p - lvl) <= level_tol]
            if len(nearby) < required_n:
                continue

            completed = True
            stop = float(l[i]) if long_ else float(h[i])

            scan_end = min(i + 1 + K, n)
            for j in range(i + 1, scan_end):
                if confirm[j]:
                    events.append({"confirm_idx": j, "stop": stop,
                                   "name": "passive_wall",
                                   "notes": {
                                       "wall_levels": [round(p, 2) for p, _ in nearby],
                                       "wall_times":  [tw.ts(t).strftime("%H:%M") for _, t in nearby],
                                       "wall_count":  len(nearby),
                                   }})
                    break

    return events


def _cvd_param(params, mode, name):
    """CVD divergence params are per-flavour: the exhaustion detector reads its own cvd_exh_*
    keys, the absorption detector reads cvd_*. `name` is the unprefixed suffix (pivot_k, etc.)."""
    prefix = "cvd_exh_" if mode == "exhaustion" else "cvd_"
    return params[prefix + name]


def _trail_cvd(tw, params, mode):
    """CVD divergence (absorption or exhaustion flavour, per `mode`) + confirming candle.
    Same left-k fractal pivot machinery as the entry finders, run over the trade bars.
    Stop = the 2nd pivot's price. Disabled when CVD is absent."""
    day = tw.day
    if day.cvd is None or day.cvd_std is None or tw.n == 0:
        return []

    name = "cvd_divergence_absorption" if mode == "absorption" else "cvd_divergence_exhaustion"
    n = tw.n
    o, h, l, c = tw.o, tw.h, tw.l, tw.c
    long_ = tw.direction == "long"

    # --- stage 1: vectorized candidate pivots (left-side k-bar fractal) ---
    k = _cvd_param(params, mode, "pivot_k")
    if long_:
        cand_piv    = l < prev_rolling_min(l, k)
        conf_candle = c > o
    else:
        cand_piv    = h > prev_rolling_max(h, k)
        conf_candle = c < o

    cvd     = day.cvd
    cvd_std = day.cvd_std
    start   = tw.start
    confirm = tw.confirm_dir
    K         = params["entry_after_absorption"]
    tol       = _cvd_param(params, mode, "wick_tolerance_ticks") * params["tick_size"]
    min_sep   = _cvd_param(params, mode, "min_separation")
    max_sep   = _cvd_param(params, mode, "max_separation")
    min_score = _cvd_param(params, mode, "min_score")
    absorption = mode == "absorption"

    # --- stage 2: forward scan maintaining the last-two confirmed pivots + active setup ---
    pivots: list[tuple] = []             # (idx, price, cvd)
    active = None                        # (p1, p2, score, std, conf_idx)
    events = []

    for i in range(1, n):
        if cand_piv[i - 1] and conf_candle[i]:
            # true swing extreme = the more-extreme of candidate bar (i-1) and confirming
            # bar (i); ties go to the candidate
            if long_:
                pivot_idx   = i - 1 if l[i - 1] <= l[i] else i
                pivot_price = float(l[pivot_idx])
            else:
                pivot_idx   = i - 1 if h[i - 1] >= h[i] else i
                pivot_price = float(h[pivot_idx])

            pivot_cvd = cvd[start + pivot_idx]
            if not np.isnan(pivot_cvd):
                pivots.append((pivot_idx, pivot_price, pivot_cvd))

                # a new confirmed pivot supersedes any pending setup; roll the pair forward
                active = None
                if len(pivots) >= 2:
                    p1, p2 = pivots[-2], pivots[-1]
                    sep = p2[0] - p1[0]
                    if min_sep <= sep <= max_sep:
                        if absorption:
                            # price HELD: lower/equal high (short) / higher/equal low (long)
                            price_ok = (p2[1] >= p1[1] - tol) if long_ else (p2[1] <= p1[1] + tol)
                        else:
                            # price DID extend: higher/equal high (short) / lower/equal low (long)
                            price_ok = (p2[1] <= p1[1] + tol) if long_ else (p2[1] >= p1[1] - tol)
                        if price_ok:
                            std = cvd_std[start + p2[0]]
                            if not np.isnan(std) and std > 0:
                                score = (p2[2] - p1[2]) / std
                                if absorption:
                                    score_ok = (score <= -min_score) if long_ else (score >= min_score)
                                else:
                                    score_ok = (score >= min_score) if long_ else (score <= -min_score)
                                if score_ok:
                                    active = (p1, p2, score, std, i)

        # confirming-candle scan for the active setup (this bar = candidate, incl. bar i itself)
        if active is not None:
            if i >= active[4] + K:
                active = None                                # window elapsed, abandon setup
            elif confirm[i]:
                p1, p2, score, std, _ = active
                events.append({"confirm_idx": i, "stop": float(p2[1]),
                               "name": name,
                               "notes": {
                                   "cvd_pivot1_time":  tw.ts(p1[0]).strftime("%H:%M"),
                                   "cvd_pivot1_price": round(p1[1], 2),
                                   "cvd_pivot1_cvd":   round(float(p1[2]), 2),
                                   "cvd_pivot2_time":  tw.ts(p2[0]).strftime("%H:%M"),
                                   "cvd_pivot2_price": round(p2[1], 2),
                                   "cvd_pivot2_cvd":   round(float(p2[2]), 2),
                                   "cvd_score":        round(float(score), 2),
                                   "cvd_change_std":   round(float(std), 2),
                               }})
                active = None                                # one event per setup

    return events


# order = FINDER_REGISTRY order = the trailing_entries bit order
_TRAIL_DETECTORS = [
    ("absorption_delta",             _trail_absorption_delta),
    ("consecutive_absorption",       _trail_consecutive_absorption),
    ("two_bar_absorption",           _trail_two_bar_absorption),
    ("passive_absorption_size_only", _trail_passive_size),
    ("passive_wall",                 _trail_passive_wall),
    ("cvd_divergence_absorption",    lambda tw, p: _trail_cvd(tw, p, "absorption")),
    ("cvd_divergence_exhaustion",    lambda tw, p: _trail_cvd(tw, p, "exhaustion")),
]


def _build_trailing_sl(tw, base_sl, entry_price, params):
    """Detect all trail signals and build the per-bar effective stop.

    The chronological signal log obeys `trailing_in_profit`: when 1, a signal whose candidate
    stop is still in loss (long: below entry_price, short: above) is not even logged; when 0,
    every signal is logged. `late_trailing` shifts the application one signal back: the k-th
    logged signal trails the stop to the (k-1)-th logged signal's level, so the first logged
    signal only arms the log. Either way a signal confirmed on bar j applies from bar j+1 and
    the stop only ever tightens.

    Returns (sl_values, trailed, trail_log):
      sl_values — array: the stop in effect at each bar
      trailed   — bool array: True where the effective stop came from a trail signal
      trail_log — ordered [{eff_ts, type, stop, notes (+trigger_type/trigger_time when
                  late)}] of the trails actually applied
    """
    n     = tw.n
    long_ = tw.direction == "long"
    flags = str(params.get("trailing_entries", "0" * len(_TRAIL_DETECTORS)))
    flags = flags.ljust(len(_TRAIL_DETECTORS), "0")

    in_profit_only = int(params.get("trailing_in_profit", 1)) == 1
    late           = int(params.get("late_trailing", 0)) == 1

    events = []
    for (label, detector), flag in zip(_TRAIL_DETECTORS, flags):
        if flag == "1":
            with timed(f"risk4:trail:{label}"):
                events.extend(detector(tw, params))
    events.sort(key=lambda e: e["confirm_idx"])   # chronological across detectors (stable)

    # --- the signal log: the in-profit filter applies BEFORE logging ---
    log = []
    for e in events:
        if in_profit_only:
            in_profit = e["stop"] >= entry_price if long_ \
                   else e["stop"] <= entry_price
            if not in_profit:
                continue
        log.append(e)

    # bucket by effective bar (confirm_idx + 1); signals confirmed on the last bar never apply
    by_bar: dict = {}
    for k, e in enumerate(log):
        eff = e["confirm_idx"] + 1
        if eff < n:
            by_bar.setdefault(eff, []).append((k, e))

    sl_values = np.full(n, base_sl)
    trailed   = np.zeros(n, dtype=bool)
    trail_log = []
    current         = base_sl
    current_trailed = False

    # the stop only changes at event bars — fill forward from each change point
    for i in sorted(by_bar):
        for k, e in by_bar[i]:
            if late:
                if k == 0:
                    continue          # first logged signal: log only, nothing to trail to yet
                src = log[k - 1]      # trail to the PREVIOUS logged signal's level
            else:
                src = e
            tighter = src["stop"] > current if long_ else src["stop"] < current
            if tighter:
                current         = src["stop"]
                current_trailed = True
                applied = {"eff_ts": tw.index[i],
                           "type":   src["name"],
                           "stop":   src["stop"],
                           "notes":  src.get("notes", {})}
                if late:
                    # the level came from the previous signal; this one pulled the trigger
                    applied["trigger_type"] = e["name"]
                    applied["trigger_time"] = tw.index[e["confirm_idx"]].strftime("%H:%M")
                trail_log.append(applied)
        sl_values[i:] = current
        trailed[i:]   = current_trailed

    return sl_values, trailed, trail_log


# ---------------------------------------------------------------------------
# Fill simulation (vwap_tp_risk's simulators with a per-bar stop)
# ---------------------------------------------------------------------------

def _run_trade(trade_win, entry_ts, entry_price, direction, base_sl,
               sl_series, trail_flags, tp, params) -> dict:
    """Like vwap_tp_risk._run_trade (fixed TP), but the stop is the per-bar sl_series.
    A stop hit at a trailed level (trail_flags True) reports exit_reason = "trailing_sl" and
    exits at the trailed level; the recorded `sl` stays the original base_sl."""
    timeout = params["trade_timeout"]
    n_all   = trade_win.n
    t_end   = min(timeout, n_all)              # len(pre_timeout)
    low     = trade_win.l
    high    = trade_win.h
    index   = trade_win.index

    sl_vals = sl_series[:t_end]
    fl_vals = trail_flags[:t_end]

    if direction == "long":
        sl_hit = low[:t_end]  <= sl_vals
        tp_hit = high[:t_end] >= tp
    else:
        sl_hit = high[:t_end] >= sl_vals
        tp_hit = low[:t_end]  <= tp

    sl_pos = int(sl_hit.argmax()) if sl_hit.any() else None
    tp_pos = int(tp_hit.argmax()) if tp_hit.any() else None

    def make_trade(exit_ts, exit_price, exit_reason, used_sl, used_tp):
        pnl = (exit_price - entry_price) if direction == "long" \
         else (entry_price - exit_price)
        return {
            "direction":   direction,
            "entry_time":  entry_ts,
            "exit_time":   exit_ts,
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "sl":          used_sl,
            "tp":          used_tp,
            "exit_reason": exit_reason,
            "pnl_points":  pnl,
        }

    def sl_exit(pos):
        hit_sl = float(sl_vals[pos])
        reason = "trailing_sl" if bool(fl_vals[pos]) else "sl"
        return make_trade(index[pos], hit_sl, reason, base_sl, tp)

    if sl_pos is not None or tp_pos is not None:
        if sl_pos is None:
            return make_trade(index[tp_pos], tp, "tp", base_sl, tp)
        if tp_pos is None:
            return sl_exit(sl_pos)
        if tp_pos <= sl_pos:
            return make_trade(index[tp_pos], tp, "tp", base_sl, tp)
        else:
            return sl_exit(sl_pos)

    if n_all < timeout:
        return make_trade(index[-1], float(trade_win.c[-1]), "eod", base_sl, tp)

    timeout_close = float(trade_win.c[t_end - 1])
    in_profit     = (timeout_close > entry_price) if direction == "long" \
               else (timeout_close < entry_price)

    if in_profit:
        return make_trade(index[t_end - 1], timeout_close, "tp_timeout", base_sl, tp)

    new_tp   = entry_price
    swing_sl = float(low[:t_end].min())  if direction == "long" \
          else float(high[:t_end].max())

    sl_vals2 = sl_series[timeout:]
    fl_vals2 = trail_flags[timeout:]

    # the swing SL may never loosen the ratchet: keep whichever is tighter per bar
    if direction == "long":
        eff_sl2 = np.maximum(sl_vals2, swing_sl)
        fl2     = fl_vals2 & (sl_vals2 > swing_sl)
    else:
        eff_sl2 = np.minimum(sl_vals2, swing_sl)
        fl2     = fl_vals2 & (sl_vals2 < swing_sl)

    if n_all == timeout:                       # post_timeout empty
        return make_trade(index[-1], float(trade_win.c[-1]), "eod", swing_sl, new_tp)

    if direction == "long":
        sl_hit2 = low[timeout:]  <= eff_sl2
        tp_hit2 = high[timeout:] >= new_tp
    else:
        sl_hit2 = high[timeout:] >= eff_sl2
        tp_hit2 = low[timeout:]  <= new_tp

    sl_pos2 = int(sl_hit2.argmax()) if sl_hit2.any() else None
    tp_pos2 = int(tp_hit2.argmax()) if tp_hit2.any() else None

    def sl_exit2(pos):
        hit_sl = float(eff_sl2[pos])
        reason = "trailing_sl" if bool(fl2[pos]) else "sl_timeout"
        return make_trade(index[timeout + pos], hit_sl, reason, swing_sl, new_tp)

    if sl_pos2 is None and tp_pos2 is None:
        return make_trade(index[-1], float(trade_win.c[-1]), "eod", swing_sl, new_tp)

    if sl_pos2 is None:
        return make_trade(index[timeout + tp_pos2], new_tp, "tp_timeout", swing_sl, new_tp)
    if tp_pos2 is None:
        return sl_exit2(sl_pos2)

    if tp_pos2 <= sl_pos2:
        return make_trade(index[timeout + tp_pos2], new_tp, "tp_timeout", swing_sl, new_tp)
    else:
        return sl_exit2(sl_pos2)


def _run_trade_trailing(trade_win, entry_ts, entry_price, direction, base_sl,
                        sl_series, trail_flags, band, params) -> dict:
    """Like vwap_tp_risk._run_trade_trailing (band TP), but the stop is the per-bar sl_series.

    Same-bar race stays PESSIMISTIC: if the stop and the trailing TP both trigger on one bar, the
    stop wins. Bars with a NaN band value can only trigger the stop. A stop hit at a trailed
    level (trail_flags True) reports exit_reason = "trailing_sl" and exits at the trailed level;
    the recorded `sl` stays the original base_sl."""
    timeout = params["trade_timeout"]
    n_all   = trade_win.n
    t_end   = min(timeout, n_all)              # len(pre_timeout)
    low     = trade_win.l
    high    = trade_win.h
    index   = trade_win.index

    sl_vals = sl_series[:t_end]
    fl_vals = trail_flags[:t_end]
    b       = band[:t_end]
    valid   = ~np.isnan(b)

    if direction == "long":
        sl_hit = low[:t_end]  <= sl_vals
        tp_hit = valid & (high[:t_end] >= b)
    else:
        sl_hit = high[:t_end] >= sl_vals
        tp_hit = valid & (low[:t_end]  <= b)

    sl_pos = int(sl_hit.argmax()) if sl_hit.any() else None
    tp_pos = int(tp_hit.argmax()) if tp_hit.any() else None

    def make_trade(exit_ts, exit_price, exit_reason, used_sl, used_tp):
        pnl = (exit_price - entry_price) if direction == "long" \
         else (entry_price - exit_price)
        return {
            "direction":   direction,
            "entry_time":  entry_ts,
            "exit_time":   exit_ts,
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "sl":          used_sl,
            "tp":          used_tp,
            "exit_reason": exit_reason,
            "pnl_points":  pnl,
        }

    if sl_pos is not None or tp_pos is not None:
        # pessimistic: on a same-bar tie (sl_pos == tp_pos) the stop wins
        sl_first = sl_pos is not None and (tp_pos is None or sl_pos <= tp_pos)
        if sl_first:
            hit_sl = float(sl_vals[sl_pos])
            reason = "trailing_sl" if bool(fl_vals[sl_pos]) else "sl"
            # the live trail level when stopped (informational; may be NaN)
            trail_at_sl = float(b[sl_pos])
            return make_trade(index[sl_pos], hit_sl, reason, base_sl, trail_at_sl)
        else:
            band_exit = float(b[tp_pos])
            return make_trade(index[tp_pos], band_exit, "tp", base_sl, band_exit)

    if n_all < timeout:
        exit_price = float(trade_win.c[-1])
        tp_at_exit = float(b[-1])
        if np.isnan(tp_at_exit):
            tp_at_exit = exit_price
        return make_trade(index[-1], exit_price, "eod", base_sl, tp_at_exit)

    # --- timeout tail: mirror _run_trade (breakeven TP + swing SL, ratchet kept) ---
    timeout_close = float(trade_win.c[t_end - 1])
    in_profit     = (timeout_close > entry_price) if direction == "long" \
               else (timeout_close < entry_price)

    if in_profit:
        tp_at_exit = float(b[-1])
        if np.isnan(tp_at_exit):
            tp_at_exit = timeout_close
        return make_trade(index[t_end - 1], timeout_close, "tp_timeout",
                          base_sl, tp_at_exit)

    new_tp   = entry_price
    swing_sl = float(low[:t_end].min())  if direction == "long" \
          else float(high[:t_end].max())

    sl_vals2 = sl_series[timeout:]
    fl_vals2 = trail_flags[timeout:]

    if direction == "long":
        eff_sl2 = np.maximum(sl_vals2, swing_sl)
        fl2     = fl_vals2 & (sl_vals2 > swing_sl)
    else:
        eff_sl2 = np.minimum(sl_vals2, swing_sl)
        fl2     = fl_vals2 & (sl_vals2 < swing_sl)

    if n_all == timeout:                       # post_timeout empty
        return make_trade(index[-1], float(trade_win.c[-1]), "eod", swing_sl, new_tp)

    if direction == "long":
        sl_hit2 = low[timeout:]  <= eff_sl2
        tp_hit2 = high[timeout:] >= new_tp
    else:
        sl_hit2 = high[timeout:] >= eff_sl2
        tp_hit2 = low[timeout:]  <= new_tp

    sl_pos2 = int(sl_hit2.argmax()) if sl_hit2.any() else None
    tp_pos2 = int(tp_hit2.argmax()) if tp_hit2.any() else None

    def sl_exit2(pos):
        hit_sl = float(eff_sl2[pos])
        reason = "trailing_sl" if bool(fl2[pos]) else "sl_timeout"
        return make_trade(index[timeout + pos], hit_sl, reason, swing_sl, new_tp)

    if sl_pos2 is None and tp_pos2 is None:
        return make_trade(index[-1], float(trade_win.c[-1]), "eod", swing_sl, new_tp)

    if sl_pos2 is None:
        return make_trade(index[timeout + tp_pos2], new_tp, "tp_timeout", swing_sl, new_tp)
    if tp_pos2 is None:
        return sl_exit2(sl_pos2)

    if tp_pos2 <= sl_pos2:
        return make_trade(index[timeout + tp_pos2], new_tp, "tp_timeout", swing_sl, new_tp)
    else:
        return sl_exit2(sl_pos2)


def run(entry_win, trade_win, entry_pos, entry_price, direction, levels, params):
    """Returns the standard trade dict (with risk_notes), or None."""
    # --- indicators required: no VWAP bands => no trade ---
    vwap_bands = trade_win.day.vwap_bands
    if vwap_bands is None:
        return None

    # --- SL placement (the trailing ratchet starts from here) ---
    placement = params["sl_placement"]
    if placement == "VAL/VAH":
        sl = levels["val"] if direction == "long" else levels["vah"]
    elif placement == "swing_low":
        sl = _swing_sl(entry_win, entry_pos, direction, levels)
    else:   # "zone_logic" — the default; unknown / legacy values fall back here
        sl = _zone_sl(entry_win, entry_pos, direction, levels)

    risk = abs(entry_price - sl)
    if risk <= 0:
        return None

    # --- band selection: pick the σ2 and σ3 columns for this session + direction ---
    session = params["vwap_session"]
    ud      = "up" if direction == "long" else "dn"
    col2    = f"vwap_tick_{session}_std2_{ud}"
    col3    = f"vwap_tick_{session}_std3_{ud}"

    if col2 not in vwap_bands or col3 not in vwap_bands:
        return None

    band2_e = vwap_bands[col2][entry_pos]
    band3_e = vwap_bands[col3][entry_pos]
    if np.isnan(band2_e) or np.isnan(band3_e):
        return None

    # --- entry-time escalation: if price already sits past the target band ---
    if direction == "long":
        past2 = entry_price >= band2_e
        past3 = entry_price >= band3_e
    else:
        past2 = entry_price <= band2_e
        past3 = entry_price <= band3_e

    eff_std  = params["vwap_std"]
    fallback = False
    if eff_std == 2:
        if past3:
            fallback = True       # past 3σ too => plain 1:1
        elif past2:
            eff_std = 3           # past 2σ only => bump target to 3σ
    else:  # eff_std == 3
        if past3:
            fallback = True

    escalated = fallback or (eff_std != params["vwap_std"])

    # --- trailing stop series (shared by all TP modes) ---
    with timed("risk4:build_trailing_sl"):
        sl_series, trail_flags, trail_log = _build_trailing_sl(
            trade_win, sl, entry_price, params
        )

    entry_ts = trade_win.day.index[entry_pos]

    # --- TP + execution ---
    with timed("risk4:fill_sim"):
        if fallback:
            tp = entry_price + risk if direction == "long" else entry_price - risk
            trade   = _run_trade(trade_win, entry_ts, entry_price, direction,
                                 sl, sl_series, trail_flags, tp, params)
            tp_type = "1:1"
        else:
            col_eff = col3 if eff_std == 3 else col2
            tp_type = f"tp_vwap_{eff_std}"

            if params["vwap_tp_mode"] == "trailing":
                band  = vwap_bands[col_eff][entry_pos:]
                trade = _run_trade_trailing(trade_win, entry_ts, entry_price, direction,
                                            sl, sl_series, trail_flags, band, params)
            else:  # "now": freeze the band value at entry
                tp    = float(vwap_bands[col_eff][entry_pos])
                trade = _run_trade(trade_win, entry_ts, entry_price, direction,
                                   sl, sl_series, trail_flags, tp, params)

    if trade is None:
        return None

    # only the trail events that were in effect by the exit bar are relevant
    applied = [e for e in trail_log if e["eff_ts"] <= trade["exit_time"]]

    risk_notes = {
        "tp_type":     tp_type,
        "escalated":   bool(escalated),
        "trail_count": len(applied),
    }

    # flat trailN_* keys so each renders as its own note tile, like entry notes
    for idx, e in enumerate(applied, start=1):
        prefix = f"trail{idx}_"
        risk_notes[prefix + "time"] = e["eff_ts"].strftime("%H:%M")
        risk_notes[prefix + "type"] = e["type"]
        risk_notes[prefix + "stop"] = round(float(e["stop"]), 2)
        for k, v in e.get("notes", {}).items():
            risk_notes[prefix + k] = v
        if "trigger_type" in e:
            risk_notes[prefix + "trigger_type"] = e["trigger_type"]
            risk_notes[prefix + "trigger_time"] = e["trigger_time"]

    trade["risk_notes"] = risk_notes

    return trade
