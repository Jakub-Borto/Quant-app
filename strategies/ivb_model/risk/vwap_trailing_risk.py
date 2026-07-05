"""VWAP-target risk script with a signal-driven trailing stop.

risk_script: 4. Same parameters and TP behaviour as vwap_tp_risk (sl_placement / vwap_std /
vwap_session / vwap_tp_mode, incl. the entry-time 2σ->3σ escalation and the 1:1 fallback), plus:
while the trade is open, the seven entry-style signals are re-detected on the live bars
(gated by the `trailing_entries` bit string, same order as `valid_entries`). When a signal is
confirmed by a candle meeting BOTH `body_threshold` and `delta_threshold` (in the trade
direction), the stop ratchets to the signal candle's extreme (low for longs / high for shorts)
starting from the bar AFTER the confirming candle. The stop only ever tightens.

Two switches shape the signal log:
  trailing_in_profit (default 1) — 1: a signal whose candidate stop is still in loss (below
    entry for longs / above for shorts) is not even logged; 0: every signal is logged and may
    trail, loss or not.
  late_trailing (default 0) — 1: each logged signal trails the stop to the PREVIOUS logged
    signal's level (the first logged signal only arms the log); 0: a signal trails to its own
    level immediately.

Unlike the entry finders, EVERY trailing signal requires the confirming candle (also
consecutive_absorption and passive_wall, which enter without one), and there is no VAL/VAH
invalidation — the stop manages the exit. A stop hit at a trailed level reports
exit_reason = "trailing_sl" with the trailed level as exit_price; the `sl` column always keeps
the originally placed stop.

Self-contained per the risk-script convention: own copies of _zone_sl, both fill simulators and
the 2-bar/CVD helpers; the only package import is the shared absorption grader the entry finders
use themselves. The day-level baselines + CVD series arrive via `levels` (see core.process_day).

INDICATORS REQUIRED: if the VWAP bands are unavailable for the day (no indicators / missing
columns / NaN at entry), this script returns None (no trade), exactly like vwap_tp_risk. The two
CVD trail detectors additionally need cvd_series / cvd_change_std and disable themselves when
those are absent.

exit_reason: tp / sl / eod (+ tp_timeout / sl_timeout) + trailing_sl. risk_notes records
tp_type / escalated / trail_count plus flat trailN_* keys per applied trail (trailN_time /
trailN_type / trailN_stop and the same fields that finder's entry_notes would carry —
absorption/trigger/passive/wall/CVD-pivot details — plus trailN_trigger_type/_trigger_time for
late trails), so every value renders as a plain note tile.
"""

import json
import pandas as pd

from ..absorption import is_absorption_candle, find_absorption_trigger


def _zone_sl(post_retest, entry_ts, direction, levels):
    """Pick the stop from the pullback window's extremes vs the VAL/POC/VAH zones."""
    poc = levels["poc"]
    vah = levels["vah"]
    val = levels["val"]

    # --- SL window: drop the breakout bar (index 0), keep retest .. bar before entry ---
    # entry is taken at the entry bar's OPEN, so that bar's low/close are future data => excluded.
    window = post_retest.iloc[1:]
    window = window[window.index < entry_ts]

    # degenerate: nothing to measure => fall back to the basic VAL/VAH stop
    if window.empty:
        return val if direction == "long" else vah

    if direction == "long":
        lowest_close = float(window["close"].min())   # where the pullback bottomed (by close)
        lowest_low   = float(window["low"].min())      # how far the wick reached

        if poc <= lowest_close <= vah:                 # bottomed in the UPPER zone
            return poc if lowest_low >= poc else lowest_low
        elif val <= lowest_close < poc:                # bottomed in the LOWER zone
            return val if lowest_low >= val else lowest_low
        else:                                          # shouldn't occur (pullback re-enters VA)
            return val

    else:
        highest_close = float(window["close"].max())   # where the pullback topped (by close)
        highest_high  = float(window["high"].max())     # how far the wick reached

        if val <= highest_close <= poc:                # topped in the LOWER zone
            return poc if highest_high <= poc else highest_high
        elif poc < highest_close <= vah:               # topped in the UPPER zone
            return vah if highest_high <= vah else highest_high
        else:                                          # shouldn't occur (pullback re-enters VA)
            return vah


# ---------------------------------------------------------------------------
# Trail-event detection (mirrors the entry finders, minus invalidation,
# collecting EVERY event instead of returning at the first one)
# ---------------------------------------------------------------------------

def _is_confirm_candle(bar: pd.Series, direction: str, params: dict) -> bool:
    """Confirming candle: correct direction + body_threshold + delta_threshold."""
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


def _trail_absorption_delta(post_entry, direction, levels, params):
    """Absorption candle + confirming candle. Stop = the absorption candle's extreme."""
    baseline_series = levels["sell_baseline"] if direction == "long" else levels["buy_baseline"]
    if baseline_series is None:
        return []

    events = []
    n      = len(post_entry)

    for i in range(n):
        bar = post_entry.iloc[i]
        ts  = post_entry.index[i]

        baseline = baseline_series.get(ts, float("nan"))
        if not is_absorption_candle(bar, baseline, direction, params):
            continue

        absorption_level = float(bar["low"]) if direction == "long" else float(bar["high"])
        trigger_price, trigger_volume = find_absorption_trigger(bar, baseline, direction, params)

        scan_end = min(i + 1 + params["entry_after_absorption"], n)
        for j in range(i + 1, scan_end):
            next_bar = post_entry.iloc[j]

            # close back through the absorption level kills this signal (as in the entry finder)
            if direction == "long":
                if float(next_bar["close"]) < absorption_level:
                    break
            else:
                if float(next_bar["close"]) > absorption_level:
                    break

            if _is_confirm_candle(next_bar, direction, params):
                events.append({"confirm_idx": j, "stop": absorption_level,
                               "name": "absorption_delta",
                               "notes": {
                                   "absorption_time": ts.strftime("%H:%M"),
                                   "abs_baseline":    round(baseline, 2),
                                   "trigger_price":   trigger_price,
                                   "trigger_volume":  trigger_volume,
                               }})
                break

    return events


def _trail_consecutive_absorption(post_entry, direction, levels, params):
    """N clustered absorption candles + confirming candle (confirmation is trailing-only —
    the entry finder needs none). Stop = the nth (triggering) candle's extreme."""
    baseline_series = levels["sell_baseline"] if direction == "long" else levels["buy_baseline"]
    if baseline_series is None:
        return []

    n          = len(post_entry)
    tick_size  = params["tick_size"]
    level_tol  = params["consec_abs_ticks"] * tick_size
    required_n = params["consec_abs_n"]

    consec_params = {
        **params,
        "absorption_mult":  params["consec_abs_mult"],
        "wick_threshold":   params["consec_wick_threshold"],
    }

    seen: list[tuple[float, float, pd.Timestamp]] = []   # (abs_level, body_mid, ts)
    events = []

    for i in range(n):
        bar = post_entry.iloc[i]
        ts  = post_entry.index[i]

        # an absorption level price has since CLOSED through is dead (mirrors the entry finder):
        # long => any later close strictly below the level runs it out; closing AT it keeps it
        close_i = float(bar["close"])
        if direction == "long":
            seen = [(lvl, bm, t) for lvl, bm, t in seen if close_i >= lvl]
        else:
            seen = [(lvl, bm, t) for lvl, bm, t in seen if close_i <= lvl]

        baseline = baseline_series.get(ts, float("nan"))
        if not is_absorption_candle(bar, baseline, direction, consec_params):
            continue

        abs_level = float(bar["low"]) if direction == "long" else float(bar["high"])
        body_mid  = (
            min(float(bar["open"]), float(bar["close"])) +
            max(float(bar["open"]), float(bar["close"]))
        ) / 2

        seen.append((abs_level, body_mid, ts))

        nearby = [
            (lvl, t) for lvl, bm, t in seen
            if abs(lvl - abs_level) <= level_tol
            and abs(bm - body_mid) <= level_tol
        ]

        if len(nearby) < required_n:
            continue

        # trigger reflects the nth (triggering) candle, as in the entry finder
        trigger_price, trigger_volume = find_absorption_trigger(bar, baseline, direction, consec_params)

        scan_end = min(i + 1 + params["entry_after_absorption"], n)
        for j in range(i + 1, scan_end):
            if _is_confirm_candle(post_entry.iloc[j], direction, params):
                events.append({"confirm_idx": j, "stop": abs_level,
                               "name": "consecutive_absorption",
                               "notes": {
                                   "absorption_time": [t.strftime("%H:%M") for _, t in nearby],
                                   "abs_baseline":    round(baseline, 2),
                                   "trigger_price":   trigger_price,
                                   "trigger_volume":  trigger_volume,
                               }})
                break

    return events


def _merge_tick_volume(tv1: str, tv2: str) -> str:
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


def _build_two_bar_baseline(pre_bars: pd.DataFrame, direction: str, params: dict) -> float:
    """Baseline from merged 2-bar pairs anchored backwards from the pattern bars."""
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


def _trail_two_bar_absorption(post_entry, direction, levels, params):
    """Reversal pair merged into a synthetic candle + confirming candle (baseline pairs come
    from the post-entry bars before the pattern). Stop = the merged candle's extreme."""
    n             = len(post_entry)
    tick_size     = params["tick_size"]
    max_wick      = params["two_bar_wick_ticks"] * tick_size
    required_mult = params["two_bar_abs_mult"]

    events = []

    for i in range(n - 1):
        bar1 = post_entry.iloc[i]
        bar2 = post_entry.iloc[i + 1]

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
        merged_tv    = _merge_tick_volume(
            bar1.get("tick_volume", "{}"),
            bar2.get("tick_volume", "{}"),
        )

        # --- 2-bar baseline from the post-entry bars before the pair ---
        pre_bars = post_entry.iloc[:i]
        baseline = _build_two_bar_baseline(pre_bars, direction, params)

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

        # absorption must register in the defended half of the full merged candle
        mid_price = (merged_low + merged_high) / 2.0

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
            if direction == "long" and price > mid_price:
                continue
            if direction == "short" and price < mid_price:
                continue
            volume_at_level = sell_qty if direction == "long" else buy_qty
            if volume_at_level >= required:
                absorbed       = True
                trigger_price  = price
                trigger_volume = volume_at_level
                break

        if not absorbed:
            continue

        stop = merged_low if direction == "long" else merged_high
        ts1  = post_entry.index[i]
        ts2  = post_entry.index[i + 1]

        # --- confirming candle scan starting from bar2 (inclusive), as in the entry finder ---
        scan_end = min(i + 1 + params["entry_after_absorption"], n)
        for j in range(i + 1, scan_end):
            if _is_confirm_candle(post_entry.iloc[j], direction, params):
                events.append({"confirm_idx": j, "stop": stop,
                               "name": "two_bar_absorption",
                               "notes": {
                                   "absorption_time":  [ts1.strftime("%H:%M"), ts2.strftime("%H:%M")],
                                   "two_bar_baseline": round(baseline, 2),
                                   "trigger_price":    trigger_price,
                                   "trigger_volume":   trigger_volume,
                               }})
                break

    return events


def _trail_passive_size(post_entry, direction, levels, params):
    """Big resting order (raw size) + absorption on the same candle + confirming candle.
    Stop = the signal candle's extreme."""
    if direction == "long":
        passive_baseline    = levels["passive_baseline_long"]
        abs_baseline_series = levels["sell_baseline"]
    else:
        passive_baseline    = levels["passive_baseline_short"]
        abs_baseline_series = levels["buy_baseline"]

    if passive_baseline is None or abs_baseline_series is None:
        return []

    n = len(post_entry)

    passive_params = {
        **params,
        "absorption_mult": params["passive_size_absorption_mult"],
        "wick_threshold":  params["passive_size_wick_threshold"],
    }

    events = []

    for i in range(n):
        bar = post_entry.iloc[i]
        ts  = post_entry.index[i]

        # --- passive order check (raw size only) ---
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
        required_po           = p_baseline * params["passive_size_order_mult"]
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
            if size >= required_po:
                has_big_passive       = True
                passive_trigger_price = price
                passive_trigger_size  = size
                passive_trigger_count = count
                break

        if not has_big_passive:
            continue

        # --- absorption check on the same candle ---
        abs_baseline = abs_baseline_series.get(ts, float("nan"))
        if not is_absorption_candle(bar, abs_baseline, direction, passive_params):
            continue

        stop = float(bar["low"]) if direction == "long" else float(bar["high"])

        scan_end = min(i + 1 + params["entry_after_absorption"], n)
        for j in range(i + 1, scan_end):
            if _is_confirm_candle(post_entry.iloc[j], direction, params):
                events.append({"confirm_idx": j, "stop": stop,
                               "name": "passive_absorption_size_only",
                               "notes": {
                                   "absorption_time": ts.strftime("%H:%M"),
                                   "passive_price":   passive_trigger_price,
                                   "passive_size":    passive_trigger_size,
                                   "passive_count":   passive_trigger_count,
                               }})
                break

    return events


def _trail_passive_wall(post_entry, direction, levels, params):
    """Cluster of big defended-side passive orders + confirming candle (confirmation is
    trailing-only — the entry finder needs none). Stop = the wall-completing bar's extreme."""
    passive_baseline = (
        levels["passive_baseline_long"] if direction == "long"
        else levels["passive_baseline_short"]
    )
    if passive_baseline is None:
        return []

    n          = len(post_entry)
    tick_size  = params["tick_size"]
    level_tol  = params["passive_wall_ticks"] * tick_size
    required_n = params["passive_wall_n"]

    seen: list[tuple[float, pd.Timestamp]] = []   # every big passive level (price, ts) — never cleared
    events = []

    for i in range(n):
        bar = post_entry.iloc[i]
        ts  = post_entry.index[i]

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

        # append each level; at most one trail event per bar (same stop either way)
        completed = False
        for lvl in new_levels:
            seen.append((lvl, ts))
            if completed:
                continue

            nearby = [(p, t) for p, t in seen if abs(p - lvl) <= level_tol]
            if len(nearby) < required_n:
                continue

            completed = True
            stop = float(bar["low"]) if direction == "long" else float(bar["high"])

            scan_end = min(i + 1 + params["entry_after_absorption"], n)
            for j in range(i + 1, scan_end):
                if _is_confirm_candle(post_entry.iloc[j], direction, params):
                    events.append({"confirm_idx": j, "stop": stop,
                                   "name": "passive_wall",
                                   "notes": {
                                       "wall_levels": [round(p, 2) for p, _ in nearby],
                                       "wall_times":  [t.strftime("%H:%M") for _, t in nearby],
                                       "wall_count":  len(nearby),
                                   }})
                    break

    return events


def _cvd_param(params, mode, name):
    """CVD divergence params are per-flavour: the exhaustion finder reads its own cvd_exh_*
    keys, the absorption finder reads cvd_*. `name` is the unprefixed suffix (pivot_k, etc.)."""
    prefix = "cvd_exh_" if mode == "exhaustion" else "cvd_"
    return params[prefix + name]


def _test_cvd_divergence(p1, p2, direction, params, cvd_change_std, mode):
    """Grade the (P1 older, P2 newer) pivot pair. Returns a setup dict or None.
    mode = "absorption" (price held, CVD pushed) or "exhaustion" (price extended, CVD didn't)."""
    min_score = _cvd_param(params, mode, "min_score")

    sep = p2["idx"] - p1["idx"]
    if sep < _cvd_param(params, mode, "min_separation") or sep > _cvd_param(params, mode, "max_separation"):
        return None

    tol = _cvd_param(params, mode, "wick_tolerance_ticks") * params["tick_size"]
    if mode == "absorption":
        # lower/equal high (short) or higher/equal low (long)
        if direction == "short":
            if not (p2["price"] <= p1["price"] + tol):
                return None
        else:
            if not (p2["price"] >= p1["price"] - tol):
                return None
    else:
        # higher/equal high (short) or lower/equal low (long) — price DID extend
        if direction == "short":
            if not (p2["price"] >= p1["price"] - tol):
                return None
        else:
            if not (p2["price"] <= p1["price"] + tol):
                return None

    std = cvd_change_std.get(p2["ts"], float("nan"))
    if pd.isna(std) or std <= 0:
        return None

    score = (p2["cvd"] - p1["cvd"]) / std
    if mode == "absorption":
        # CVD rose into a lower/equal high (short) / fell into a higher/equal low (long)
        if direction == "short":
            if not (score >= min_score):
                return None
        else:
            if not (score <= -min_score):
                return None
    else:
        # CVD fell into a higher/equal high (short) / rose into a lower/equal low (long)
        if direction == "short":
            if not (score <= -min_score):
                return None
        else:
            if not (score >= min_score):
                return None

    return {"p1": p1, "p2": p2, "score": score, "std": std}


def _trail_cvd(post_entry, direction, levels, params, mode):
    """CVD divergence (absorption or exhaustion flavour, per `mode`) + confirming candle.
    Same left-k fractal pivot machinery as the entry finders, run over the trade bars.
    Stop = the 2nd pivot's price. Disabled when CVD is absent."""
    cvd_series     = levels.get("cvd_series")
    cvd_change_std = levels.get("cvd_change_std")
    if cvd_series is None or cvd_change_std is None or post_entry.empty:
        return []

    name = "cvd_divergence_absorption" if mode == "absorption" else "cvd_divergence_exhaustion"
    n    = len(post_entry)

    # --- stage 1: vectorized candidate pivots (left-side k-bar fractal) ---
    k = _cvd_param(params, mode, "pivot_k")
    prev_high_max = post_entry["high"].rolling(k).max().shift(1)
    prev_low_min  = post_entry["low"].rolling(k).min().shift(1)
    cand_high = (post_entry["high"] > prev_high_max).values
    cand_low  = (post_entry["low"]  < prev_low_min).values
    bearish   = (post_entry["close"] < post_entry["open"]).values
    bullish   = (post_entry["close"] > post_entry["open"]).values

    # --- stage 2: forward scan maintaining the last-two confirmed pivots + active setup ---
    pivots: list[dict] = []
    active = None
    events = []

    for i in range(1, n):
        if direction == "short":
            new_pivot = cand_high[i - 1] and bearish[i]
        else:
            new_pivot = cand_low[i - 1] and bullish[i]

        if new_pivot:
            # true swing extreme = the more-extreme of candidate bar (i-1) and confirming bar (i)
            if direction == "short":
                pivot_idx = i - 1 if float(post_entry.iloc[i - 1]["high"]) >= float(post_entry.iloc[i]["high"]) else i
                pivot_price = float(post_entry.iloc[pivot_idx]["high"])
            else:
                pivot_idx = i - 1 if float(post_entry.iloc[i - 1]["low"]) <= float(post_entry.iloc[i]["low"]) else i
                pivot_price = float(post_entry.iloc[pivot_idx]["low"])

            pivot_ts  = post_entry.index[pivot_idx]
            pivot_cvd = cvd_series.get(pivot_ts, float("nan"))
            if not pd.isna(pivot_cvd):
                pivots.append({"idx": pivot_idx, "ts": pivot_ts, "price": pivot_price, "cvd": pivot_cvd})

                # a new confirmed pivot supersedes any pending setup; roll the pair forward
                active = None
                if len(pivots) >= 2:
                    setup = _test_cvd_divergence(pivots[-2], pivots[-1], direction, params,
                                                 cvd_change_std, mode)
                    if setup is not None:
                        setup["conf_idx"] = i
                        active = setup

        # confirming-candle scan for the active setup (this bar = candidate, incl. bar i itself)
        if active is not None:
            if i >= active["conf_idx"] + params["entry_after_absorption"]:
                active = None                                # window elapsed, abandon setup
            elif _is_confirm_candle(post_entry.iloc[i], direction, params):
                p1, p2 = active["p1"], active["p2"]
                events.append({"confirm_idx": i, "stop": float(p2["price"]),
                               "name": name,
                               "notes": {
                                   "cvd_pivot1_time":  p1["ts"].strftime("%H:%M"),
                                   "cvd_pivot1_price": round(p1["price"], 2),
                                   "cvd_pivot1_cvd":   round(float(p1["cvd"]), 2),
                                   "cvd_pivot2_time":  p2["ts"].strftime("%H:%M"),
                                   "cvd_pivot2_price": round(p2["price"], 2),
                                   "cvd_pivot2_cvd":   round(float(p2["cvd"]), 2),
                                   "cvd_score":        round(float(active["score"]), 2),
                                   "cvd_change_std":   round(float(active["std"]), 2),
                               }})
                active = None                                # one event per setup

    return events


# order = FINDER_REGISTRY order = the trailing_entries bit order
_TRAIL_DETECTORS = [
    _trail_absorption_delta,
    _trail_consecutive_absorption,
    _trail_two_bar_absorption,
    _trail_passive_size,
    _trail_passive_wall,
    lambda pe, d, lv, p: _trail_cvd(pe, d, lv, p, "absorption"),
    lambda pe, d, lv, p: _trail_cvd(pe, d, lv, p, "exhaustion"),
]


def _build_trailing_sl(post_entry, direction, base_sl, entry_price, levels, params):
    """Detect all trail signals and build the per-bar effective stop.

    The chronological signal log obeys `trailing_in_profit`: when 1, a signal whose candidate
    stop is still in loss (long: below entry_price, short: above) is not even logged; when 0,
    every signal is logged. `late_trailing` shifts the application one signal back: the k-th
    logged signal trails the stop to the (k-1)-th logged signal's level, so the first logged
    signal only arms the log. Either way a signal confirmed on bar j applies from bar j+1 and
    the stop only ever tightens.

    Returns (sl_series, trail_flags, trail_log):
      sl_series   — the stop in effect at each bar
      trail_flags — True where the effective stop came from a trail signal
      trail_log   — ordered [{eff_ts, type, stop, notes (+trigger_type/trigger_time when
                    late)}] of the trails actually applied
    """
    n     = len(post_entry)
    flags = str(params.get("trailing_entries", "0" * len(_TRAIL_DETECTORS)))
    flags = flags.ljust(len(_TRAIL_DETECTORS), "0")

    in_profit_only = int(params.get("trailing_in_profit", 1)) == 1
    late           = int(params.get("late_trailing", 0)) == 1

    events = []
    for detector, flag in zip(_TRAIL_DETECTORS, flags):
        if flag == "1":
            events.extend(detector(post_entry, direction, levels, params))
    events.sort(key=lambda e: e["confirm_idx"])   # chronological across detectors

    # --- the signal log: the in-profit filter applies BEFORE logging ---
    log = []
    for e in events:
        if in_profit_only:
            in_profit = e["stop"] >= entry_price if direction == "long" \
                   else e["stop"] <= entry_price
            if not in_profit:
                continue
        log.append(e)

    # bucket by effective bar (confirm_idx + 1); signals confirmed on the last bar never apply
    by_bar: dict[int, list] = {}
    for k, e in enumerate(log):
        eff = e["confirm_idx"] + 1
        if eff < n:
            by_bar.setdefault(eff, []).append((k, e))

    sl_values = [base_sl] * n
    trailed   = [False] * n
    trail_log = []
    current         = base_sl
    current_trailed = False

    for i in range(n):
        for k, e in by_bar.get(i, []):
            if late:
                if k == 0:
                    continue          # first logged signal: log only, nothing to trail to yet
                src = log[k - 1]      # trail to the PREVIOUS logged signal's level
            else:
                src = e
            tighter = src["stop"] > current if direction == "long" else src["stop"] < current
            if tighter:
                current         = src["stop"]
                current_trailed = True
                applied = {"eff_ts": post_entry.index[i],
                           "type":   src["name"],
                           "stop":   src["stop"],
                           "notes":  src.get("notes", {})}
                if late:
                    # the level came from the previous signal; this one pulled the trigger
                    applied["trigger_type"] = e["name"]
                    applied["trigger_time"] = post_entry.index[e["confirm_idx"]].strftime("%H:%M")
                trail_log.append(applied)
        sl_values[i] = current
        trailed[i]   = current_trailed

    sl_series   = pd.Series(sl_values, index=post_entry.index)
    trail_flags = pd.Series(trailed,   index=post_entry.index)
    return sl_series, trail_flags, trail_log


# ---------------------------------------------------------------------------
# Fill simulation (vwap_tp_risk's simulators with a per-bar stop)
# ---------------------------------------------------------------------------

def _run_trade(
    post_entry:   pd.DataFrame,
    entry_ts:     pd.Timestamp,
    entry_price:  float,
    direction:    str,
    base_sl:      float,
    sl_series:    pd.Series,
    trail_flags:  pd.Series,
    tp:           float,
    params:       dict,
) -> dict:
    """Like vwap_tp_risk._run_trade (fixed TP), but the stop is the per-bar sl_series.
    A stop hit at a trailed level (trail_flags True) reports exit_reason = "trailing_sl" and
    exits at the trailed level; the recorded `sl` stays the original base_sl."""
    timeout     = params["trade_timeout"]
    pre_timeout = post_entry.iloc[:timeout]
    sl_vals     = sl_series.iloc[:timeout]
    fl_vals     = trail_flags.iloc[:timeout]
    n           = len(pre_timeout)

    if direction == "long":
        sl_hit = pre_timeout["low"]  <= sl_vals
        tp_hit = pre_timeout["high"] >= tp
    else:
        sl_hit = pre_timeout["high"] >= sl_vals
        tp_hit = pre_timeout["low"]  <= tp

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
        hit_sl = float(sl_vals.iloc[pos])
        reason = "trailing_sl" if bool(fl_vals.iloc[pos]) else "sl"
        return make_trade(pre_timeout.index[pos], hit_sl, reason, base_sl, tp)

    if sl_pos is not None or tp_pos is not None:
        if sl_pos is None:
            return make_trade(pre_timeout.index[tp_pos], tp, "tp", base_sl, tp)
        if tp_pos is None:
            return sl_exit(sl_pos)
        if tp_pos <= sl_pos:
            return make_trade(pre_timeout.index[tp_pos], tp, "tp", base_sl, tp)
        else:
            return sl_exit(sl_pos)

    if n < timeout:
        exit_ts    = post_entry.index[-1]
        exit_price = float(post_entry.iloc[-1]["close"])
        return make_trade(exit_ts, exit_price, "eod", base_sl, tp)

    timeout_close = float(pre_timeout.iloc[-1]["close"])
    in_profit     = (timeout_close > entry_price) if direction == "long" \
               else (timeout_close < entry_price)

    if in_profit:
        return make_trade(pre_timeout.index[-1], timeout_close, "tp_timeout", base_sl, tp)

    new_tp   = entry_price
    swing_sl = float(pre_timeout["low"].min())  if direction == "long" \
          else float(pre_timeout["high"].max())

    post_timeout = post_entry.iloc[timeout:]
    sl_vals2     = sl_series.iloc[timeout:]
    fl_vals2     = trail_flags.iloc[timeout:]

    # the swing SL may never loosen the ratchet: keep whichever is tighter per bar
    if direction == "long":
        eff_sl2 = sl_vals2.clip(lower=swing_sl)
        fl2     = fl_vals2 & (sl_vals2 > swing_sl)
    else:
        eff_sl2 = sl_vals2.clip(upper=swing_sl)
        fl2     = fl_vals2 & (sl_vals2 < swing_sl)

    if post_timeout.empty:
        exit_ts    = post_entry.index[-1]
        exit_price = float(post_entry.iloc[-1]["close"])
        return make_trade(exit_ts, exit_price, "eod", swing_sl, new_tp)

    if direction == "long":
        sl_hit2 = post_timeout["low"]  <= eff_sl2
        tp_hit2 = post_timeout["high"] >= new_tp
    else:
        sl_hit2 = post_timeout["high"] >= eff_sl2
        tp_hit2 = post_timeout["low"]  <= new_tp

    sl_pos2 = int(sl_hit2.argmax()) if sl_hit2.any() else None
    tp_pos2 = int(tp_hit2.argmax()) if tp_hit2.any() else None

    def sl_exit2(pos):
        hit_sl = float(eff_sl2.iloc[pos])
        reason = "trailing_sl" if bool(fl2.iloc[pos]) else "sl_timeout"
        return make_trade(post_timeout.index[pos], hit_sl, reason, swing_sl, new_tp)

    if sl_pos2 is None and tp_pos2 is None:
        exit_ts    = post_entry.index[-1]
        exit_price = float(post_entry.iloc[-1]["close"])
        return make_trade(exit_ts, exit_price, "eod", swing_sl, new_tp)

    if sl_pos2 is None:
        return make_trade(post_timeout.index[tp_pos2], new_tp, "tp_timeout", swing_sl, new_tp)
    if tp_pos2 is None:
        return sl_exit2(sl_pos2)

    if tp_pos2 <= sl_pos2:
        return make_trade(post_timeout.index[tp_pos2], new_tp, "tp_timeout", swing_sl, new_tp)
    else:
        return sl_exit2(sl_pos2)


def _run_trade_trailing(
    post_entry:   pd.DataFrame,
    entry_ts:     pd.Timestamp,
    entry_price:  float,
    direction:    str,
    base_sl:      float,
    sl_series:    pd.Series,
    trail_flags:  pd.Series,
    band_series:  pd.Series,
    params:       dict,
) -> dict:
    """Like vwap_tp_risk._run_trade_trailing (band TP), but the stop is the per-bar sl_series.

    Same-bar race stays PESSIMISTIC: if the stop and the trailing TP both trigger on one bar, the
    stop wins. Bars with a NaN band value can only trigger the stop. A stop hit at a trailed
    level (trail_flags True) reports exit_reason = "trailing_sl" and exits at the trailed level;
    the recorded `sl` stays the original base_sl."""
    timeout     = params["trade_timeout"]
    pre_timeout = post_entry.iloc[:timeout]
    sl_vals     = sl_series.iloc[:timeout]
    fl_vals     = trail_flags.iloc[:timeout]
    n           = len(pre_timeout)

    band = band_series.reindex(pre_timeout.index)

    if direction == "long":
        sl_hit = pre_timeout["low"]  <= sl_vals
        tp_hit = band.notna() & (pre_timeout["high"] >= band)
    else:
        sl_hit = pre_timeout["high"] >= sl_vals
        tp_hit = band.notna() & (pre_timeout["low"]  <= band)

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
            hit_sl = float(sl_vals.iloc[sl_pos])
            reason = "trailing_sl" if bool(fl_vals.iloc[sl_pos]) else "sl"
            # the live trail level when stopped (informational; may be NaN)
            trail_at_sl = float(band.iloc[sl_pos])
            return make_trade(pre_timeout.index[sl_pos], hit_sl, reason, base_sl, trail_at_sl)
        else:
            band_exit = float(band.iloc[tp_pos])
            return make_trade(pre_timeout.index[tp_pos], band_exit, "tp", base_sl, band_exit)

    if n < timeout:
        exit_ts    = post_entry.index[-1]
        exit_price = float(post_entry.iloc[-1]["close"])
        tp_at_exit = float(band.iloc[-1])
        if pd.isna(tp_at_exit):
            tp_at_exit = exit_price
        return make_trade(exit_ts, exit_price, "eod", base_sl, tp_at_exit)

    # --- timeout tail: mirror _run_trade (breakeven TP + swing SL, ratchet kept) ---
    timeout_close = float(pre_timeout.iloc[-1]["close"])
    in_profit     = (timeout_close > entry_price) if direction == "long" \
               else (timeout_close < entry_price)

    if in_profit:
        tp_at_exit = float(band.iloc[-1])
        if pd.isna(tp_at_exit):
            tp_at_exit = timeout_close
        return make_trade(pre_timeout.index[-1], timeout_close, "tp_timeout",
                          base_sl, tp_at_exit)

    new_tp   = entry_price
    swing_sl = float(pre_timeout["low"].min())  if direction == "long" \
          else float(pre_timeout["high"].max())

    post_timeout = post_entry.iloc[timeout:]
    sl_vals2     = sl_series.iloc[timeout:]
    fl_vals2     = trail_flags.iloc[timeout:]

    if direction == "long":
        eff_sl2 = sl_vals2.clip(lower=swing_sl)
        fl2     = fl_vals2 & (sl_vals2 > swing_sl)
    else:
        eff_sl2 = sl_vals2.clip(upper=swing_sl)
        fl2     = fl_vals2 & (sl_vals2 < swing_sl)

    if post_timeout.empty:
        exit_ts    = post_entry.index[-1]
        exit_price = float(post_entry.iloc[-1]["close"])
        return make_trade(exit_ts, exit_price, "eod", swing_sl, new_tp)

    if direction == "long":
        sl_hit2 = post_timeout["low"]  <= eff_sl2
        tp_hit2 = post_timeout["high"] >= new_tp
    else:
        sl_hit2 = post_timeout["high"] >= eff_sl2
        tp_hit2 = post_timeout["low"]  <= new_tp

    sl_pos2 = int(sl_hit2.argmax()) if sl_hit2.any() else None
    tp_pos2 = int(tp_hit2.argmax()) if tp_hit2.any() else None

    def sl_exit2(pos):
        hit_sl = float(eff_sl2.iloc[pos])
        reason = "trailing_sl" if bool(fl2.iloc[pos]) else "sl_timeout"
        return make_trade(post_timeout.index[pos], hit_sl, reason, swing_sl, new_tp)

    if sl_pos2 is None and tp_pos2 is None:
        exit_ts    = post_entry.index[-1]
        exit_price = float(post_entry.iloc[-1]["close"])
        return make_trade(exit_ts, exit_price, "eod", swing_sl, new_tp)

    if sl_pos2 is None:
        return make_trade(post_timeout.index[tp_pos2], new_tp, "tp_timeout", swing_sl, new_tp)
    if tp_pos2 is None:
        return sl_exit2(sl_pos2)

    if tp_pos2 <= sl_pos2:
        return make_trade(post_timeout.index[tp_pos2], new_tp, "tp_timeout", swing_sl, new_tp)
    else:
        return sl_exit2(sl_pos2)


def run(post_retest, post_entry, entry_ts, entry_price, direction, levels, params):
    """Returns the standard trade dict (with risk_notes), or None."""
    # --- indicators required: no VWAP bands => no trade ---
    vwap_bands = levels.get("vwap_bands")
    if vwap_bands is None:
        return None

    # --- SL placement (the trailing ratchet starts from here) ---
    if params["sl_placement"] == 1:
        sl = levels["val"] if direction == "long" else levels["vah"]
    else:
        sl = _zone_sl(post_retest, entry_ts, direction, levels)

    risk = abs(entry_price - sl)
    if risk <= 0:
        return None

    # --- band selection: pick the σ2 and σ3 columns for this session + direction ---
    session = params["vwap_session"]
    ud      = "up" if direction == "long" else "dn"
    col2    = f"vwap_tick_{session}_std2_{ud}"
    col3    = f"vwap_tick_{session}_std3_{ud}"

    if col2 not in vwap_bands.columns or col3 not in vwap_bands.columns:
        return None

    band2_e = vwap_bands[col2].get(entry_ts, float("nan"))
    band3_e = vwap_bands[col3].get(entry_ts, float("nan"))
    if pd.isna(band2_e) or pd.isna(band3_e):
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
    sl_series, trail_flags, trail_log = _build_trailing_sl(
        post_entry, direction, sl, entry_price, levels, params
    )

    # --- TP + execution ---
    if fallback:
        tp = entry_price + risk if direction == "long" else entry_price - risk
        trade   = _run_trade(post_entry, entry_ts, entry_price, direction,
                             sl, sl_series, trail_flags, tp, params)
        tp_type = "1:1"
    else:
        col_eff = col3 if eff_std == 3 else col2
        tp_type = f"tp_vwap_{eff_std}"

        if params["vwap_tp_mode"] == "trailing":
            trade = _run_trade_trailing(post_entry, entry_ts, entry_price, direction,
                                        sl, sl_series, trail_flags, vwap_bands[col_eff], params)
        else:  # "now": freeze the band value at entry
            tp    = float(vwap_bands[col_eff].get(entry_ts, float("nan")))
            trade = _run_trade(post_entry, entry_ts, entry_price, direction,
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
