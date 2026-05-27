import json
import pandas as pd
from pathlib import Path
from datetime import time

PARAMS = {
    "ib_minutes":               30,     # IB range duration: 15, 30, or 60
    "delta_threshold":          30.0,   # minimum volume_delta_pct for entry candle
    "body_threshold":           0.5,    # body must cover 50% of bar range
    "rr":                       1.0,    # fixed risk to reward ratio
    "sl_type":                  0,      # 0 = VAL, 1 = swing low
    "retest_window":            30,     # max bars to wait for retest after breakout
    "entry_window":             15,     # bars to scan for entry after retest
    "entry_after_absorption":   5,      # max bars to scan for entry candle after absorption
    "trade_timeout":            999,    # bars before timeout logic kicks in
    "max_flips":                4,      # max direction flips per day after invalidation
    # --- absorption params ---
    "wick_threshold":           0.4,    # lower wick must be >= this fraction of total bar range
    "absorption_mult":          2.0,    # wick level sell volume must be >= this x rolling avg
    "absorption_window":        20,     # rolling N bars for avg sell_per_tick baseline (RTH, post 09:35)
    "tick_size":                0.25,   # ES tick size
    # --- consecutive absorption params ---
    "consec_abs_n":             3,      # number of absorption candles required at same level
    "consec_abs_mult":          1.5,    # absorption multiplier for consecutive absorption finder
    "consec_abs_ticks":         5,      # ±ticks tolerance for grouping absorption levels
}

_OUTPUT_COLUMNS = [
    "date",
    "direction",
    "trade_type",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "sl",
    "tp",
    "exit_reason",
    "pnl_points",
    "notes",
]


# ---------------------------------------------------------------------------
# Data layer helpers
# ---------------------------------------------------------------------------

def _compute_ivb_profile(ib_bars: pd.DataFrame) -> tuple:
    """
    Compute POC, VAH, VAL from tick_volume within the IB bars.

    Algorithm:
      1. Build raw volume-at-price dict from tick_volume JSON.
      2. Smooth with a 3-tick rolling average to remove single-tick spikes.
      3. Find local maxima (smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]).
      4. Cluster peaks within 4 ticks of each other — keep only the highest per cluster.
      5. Take up to 5 peak candidates. For each, scan ±3 ticks in the RAW profile
         to find the actual highest-volume tick — that becomes the POC candidate.
      6. Expand VA outward from each POC candidate until 70% of total volume captured.
      7. Pick the VA with the tightest price range (smallest VAH - VAL).
      8. Return (poc, vah, val) for the winning candidate.

    Falls back to simple max-volume POC if fewer than 2 price levels exist.
    """
    # --- Step 1: build raw levels -------------------------------------------
    levels = {}

    for _, row in ib_bars.iterrows():
        tv = row["tick_volume"]
        if not tv or tv == "{}":
            continue
        try:
            raw = json.loads(tv)
        except Exception:
            continue
        for price_str, (buy_qty, sell_qty) in raw.items():
            price = float(price_str)
            total = buy_qty + sell_qty
            if price not in levels:
                levels[price] = 0
            levels[price] += total

    if not levels:
        return None, None, None

    sorted_prices = sorted(levels.keys())
    n             = len(sorted_prices)

    # Fallback for very thin profiles
    if n < 3:
        poc           = max(levels, key=levels.get)
        total_volume  = sum(levels.values())
        target        = total_volume * 0.70
        poc_idx       = sorted_prices.index(poc)
        lo_idx        = poc_idx
        hi_idx        = poc_idx
        va_volume     = levels[poc]
        while va_volume < target:
            down_vol = levels[sorted_prices[lo_idx - 1]] if lo_idx > 0 else 0
            up_vol   = levels[sorted_prices[hi_idx + 1]] if hi_idx < n - 1 else 0
            if down_vol == 0 and up_vol == 0:
                break
            if up_vol >= down_vol:
                hi_idx   += 1
                va_volume += up_vol
            else:
                lo_idx   -= 1
                va_volume += down_vol
        return poc, sorted_prices[hi_idx], sorted_prices[lo_idx]

    volumes = [levels[p] for p in sorted_prices]

    # --- Step 2: smooth with 3-tick rolling average -------------------------
    smoothed = []
    for i in range(n):
        lo = max(0, i - 1)
        hi = min(n, i + 2)
        smoothed.append(sum(volumes[lo:hi]) / (hi - lo))

    # --- Step 3: find local maxima ------------------------------------------
    raw_peaks = []
    for i in range(1, n - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            raw_peaks.append(i)

    if smoothed[0] > smoothed[1]:
        raw_peaks.insert(0, 0)
    if smoothed[-1] > smoothed[-2]:
        raw_peaks.append(n - 1)

    if not raw_peaks:
        raw_peaks = [int(max(range(n), key=lambda i: volumes[i]))]

    # --- Step 4: cluster peaks within 4 ticks, keep highest per cluster -----
    cluster_size = 4
    raw_peaks    = sorted(raw_peaks)
    clusters     = []
    current      = [raw_peaks[0]]

    for idx in raw_peaks[1:]:
        if idx - current[-1] <= cluster_size:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)

    cluster_peaks = [max(c, key=lambda i: smoothed[i]) for c in clusters]
    cluster_peaks = sorted(cluster_peaks, key=lambda i: smoothed[i], reverse=True)[:5]

    # --- Step 5: refine each peak to actual highest raw tick within ±3 ------
    poc_candidates = []
    for peak_idx in cluster_peaks:
        lo = max(0, peak_idx - 3)
        hi = min(n, peak_idx + 4)
        best_idx = max(range(lo, hi), key=lambda i: volumes[i])
        poc_candidates.append(sorted_prices[best_idx])

    poc_candidates = list(dict.fromkeys(poc_candidates))

    # --- Step 6 & 7: expand VA from each candidate, pick tightest ----------
    total_volume = sum(volumes)
    target       = total_volume * 0.70

    best_poc = None
    best_vah = None
    best_val = None
    best_range = float("inf")

    for poc_price in poc_candidates:
        poc_idx   = sorted_prices.index(poc_price)
        lo_idx    = poc_idx
        hi_idx    = poc_idx
        va_volume = levels[poc_price]

        while va_volume < target:
            down_vol = levels[sorted_prices[lo_idx - 1]] if lo_idx > 0 else 0
            up_vol   = levels[sorted_prices[hi_idx + 1]] if hi_idx < n - 1 else 0

            if down_vol == 0 and up_vol == 0:
                break

            if up_vol >= down_vol:
                hi_idx   += 1
                va_volume += levels[sorted_prices[hi_idx]]
            else:
                lo_idx   -= 1
                va_volume += levels[sorted_prices[lo_idx]]

        vah        = sorted_prices[hi_idx]
        val        = sorted_prices[lo_idx]
        va_range   = vah - val

        if va_range < best_range:
            best_range = va_range
            best_poc   = poc_price
            best_vah   = vah
            best_val   = val

    if best_poc is None:
        return None, None, None

    best_poc = max(
        (p for p in sorted_prices if best_val <= p <= best_vah),
        key=lambda p: levels[p]
    )

    return best_poc, best_vah, best_val


# ---------------------------------------------------------------------------
# Rolling baseline
# ---------------------------------------------------------------------------

def _build_rolling_baseline(rth_session: pd.DataFrame, params: dict):
    tick_size = params["tick_size"]
    window    = params["absorption_window"]

    valid = rth_session[rth_session.index.time >= time(9, 35)].copy()

    range_ticks = ((valid["high"] - valid["low"]) / tick_size).replace(0, float("nan"))

    valid["_sell_per_tick"] = valid["sell_volume"] / range_ticks
    valid["_buy_per_tick"]  = valid["buy_volume"]  / range_ticks

    sell_baseline = valid["_sell_per_tick"].rolling(window, min_periods=window).mean()
    buy_baseline  = valid["_buy_per_tick"].rolling(window, min_periods=window).mean()

    sell_baseline = sell_baseline.reindex(rth_session.index)
    buy_baseline  = buy_baseline.reindex(rth_session.index)

    return sell_baseline, buy_baseline


# ---------------------------------------------------------------------------
# Absorption grading
# ---------------------------------------------------------------------------

def _is_absorption_candle(
    bar:       pd.Series,
    baseline:  float,
    direction: str,
    params:    dict,
) -> bool:
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


# ---------------------------------------------------------------------------
# Strategy steps
# ---------------------------------------------------------------------------

def _detect_breakout(post_ib: pd.DataFrame, ivb_high: float, ivb_low: float) -> tuple:
    long_breakout  = post_ib["close"] > ivb_high
    short_breakout = post_ib["close"] < ivb_low

    long_pos  = int(long_breakout.argmax())  if long_breakout.any()  else None
    short_pos = int(short_breakout.argmax()) if short_breakout.any() else None

    if long_pos is None and short_pos is None:
        return None, None

    if long_pos is not None and short_pos is not None:
        direction = "long" if long_pos <= short_pos else "short"
    elif long_pos is not None:
        direction = "long"
    else:
        direction = "short"

    breakout_pos = long_pos if direction == "long" else short_pos
    return direction, breakout_pos


def _detect_retest(
    post_ib:       pd.DataFrame,
    breakout_pos:  int,
    direction:     str,
    vah:           float,
    val:           float,
    poc:           float,
    retest_window: int,
) -> int:
    scan_start = breakout_pos + 1
    scan_end   = scan_start + retest_window
    scan       = post_ib.iloc[scan_start:scan_end]

    if scan.empty:
        return None

    if direction == "long":
        retest_mask = scan["low"] <= vah
    else:
        retest_mask = scan["high"] >= val

    if not retest_mask.any():
        return None

    return scan_start + int(retest_mask.argmax())


# ---------------------------------------------------------------------------
# Entry sub-finders
# Each returns: (entry_ts, entry_price, invalidation_ts, absorption_ts, trade_type)
# entry_ts=None means no entry found.
# invalidation_ts=None means no invalidation hit (scanned to end without entry).
# ---------------------------------------------------------------------------

def _find_entry_pure_absorption(
    post_retest:   pd.DataFrame,
    direction:     str,
    ivb_high:      float,
    ivb_low:       float,
    poc:           float,
    vah:           float,
    val:           float,
    sell_baseline: pd.Series,
    buy_baseline:  pd.Series,
    params:        dict,
) -> tuple:
    """
    Returns (entry_ts, entry_price, invalidation_ts, absorption_ts, trade_type).
    Looks for an absorption candle followed by a confirming entry candle.
    """
    if post_retest.empty:
        return None, None, None, None, None

    if direction == "long":
        invalid_whole = post_retest["close"] < val
    else:
        invalid_whole = post_retest["close"] > vah

    n = len(post_retest)

    if invalid_whole.iloc[0]:
        return None, None, post_retest.index[0], None, None

    for i in range(1, n):
        bar = post_retest.iloc[i]
        ts  = post_retest.index[i]

        if invalid_whole.iloc[i]:
            return None, None, ts, None, None

        baseline = (
            sell_baseline.get(ts, float("nan"))
            if direction == "long"
            else buy_baseline.get(ts, float("nan"))
        )

        if not _is_absorption_candle(bar, baseline, direction, params):
            continue

        absorption_ts    = ts
        absorption_level = float(bar["low"]) if direction == "long" else float(bar["high"])

        abs_scan_end = min(i + 1 + params["entry_after_absorption"], n)
        for j in range(i + 1, abs_scan_end):
            next_bar = post_retest.iloc[j]

            if invalid_whole.iloc[j]:
                return None, None, post_retest.index[j], None, None

            if direction == "long":
                if float(next_bar["close"]) < absorption_level:
                    break
            else:
                if float(next_bar["close"]) > absorption_level:
                    break

            bar_range = float(next_bar["high"]) - float(next_bar["low"])
            if bar_range <= 0:
                continue

            body    = abs(float(next_bar["close"]) - float(next_bar["open"]))
            body_ok = (body / bar_range) >= params["body_threshold"]

            if direction == "long":
                delta_ok = float(next_bar["volume_delta_pct"]) >= params["delta_threshold"]
            else:
                delta_ok = float(next_bar["volume_delta_pct"]) <= -params["delta_threshold"]

            if not (body_ok and delta_ok):
                continue

            entry_bar_idx = j + 1
            if entry_bar_idx >= n:
                return None, None, None, None, None

            if invalid_whole.iloc[entry_bar_idx]:
                return None, None, post_retest.index[entry_bar_idx], None, None

            entry_ts    = post_retest.index[entry_bar_idx]
            entry_price = float(post_retest.iloc[entry_bar_idx]["open"])
            return entry_ts, entry_price, None, absorption_ts, "pure_absorption"

    return None, None, None, None, None


def _find_entry_consecutive_absorption(
    post_retest:   pd.DataFrame,
    direction:     str,
    ivb_high:      float,
    ivb_low:       float,
    poc:           float,
    vah:           float,
    val:           float,
    sell_baseline: pd.Series,
    buy_baseline:  pd.Series,
    params:        dict,
) -> tuple:
    """
    Returns (entry_ts, entry_price, invalidation_ts, absorption_ts, trade_type).

    Scans post_retest bar by bar. For each absorption candle found, checks whether
    n-1 prior absorption candles exist within ±consec_abs_ticks of its absorption
    level. If so, enters on the open of the next bar. No confirmation candle required.

    Uses consec_abs_mult instead of absorption_mult — expected to be lower since
    repetition itself is the signal.
    """
    if post_retest.empty:
        return None, None, None, None, None

    if direction == "long":
        invalid_whole = post_retest["close"] < val
    else:
        invalid_whole = post_retest["close"] > vah

    n            = len(post_retest)
    tick_size    = params["tick_size"]
    level_tol    = params["consec_abs_ticks"] * tick_size
    required_n   = params["consec_abs_n"]

    if invalid_whole.iloc[0]:
        return None, None, post_retest.index[0], None, None

    # Override absorption_mult with the consecutive-specific multiplier
    consec_params = {**params, "absorption_mult": params["consec_abs_mult"]}

    # Running list of (absorption_level, bar_ts) for candles seen so far
    seen: list[tuple[float, pd.Timestamp]] = []

    for i in range(1, n):
        bar = post_retest.iloc[i]
        ts  = post_retest.index[i]

        if invalid_whole.iloc[i]:
            return None, None, ts, None, None

        baseline = (
            sell_baseline.get(ts, float("nan"))
            if direction == "long"
            else buy_baseline.get(ts, float("nan"))
        )

        if not _is_absorption_candle(bar, baseline, direction, consec_params):
            continue

        abs_level = float(bar["low"]) if direction == "long" else float(bar["high"])

        # Record this absorption candle
        body_mid = (min(float(bar["open"]), float(bar["close"])) +
            max(float(bar["open"]), float(bar["close"]))) / 2
        seen.append((abs_level, body_mid, ts))

        # Count how many prior seen levels are within ±level_tol of this one
        nearby = [
            (lvl, t) for lvl, bm, t in seen
            if abs(lvl - abs_level) <= level_tol
            and abs(bm - body_mid) <= level_tol
        ]

        if len(nearby) < required_n:
            continue

        # collect timestamps of all contributing absorption candles
        nearby_ts = [
            t for lvl, bm, t in seen
            if abs(lvl - abs_level) <= level_tol
            and abs(bm - body_mid) <= level_tol
        ]

        # nth hit — enter on open of next bar
        entry_bar_idx = i + 1
        if entry_bar_idx >= n:
            return None, None, None, None, None

        if invalid_whole.iloc[entry_bar_idx]:
            return None, None, post_retest.index[entry_bar_idx], None, None

        entry_ts    = post_retest.index[entry_bar_idx]
        entry_price = float(post_retest.iloc[entry_bar_idx]["open"])
        return entry_ts, entry_price, None, nearby_ts, "consecutive_absorption"

    return None, None, None, None, None


# ---------------------------------------------------------------------------
# Entry dispatcher
# ---------------------------------------------------------------------------

def _find_entry(
    post_retest:   pd.DataFrame,
    direction:     str,
    ivb_high:      float,
    ivb_low:       float,
    poc:           float,
    vah:           float,
    val:           float,
    sell_baseline: pd.Series,
    buy_baseline:  pd.Series,
    params:        dict,
) -> tuple:
    """
    Calls all entry sub-finders and returns the one with the earliest entry_ts.
    If no sub-finder finds an entry, returns the earliest invalidation_ts across all.

    Returns: (entry_ts, entry_price, invalidation_ts, absorption_ts, trade_type)
    """
    shared = dict(
        post_retest   = post_retest,
        direction     = direction,
        ivb_high      = ivb_high,
        ivb_low       = ivb_low,
        poc           = poc,
        vah           = vah,
        val           = val,
        sell_baseline = sell_baseline,
        buy_baseline  = buy_baseline,
        params        = params,
    )

    candidates = [
        _find_entry_pure_absorption(**shared),
        _find_entry_consecutive_absorption(**shared),
        # _find_entry_passive_order(**shared),  # future
        # _find_entry_cvd_confirm(**shared),     # future
    ]

    # Separate hits from misses
    entries       = [c for c in candidates if c[0] is not None]
    invalidations = [c for c in candidates if c[0] is None and c[2] is not None]

    if entries:
        # Pick the earliest entry across all finders
        return min(entries, key=lambda c: c[0])

    if invalidations:
        # Return the earliest invalidation so flip logic triggers as soon as possible
        return min(invalidations, key=lambda c: c[2])

    return None, None, None, None, None


# ---------------------------------------------------------------------------
# SL/TP and trade runner (unchanged)
# ---------------------------------------------------------------------------

def _compute_sl_tp(
    post_retest:  pd.DataFrame,
    entry_ts:     pd.Timestamp,
    entry_price:  float,
    direction:    str,
    val:          float,
    vah:          float,
    params:       dict,
) -> tuple:
    if params["sl_type"] == 0:
        sl = val if direction == "long" else vah
    else:
        bars_to_entry = post_retest.loc[:entry_ts]
        if bars_to_entry.empty:
            sl = val if direction == "long" else vah
        else:
            sl = float(bars_to_entry["low"].min())  if direction == "long" \
            else float(bars_to_entry["high"].max())

    risk = abs(entry_price - sl)
    if risk <= 0:
        return None, None

    tp = entry_price + risk * params["rr"] if direction == "long" \
    else entry_price - risk * params["rr"]

    return sl, tp


def _run_trade(
    post_entry:   pd.DataFrame,
    entry_ts:     pd.Timestamp,
    entry_price:  float,
    direction:    str,
    sl:           float,
    tp:           float,
    params:       dict,
) -> dict:
    timeout     = params["trade_timeout"]
    pre_timeout = post_entry.iloc[:timeout]
    n           = len(pre_timeout)

    if direction == "long":
        sl_hit = pre_timeout["low"]  <= sl
        tp_hit = pre_timeout["high"] >= tp
    else:
        sl_hit = pre_timeout["high"] >= sl
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

    if sl_pos is not None or tp_pos is not None:
        if sl_pos is None:
            return make_trade(pre_timeout.index[tp_pos], tp, "tp", sl, tp)
        if tp_pos is None:
            return make_trade(pre_timeout.index[sl_pos], sl, "sl", sl, tp)
        if tp_pos <= sl_pos:
            return make_trade(pre_timeout.index[tp_pos], tp, "tp", sl, tp)
        else:
            return make_trade(pre_timeout.index[sl_pos], sl, "sl", sl, tp)

    if n < timeout:
        exit_ts    = post_entry.index[-1]
        exit_price = float(post_entry.iloc[-1]["close"])
        return make_trade(exit_ts, exit_price, "eod", sl, tp)

    timeout_close = float(pre_timeout.iloc[-1]["close"])
    in_profit     = (timeout_close > entry_price) if direction == "long" \
               else (timeout_close < entry_price)

    if in_profit:
        return make_trade(pre_timeout.index[-1], timeout_close, "tp_timeout", sl, tp)

    new_tp = entry_price
    new_sl = float(pre_timeout["low"].min())  if direction == "long" \
        else float(pre_timeout["high"].max())

    post_timeout = post_entry.iloc[timeout:]

    if post_timeout.empty:
        exit_ts    = post_entry.index[-1]
        exit_price = float(post_entry.iloc[-1]["close"])
        return make_trade(exit_ts, exit_price, "eod", new_sl, new_tp)

    if direction == "long":
        sl_hit2 = post_timeout["low"]  <= new_sl
        tp_hit2 = post_timeout["high"] >= new_tp
    else:
        sl_hit2 = post_timeout["high"] >= new_sl
        tp_hit2 = post_timeout["low"]  <= new_tp

    sl_pos2 = int(sl_hit2.argmax()) if sl_hit2.any() else None
    tp_pos2 = int(tp_hit2.argmax()) if tp_hit2.any() else None

    if sl_pos2 is None and tp_pos2 is None:
        exit_ts    = post_entry.index[-1]
        exit_price = float(post_entry.iloc[-1]["close"])
        return make_trade(exit_ts, exit_price, "eod", new_sl, new_tp)

    if sl_pos2 is None:
        return make_trade(post_timeout.index[tp_pos2], new_tp, "tp_timeout", new_sl, new_tp)
    if tp_pos2 is None:
        return make_trade(post_timeout.index[sl_pos2], new_sl, "sl_timeout", new_sl, new_tp)

    if tp_pos2 <= sl_pos2:
        return make_trade(post_timeout.index[tp_pos2], new_tp, "tp_timeout", new_sl, new_tp)
    else:
        return make_trade(post_timeout.index[sl_pos2], new_sl, "sl_timeout", new_sl, new_tp)


# ---------------------------------------------------------------------------
# Day processor
# ---------------------------------------------------------------------------

def _process_day(session: pd.DataFrame, params: dict):
    rth_session = session[
        (session.index.time >= time(9, 30)) &
        (session.index.time <  time(16, 0))
    ]

    if len(rth_session) < params["ib_minutes"]:
        return None

    ib_bars   = rth_session.iloc[:params["ib_minutes"]]
    ivb_high  = float(ib_bars["high"].max())
    ivb_low   = float(ib_bars["low"].min())
    ivb_range = ivb_high - ivb_low

    if ivb_range <= 0:
        return None

    poc, vah, val = _compute_ivb_profile(ib_bars)
    if poc is None:
        return None

    sell_baseline, buy_baseline = _build_rolling_baseline(rth_session, params)

    post_ib = rth_session.iloc[params["ib_minutes"]:]
    if post_ib.empty:
        return None

    direction, breakout_pos = _detect_breakout(post_ib, ivb_high, ivb_low)
    if direction is None:
        return None

    max_flips  = params["max_flips"]
    flip_count = 0

    while True:
        breakout_ts = post_ib.index[breakout_pos]

        retest_pos = _detect_retest(
            post_ib       = post_ib,
            breakout_pos  = breakout_pos,
            direction     = direction,
            vah           = vah,
            val           = val,
            poc           = poc,
            retest_window = params["retest_window"],
        )

        if retest_pos is None:
            return None

        retest_ts = post_ib.index[retest_pos]

        post_retest = pd.concat([
            post_ib.iloc[[breakout_pos]],
            post_ib.iloc[retest_pos : retest_pos + params["entry_window"]]
        ])

        entry_ts, entry_price, invalidation_ts, absorption_ts, trade_type = _find_entry(
            post_retest   = post_retest,
            direction     = direction,
            ivb_high      = ivb_high,
            ivb_low       = ivb_low,
            poc           = poc,
            vah           = vah,
            val           = val,
            sell_baseline = sell_baseline,
            buy_baseline  = buy_baseline,
            params        = params,
        )

        if entry_ts is not None:
            break

        if invalidation_ts is None:
            return None

        if flip_count >= max_flips:
            return None

        flip_count += 1
        direction = "short" if direction == "long" else "long"

        resume_pos = post_ib.index.searchsorted(invalidation_ts)
        post_ib    = post_ib.iloc[resume_pos:]

        if post_ib.empty:
            return None

        direction_found, breakout_pos = _detect_breakout(post_ib, ivb_high, ivb_low)

        if direction_found != direction:
            return None

    sl, tp = _compute_sl_tp(
        post_retest = post_retest,
        entry_ts    = entry_ts,
        entry_price = entry_price,
        direction   = direction,
        val         = val,
        vah         = vah,
        params      = params,
    )

    if sl is None:
        return None

    post_entry = post_ib.loc[entry_ts:]

    trade = _run_trade(
        post_entry  = post_entry,
        entry_ts    = entry_ts,
        entry_price = entry_price,
        direction   = direction,
        sl          = sl,
        tp          = tp,
        params      = params,
    )

    trade["trade_type"] = trade_type

    trade["notes"] = json.dumps({
        "breakout_time":   str(breakout_ts),
        "retest_time":     str(retest_ts),
        "absorption_time": [str(t) for t in absorption_ts] if isinstance(absorption_ts, list) else str(absorption_ts),
        "flip_count":      flip_count,
        "ivb_high":        ivb_high,
        "ivb_low":         ivb_low,
        "poc":             poc,
        "vah":             vah,
        "val":             val,
    })

    return trade


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    folder_path: Path,
    start_date:  pd.Timestamp,
    end_date:    pd.Timestamp,
    params:      dict | None = None,
) -> pd.DataFrame:
    merged_params = {**PARAMS, **(params or {})}

    files = sorted(Path(folder_path).glob("*.parquet"))
    files = [
        f for f in files
        if f.stem[0].isdigit()
        and start_date.date() <= pd.Timestamp(f.stem).date() <= end_date.date()
    ]

    if not files:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    trades = []
    for f in files:
        session = pd.read_parquet(f)
        if session.empty:
            continue
        if session.index.tz is None:
            continue
        trade = _process_day(session, merged_params)
        if trade is not None:
            trade["date"] = pd.Timestamp(f.stem).date()
            trades.append(trade)

    if not trades:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    return pd.DataFrame(trades)[_OUTPUT_COLUMNS]


'''
best results
ib_minutes 30
delta_threshold 30
body_threshold 0.5
rr 2
sl_type 1
retest_window 40
entry_window 25
entry_after_absorption 5
trade_timeout 120
max_flips 4
wick_threshold 0.3
absorption_mult 2
absorption_window 20
tick_size 0.25
'''