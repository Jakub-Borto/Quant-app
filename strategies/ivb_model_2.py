import json
import pandas as pd
from pathlib import Path
from datetime import time


# FOR BACKTESTER TO SHOW IT NICELY
PARAM_SECTIONS = {
    "General":                  ["ib_minutes", "rr", "sl_type", "trade_timeout", "max_flips", "valid_entries"],
    "Entry Windows":            ["retest_window", "entry_window", "entry_after_absorption"],
    "Candle Filters":           ["delta_threshold", "body_threshold"],
    "Absorption":               ["wick_threshold", "absorption_mult", "absorption_window"],
    "Consecutive Absorption":   ["consec_abs_n", "consec_abs_mult", "consec_abs_ticks"],
    "Two Bar Absorption":       ["two_bar_wick_ticks", "two_bar_abs_mult"],
    "Passive Absorption":       ["passive_order_mult", "passive_absorption_mult"],
}


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
    "consec_abs_n":             2,      # number of absorption candles required at same level
    "consec_abs_mult":          2.0,    # absorption multiplier for consecutive absorption finder
    "consec_abs_ticks":         4,      # ±ticks tolerance for grouping absorption levels
    # --- two bar absorption params ---
    "two_bar_wick_ticks":       3,      # max wick size in ticks on defended side for both candles
    "two_bar_abs_mult":         2.0,    # absorption multiplier for merged 2-bar candle
    # --- passive order + absorption params ---
    "passive_order_mult":       3.0,    # passive avg order size must be >= this x rolling baseline
    "passive_absorption_mult":  1.5,    # absorption mult for passive+absorption finder
    # --- which entries to look for (1=on, 0=off): pure, consec, two_bar, passive) ---
    "valid_entries":            "1111",
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

    smoothed = []
    for i in range(n):
        lo = max(0, i - 1)
        hi = min(n, i + 2)
        smoothed.append(sum(volumes[lo:hi]) / (hi - lo))

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

    poc_candidates = []
    for peak_idx in cluster_peaks:
        lo = max(0, peak_idx - 3)
        hi = min(n, peak_idx + 4)
        best_idx = max(range(lo, hi), key=lambda i: volumes[i])
        poc_candidates.append(sorted_prices[best_idx])

    poc_candidates = list(dict.fromkeys(poc_candidates))

    total_volume = sum(volumes)
    target       = total_volume * 0.70

    best_poc   = None
    best_vah   = None
    best_val   = None
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

        vah      = sorted_prices[hi_idx]
        val      = sorted_prices[lo_idx]
        va_range = vah - val

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
# Rolling baselines
# ---------------------------------------------------------------------------

def _build_rolling_baseline(rth_session: pd.DataFrame, params: dict):
    """Volume-per-tick rolling baseline for absorption detection."""
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


def _build_passive_baseline(rth_session: pd.DataFrame, direction: str, params: dict) -> pd.Series:
    window = params["absorption_window"]

    valid = rth_session[rth_session.index.time >= time(9, 35)].copy()

    def max_avg_order_size(row):
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
            avg = size / count
            if pd.isna(best) or avg > best:
                best = avg

        return best

    per_bar = valid.apply(max_avg_order_size, axis=1)

    # rolling over valid (non-NaN) observations only
    sparse  = per_bar.dropna()
    rolling = sparse.rolling(window, min_periods=window).mean()
    return rolling.reindex(rth_session.index)


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
    Looks for an absorption candle followed by a confirming entry candle.
    """
    if post_retest.empty:
        return None, None, None, None, None

    if direction == "long":
        invalid_whole = post_retest["close"] < val
    else:
        invalid_whole = post_retest["close"] > vah

    n = len(post_retest)

    # check breakout bar for invalidation only
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
    Scans post_retest bar by bar. For each absorption candle found, checks whether
    n-1 prior absorption candles exist within ±consec_abs_ticks of its absorption
    level. If so, enters on the open of the next bar. No confirmation candle required.
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

    consec_params = {**params, "absorption_mult": params["consec_abs_mult"]}

    seen: list[tuple[float, float, pd.Timestamp]] = []

    # check breakout bar for invalidation only
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

        if not _is_absorption_candle(bar, baseline, direction, consec_params):
            continue

        abs_level = float(bar["low"]) if direction == "long" else float(bar["high"])
        body_mid  = (min(float(bar["open"]), float(bar["close"])) +
                     max(float(bar["open"]), float(bar["close"]))) / 2

        seen.append((abs_level, body_mid, ts))

        nearby = [
            (lvl, t) for lvl, bm, t in seen
            if abs(lvl - abs_level) <= level_tol
            and abs(bm - body_mid) <= level_tol
        ]

        if len(nearby) < required_n:
            continue

        entry_bar_idx = i + 1
        if entry_bar_idx >= n:
            return None, None, None, None, None

        if invalid_whole.iloc[entry_bar_idx]:
            return None, None, post_retest.index[entry_bar_idx], None, None

        nearby_ts   = [t for _, t in nearby]
        entry_ts    = post_retest.index[entry_bar_idx]
        entry_price = float(post_retest.iloc[entry_bar_idx]["open"])
        return entry_ts, entry_price, None, nearby_ts, "consecutive_absorption"

    return None, None, None, None, None


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


def _build_two_bar_baseline(
    pre_bars:  pd.DataFrame,
    direction: str,
    params:    dict,
) -> float:
    """
    Build baseline from merged 2-bar pairs ending at the last bar of pre_bars.
    Window = absorption_window // 2 pairs. Anchored backwards from the pattern bars.
    """
    tick_size = params["tick_size"]
    n_pairs   = params["absorption_window"] // 2

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


def _find_entry_two_bar_absorption(
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
    Pattern (long bias):
      - Bar i:   bearish, small bottom wick (<= two_bar_wick_ticks)
      - Bar i+1: bullish, small bottom wick (<= two_bar_wick_ticks), just positive delta

    Pattern (short bias):
      - Bar i:   bullish, small upper wick (<= two_bar_wick_ticks)
      - Bar i+1: bearish, small upper wick (<= two_bar_wick_ticks), just negative delta

    Merges both bars, checks absorption in defended wick against 2-bar paired baseline.
    Then scans from bar i+1 (inclusive) for a confirmation candle:
    correct direction + body_threshold + delta_threshold.
    Enters on open of the bar after the confirmation candle.
    """
    if post_retest.empty:
        return None, None, None, None, None

    if direction == "long":
        invalid_whole = post_retest["close"] < val
    else:
        invalid_whole = post_retest["close"] > vah

    n             = len(post_retest)
    tick_size     = params["tick_size"]
    max_wick      = params["two_bar_wick_ticks"] * tick_size
    required_mult = params["two_bar_abs_mult"]

    # check breakout bar for invalidation only
    if invalid_whole.iloc[0]:
        return None, None, post_retest.index[0], None, None

    for i in range(1, n - 1):
        bar1 = post_retest.iloc[i]
        bar2 = post_retest.iloc[i + 1]
        ts1  = post_retest.index[i]
        ts2  = post_retest.index[i + 1]

        if invalid_whole.iloc[i] or invalid_whole.iloc[i + 1]:
            return None, None, post_retest.index[i], None, None

        # --- bar1 direction + small wick ---
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

        # --- bar2 direction + small wick + just positive/negative delta ---
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

        # --- build 2-bar baseline from bars before bar i ---
        pre_bars = post_retest.iloc[1:i]
        baseline = _build_two_bar_baseline(pre_bars, direction, params)

        if pd.isna(baseline) or baseline <= 0:
            continue

        required = baseline * required_mult

        # --- check absorption in defended wick of merged candle ---
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

        try:
            raw_tv = json.loads(merged_tv)
        except Exception:
            continue

        absorbed = False
        for price_str, (buy_qty, sell_qty) in raw_tv.items():
            price = float(price_str)
            if not (wick_low <= price <= wick_high):
                continue
            volume_at_level = sell_qty if direction == "long" else buy_qty
            if volume_at_level >= required:
                absorbed = True
                break

        if not absorbed:
            continue

        # --- scan for confirmation candle starting from bar2 (inclusive) ---
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
            return entry_ts, entry_price, None, [ts1, ts2], "two_bar_absorption"

    return None, None, None, None, None


def _find_entry_passive_absorption(
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

    # check breakout bar for invalidation only
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

        bar_open        = float(bar["open"])
        required_po     = p_baseline * params["passive_order_mult"]
        has_big_passive = False

        for price_str, (size, count) in raw_po.items():
            price = float(price_str)
            if count <= 0:
                continue
            if direction == "long" and price >= bar_open:
                continue
            if direction == "short" and price <= bar_open:
                continue
            if (size / count) >= required_po:
                has_big_passive = True
                break

        if not has_big_passive:
            continue

        # --- absorption check on the same candle ---
        abs_baseline = (
            sell_baseline.get(ts, float("nan"))
            if direction == "long"
            else buy_baseline.get(ts, float("nan"))
        )

        if not _is_absorption_candle(bar, abs_baseline, direction, passive_params):
            continue

        absorption_ts = ts

        # --- scan for confirmation candle ---
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
            return entry_ts, entry_price, None, absorption_ts, "passive_absorption"

    return None, None, None, None, None


# ---------------------------------------------------------------------------
# Entry dispatcher
# ---------------------------------------------------------------------------

def _find_entry(
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
    Calls all entry sub-finders and returns the one with the earliest entry_ts.
    If no sub-finder finds an entry, returns the earliest invalidation_ts across all.

    Returns: (entry_ts, entry_price, invalidation_ts, absorption_ts, trade_type)
    """
    shared = dict(
        post_retest            = post_retest,
        direction              = direction,
        ivb_high               = ivb_high,
        ivb_low                = ivb_low,
        poc                    = poc,
        vah                    = vah,
        val                    = val,
        sell_baseline          = sell_baseline,
        buy_baseline           = buy_baseline,
        passive_baseline_long  = passive_baseline_long,
        passive_baseline_short = passive_baseline_short,
        params                 = params,
    )

    finder_registry = [
        _find_entry_pure_absorption,
        _find_entry_consecutive_absorption,
        _find_entry_two_bar_absorption,
        _find_entry_passive_absorption,
    ]

    valid_entries = params.get("valid_entries", "1111")
    flags         = valid_entries.ljust(len(finder_registry), "0")

    candidates = [
        fn(**shared)
        for fn, flag in zip(finder_registry, flags)
        if flag == "1"
    ]

    entries       = [c for c in candidates if c[0] is not None]
    invalidations = [c for c in candidates if c[0] is None and c[2] is not None]

    if entries:
        return min(entries, key=lambda c: c[0])

    if invalidations:
        return min(invalidations, key=lambda c: c[2])

    return None, None, None, None, None


# ---------------------------------------------------------------------------
# SL/TP and trade runner
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
    passive_baseline_long       = _build_passive_baseline(rth_session, "long",  params)
    passive_baseline_short      = _build_passive_baseline(rth_session, "short", params)

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
            post_retest            = post_retest,
            direction              = direction,
            ivb_high               = ivb_high,
            ivb_low                = ivb_low,
            poc                    = poc,
            vah                    = vah,
            val                    = val,
            sell_baseline          = sell_baseline,
            buy_baseline           = buy_baseline,
            passive_baseline_long  = passive_baseline_long,
            passive_baseline_short = passive_baseline_short,
            params                 = params,
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

    abs_time = absorption_ts
    if isinstance(abs_time, list):
        abs_time = [t.strftime("%H:%M") if hasattr(t, "strftime") else str(t) for t in abs_time]
    elif hasattr(abs_time, "strftime"):
        abs_time = abs_time.strftime("%H:%M")
    else:
        abs_time = str(abs_time)

    trade["notes"] = json.dumps({
        "breakout_time":   breakout_ts.strftime("%H:%M"),
        "retest_time":     retest_ts.strftime("%H:%M"),
        "absorption_time": abs_time,
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


# OPEN QUESTION — direction reclaim after invalidation
# Current behaviour: long bias invalidated (close < VAL) -> flip to short ->
# if price then breaks IVB high again (close > ivb_high), the day is killed (return None).
# No re-entry on the original direction is allowed.
#
# Question: should a reclaim of IVB high after invalidation re-enable the long thesis?
# Options:
#   A) Keep current — once invalidated, original direction is dead for the day (conservative)
#   B) Allow re-entry long if price reclaims ivb_high — but this risks chasing
#      a choppy range-bound day with multiple false breakouts
#   C) Allow re-entry only if price reclaims ivb_high AND retests VAH cleanly again
#
# Revisit when sample size is larger and you can see how often this scenario occurs.


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