"""IVB volume profile computation (peak-based POC / VAH / VAL)."""


def compute_ivb_profile(day, ib_end: int) -> tuple:
    """
    Compute POC, VAH, VAL from tick_volume within the IB bars (day positions [0, ib_end)).

    Algorithm:
      1. Build raw volume-at-price dict from the pre-parsed tick_volume arrays.
      2. Smooth with a 3-tick rolling average to remove single-tick spikes.
      3. Find local maxima (smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]).
      4. Cluster peaks within 4 ticks of each other — keep only the highest per cluster.
      5. Take up to 5 peak candidates. For each, scan ±3 ticks in the RAW profile
         to find the actual highest-volume tick — that becomes the POC candidate.
      6. Expand VA outward from each POC candidate until 70% of total volume captured.
      7. Pick the VA with the tightest price range (smallest VAH - VAL).
      8. Return (poc, vah, val) for the winning candidate.

    Falls back to simple max-volume POC if fewer than 3 price levels exist.

    The dict accumulation iterates bars then levels in document order (python ints), so
    values, insertion order and the rare tie-breaks stay identical to the JSON-loop original;
    only the repeated json.loads is gone. The peak / VA-expansion logic is O(#levels) and
    FP-order sensitive, so it is kept as-is.
    """
    levels = {}

    tick_volume = day.tick_volume
    for i in range(ib_end):
        tv = tick_volume[i]
        if tv is None:
            continue
        prices, buys, sells = tv
        totals = buys + sells
        for price, total in zip(prices.tolist(), totals.tolist()):
            if price in levels:
                levels[price] += total
            else:
                levels[price] = total

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
