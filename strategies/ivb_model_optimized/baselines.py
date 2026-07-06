"""Day-level rolling baselines for absorption / passive / CVD-change grading.

All three builders now take the DayData context plus `valid_pos` (positions of the bars at or
after the warm-up start = session_start + BASELINE_WARMUP_MINUTES) and return full-length
numpy arrays positionally aligned to the
RTH session (NaN outside the warm window / during rolling warmup). The pandas rolling
mean/std pipelines are kept verbatim so the float results stay bit-identical to the original
Series versions — only the per-row JSON parsing / .apply overhead is gone (the passive
per-bar max sizes are computed once in DayData for both directions).
"""

import numpy as np
import pandas as pd

# Skip the first 5 minutes of the session before baselining — the opening bar(s) are abnormally
# heavy and would distort the rolling volume-per-tick averages. Shared by all baselines below;
# the absolute warm-up start is session_start + this offset (core passes `valid_pos` in).
BASELINE_WARMUP_MINUTES = 5


def build_rolling_baseline(day, valid_pos: np.ndarray, params: dict):
    """Volume-per-tick rolling baseline for absorption detection.

    Returns (sell_baseline, buy_baseline) arrays aligned to the session.
    """
    tick_size = params["tick_size"]
    window    = params["absorption_baseline_window"]

    h = day.high[valid_pos]
    l = day.low[valid_pos]
    range_ticks = (h - l) / tick_size
    range_ticks[range_ticks == 0] = np.nan

    sell_rolled = (
        pd.Series(day.sell_vol[valid_pos] / range_ticks)
        .rolling(window, min_periods=window).mean().to_numpy()
    )
    buy_rolled = (
        pd.Series(day.buy_vol[valid_pos] / range_ticks)
        .rolling(window, min_periods=window).mean().to_numpy()
    )

    sell_baseline = np.full(day.n, np.nan)
    buy_baseline  = np.full(day.n, np.nan)
    sell_baseline[valid_pos] = sell_rolled
    buy_baseline[valid_pos]  = buy_rolled
    return sell_baseline, buy_baseline


def build_passive_baseline(day, valid_pos: np.ndarray, direction: str, params: dict) -> np.ndarray:
    """Rolling baseline of the max raw resting size on the defended side per bar.

    Long: passive buy orders below candle open. Short: passive sell orders above candle open.
    Only bars with valid passive data on the defended side contribute to the window (the
    rolling window slides over the SPARSE sequence of bars-with-data, as before).
    """
    window = params["absorption_baseline_window"]

    best  = day.best_passive_long if direction == "long" else day.best_passive_short
    vals  = best[valid_pos]
    notna = ~np.isnan(vals)

    rolled = (
        pd.Series(vals[notna])
        .rolling(window, min_periods=window).mean().to_numpy()
    )

    out = np.full(day.n, np.nan)
    out[valid_pos[notna]] = rolled
    return out


def build_cvd_change_baseline(day, valid_pos: np.ndarray, params: dict) -> np.ndarray:
    """Rolling std of bar-to-bar CVD changes, aligned to the session (NaN during warmup)."""
    window = params["absorption_baseline_window"]

    std = (
        pd.Series(day.cvd[valid_pos])
        .diff().rolling(window, min_periods=window).std().to_numpy()
    )

    out = np.full(day.n, np.nan)
    out[valid_pos] = std
    return out
