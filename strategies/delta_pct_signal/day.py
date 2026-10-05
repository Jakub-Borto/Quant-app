"""prepare_day: the day's arrays + the rolling volume median (cached by the engine).

Everything here depends only on the files and on `volume_lookback` (the one
PREPARE_PARAM), never on the sweepable signal / trade params.
"""

import numpy as np


class DeltaDay:
    """One day file as positional numpy arrays (read-only once built)."""
    __slots__ = ("date", "index", "open", "close", "volume", "delta", "median", "n")

    def __init__(self, date, frame, lookback: int):
        self.date = date
        self.index = frame.index
        self.open = frame["open"].to_numpy(dtype=np.float64)
        self.close = frame["close"].to_numpy(dtype=np.float64)
        self.volume = frame["volume"].to_numpy(dtype=np.float64)
        self.delta = frame["volume_delta_pct"].to_numpy(dtype=np.float64)
        self.n = len(self.index)
        # median of the `lookback` bars BEFORE each bar (zero-volume filled bars
        # included); NaN during the warm-up (fewer than `lookback` earlier bars)
        median = np.full(self.n, np.nan)
        if self.n > lookback:
            windows = np.lib.stride_tricks.sliding_window_view(self.volume, lookback)
            median[lookback:] = np.median(windows[:-1], axis=1)
        self.median = median
        for arr in (self.open, self.close, self.volume, self.delta, self.median):
            arr.setflags(write=False)


def build(date, frame, lookback):
    # validated here too: prepare_day runs before process_day's validation
    if isinstance(lookback, bool) or not float(lookback).is_integer() or lookback < 1:
        raise ValueError(f"volume_lookback must be a whole number >= 1 (got {lookback!r})")
    if len(frame) < 2 or frame.index.tz is None:
        return None
    return DeltaDay(date, frame, int(lookback))
