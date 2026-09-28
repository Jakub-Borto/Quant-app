"""prepare_day's result: one RTH session as numpy arrays + its summary.

Everything here depends only on the day's files and `rth_start` (declared in
PREPARE_PARAMS), so the engine caches it and reuses it for every later run,
every optimizer combo, and for `day.previous` lookups from the next days.
Nothing in here may be modified after it is built.
"""

import numpy as np
import pandas as pd

RTH_END = "16:00"


class RthDay:
    """The RTH bars [rth_start, 16:00) of one day."""

    __slots__ = ("index", "open", "high", "low", "close", "vwap", "cvd",
                 "rth_high", "rth_low", "rth_close", "rth_range")

    def __init__(self, candles: pd.DataFrame, indicators: pd.DataFrame, date, rth_start: str):
        tz = candles.index.tz
        start = pd.Timestamp(f"{date} {rth_start}", tz=tz)
        end = pd.Timestamp(f"{date} {RTH_END}", tz=tz)
        rth = candles[(candles.index >= start) & (candles.index < end)]
        # indicators are aligned to the candles by timestamp (same 1-minute grid)
        ind = indicators.reindex(rth.index)

        self.index = rth.index
        self.open = rth["open"].to_numpy(dtype=np.float64)
        self.high = rth["high"].to_numpy(dtype=np.float64)
        self.low = rth["low"].to_numpy(dtype=np.float64)
        self.close = rth["close"].to_numpy(dtype=np.float64)
        self.vwap = ind["vwap_bar_rth"].to_numpy(dtype=np.float64)
        self.cvd = ind["cumulative_delta"].to_numpy(dtype=np.float64)
        self.rth_high = float(self.high.max())
        self.rth_low = float(self.low.min())
        self.rth_close = float(self.close[-1])
        self.rth_range = self.rth_high - self.rth_low


def build(date, candles: pd.DataFrame, indicators: pd.DataFrame, rth_start: str):
    """RthDay, or None when the day has fewer than 30 RTH bars (the engine then
    skips the day, and later days see it as a previous day with prepared=None)."""
    tz = candles.index.tz
    start = pd.Timestamp(f"{date} {rth_start}", tz=tz)
    end = pd.Timestamp(f"{date} {RTH_END}", tz=tz)
    n = int(((candles.index >= start) & (candles.index < end)).sum())
    if n < 30:
        return None
    return RthDay(candles, indicators, date, rth_start)
