"""The param-independent day core: the RTH session (09:30-16:00 NY) as numpy
arrays. Built once per day by prepare_day() and cached by the engine."""

import numpy as np
import pandas as pd

RTH_OPEN  = "09:30"
RTH_CLOSE = "16:00"

CANDLE_COLUMNS = ["open", "high", "low", "close"]


class DayCore:
    """One RTH session as positional numpy arrays."""

    __slots__ = ("index", "i8", "open", "high", "low", "close", "rth_start")

    def __init__(self, session: pd.DataFrame, rth_date):
        idx = session.index
        rth_start = pd.Timestamp(f"{rth_date} {RTH_OPEN}",  tz=idx.tz)
        rth_end   = pd.Timestamp(f"{rth_date} {RTH_CLOSE}", tz=idx.tz)
        # asi8 is in the index's own unit (some datasets store datetime64[us]);
        # Timestamp/Timedelta .value are always ns, so compare in ns
        i8 = idx.asi8 if idx.unit == "ns" else idx.as_unit("ns").asi8
        i0 = int(i8.searchsorted(rth_start.value, side="left"))
        i1 = int(i8.searchsorted(rth_end.value,   side="right"))
        self.index = idx[i0:i1]
        self.i8    = i8[i0:i1]
        self.open  = session["open"].to_numpy(dtype=np.float64)[i0:i1]
        self.high  = session["high"].to_numpy(dtype=np.float64)[i0:i1]
        self.low   = session["low"].to_numpy(dtype=np.float64)[i0:i1]
        self.close = session["close"].to_numpy(dtype=np.float64)[i0:i1]
        self.rth_start = rth_start


def build_day_core(session: pd.DataFrame, rth_date) -> "DayCore | None":
    """None for an empty file (the engine then skips the day)."""
    if session.empty:
        return None
    return DayCore(session, rth_date)
