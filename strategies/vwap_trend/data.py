"""The param-independent day core, built once per day by prepare_day() and
cached by the engine (modules.engine) across runs and optimizer combos.

A DayCore holds one FULL joined session (candles ∩ indicators) as numpy
arrays; the per-run trading-window slice is two searchsorted calls on top of
it. It depends only on the two day files, never on strategy params. Both
vwap_bar_* columns are stored, so `vwap_anchor` is not a cache dimension.
"""

import numpy as np
import pandas as pd

CANDLE_COLUMNS    = ["open", "close", "volume"]          # high/low unused: no intrabar logic
INDICATOR_COLUMNS = ["vwap_bar_rth", "vwap_bar_globex"]


def _ns_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Normalize to ns resolution — some datasets store datetime64[us], and
    int64 position math must agree with Timestamp.value (always ns)."""
    idx = df.index
    return idx if idx.unit == "ns" else idx.as_unit("ns")


class DayCore:
    """One joined session as positional arrays. Param-independent."""

    __slots__ = ("index", "i8", "open", "close", "volume", "vwap")

    def __init__(self, session: pd.DataFrame, ind: pd.DataFrame):
        session = session.set_axis(_ns_index(session))
        ind     = ind.set_axis(_ns_index(ind))
        vwap_cols = [c for c in INDICATOR_COLUMNS if c in ind.columns]
        joined = session[CANDLE_COLUMNS].join(ind[vwap_cols], how="inner")

        self.index  = joined.index
        self.i8     = joined.index.asi8
        self.open   = joined["open"].to_numpy(dtype=np.float64)
        self.close  = joined["close"].to_numpy(dtype=np.float64)
        self.volume = joined["volume"].to_numpy(dtype=np.float64)
        self.vwap   = {c: joined[c].to_numpy(dtype=np.float64) for c in vwap_cols}


def build_day_core(session: pd.DataFrame, ind: pd.DataFrame) -> "DayCore | None":
    """(candles, indicators) -> DayCore, or None for an empty session (the
    engine then skips the day)."""
    if session.empty:
        return None
    return DayCore(session, ind)
