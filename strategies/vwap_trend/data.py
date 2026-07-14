"""Day loading + the param-independent day-core cache.

A DayCore holds one FULL joined session (candles ∩ indicators) as numpy
arrays; the per-run trading-window slice is two searchsorted calls on top of
it. It depends only on the two files, never on strategy params — the same
holder-module trick as ivb_model / orb keeps the cache alive across the
backtester's plugin re-loads, so optimizer sweeps re-read nothing.

Both vwap_bar_* columns are stored (when present), so `vwap_anchor` is not a
cache dimension either.
"""

import sys
import types
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

CANDLE_COLUMNS    = ["open", "close", "volume"]          # high/low unused: no intrabar logic
INDICATOR_COLUMNS = ["vwap_bar_rth", "vwap_bar_globex"]

_STORE_NAME = "_vwap_trend_day_cache_store"
_store = sys.modules.get(_STORE_NAME)
if _store is None:
    _store = types.ModuleType(_STORE_NAME)
    _store.cache = OrderedDict()
    sys.modules[_STORE_NAME] = _store
DAY_CACHE: OrderedDict = _store.cache
DAY_CACHE_MAX_DAYS = 5000        # ~65 KB per cached globex day -> ~325 MB ceiling
MISS = object()                  # cache-miss sentinel
SKIP = "skip"                    # cached marker: indicators missing/unreadable for this day


def cache_key(candle_file: Path, ind_file: Path):
    """Identity of a day: both files' (path, mtime, size). Editing either file
    (or re-running a transform) invalidates naturally. Params play no part."""
    st = candle_file.stat()
    ind_id = None
    if ind_file.exists():
        ist = ind_file.stat()
        ind_id = (str(ind_file), ist.st_mtime_ns, ist.st_size)
    return (str(candle_file), st.st_mtime_ns, st.st_size, ind_id)


def read_candles(f: Path) -> pd.DataFrame:
    """Column-pruned read; full-read fallback for files missing any column."""
    try:
        return pd.read_parquet(f, columns=CANDLE_COLUMNS)
    except Exception:
        return pd.read_parquet(f)


def read_indicators(f: Path):
    """Column-pruned read with full-read fallback; any problem -> None
    (the caller skips the day with a warning)."""
    try:
        return pd.read_parquet(f, columns=INDICATOR_COLUMNS)
    except Exception:
        try:
            return pd.read_parquet(f)
        except Exception:
            return None


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
        joined = session[["open", "close", "volume"]].join(ind[vwap_cols], how="inner")

        self.index  = joined.index
        self.i8     = joined.index.asi8
        self.open   = joined["open"].to_numpy(dtype=np.float64)
        self.close  = joined["close"].to_numpy(dtype=np.float64)
        self.volume = joined["volume"].to_numpy(dtype=np.float64)
        self.vwap   = {c: joined[c].to_numpy(dtype=np.float64) for c in vwap_cols}


def build_day_core(session: pd.DataFrame, ind) -> "DayCore | str":
    """(candles, indicators) -> DayCore, or SKIP when the day is unusable
    (empty candles / no indicators frame)."""
    if session is None or session.empty or ind is None:
        return SKIP
    return DayCore(session, ind)
