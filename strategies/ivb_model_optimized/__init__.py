"""
IVB Model 2 — modular package (vectorized).

The backtester loads this package via __init__.py and expects:
  - run(folder_path, start_date, end_date, params) -> pd.DataFrame
  - PARAMS, PARAM_SECTIONS

Internal layout:
  params.py      PARAMS, PARAM_SECTIONS, OUTPUT_COLUMNS
  _timing.py     accumulating stage timers (one report printed per run)
  _daydata.py    per-day numpy context: parsed JSON + positional windows/masks
  profile.py     compute_ivb_profile
  baselines.py   rolling / passive / cvd-change day-level baselines
  absorption.py  shared absorption level scan on pre-parsed tick_volume
  entries/       one module per entry type (7), registered in FINDER_REGISTRY
  risk/          self-contained risk scripts in RISK_REGISTRY (selected by risk_script)
  core.py        breakout/retest detection, entry dispatcher, process_day
"""

import sys
import time
import types
from collections import deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

from .params  import PARAMS, PARAM_SECTIONS, OUTPUT_COLUMNS
from .core    import process_day, build_day_core, VWAP_BAND_COLUMNS
from . import _timing
from ._timing import timed


# ---------------------------------------------------------------------------
# In-memory day-core cache (persists across run() calls in the same process)
# ---------------------------------------------------------------------------
# A DayData is param-independent (see _daydata), so re-running the backtester with different
# params — the normal research loop — skips file reads, JSON parsing and array building
# entirely. Keyed by (path, mtime_ns, size) of BOTH the candle and indicator file, so
# re-running a transform invalidates naturally. LRU-capped in days (a fully parsed enriched
# ES day is ~0.3 MB, so the default cap is a ~200 MB ceiling).
#
# The backtester's plugin loader RE-EXECUTES this __init__ on every backtest run
# (spec_from_file_location + exec_module), which would wipe a plain module-level dict. The
# dict therefore lives on a tiny holder module registered once in sys.modules — that survives
# for the lifetime of the Python process, however many times the package is re-loaded.
_STORE_NAME = "_ivb_day_cache_store"
_store = sys.modules.get(_STORE_NAME)
if _store is None:
    _store = types.ModuleType(_STORE_NAME)
    _store.cache = OrderedDict()
    sys.modules[_STORE_NAME] = _store
_DAY_CACHE: OrderedDict = _store.cache
_DAY_CACHE_MAX_DAYS = 600
_MISS = object()


# only these candle columns are consumed (see CLAUDE.md) — `volume` / `volume_delta` are not
CANDLE_COLUMNS = [
    "open", "high", "low", "close",
    "buy_volume", "sell_volume", "volume_delta_pct",
    "tick_volume", "passive_orders",
]

# indicators: CVD + the 8 tick-vwap ±2σ/±3σ band columns (of ~32 in the file)
INDICATOR_COLUMNS = ["cumulative_delta"] + VWAP_BAND_COLUMNS


def _read_candles(f: Path) -> pd.DataFrame:
    """Column-pruned read; falls back to a full read for files missing any of the columns."""
    try:
        return pd.read_parquet(f, columns=CANDLE_COLUMNS)
    except Exception:
        return pd.read_parquet(f)


def _read_indicators(f: Path):
    """Column-pruned read with full-read fallback; any problem => None (day runs without
    indicators, exactly as before)."""
    try:
        return pd.read_parquet(f, columns=INDICATOR_COLUMNS)
    except Exception:
        try:
            return pd.read_parquet(f)
        except Exception:
            return None


def run(
    folder_path: Path,
    start_date:  pd.Timestamp,
    end_date:    pd.Timestamp,
    params:      dict | None = None,
) -> pd.DataFrame:
    t0 = time.perf_counter()
    _timing.reset()

    merged_params = {**PARAMS, **(params or {})}

    folder_path = Path(folder_path)

    files = sorted(folder_path.glob("*.parquet"))
    files = [
        f for f in files
        if f.stem[0].isdigit()
        and start_date.date() <= pd.Timestamp(f.stem).date() <= end_date.date()
    ]

    if not files:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # --- resolve the indicators folder (sibling dataset under same type/asset) ---
    # candle folder_path is .../parquet/{type}/{asset}/{dataset}; the indicators live in a
    # folder the user names. Empty param => no indicators (CVD entries + vwap risk disabled).
    ind_folder_name   = merged_params.get("indicators_folder", "")
    indicators_folder = folder_path.parent / ind_folder_name if ind_folder_name else None

    def _key(f: Path):
        """Cache identity of a day: candle file + its indicator sibling (or None)."""
        st = f.stat()
        ind_id = None
        if indicators_folder is not None:
            ind_file = indicators_folder / f.name
            if ind_file.exists():
                ist    = ind_file.stat()
                ind_id = (str(ind_file), ist.st_mtime_ns, ist.st_size)
        return (str(f), st.st_mtime_ns, st.st_size, ind_id)

    def _load(f: Path):
        """Read one day's candles + indicators (runs on a prefetch thread)."""
        with timed("io:read_candles"):
            session = _read_candles(f)

        # per-day indicators: matching YYYY-MM-DD.parquet in the indicators folder. Any problem
        # (no folder, missing file, bad read) => None => CVD entries + vwap risk disable this day.
        ind_df = None
        if indicators_folder is not None:
            ind_file = indicators_folder / f.name
            if ind_file.exists():
                with timed("io:read_indicators"):
                    ind_df = _read_indicators(ind_file)
        return session, ind_df

    def _submit(f: Path, executor):
        """Cache lookup at prefetch time; only misses hit the reader thread."""
        key  = _key(f)
        core = _DAY_CACHE.get(key, _MISS)
        if core is not _MISS:
            _DAY_CACHE.move_to_end(key)
            return (f, key, core, None)
        return (f, key, _MISS, executor.submit(_load, f))

    # sliding-window prefetch: the next few days' parquet reads overlap the current day's
    # compute (pyarrow releases the GIL). Days are still consumed strictly in file order,
    # so the output is identical to the sequential loop.
    PREFETCH = 4
    trades = []
    hits = misses = 0
    with ThreadPoolExecutor(max_workers=2) as executor:
        queue: deque = deque()
        file_iter = iter(files)
        for f in file_iter:
            queue.append(_submit(f, executor))
            if len(queue) >= PREFETCH:
                break

        while queue:
            f, key, core, fut = queue.popleft()
            nxt = next(file_iter, None)
            if nxt is not None:
                queue.append(_submit(nxt, executor))

            if core is _MISS:
                misses += 1
                with timed("io:stall"):
                    session, ind_df = fut.result()
                with timed("day:build_core"):
                    core = build_day_core(session, ind_df)
                _DAY_CACHE[key] = core
                while len(_DAY_CACHE) > _DAY_CACHE_MAX_DAYS:
                    _DAY_CACHE.popitem(last=False)
            else:
                hits += 1

            if core is None:            # unusable day (empty / tz-naive)
                continue

            with timed("day:process_day"):
                trade = process_day(core, merged_params)
            if trade is not None:
                trade["date"] = pd.Timestamp(f.stem).date()
                trades.append(trade)

    print(f"[ivb cache] day-cores: {hits} from memory, {misses} built, "
          f"{len(_DAY_CACHE)} cached", flush=True)

    if not trades:
        result = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        result = pd.DataFrame(trades)[OUTPUT_COLUMNS]

    _timing.report(time.perf_counter() - t0)
    return result


__all__ = ["run", "PARAMS", "PARAM_SECTIONS"]
