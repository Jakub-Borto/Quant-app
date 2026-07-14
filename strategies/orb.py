# strategies/orb.py — Opening Range Breakout, vectorized.
#
# Trade logic is identical to the original per-bar version (verified exactly over
# ES_1m_advanced + ES_1m_ohlcv_globex); the internals follow ivb_model's playbook:
#   - column-pruned parquet reads (only OHLC — skips the enriched JSON columns),
#   - per-day numpy arrays instead of DataFrame slicing / iterrows,
#   - a persistent day-core cache so re-running with different params skips
#     file reads and array building entirely,
#   - a small prefetch pool overlapping the next days' reads with compute.

import sys
import time
import types
from collections import deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

PARAMS = {
    "range_minutes": 15,
    "sl_factor": 0.5,
    "rr": 2.0,
    "timeout_minutes": 240,
}

RTH_OPEN  = "09:30"
RTH_CLOSE = "16:00"

CANDLE_COLUMNS = ["open", "high", "low", "close"]


# ---------------------------------------------------------------------------
# Timing (one aggregated table printed per run, like ivb_model's _timing)
# ---------------------------------------------------------------------------

_TIMES: dict = {}   # name -> [total_seconds, calls]


class _timed:
    __slots__ = ("name", "t0")

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, *exc):
        dt  = time.perf_counter() - self.t0
        rec = _TIMES.get(self.name)
        if rec is None:
            _TIMES[self.name] = [dt, 1]
        else:
            rec[0] += dt
            rec[1] += 1
        return False


def _report(wall):
    if not _TIMES:
        return
    print(f"[orb timing] wall {wall:.3f}s")
    print(f"  {'section':<20} {'total s':>9} {'calls':>7} {'ms/call':>9} {'% wall':>7}")
    for name, (tot, calls) in sorted(_TIMES.items(), key=lambda kv: kv[1][0], reverse=True):
        print(f"  {name:<20} {tot:>9.3f} {calls:>7} {tot / calls * 1e3:>9.3f} "
              f"{tot / wall * 100:>6.1f}%")
    print(flush=True)


# ---------------------------------------------------------------------------
# Day-core cache (persists across run() calls in the same process)
# ---------------------------------------------------------------------------
# A day core (RTH-sliced OHLC arrays) depends only on the file and the fixed RTH
# bounds, so it is fully param-independent: every re-run — the normal research /
# optimizer loop — skips reads and slicing. Keyed by (path, mtime_ns, size) so
# re-running a transform invalidates naturally. The backtester's plugin loader
# re-executes this module on every run, which would wipe a module-level dict, so
# the dict lives on a holder module registered once in sys.modules (same trick
# as ivb_model). A cached day is ~16 KB (5 arrays x ~390 bars), so even 10k days
# stay under ~200 MB.

_STORE_NAME = "_orb_day_cache_store"
_store = sys.modules.get(_STORE_NAME)
if _store is None:
    _store = types.ModuleType(_STORE_NAME)
    _store.cache = OrderedDict()
    sys.modules[_STORE_NAME] = _store
_DAY_CACHE: OrderedDict = _store.cache
_DAY_CACHE_MAX_DAYS = 10_000
_MISS = object()


class _DayCore:
    """One RTH session as positional numpy arrays. Param-independent."""

    __slots__ = ("index", "i8", "open", "high", "low", "close", "rth_start", "date_str")

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
        self.date_str  = str(rth_date)


def _read_candles(f: Path) -> pd.DataFrame:
    """Column-pruned read; falls back to a full read for files missing any column."""
    try:
        return pd.read_parquet(f, columns=CANDLE_COLUMNS)
    except Exception:
        return pd.read_parquet(f)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

def _first_true(mask: np.ndarray) -> int:
    """Index of the first True, or -1."""
    return int(mask.argmax()) if mask.any() else -1


def _process_day(core: _DayCore, p: dict) -> dict | None:
    n = len(core.i8)
    if n < 2:
        return None

    # opening range / after range split (rth.index < range_end vs >=)
    range_end_ns = core.rth_start.value + pd.Timedelta(minutes=p["range_minutes"]).value
    split = int(core.i8.searchsorted(range_end_ns, side="left"))
    if split == 0 or split == n:
        return None

    orb_high  = np.nanmax(core.high[:split])
    orb_low   = np.nanmin(core.low[:split])
    orb_range = orb_high - orb_low
    if orb_range <= 0:
        return None

    # signal scan over after_range minus its last bar (entry needs the next bar);
    # long is checked first on each bar, and the earliest signal wins
    o, c = core.open, core.close
    cs, os_ = c[split:n - 1], o[split:n - 1]
    long_m  = (cs > orb_high) & (cs > os_)
    short_m = (cs < orb_low) & (cs < os_)
    rel = _first_true(long_m | short_m)
    if rel == -1:
        return None
    direction = "long" if long_m[rel] else "short"

    entry_pos   = split + rel + 1
    entry_price = core.open[entry_pos]
    entry_time  = core.index[entry_pos]

    sl_distance = orb_range * p["sl_factor"]
    if direction == "long":
        sl = entry_price - sl_distance
        tp = entry_price + sl_distance * p["rr"]
    else:
        sl = entry_price + sl_distance
        tp = entry_price - sl_distance * p["rr"]

    # trade simulation over [entry bar .. end of day]
    th = core.high[entry_pos:]
    tl = core.low[entry_pos:]
    tc = core.close[entry_pos:]
    ti8  = core.i8[entry_pos:]
    tidx = core.index[entry_pos:]
    m = n - entry_pos

    if direction == "long":
        sl_hit = tl <= sl
        tp_hit = th >= tp
    else:
        sl_hit = th >= sl
        tp_hit = tl <= tp

    # first bar at/after the timeout; before it plain SL/TP applies (SL checked
    # first within a bar, so it wins a same-bar tie)
    timeout_ns = ti8[0] + pd.Timedelta(minutes=p["timeout_minutes"]).value
    tpos = int(ti8.searchsorted(timeout_ns, side="left"))

    f_sl = _first_true(sl_hit[:tpos])
    f_tp = _first_true(tp_hit[:tpos])
    if f_sl != -1 and (f_tp == -1 or f_sl <= f_tp):
        exit_price, exit_time, exit_reason = sl, tidx[f_sl], "sl"
    elif f_tp != -1:
        exit_price, exit_time, exit_reason = tp, tidx[f_tp], "tp"
    elif tpos < m:
        # timeout bar: exit at close if in profit, else move TP to breakeven and
        # keep running (the timeout bar itself is re-checked against the new TP)
        in_profit = tc[tpos] > entry_price if direction == "long" else tc[tpos] < entry_price
        if in_profit:
            exit_price, exit_time, exit_reason = tc[tpos], tidx[tpos], "timeout_profit"
        else:
            tp = entry_price
            if direction == "long":
                be_hit = th[tpos:] >= tp
            else:
                be_hit = tl[tpos:] <= tp
            f_sl2 = _first_true(sl_hit[tpos:])
            f_tp2 = _first_true(be_hit)
            if f_sl2 != -1 and (f_tp2 == -1 or f_sl2 <= f_tp2):
                exit_price, exit_time, exit_reason = sl, tidx[tpos + f_sl2], "sl"
            elif f_tp2 != -1:
                exit_price, exit_time, exit_reason = tp, tidx[tpos + f_tp2], "tp"
            else:
                exit_price, exit_time, exit_reason = tc[m - 1], tidx[m - 1], "eod"
    else:
        exit_price, exit_time, exit_reason = tc[m - 1], tidx[m - 1], "eod"

    pnl_points = exit_price - entry_price if direction == "long" else entry_price - exit_price

    return {
        "date":        core.date_str,
        "direction":   direction,
        "entry_time":  entry_time,
        "exit_time":   exit_time,
        "entry_price": entry_price,
        "exit_price":  exit_price,
        "sl":          sl,
        "tp":          tp,
        "exit_reason": exit_reason,
        "pnl_points":  pnl_points,
    }


def run(folder_path: Path, start_date: pd.Timestamp,
        end_date: pd.Timestamp, params: dict = None) -> pd.DataFrame:
    t0 = time.perf_counter()
    _TIMES.clear()
    p = {**PARAMS, **(params or {})}

    with _timed("glob+filter"):
        files = sorted(Path(folder_path).glob("*.parquet"))
        files = [
            f for f in files
            if f.stem[0].isdigit()
            and start_date.date() <= pd.Timestamp(f.stem).date() <= end_date.date()
        ]

    def _key(f: Path):
        st = f.stat()
        return (str(f), st.st_mtime_ns, st.st_size, RTH_OPEN, RTH_CLOSE)

    def _load(f: Path) -> pd.DataFrame:
        with _timed("io:read_candles"):
            return _read_candles(f)

    def _submit(f: Path, executor):
        key  = _key(f)
        core = _DAY_CACHE.get(key, _MISS)
        if core is not _MISS:
            _DAY_CACHE.move_to_end(key)
            return (f, key, core, None)
        return (f, key, _MISS, executor.submit(_load, f))

    # sliding-window prefetch: the next days' reads overlap the current day's
    # compute; days are consumed strictly in file order, so output order is
    # identical to the sequential loop
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
                with _timed("io:stall"):
                    session = fut.result()
                with _timed("day:build_core"):
                    core = None if session.empty else _DayCore(session, pd.Timestamp(f.stem).date())
                _DAY_CACHE[key] = core
                while len(_DAY_CACHE) > _DAY_CACHE_MAX_DAYS:
                    _DAY_CACHE.popitem(last=False)
            else:
                hits += 1

            if core is None:            # empty session
                continue

            with _timed("day:process"):
                trade = _process_day(core, p)
            if trade:
                trades.append(trade)

    print(f"[orb cache] day-cores: {hits} from memory, {misses} built, "
          f"{len(_DAY_CACHE)} cached", flush=True)

    with _timed("build_output_df"):
        result = pd.DataFrame(trades)
    _report(time.perf_counter() - t0)
    return result
