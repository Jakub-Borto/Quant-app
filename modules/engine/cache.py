"""
The engine's RAM cache — day data "floats" in memory between runs, because
disk reads are the bottleneck.

ONE least-recently-used store with ONE byte budget, holding two kinds of
entries:

  raw       ("raw", file path, mtime_ns, size, columns)  -> DataFrame
            the declared columns of one day file, exactly as read. Shared by
            every strategy that reads the same file + columns.
  prepared  ("prep", strategy key, date, slot files' identities,
             prepare-params)                             -> prepare_day result
            what a strategy's prepare_day() built for one day.

When the budget is exceeded, entries are dropped in this order (see
DayCache): raw frames already turned into a cached prepared day, then entries
the current run hasn't used (least recently used first). If everything left
is in use by the current run, the NEW entry is not kept instead. That makes
the cache scan-resistant: a run larger than the budget keeps its first days
for the next run rather than evicting exactly the days the next run needs
(a plain LRU would make every optimizer combo cold). The run then reports
cache_rejected / cache_note.

There is deliberately NO invalidation by source code: after changing a
strategy's prepare_day(), press "Free cached data" (clear()). File changes
ARE detected — mtime and size are part of every key.

One process-wide instance (get_cache()); optimizer workers each have their
own, sized to their share of the optimizer's budget.
"""

import sys
import threading
from collections import OrderedDict

import numpy as np
import pandas as pd

GB = 1024 ** 3
DEFAULT_BUDGET_GB = 4.0

_MISS = object()


_ARRAY_HEADER = 112          # sys.getsizeof of an empty ndarray object
_SCALARS = (int, float, bool, complex, np.generic)


def _frame_nbytes(df: pd.DataFrame) -> int:
    """== df.memory_usage(index=True, deep=True).sum(), ~40x faster: sums the
    column arrays directly (arrow-backed strings report their buffer size;
    only object columns need the per-element walk)."""
    try:
        idx = df.index
        total = (int(idx.nbytes) if idx.dtype != object and not isinstance(idx, pd.MultiIndex)
                 else int(idx.memory_usage(deep=True)))
        for arr in df._mgr.arrays:
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                total += int(pd.Series(arr, copy=False).memory_usage(deep=True, index=False))
            else:
                total += int(arr.nbytes)
        return total
    except Exception:  # noqa: BLE001 — pandas internals changed: the public, slower way
        return int(df.memory_usage(index=True, deep=True).sum())


def estimate_nbytes(obj, _depth: int = 0) -> int:
    """Approximate memory held by a cached value.

    DataFrame/Series: pandas' deep memory usage (counts string contents).
    numpy arrays: nbytes + the array object itself. Objects may define
    __cache_nbytes__() to answer precisely; otherwise their __dict__ /
    __slots__ are walked for arrays, frames, strings and containers (4 levels
    deep). Containers of numeric arrays (e.g. per-bar parsed JSON) take a
    fast path — this runs for every cached entry."""
    if obj is None:
        return 0
    t = type(obj)
    if t is np.ndarray:
        if obj.dtype == object and _depth < 4:
            return _ARRAY_HEADER + int(obj.nbytes) + sum(estimate_nbytes(x, _depth + 1)
                                                         for x in obj.flat)
        return _ARRAY_HEADER + int(obj.nbytes)
    if t is str or t is bytes:
        return sys.getsizeof(obj)
    if isinstance(obj, _SCALARS):
        return 32
    if isinstance(obj, pd.DataFrame):
        return _frame_nbytes(obj)
    if isinstance(obj, (pd.Series, pd.Index)):
        return int(obj.memory_usage(deep=True))
    custom = getattr(obj, "__cache_nbytes__", None)
    if callable(custom):
        return int(custom())
    if _depth >= 4:
        return sys.getsizeof(obj)
    if isinstance(obj, dict):
        return sys.getsizeof(obj) + sum(estimate_nbytes(k, _depth + 1) + estimate_nbytes(v, _depth + 1)
                                        for k, v in obj.items())
    if isinstance(obj, (list, tuple, set, frozenset)):
        total = sys.getsizeof(obj)
        for x in obj:
            tx = type(x)
            if tx is np.ndarray and x.dtype != object:
                total += _ARRAY_HEADER + x.nbytes
            elif x is None:
                continue
            elif tx is tuple:              # e.g. (prices, sizes, counts) per bar
                total += sys.getsizeof(x)
                for y in x:
                    if type(y) is np.ndarray and y.dtype != object:
                        total += _ARRAY_HEADER + y.nbytes
                    else:
                        total += estimate_nbytes(y, _depth + 2)
            else:
                total += estimate_nbytes(x, _depth + 1)
        return total
    total = sys.getsizeof(obj)
    fields = []
    if hasattr(obj, "__dict__"):
        fields.extend(vars(obj).values())
    for cls in type(obj).__mro__:
        for name in getattr(cls, "__slots__", ()):
            if isinstance(name, str) and hasattr(obj, name):
                fields.append(getattr(obj, name))
    return total + sum(estimate_nbytes(v, _depth + 1) for v in fields)


class DayCache:
    """Thread-safe, scan-resistant LRU keyed by tuples, bounded by an
    approximate byte budget.

    Eviction order when an insert pushes the cache over its budget:
      1. DEMOTED entries (raw frames a prepared day was already built from —
         the engine demotes them; they are only needed to rebuild it);
      2. entries not touched during the CURRENT run (begin_run() starts one);
      3. otherwise the INCOMING entry is not stored (rejected) — everything
         left is in use by this run.
    Step 3 is what makes the cache scan-resistant: a run walks the days in
    order, and a plain LRU smaller than the run would evict exactly the days
    the next run (optimizer combo) needs first — every run cold. Keeping the
    first days instead means a too-small cache still serves most of them.
    """

    def __init__(self, budget_bytes: int = int(DEFAULT_BUDGET_GB * GB)):
        self._lock = threading.Lock()
        self._items: OrderedDict = OrderedDict()   # key -> [value, nbytes, epoch]
        self._demoted: OrderedDict = OrderedDict() # keys to evict first
        self._bytes = 0
        self._budget = int(budget_bytes)
        self._epoch = 0
        self.hits = 0
        self.misses = 0
        self.rejected = 0

    # ── configuration ────────────────────────────────────────────────────────
    @property
    def budget_bytes(self) -> int:
        return self._budget

    @property
    def bytes_used(self) -> int:
        return self._bytes

    def set_budget(self, budget_bytes: int) -> None:
        """Change the budget; shrinking evicts immediately (LRU)."""
        with self._lock:
            self._budget = max(0, int(budget_bytes))
            self._make_room_locked(incoming=None)

    def begin_run(self) -> int:
        """Start a new run: entries touched before now become evictable
        before anything the new run touches. Returns the run's epoch."""
        with self._lock:
            self._epoch += 1
            return self._epoch

    # ── access ───────────────────────────────────────────────────────────────
    def get(self, key, default=_MISS):
        """The cached value (refreshing its recency), or `default`."""
        with self._lock:
            item = self._items.get(key)
            if item is None:
                self.misses += 1
                return default
            self._items.move_to_end(key)
            item[2] = self._epoch
            self._demoted.pop(key, None)       # used again -> worth keeping
            self.hits += 1
            return item[0]

    def contains(self, key) -> bool:
        with self._lock:
            return key in self._items

    def put(self, key, value, nbytes: int | None = None) -> bool:
        """Store `value`; False when it was not kept (the cache is full of
        entries the current run is using — see the class docstring)."""
        size = estimate_nbytes(value) if nbytes is None else int(nbytes)
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
                self._demoted.pop(key, None)
            self._items[key] = [value, size, self._epoch]
            self._bytes += size
            kept = self._make_room_locked(incoming=key)
            if not kept:
                self.rejected += 1
            return kept

    def resize(self, key, nbytes: int) -> bool:
        """Correct an entry's size after its value grew (e.g. a strategy filled
        lazy caches on a prepared day). May evict like put(); False when the
        entry itself had to go."""
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return False
            self._bytes += int(nbytes) - item[1]
            item[1] = int(nbytes)
            kept = self._make_room_locked(incoming=key)
            if not kept:
                self.rejected += 1
            return kept

    def demote(self, keys) -> None:
        """Mark entries as first to evict (e.g. raw frames already turned into
        a cached prepared day)."""
        with self._lock:
            for key in keys:
                if key in self._items:
                    self._demoted[key] = None

    def _drop_locked(self, key) -> None:
        item = self._items.pop(key)
        self._bytes -= item[1]
        self._demoted.pop(key, None)

    def _make_room_locked(self, incoming) -> bool:
        """Evict until within budget; returns False when `incoming` itself had
        to be dropped."""
        if self._bytes <= self._budget:
            return True
        # 1. demoted entries, in demotion order
        while self._bytes > self._budget and self._demoted:
            key = next(iter(self._demoted))
            if key == incoming:
                self._demoted.pop(key)
                continue
            self._drop_locked(key)
        # 2. entries from earlier runs: they sit at the LRU front, because
        #    touching an entry moves it to the end
        while self._bytes > self._budget and self._items:
            key = next(iter(self._items))
            if key == incoming or self._items[key][2] >= self._epoch:
                break
            self._drop_locked(key)
        if self._bytes <= self._budget:
            return True
        # 3. everything left is in use by this run
        if incoming is not None:
            self._drop_locked(incoming)
            return False
        while self._bytes > self._budget and self._items:   # budget shrink
            self._drop_locked(next(iter(self._items)))
        return True

    # ── housekeeping ─────────────────────────────────────────────────────────
    def clear(self) -> int:
        """Drop everything; returns the bytes freed (approximate)."""
        with self._lock:
            freed = self._bytes
            self._items.clear()
            self._demoted.clear()
            self._bytes = 0
            self.hits = self.misses = self.rejected = 0
        return freed

    def stats(self) -> dict:
        with self._lock:
            kinds: dict[str, int] = {}
            for key in self._items:
                kinds[key[0]] = kinds.get(key[0], 0) + 1
            return {"entries": len(self._items), "bytes": self._bytes,
                    "budget": self._budget, "by_kind": kinds,
                    "hits": self.hits, "misses": self.misses,
                    "rejected": self.rejected}


_CACHE = DayCache()


def get_cache() -> DayCache:
    """The process-wide cache (the Backtester, a serial Optimizer run and each
    optimizer worker process all use the instance of their own process)."""
    return _CACHE


def set_budget_gb(gb: float) -> None:
    _CACHE.set_budget(int(float(gb) * GB))


def clear_cache() -> int:
    """Free everything cached in this process; returns bytes freed."""
    return _CACHE.clear()
