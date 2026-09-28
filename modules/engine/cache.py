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

When the budget is exceeded, the least recently USED entries are dropped
(a hit refreshes an entry). An entry larger than the whole budget is still
kept until the next insert pushes it out, so a run never thrashes forever.

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


def estimate_nbytes(obj, _depth: int = 0) -> int:
    """Approximate memory held by a cached value.

    DataFrame/Series: pandas' deep memory_usage (counts string contents).
    numpy arrays: nbytes. Objects may define __cache_nbytes__() to answer
    precisely; otherwise their __dict__ / __slots__ are walked for arrays,
    frames, strings and containers (4 levels deep)."""
    if obj is None:
        return 0
    if isinstance(obj, pd.DataFrame):
        return int(obj.memory_usage(index=True, deep=True).sum())
    if isinstance(obj, (pd.Series, pd.Index)):
        return int(obj.memory_usage(deep=True))
    if isinstance(obj, np.ndarray):
        if obj.dtype == object and _depth < 4:
            return int(obj.nbytes) + sum(estimate_nbytes(x, _depth + 1) for x in obj.flat)
        return int(obj.nbytes)
    if isinstance(obj, (str, bytes)):
        return sys.getsizeof(obj)
    if isinstance(obj, (int, float, bool)):
        return 32
    custom = getattr(obj, "__cache_nbytes__", None)
    if callable(custom):
        return int(custom())
    if _depth >= 4:
        return sys.getsizeof(obj)
    if isinstance(obj, dict):
        return sys.getsizeof(obj) + sum(estimate_nbytes(k, _depth + 1) + estimate_nbytes(v, _depth + 1)
                                        for k, v in obj.items())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return sys.getsizeof(obj) + sum(estimate_nbytes(x, _depth + 1) for x in obj)
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
    """Thread-safe LRU keyed by tuples, bounded by an approximate byte budget."""

    def __init__(self, budget_bytes: int = int(DEFAULT_BUDGET_GB * GB)):
        self._lock = threading.Lock()
        self._items: OrderedDict = OrderedDict()   # key -> (value, nbytes)
        self._bytes = 0
        self._budget = int(budget_bytes)
        self.hits = 0
        self.misses = 0

    # ── configuration ────────────────────────────────────────────────────────
    @property
    def budget_bytes(self) -> int:
        return self._budget

    def set_budget(self, budget_bytes: int) -> None:
        """Change the budget; shrinking evicts immediately."""
        with self._lock:
            self._budget = max(0, int(budget_bytes))
            self._evict_locked(keep=None)

    # ── access ───────────────────────────────────────────────────────────────
    def get(self, key, default=_MISS):
        """The cached value (refreshing its recency), or `default`."""
        with self._lock:
            item = self._items.get(key)
            if item is None:
                self.misses += 1
                return default
            self._items.move_to_end(key)
            self.hits += 1
            return item[0]

    def contains(self, key) -> bool:
        with self._lock:
            return key in self._items

    def put(self, key, value, nbytes: int | None = None) -> None:
        size = estimate_nbytes(value) if nbytes is None else int(nbytes)
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            self._items[key] = (value, size)
            self._bytes += size
            self._evict_locked(keep=key)

    def _evict_locked(self, keep) -> None:
        # `keep` (the entry just inserted) is always the newest, so it is only
        # ever the oldest when it is the last entry left — an oversize value
        # stays until the next insert pushes it out
        while self._bytes > self._budget and self._items:
            oldest = next(iter(self._items))
            if oldest == keep:
                return
            _value, size = self._items.pop(oldest)
            self._bytes -= size

    # ── housekeeping ─────────────────────────────────────────────────────────
    def clear(self) -> int:
        """Drop everything; returns the bytes freed (approximate)."""
        with self._lock:
            freed = self._bytes
            self._items.clear()
            self._bytes = 0
            self.hits = self.misses = 0
        return freed

    def stats(self) -> dict:
        with self._lock:
            kinds: dict[str, int] = {}
            for key in self._items:
                kinds[key[0]] = kinds.get(key[0], 0) + 1
            return {"entries": len(self._items), "bytes": self._bytes,
                    "budget": self._budget, "by_kind": kinds,
                    "hits": self.hits, "misses": self.misses}


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
