"""
`Day` — what a strategy's process_day() receives: one trading day, with every
declared dataset, the cached prepare_day() result, and handles to earlier days.

    day.date                  datetime.date — the RTH date the day file is named after
    day.data["main"]          DataFrame of the declared columns of the main dataset
    day.data["indicators"]    same, for an additional data slot
    day.prepared              what prepare_day() returned for this day (None if the
                              strategy has no prepare_day)
    day.previous              the previous trading day (a Day) or None
    day.previous_days(n)      up to n earlier trading days, oldest first
    day.missing               additional slots with NO file for this day ([] = complete)

Everything is lazy: a slot's DataFrame is read (or fetched from the RAM cache)
the first time it is touched, and earlier days only when asked for. Earlier
days may lie before the run's start date — the lookback uses every file in
the main dataset folder, not just the run's range.

NEVER modify what you get here (no in-place writes to the DataFrames or to
arrays inside `prepared`): the same objects are served from the cache to later
days, later runs and other strategies. Copy first if you need to change data.
"""

import datetime


class DayData:
    """Read-only mapping slot -> DataFrame for one day (lazy)."""
    __slots__ = ("_ctx", "_i")

    def __init__(self, ctx, index: int):
        self._ctx = ctx
        self._i = index

    def __getitem__(self, slot: str):
        return self._ctx.raw(slot, self._i)

    def __contains__(self, slot) -> bool:
        return slot in self._ctx.slots

    def keys(self):
        return list(self._ctx.slots)

    def __iter__(self):
        return iter(self._ctx.slots)

    def __len__(self) -> int:
        return len(self._ctx.slots)

    def __repr__(self) -> str:
        return f"DayData({self._ctx.dates[self._i]}, slots={list(self._ctx.slots)})"


class Day:
    """One trading day as seen by a strategy. See the module docstring."""
    __slots__ = ("_ctx", "_i", "date")

    def __init__(self, ctx, index: int):
        self._ctx = ctx
        self._i = index
        self.date: datetime.date = ctx.dates[index]

    @property
    def data(self) -> DayData:
        return DayData(self._ctx, self._i)

    @property
    def prepared(self):
        return self._ctx.prepared(self._i)

    @property
    def missing(self) -> list:
        """Additional DATA slots whose file does not exist for this day. Always
        [] for the day being processed (the engine skips incomplete days);
        check it before reading a slot of an EARLIER day — reading a missing
        slot raises EngineError."""
        return self._ctx.missing_slots(self._i)

    @property
    def previous(self):
        """The previous trading day (the previous file in the main dataset),
        or None when this is the first file."""
        return Day(self._ctx, self._i - 1) if self._i > 0 else None

    def previous_days(self, n: int) -> list:
        """Up to `n` earlier trading days, OLDEST first (so the last element is
        day.previous). Fewer than n near the start of the data; [] for n <= 0."""
        n = int(n)
        if n <= 0:
            return []
        start = max(0, self._i - n)
        return [Day(self._ctx, i) for i in range(start, self._i)]

    def __repr__(self) -> str:
        return f"Day({self.date})"
