"""Lightweight accumulating stage timers for the whole package.

Usage:
    from ._timing import timed
    with timed("risk4:trail:two_bar_absorption"):
        ...

`reset()` is called at the start of `run()` and `report(wall)` at its end, printing ONE
aggregated table per backtest run (totals across all days — never per-day spam). Sections
nest freely: a child's time is also inside its parent's total, so percentages are a map of
where time goes, not a disjoint partition.

Timers live at stage level (per day / per window / per detector), never per bar, so the
overhead is negligible next to the work they wrap.
"""

import time

_TIMES: dict[str, list] = {}    # name -> [total_seconds, calls]


def reset():
    _TIMES.clear()


class timed:
    """Context manager accumulating wall time under a section name."""

    __slots__ = ("name", "t0")

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt  = time.perf_counter() - self.t0
        rec = _TIMES.get(self.name)
        if rec is None:
            _TIMES[self.name] = [dt, 1]
        else:
            rec[0] += dt
            rec[1] += 1
        return False


def report(wall: float):
    """Print the accumulated table, slowest section first."""
    if not _TIMES:
        return
    print(f"\n[ivb timing] wall {wall:.3f}s — accumulated sections (nested: children also count inside parents)")
    print(f"  {'section':<38} {'total s':>9} {'calls':>7} {'ms/call':>9} {'% wall':>7}")
    for name, (tot, calls) in sorted(_TIMES.items(), key=lambda kv: kv[1][0], reverse=True):
        print(f"  {name:<38} {tot:>9.3f} {calls:>7} {tot / calls * 1e3:>9.3f} {tot / wall * 100:>6.1f}%")
    print(flush=True)
