"""
Accumulating stage timers — one aggregated table printed per strategy run.

The engine times its own stages (reads, prepare, process, output) and
strategies may time their inner sections with the same `timed`, so everything
lands in ONE table:

    from modules.engine import timed
    with timed("entry:absorption_delta"):
        ...

Sections nest (a strategy's sections run inside the engine's "day:process"),
so percentages overlap. Thread-safe: the engine's prefetch threads time their
reads concurrently with the main loop.
"""

import threading
import time

_TIMES: dict[str, list] = {}     # name -> [total_seconds, calls]
_LOCK = threading.Lock()


class timed:
    """`with timed("section"):` — adds the block's wall time to the table."""
    __slots__ = ("name", "t0")

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        with _LOCK:
            rec = _TIMES.get(self.name)
            if rec is None:
                _TIMES[self.name] = [dt, 1]
            else:
                rec[0] += dt
                rec[1] += 1
        return False


def reset() -> None:
    with _LOCK:
        _TIMES.clear()


def report(title: str, wall: float) -> str:
    """The table as text (also printed). Empty string when nothing was timed."""
    with _LOCK:
        items = sorted(_TIMES.items(), key=lambda kv: kv[1][0], reverse=True)
    if not items:
        return ""
    wall = max(wall, 1e-9)
    lines = [f"[{title} timing] wall {wall:.3f}s",
             f"  {'section':<32} {'total s':>9} {'calls':>7} {'ms/call':>9} {'% wall':>7}"]
    for name, (tot, calls) in items:
        lines.append(f"  {name:<32} {tot:>9.3f} {calls:>7} "
                     f"{tot / calls * 1e3:>9.3f} {tot / wall * 100:>6.1f}%")
    text = "\n".join(lines)
    print(text + "\n", flush=True)
    return text
