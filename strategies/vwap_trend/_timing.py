"""Accumulating stage timers — one aggregated table printed per run()
(same shape as ivb_model's _timing; sections nest, so %s overlap)."""

import time

_TIMES: dict = {}   # name -> [total_seconds, calls]


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
    if not _TIMES:
        return
    print(f"[vwap_trend timing] wall {wall:.3f}s")
    print(f"  {'section':<22} {'total s':>9} {'calls':>7} {'ms/call':>9} {'% wall':>7}")
    for name, (tot, calls) in sorted(_TIMES.items(), key=lambda kv: kv[1][0], reverse=True):
        print(f"  {name:<22} {tot:>9.3f} {calls:>7} {tot / calls * 1e3:>9.3f} "
              f"{tot / wall * 100:>6.1f}%")
    print(flush=True)
