"""
VWAP Trend Trading — stop-and-reverse, always-in-market intraday trend follower.

After Zarattini & Aziz, "Volume Weighted Average Price (VWAP): The Holy Grail
for Day Trading Systems" (SSRN 4631351): hold long while 1-minute closes are
above VWAP, short while below, flip on the first close on the opposite side,
flatten at the end of the trading window. Generalized with a configurable
VWAP anchor (rth/globex, decoupled from the trading window), a neutral band
around VWAP (`vwap_band_ticks` + `band_rule`), an optional midday exclusion
window, and a zero-volume-bar signal filter. `vwap_band_ticks = 0` reproduces
the paper's exact rules.

Signals are evaluated on bar CLOSES; fills happen at the NEXT bar's open.
`trade_start_time` is the first bar eligible to OPEN a position — the first
signal comes from the close of the bar before it (the paper's "wait for the
9:30 candle to close, enter at 9:31" with the default 09:31).

⚠ COST CAVEAT (read before trusting results): this system trades ~15x/day.
No transaction costs are modelled (platform convention — pnl_points is a pure
price difference). On ES, commission + one tick of slippage is ≈ 1.3 ticks
per round-turn ≈ 20 ticks/day of drag, which is the same order as the edge
the paper reports. Raw results are materially optimistic and must not be
compared like-for-like against low-frequency strategies.

The backtester loads this package via __init__.py and expects:
  run(folder_path, start_date, end_date, params) -> pd.DataFrame
  PARAMS, PARAM_SECTIONS (PARAM_SPACE here is advisory documentation only —
  the optimizer infers sweep axes from PARAMS defaults' types).
"""

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import _timing, data
from ._timing import timed
from .engine import run_day
from .params import (OUTPUT_COLUMNS, PARAM_SECTIONS, PARAM_SPACE, PARAMS,
                     validate)

RTH_VWAP_ANCHOR_MIN = 9 * 60 + 30    # vwap_bar_rth is NaN before 09:30 NY


def run(folder_path: Path, start_date: pd.Timestamp,
        end_date: pd.Timestamp, params: dict = None) -> pd.DataFrame:
    t0 = time.perf_counter()
    _timing.reset()
    cfg = validate(params)

    folder_path = Path(folder_path)
    indicators_folder = folder_path.parent / cfg["indicators_dataset"]
    if not indicators_folder.is_dir():
        raise FileNotFoundError(
            f"vwap_trend: indicators folder not found: {indicators_folder} — expected a "
            f"sibling dataset of {folder_path.name} named '{cfg['indicators_dataset']}' "
            f"holding the {cfg['anchor_col']} column (set the 'indicators_dataset' param)"
        )

    if cfg["anchor"] == "rth" and cfg["start_min"] < RTH_VWAP_ANCHOR_MIN:
        print(f"[vwap_trend] WARNING: vwap_anchor='rth' with trade_start_time "
              f"{cfg['trade_start']} — vwap_bar_rth is NaN before 09:30 NY, so no "
              f"signals (flat) until 09:30", flush=True)

    with timed("glob+filter"):
        files = sorted(folder_path.glob("*.parquet"))
        files = [
            f for f in files
            if f.stem[0].isdigit()
            and start_date.date() <= pd.Timestamp(f.stem).date() <= end_date.date()
        ]

    def _load(f: Path):
        """Read one day's candles + indicators (runs on a prefetch thread)."""
        with timed("io:read_candles"):
            session = data.read_candles(f)
        ind_file = indicators_folder / f.name
        ind = None
        if ind_file.exists():
            with timed("io:read_indicators"):
                ind = data.read_indicators(ind_file)
        return session, ind

    def _submit(f: Path, executor):
        key  = data.cache_key(f, indicators_folder / f.name)
        core = data.DAY_CACHE.get(key, data.MISS)
        if core is not data.MISS:
            data.DAY_CACHE.move_to_end(key)
            return (f, key, core, None)
        return (f, key, data.MISS, executor.submit(_load, f))

    excl = cfg["exclusion"]
    trades = []
    skipped = []            # dates with missing/unreadable indicators
    hits = misses = 0

    # sliding-window prefetch (same pattern as ivb_model/orb): the next days'
    # reads overlap the current day's compute; consumption stays in file order
    PREFETCH = 4
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

            if core is data.MISS:
                misses += 1
                with timed("io:stall"):
                    session, ind = fut.result()
                with timed("day:build_core"):
                    core = data.build_day_core(session, ind)
                data.DAY_CACHE[key] = core
                while len(data.DAY_CACHE) > data.DAY_CACHE_MAX_DAYS:
                    data.DAY_CACHE.popitem(last=False)
            else:
                hits += 1

            if core is data.SKIP:
                skipped.append(f.stem)
                continue

            vwap = core.vwap.get(cfg["anchor_col"])
            if vwap is None:
                raise ValueError(
                    f"vwap_trend: column '{cfg['anchor_col']}' not found in "
                    f"{indicators_folder / f.name} (available: {sorted(core.vwap)}); "
                    f"check the 'vwap_anchor' / 'indicators_dataset' params"
                )

            with timed("day:process"):
                rth_date = pd.Timestamp(f.stem).date()
                tz = core.index.tz
                start_ns = pd.Timestamp(f"{rth_date} {cfg['trade_start']}", tz=tz).value
                end_ns   = pd.Timestamp(f"{rth_date} {cfg['trade_end']}",   tz=tz).value

                fill0 = int(core.i8.searchsorted(start_ns, side="left"))
                i1    = int(core.i8.searchsorted(end_ns,   side="right"))
                if fill0 >= i1:
                    continue                        # no fill-eligible bars in the window
                sig0 = max(fill0 - 1, 0)            # the bar whose close is the first signal

                i8_w = core.i8[sig0:i1]
                if excl is not None:
                    e0 = pd.Timestamp(f"{rth_date} {excl[0]}", tz=tz).value
                    e1 = pd.Timestamp(f"{rth_date} {excl[1]}", tz=tz).value
                    excl_mask = (i8_w >= e0) & (i8_w < e1)
                else:
                    excl_mask = np.zeros(i1 - sig0, dtype=bool)

                trades.extend(run_day(
                    core.index[sig0:i1], core.open[sig0:i1], core.close[sig0:i1],
                    core.volume[sig0:i1], vwap[sig0:i1], excl_mask, rth_date, cfg,
                ))

    print(f"[vwap_trend cache] day-cores: {hits} from memory, {misses} built, "
          f"{len(data.DAY_CACHE)} cached", flush=True)
    if skipped:
        shown = ", ".join(skipped[:5]) + ("…" if len(skipped) > 5 else "")
        print(f"[vwap_trend] WARNING: skipped {len(skipped)} day(s) with missing/"
              f"unreadable indicators in {indicators_folder}: {shown}", flush=True)

    with timed("build_output_df"):
        if trades:
            result = pd.DataFrame(trades)[OUTPUT_COLUMNS]
        else:
            result = pd.DataFrame(columns=OUTPUT_COLUMNS)
    _timing.report(time.perf_counter() - t0)
    return result


__all__ = ["run", "PARAMS", "PARAM_SECTIONS", "PARAM_SPACE"]
