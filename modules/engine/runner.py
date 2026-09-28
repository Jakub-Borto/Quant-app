"""
run_strategy() — the engine's day loop. A strategy supplies data declarations
and per-day logic; the engine does everything else:

  1. validate the strategy (DATA, PREPARE_PARAMS, prepare_day, process_day)
     and the folders given for its data slots;
  2. list the main dataset's day files (YYYY-MM-DD.parquet) and keep the ones
     inside [start, end];
  3. read ONLY the declared columns of every slot, in background threads a few
     days ahead, through the RAM cache (modules.engine.cache);
  4. per day, in date order: skip it if an additional slot's file is missing
     (collected into a warning), build/fetch prepare_day() for it, call
     process_day(day, params) and stamp the returned Trades with the date;
  5. build the 12-column trades frame, print the timing table and cache line.

The strategy contract (see STRATEGY_GUIDE.md for the full reference):

    PARAMS         = {...}                               # defaults
    DATA           = {"main": [cols...], "slot": [cols...]}  # or "*" = all columns
    PREPARE_PARAMS = ["param", ...]                      # optional
    def prepare_day(date, data: dict, params: dict): ... # optional, cached
    def process_day(day, params) -> Trade | list[Trade] | None

Pure Python — no Qt — so optimizer worker processes import it freely.
"""

import datetime
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from . import timing
from .cache import GB, DayCache, get_cache
from .day import Day
from .timing import timed
from .trade import Trade, trades_to_frame

MAIN_SLOT = "main"
ALL_COLUMNS = "*"
PREFETCH_DAYS = 4
READ_THREADS = 2
_MISS = object()


class EngineError(Exception):
    """A strategy/data contract problem, with a message meant for the user."""


@dataclass
class RunResult:
    trades: pd.DataFrame                 # the 12 OUTPUT_COLUMNS, date order
    warnings: list = field(default_factory=list)
    days_in_range: int = 0               # main-dataset day files in [start, end]
    days_run: int = 0                    # days handed to process_day
    skipped: dict = field(default_factory=dict)   # slot -> [dates missing a file]
    files_read: int = 0                  # day files read from DISK (cache misses)
    days_prepared: int = 0               # prepare_day() calls (prepared-cache misses)


# ══ strategy declarations ═════════════════════════════════════════════════════
@dataclass(frozen=True)
class StrategySpec:
    name: str
    data: dict                           # slot -> tuple(columns) | None (= all)
    prepare_params: tuple
    has_prepare: bool


def strategy_spec(module) -> StrategySpec:
    """Validate and normalise a strategy module's declarations."""
    name = getattr(module, "__name__", "strategy")
    if not callable(getattr(module, "process_day", None)):
        raise EngineError(f"Strategy '{name}' has no process_day(day, params) function.")
    raw = getattr(module, "DATA", None)
    if not isinstance(raw, dict) or MAIN_SLOT not in raw:
        raise EngineError(f"Strategy '{name}' must declare DATA = {{\"main\": [columns...], ...}} "
                          f"(got {raw!r}).")
    data = {}
    for slot, cols in raw.items():
        if not isinstance(slot, str) or not slot.strip():
            raise EngineError(f"Strategy '{name}': DATA keys must be non-empty strings (got {slot!r}).")
        if cols == ALL_COLUMNS:
            data[slot] = None
        elif isinstance(cols, (list, tuple)) and cols and all(isinstance(c, str) for c in cols):
            if len(set(cols)) != len(cols):
                raise EngineError(f"Strategy '{name}': DATA['{slot}'] lists a column twice.")
            data[slot] = tuple(cols)
        else:
            raise EngineError(f"Strategy '{name}': DATA['{slot}'] must be a non-empty list of "
                              f"column names or \"*\" (got {cols!r}).")
    prep = getattr(module, "PREPARE_PARAMS", ()) or ()
    if isinstance(prep, str) or not all(isinstance(p, str) for p in prep):
        raise EngineError(f"Strategy '{name}': PREPARE_PARAMS must be a list of param names.")
    has_prepare = callable(getattr(module, "prepare_day", None))
    if prep and not has_prepare:
        raise EngineError(f"Strategy '{name}' declares PREPARE_PARAMS but has no prepare_day().")
    return StrategySpec(name, data, tuple(prep), has_prepare)


def additional_slots(module) -> list[str]:
    """The strategy's additional data slots (every DATA key except "main"), in
    declaration order — one "Additional data" row per slot in the UI."""
    raw = getattr(module, "DATA", None)
    if not isinstance(raw, dict):
        return []
    return [s for s in raw if s != MAIN_SLOT]


def day_files(folder) -> list[tuple[datetime.date, Path]]:
    """Every YYYY-MM-DD.parquet in `folder`, oldest first (other files ignored)."""
    out = []
    folder = Path(folder)
    if not folder.is_dir():
        return out
    for f in folder.glob("*.parquet"):
        stem = f.stem
        if not stem[:1].isdigit():
            continue
        try:
            out.append((datetime.date.fromisoformat(stem), f))
        except ValueError:
            continue
    out.sort(key=lambda t: t[0])
    return out


def _identity(path: Path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (str(path), st.st_mtime_ns, st.st_size)


# ══ the per-run context ═══════════════════════════════════════════════════════
class _RunContext:
    """Everything a Day needs to resolve its data: the day index, the slot
    folders, the cache and the strategy. Days are identified by their position
    in the main dataset's full file list (so lookback crosses the start date)."""

    def __init__(self, module, spec: StrategySpec, key: str, folders: dict,
                 params: dict, cache: DayCache):
        self.module = module
        self.spec = spec
        self.key = key
        self.folders = folders
        self.slots = list(spec.data)
        self.params = params
        self.cache = cache
        files = day_files(folders[MAIN_SLOT])
        self.dates = [d for d, _ in files]
        self._main_paths = [p for _, p in files]
        self.prep_params = {k: params[k] for k in spec.prepare_params}
        self._prep_items = tuple(sorted((k, _hashable(v)) for k, v in self.prep_params.items()))
        self.raw_hits = self.raw_reads = 0
        self.prep_hits = self.prep_builds = 0

    # ── files ────────────────────────────────────────────────────────────────
    def path(self, slot: str, i: int) -> Path:
        if slot == MAIN_SLOT:
            return self._main_paths[i]
        return self.folders[slot] / f"{self.dates[i].isoformat()}.parquet"

    def missing_slots(self, i: int) -> list[str]:
        return [s for s in self.slots if s != MAIN_SLOT and not self.path(s, i).exists()]

    def _raw_key(self, slot: str, i: int):
        ident = _identity(self.path(slot, i))
        if ident is None:
            return None
        return ("raw",) + ident + (self.spec.data[slot],)

    # ── raw data ─────────────────────────────────────────────────────────────
    def read(self, slot: str, i: int) -> pd.DataFrame:
        """Read one slot's day file from disk (declared columns only) and cache it."""
        path = self.path(slot, i)
        key = self._raw_key(slot, i)
        if key is None:
            raise EngineError(f"The '{slot}' data file for {self.dates[i]} does not exist: {path}")
        columns = self.spec.data[slot]
        with timed(f"io:read:{slot}"):
            try:
                frame = pd.read_parquet(path, columns=list(columns) if columns else None)
            except Exception as e:
                if columns:
                    try:
                        names = set(pq.read_schema(path).names)
                    except Exception:
                        names = None
                    if names is not None:
                        missing = [c for c in columns if c not in names]
                        if missing:
                            raise EngineError(
                                f"Column(s) {missing} declared in DATA['{slot}'] are not in the "
                                f"'{slot}' dataset {path.parent.name} (file {path.name}). "
                                f"Pick a dataset that has them, or remove them from DATA."
                            ) from e
                raise
        self.raw_reads += 1
        self.cache.put(key, frame)
        return frame

    def raw(self, slot: str, i: int) -> pd.DataFrame:
        if slot not in self.spec.data:
            raise EngineError(f"No data slot '{slot}' — the strategy declares DATA slots "
                              f"{list(self.spec.data)}.")
        key = self._raw_key(slot, i)
        if key is not None:
            frame = self.cache.get(key, _MISS)
            if frame is not _MISS:
                self.raw_hits += 1
                return frame
        return self.read(slot, i)

    # ── prepared day ─────────────────────────────────────────────────────────
    def _prep_key(self, i: int):
        idents = tuple(_identity(self.path(s, i)) for s in self.slots)
        cols = tuple((s, self.spec.data[s]) for s in self.slots)
        return ("prep", self.key, self.dates[i].isoformat(), idents, cols, self._prep_items)

    def prepared_cached(self, i: int) -> bool:
        return self.spec.has_prepare and self.cache.contains(self._prep_key(i))

    def prepared(self, i: int):
        if not self.spec.has_prepare:
            return None
        key = self._prep_key(i)
        value = self.cache.get(key, _MISS)
        if value is not _MISS:
            self.prep_hits += 1
            return value
        data = {slot: self.raw(slot, i) for slot in self.slots}
        with timed("day:prepare"):
            value = self.module.prepare_day(self.dates[i], data, dict(self.prep_params))
        self.prep_builds += 1
        self.cache.put(key, value)
        return value


def _hashable(value):
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


def _as_trades(result, name: str) -> list:
    if result is None:
        return []
    if isinstance(result, Trade):
        return [result]
    try:
        items = list(result)
    except TypeError:
        raise EngineError(f"process_day of '{name}' must return a Trade, a list of Trades "
                          f"or None (got {type(result).__name__}).") from None
    for t in items:
        if not isinstance(t, Trade):
            raise EngineError(f"process_day of '{name}' returned a {type(t).__name__} — "
                              f"every trade must be a modules.engine.Trade.")
    return items


def _explain_key_error(e: KeyError, spec: StrategySpec, where: str) -> EngineError:
    name = e.args[0] if e.args else None
    declared = {s: list(c) if c else "*" for s, c in spec.data.items()}
    return EngineError(
        f"KeyError {name!r} in {where} of '{spec.name}'. If {name!r} is a data column, "
        f"it is not loaded — the strategy declares DATA = {declared}; add it to the "
        f"right slot (the dataset must have it). If it is a param, it is missing "
        f"from PARAMS.")


# ══ the loop ══════════════════════════════════════════════════════════════════
def run_strategy(module, main_folder, start_date, end_date, params: dict, *,
                 tick_size: float, extra_folders: dict | None = None,
                 strategy_key: str | None = None, on_progress=None,
                 cache: DayCache | None = None, verbose: bool = True) -> RunResult:
    """Run `module` over the main dataset's days in [start_date, end_date].

    params         the UI's values; merged over the strategy's PARAMS defaults,
                   and tick_size is always injected (from ASSET_INFO upstream)
    extra_folders  {slot: folder} for every additional DATA slot
    strategy_key   cache identity of the strategy (default: its file path)
    on_progress    callback(current_day, total_days, message) — raising inside
                   it cancels the run (the app's worker cancellation)
    """
    t0 = time.perf_counter()
    timing.reset()
    spec = strategy_spec(module)
    cache = cache or get_cache()

    folders = {MAIN_SLOT: Path(main_folder)}
    extra_folders = {k: Path(v) for k, v in (extra_folders or {}).items()}
    for slot in spec.data:
        if slot == MAIN_SLOT:
            continue
        folder = extra_folders.get(slot)
        if folder is None:
            raise EngineError(f"'{spec.name}' needs additional data '{slot}', but no dataset "
                              f"was chosen for it.")
        if not folder.is_dir():
            raise EngineError(f"The dataset chosen for '{slot}' does not exist: {folder}")
        folders[slot] = folder
    if not folders[MAIN_SLOT].is_dir():
        raise EngineError(f"The main dataset folder does not exist: {folders[MAIN_SLOT]}")

    merged = {**getattr(module, "PARAMS", {}), **(params or {}), "tick_size": float(tick_size)}
    missing_prep = [p for p in spec.prepare_params if p not in merged]
    if missing_prep:
        raise EngineError(f"PREPARE_PARAMS {missing_prep} of '{spec.name}' are not params.")
    key = strategy_key or str(Path(getattr(module, "__file__", spec.name)).resolve())
    ctx = _RunContext(module, spec, key, folders, merged, cache)

    lo, hi = pd.Timestamp(start_date).date(), pd.Timestamp(end_date).date()
    run_idx = [i for i, d in enumerate(ctx.dates) if lo <= d <= hi]
    total = len(run_idx)

    skipped: dict[str, list[str]] = {}
    rows: list = []
    days_run = 0

    def _needs_read(i: int) -> list[str]:
        """Slots whose raw frame must be read for day i (nothing when the
        prepared day is already cached and process_day may never touch raw)."""
        if ctx.missing_slots(i):
            return []
        if spec.has_prepare and ctx.prepared_cached(i):
            return []
        return [s for s in ctx.slots if not cache.contains(ctx._raw_key(s, i))]

    def _load(i: int, slots: list[str]):
        return {s: ctx.read(s, i) for s in slots}

    with ThreadPoolExecutor(max_workers=READ_THREADS) as pool:
        queue: deque = deque()
        it = iter(run_idx)

        def _submit_next():
            i = next(it, None)
            if i is None:
                return False
            need = _needs_read(i)
            queue.append((i, pool.submit(_load, i, need) if need else None))
            return True

        for _ in range(PREFETCH_DAYS):
            if not _submit_next():
                break

        done = 0
        while queue:
            i, fut = queue.popleft()
            _submit_next()
            if fut is not None:
                with timed("io:stall"):
                    fut.result()                # re-raises read errors here
            done += 1
            date = ctx.dates[i]

            missing = ctx.missing_slots(i)
            if missing:
                for slot in missing:
                    skipped.setdefault(slot, []).append(date.isoformat())
            else:
                try:
                    prepared = ctx.prepared(i)
                    if not (spec.has_prepare and prepared is None):
                        with timed("day:process"):
                            result = module.process_day(Day(ctx, i), merged)
                        for trade in _as_trades(result, spec.name):
                            rows.append((date, trade))
                        days_run += 1
                except KeyError as e:
                    raise _explain_key_error(e, spec, f"the day {date}") from e

            if on_progress is not None:
                on_progress(done, total, "")

    with timed("build_output_df"):
        trades = trades_to_frame(rows)

    warnings = []
    for slot, dates in skipped.items():
        shown = ", ".join(dates[:10]) + (f" … (+{len(dates) - 10} more)" if len(dates) > 10 else "")
        warnings.append(f"The '{slot}' dataset ({folders[slot].name}) has no file for "
                        f"{len(dates)} day(s), so those days were SKIPPED: {shown}")

    if verbose:
        st = cache.stats()
        print(f"[engine] {spec.name}: {total} days in range, {days_run} processed, "
              f"{len(rows)} trades | prepared days {ctx.prep_hits} from memory, "
              f"{ctx.prep_builds} built | raw frames {ctx.raw_hits} from memory, "
              f"{ctx.raw_reads} read | cache {st['bytes'] / GB:.2f}/{st['budget'] / GB:.2f} GB",
              flush=True)
        for w in warnings:
            print(f"[engine] WARNING: {w}", flush=True)
        timing.report(spec.name, time.perf_counter() - t0)
    return RunResult(trades=trades, warnings=warnings, days_in_range=total,
                     days_run=days_run, skipped=skipped,
                     files_read=ctx.raw_reads, days_prepared=ctx.prep_builds)
