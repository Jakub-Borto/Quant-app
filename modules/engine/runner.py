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

Progress: on_progress(current_day, total_days, text) is called at every step
of every day (and during setup / wrap-up with current = 0 / total). `text` is
several lines describing exactly what the engine is doing right now plus the
run's counters — see _Progress. Raising inside on_progress cancels the run.

Pure Python — no Qt — so optimizer worker processes import it freely.
"""

import datetime
import json
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from . import timing
from .cache import GB, DayCache, estimate_nbytes, get_cache
from .day import Day
from .timing import timed
from .trade import Trade, trades_to_frame

MAIN_SLOT = "main"
ALL_COLUMNS = "*"
PREFETCH_DAYS = 4
READ_THREADS = 2
# "heavy" days (declared columns decompress to more than this, e.g. full-book
# MBO JSON) decode mostly outside the GIL, so more threads help them; for
# small 1-minute files extra threads only fight the strategy for the GIL.
HEAVY_DAY_BYTES = 32 * 1024 ** 2
HEAVY_READ_THREADS = 4
HEAVY_PREFETCH_DAYS = 6
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
    cache_rejected: int = 0              # items the RAM cache was too full to keep
    cache_note: str | None = None        # human-readable note when cache_rejected > 0
    elapsed: float = 0.0                 # wall seconds of the whole run
    timing: str = ""                     # the per-section timing table (as printed)
    summary: str = ""                    # one-paragraph plain-English run summary
    rows: list = field(default_factory=list)      # [(date, Trade)] behind `trades`
                                         # (optimizer workers ship these: the parent
                                         # builds ONE frame, like a serial run)


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


def _memoize_arrow_tz_lookup() -> None:
    """pyarrow rebuilds a tz-aware index through pa.lib.string_to_tzinfo(name),
    which first tries `import pytz`. pytz is not installed, and a FAILED import
    is not cached, so every file read re-scanned sys.path (~0.26 ms holding the
    GIL — a third of a small file's read). The function is pure and already
    returns the same cached ZoneInfo object for a name, so memoizing it is
    observably identical."""
    import functools

    import pyarrow as pa
    fn = getattr(pa.lib, "string_to_tzinfo", None)
    if fn is None or getattr(fn, "_engine_memoized", False):
        return
    cached = functools.lru_cache(maxsize=64)(fn)
    cached._engine_memoized = True
    pa.lib.string_to_tzinfo = cached


_memoize_arrow_tz_lookup()


def read_day_file(path: Path, columns, slot: str = MAIN_SLOT) -> pd.DataFrame:
    """One day file -> DataFrame of `columns` (None = all) plus the stored index.

    Byte-identical to pd.read_parquet(path, columns=...) — it is the same
    pyarrow read + to_pandas + attrs that pandas performs — minus pandas'
    per-call dataset/import/version overhead (~40% of a small file's read, and
    it holds the GIL, so it also slowed the strategy running meanwhile)."""
    with pq.ParquetFile(path) as f:
        if columns:
            names = set(f.schema_arrow.names)
            missing = [c for c in columns if c not in names]
            if missing:
                raise EngineError(
                    f"Column(s) {missing} declared in DATA['{slot}'] are not in the "
                    f"'{slot}' dataset {path.parent.name} (file {path.name}). "
                    f"Pick a dataset that has them, or remove them from DATA.")
        table = f.read(columns=list(columns) if columns else None, use_pandas_metadata=True)
    frame = table.to_pandas()
    meta = table.schema.metadata
    if meta and b"PANDAS_ATTRS" in meta:
        frame.attrs = json.loads(meta[b"PANDAS_ATTRS"])
    return frame


def skipped_warnings(skipped: dict, dataset_names: dict) -> list[str]:
    """{slot: [iso dates without a file]} -> the run's loud warnings (one per
    slot). Shared with the optimizer, which merges the skipped days of its
    per-worker date chunks before formatting."""
    out = []
    for slot, dates in skipped.items():
        shown = ", ".join(dates[:10]) + (f" … (+{len(dates) - 10} more)" if len(dates) > 10 else "")
        out.append(f"The '{slot}' dataset ({dataset_names.get(slot, slot)}) has no file for "
                   f"{len(dates)} day(s), so those days were SKIPPED: {shown}")
    return out


def _column_memory(meta, schema, columns) -> int:
    """Approximate in-memory bytes of `columns` (None = all) of one parquet
    file from its footer: fixed-width columns = rows x width (the footer's
    sizes are dictionary-compressed for them), variable-width (strings) = the
    uncompressed size."""
    import pyarrow as pa
    rows = meta.num_rows
    total = 0
    sizes: dict = {}
    for g in range(meta.num_row_groups):
        rg = meta.row_group(g)
        for c in range(rg.num_columns):
            col = rg.column(c)
            sizes[col.path_in_schema] = sizes.get(col.path_in_schema, 0) + col.total_uncompressed_size
    for name in schema.names:
        if columns is not None and name not in columns:
            # the stored index is always read along with the columns
            pandas_meta = schema.pandas_metadata or {}
            if name not in [c for c in pandas_meta.get("index_columns", []) if isinstance(c, str)]:
                continue
        typ = schema.field(name).type
        try:
            width = typ.bit_width // 8
        except ValueError:
            width = 0
        total += rows * width if width else sizes.get(name, 0)
    return total


def estimate_run_memory(module, main_folder, start_date, end_date,
                        extra_folders: dict | None = None, samples: int = 8) -> dict:
    """What a run of `module` over [start, end] will hold in RAM, from the
    parquet footers of up to `samples` evenly spread days (no data is read).

    Returns {"n_days", "raw_bytes" (declared columns of every slot, as
    DataFrames), "prepared_bytes" (a guess: = raw when the strategy has a
    prepare_day, else 0), "total_bytes"}."""
    spec = strategy_spec(module)
    folders = {MAIN_SLOT: Path(main_folder)}
    folders.update({k: Path(v) for k, v in (extra_folders or {}).items()})
    lo, hi = pd.Timestamp(start_date).date(), pd.Timestamp(end_date).date()
    days = [d for d, _ in day_files(folders[MAIN_SLOT]) if lo <= d <= hi]
    if not days:
        return {"n_days": 0, "raw_bytes": 0, "prepared_bytes": 0, "total_bytes": 0}
    step = max(1, len(days) // max(1, samples))
    picked = days[::step][:samples]
    per_day = []
    for d in picked:
        size = 0
        for slot, cols in spec.data.items():
            path = (folders.get(slot) or Path()) / f"{d.isoformat()}.parquet"
            try:
                f = pq.ParquetFile(path)
            except Exception:  # noqa: BLE001 — missing / unreadable: not counted
                continue
            size += _column_memory(f.metadata, f.schema_arrow, cols)
        per_day.append(size)
    raw = int(sum(per_day) / len(per_day) * len(days))
    prepared = raw if spec.has_prepare else 0
    return {"n_days": len(days), "raw_bytes": raw, "prepared_bytes": prepared,
            "total_bytes": raw + prepared}


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m} min {s:02d} s"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


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
        self._lock = threading.Lock()
        self.raw_hits = self.raw_reads = 0
        self.prep_hits = self.prep_builds = 0
        self.reads_by_slot = {s: 0 for s in self.slots}
        self.read_seconds = 0.0
        self.read_threads = READ_THREADS
        # file identities (path, mtime, size) are looked up once per run: a
        # day's key is needed by the scheduler, the cache and the prepared key
        self._idents: dict = {}
        # frames the prefetch read for the day in progress: used directly, so a
        # full cache that declined to keep them never causes a second read
        self.pending: dict[int, dict] = {}
        # the prepared value of the day being processed (served to day.prepared
        # without another cache lookup) and prepared keys built this run, which
        # are re-measured after their first process_day (lazy strategy state)
        self.current: tuple | None = None
        self.fresh_prep: set = set()

    # ── files ────────────────────────────────────────────────────────────────
    def path(self, slot: str, i: int) -> Path:
        if slot == MAIN_SLOT:
            return self._main_paths[i]
        return self.folders[slot] / f"{self.dates[i].isoformat()}.parquet"

    def ident(self, slot: str, i: int):
        k = (slot, i)
        try:
            return self._idents[k]
        except KeyError:
            pass
        with timed("engine:file_stat"):
            value = _identity(self.path(slot, i))
        self._idents[k] = value
        return value

    def missing_slots(self, i: int) -> list[str]:
        return [s for s in self.slots if s != MAIN_SLOT and self.ident(s, i) is None]

    def _raw_key(self, slot: str, i: int):
        ident = self.ident(slot, i)
        if ident is None:
            return None
        return ("raw",) + ident + (self.spec.data[slot],)

    # ── raw data ─────────────────────────────────────────────────────────────
    def read(self, slot: str, i: int, background: bool = False) -> pd.DataFrame:
        """Read one slot's day file from disk (declared columns only) and cache
        it. background=True: called on a read-ahead thread (timed separately —
        that time overlaps the main thread's)."""
        path = self.path(slot, i)
        key = self._raw_key(slot, i)
        if key is None:
            raise EngineError(f"The '{slot}' data file for {self.dates[i]} does not exist: {path}")
        prefix = "bg " if background else ""
        t0 = time.perf_counter()
        with timed(f"{prefix}io:read:{slot}"):
            frame = read_day_file(path, self.spec.data[slot], slot)
        dt = time.perf_counter() - t0
        with self._lock:
            self.raw_reads += 1
            self.reads_by_slot[slot] += 1
            self.read_seconds += dt
        with timed(f"{prefix}cache:store"):
            self.cache.put(key, frame)
        return frame

    def day_bytes(self, i: int) -> int:
        """Uncompressed size of day i's declared columns across all slots,
        from the parquet footers (no data read)."""
        total = 0
        for slot in self.slots:
            if self.ident(slot, i) is None:
                continue
            cols = self.spec.data[slot]
            try:
                meta = pq.ParquetFile(self.path(slot, i)).metadata
            except Exception:  # noqa: BLE001 — the real read reports problems
                continue
            for g in range(meta.num_row_groups):
                rg = meta.row_group(g)
                for c in range(rg.num_columns):
                    col = rg.column(c)
                    if cols is None or col.path_in_schema in cols:
                        total += col.total_uncompressed_size
        return total

    def raw(self, slot: str, i: int) -> pd.DataFrame:
        if slot not in self.spec.data:
            raise EngineError(f"No data slot '{slot}' — the strategy declares DATA slots "
                              f"{list(self.spec.data)}.")
        pending = self.pending.get(i)
        if pending is not None and slot in pending:
            return pending[slot]
        key = self._raw_key(slot, i)
        if key is not None:
            with timed("cache:lookup"):
                frame = self.cache.get(key, _MISS)
            if frame is not _MISS:
                with self._lock:
                    self.raw_hits += 1
                return frame
        return self.read(slot, i)

    # ── prepared day ─────────────────────────────────────────────────────────
    def _prep_key(self, i: int):
        idents = tuple(self.ident(s, i) for s in self.slots)
        cols = tuple((s, self.spec.data[s]) for s in self.slots)
        return ("prep", self.key, self.dates[i].isoformat(), idents, cols, self._prep_items)

    def prepared_cached(self, i: int) -> bool:
        return self.spec.has_prepare and self.cache.contains(self._prep_key(i))

    def prepared(self, i: int):
        if not self.spec.has_prepare:
            return None
        cur = self.current
        if cur is not None and cur[0] == i:
            return cur[1]
        key = self._prep_key(i)
        with timed("cache:lookup"):
            value = self.cache.get(key, _MISS)
        if value is not _MISS:
            self.prep_hits += 1
            return value
        data = {slot: self.raw(slot, i) for slot in self.slots}
        with timed("day:prepare"):
            value = self.module.prepare_day(self.dates[i], data, dict(self.prep_params))
        self.prep_builds += 1
        with timed("cache:store"):
            kept = self.cache.put(key, value)
        if kept:
            # the raw frames are now only needed to rebuild this prepared day:
            # first in line for eviction (they are ~2/3 of the cached bytes)
            self.cache.demote([self._raw_key(slot, i) for slot in self.slots])
            self.fresh_prep.add(key)
        return value

    def remeasure_fresh(self, i: int) -> None:
        """A prepared day built this run was just processed for the first time:
        strategies may fill lazy caches on it (ivb parses its JSON columns on
        first use), so re-measure its size once for the cache budget."""
        if not self.fresh_prep:
            return
        key = self._prep_key(i)
        if key in self.fresh_prep:
            self.fresh_prep.discard(key)
            cur = self.current
            if cur is not None and cur[0] == i:
                with timed("cache:remeasure"):
                    self.cache.resize(key, estimate_nbytes(cur[1]))


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


# ══ progress text ═════════════════════════════════════════════════════════════
class _Progress:
    """Turns the loop's steps into on_progress(current, total, text) calls.

    text = four lines:
      1  what the engine is doing RIGHT NOW (e.g. "Day 12 of 370 · 2025-05-06 —
         running the strategy (process_day)")
      2  the read-ahead: which days the background threads are reading
      3  the counters so far: trades, processed / skipped days, disk reads per
         slot, what came from RAM, prepared days built vs reused
      4  cache fill, elapsed time and the estimated time left
    Setup and wrap-up steps send current = 0 / total and only lines 1 (+ 4).
    """

    DETAIL_EVERY = 0.1        # seconds between rebuilds of lines 2-4

    def __init__(self, callback, spec: StrategySpec, folders: dict, cache: DayCache, t0: float):
        self.cb = callback
        self.spec = spec
        self.folders = folders
        self.cache = cache
        self.t0 = t0
        self.ctx = None
        self.total = 0
        self.done = 0
        self.trades = 0
        self.skipped = 0
        self.processed = 0
        self.ahead: list = []
        self._detail = ""
        self._detail_at = float("-inf")

    def step(self, now: str, current: int | None = None, force: bool = True) -> None:
        """Report `now` (line 1, always exact). Lines 2-4 are rebuilt when
        `force` (setup / wrap-up steps) or at most every DETAIL_EVERY seconds:
        a warm run makes thousands of calls a second, and building the counter
        lines for each was measurable."""
        if self.cb is None:
            return
        with timed("engine:progress"):
            t = time.perf_counter()
            if force or t - self._detail_at >= self.DETAIL_EVERY:
                self._detail = self._details()
                self._detail_at = t
            text = now + "\n" + self._detail if self._detail else now
            self.cb(self.done if current is None else current, self.total, text)

    def _details(self) -> str:
        lines = []
        ctx = self.ctx
        if ctx is not None and self.total:
            if self.ahead:
                days = [ctx.dates[i].isoformat() for i in self.ahead]
                span = days[0] if len(days) == 1 else f"{days[0]} … {days[-1]}"
                lines.append(f"Reading ahead on {ctx.read_threads} background threads: {span}")
            else:
                lines.append("Reading ahead: nothing left to read")
            reads = ", ".join(f"{s} {n}" for s, n in ctx.reads_by_slot.items())
            prep = (f" · prepared days: {ctx.prep_builds} built, {ctx.prep_hits} from RAM"
                    if self.spec.has_prepare else "")
            skipped = f" · {self.skipped} skipped" if self.skipped else ""
            lines.append(f"So far: {self.trades} trades · {self.processed} days processed"
                         f"{skipped} · files read from disk: {reads} · files from RAM: "
                         f"{ctx.raw_hits}{prep}")
        elapsed = time.perf_counter() - self.t0
        tail = (f"RAM cache {self.cache.bytes_used / GB:.2f} / "
                f"{self.cache.budget_bytes / GB:.2f} GB · elapsed {_duration(elapsed)}")
        if self.total and 0 < self.done < self.total:
            left = elapsed / self.done * (self.total - self.done)
            tail += f" · about {_duration(left)} left"
        lines.append(tail)
        return "\n".join(lines)

    def day(self, i: int, what: str) -> None:
        if self.cb is None:
            return
        self.step(f"Day {self.done + 1} of {self.total} · {self.ctx.dates[i].isoformat()} — {what}",
                  current=self.done, force=False)


def _slots_text(slots) -> str:
    return " + ".join(slots)


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
    on_progress    callback(current_day, total_days, text) — called at every
                   step (see _Progress); raising inside it cancels the run
                   (the app's worker cancellation)
    """
    t0 = time.perf_counter()
    timing.reset()
    spec = strategy_spec(module)
    cache = cache or get_cache()
    progress = _Progress(on_progress, spec, {}, cache, t0)
    progress.step(f"Checking {spec.name}'s DATA declaration and the data folders…")
    cache.begin_run()
    rejected0 = cache.rejected

    with timed("engine:setup"):
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
    progress.folders = folders
    progress.step(f"Listing the day files of {folders[MAIN_SLOT].name}…")
    with timed("engine:list_days"):
        ctx = _RunContext(module, spec, key, folders, merged, cache)
        lo, hi = pd.Timestamp(start_date).date(), pd.Timestamp(end_date).date()
        run_idx = [i for i, d in enumerate(ctx.dates) if lo <= d <= hi]
    total = len(run_idx)
    progress.ctx, progress.total = ctx, total
    slot_desc = ", ".join(f"{s} = {f.name}" for s, f in folders.items())
    progress.step(f"Found {total} days in range ({len(ctx.dates)} files in "
                  f"{folders[MAIN_SLOT].name}) · data: {slot_desc} · starting the day loop")

    skipped: dict[str, list[str]] = {}
    rows: list = []
    days_run = 0

    def _needs_read(i: int) -> list[str]:
        """Slots whose raw frame must be read for day i (nothing when the
        prepared day is already cached and process_day may never touch raw)."""
        with timed("engine:schedule"):
            if ctx.missing_slots(i):
                return []
            if spec.has_prepare and ctx.prepared_cached(i):
                return []
            return [s for s in ctx.slots if not cache.contains(ctx._raw_key(s, i))]

    def _load(i: int, slots: list[str]):
        return {s: ctx.read(s, i, background=True) for s in slots}

    with timed("engine:read_plan"):
        heavy = bool(run_idx) and ctx.day_bytes(run_idx[0]) > HEAVY_DAY_BYTES
    ctx.read_threads = HEAVY_READ_THREADS if heavy else READ_THREADS
    prefetch_days = HEAVY_PREFETCH_DAYS if heavy else PREFETCH_DAYS
    pool = ThreadPoolExecutor(max_workers=ctx.read_threads)
    try:
        queue: deque = deque()
        it = iter(run_idx)

        def _submit_next():
            i = next(it, None)
            if i is None:
                return False
            need = _needs_read(i)
            queue.append((i, need, pool.submit(_load, i, need) if need else None))
            return True

        for _ in range(prefetch_days):
            if not _submit_next():
                break

        while queue:
            i, need, fut = queue.popleft()
            _submit_next()
            progress.ahead = [q[0] for q in queue if q[2] is not None]
            date = ctx.dates[i]
            if fut is not None:
                if not fut.done():
                    progress.day(i, f"waiting for the disk: reading {_slots_text(need)}")
                with timed("io:wait_for_read"):
                    ctx.pending[i] = fut.result()    # re-raises read errors here

            missing = ctx.missing_slots(i)
            if missing:
                for slot in missing:
                    skipped.setdefault(slot, []).append(date.isoformat())
                progress.skipped += 1
                progress.day(i, f"skipped: no {_slots_text(missing)} file for this day")
            else:
                try:
                    if spec.has_prepare and progress.cb is not None:
                        if ctx.prepared_cached(i):
                            progress.day(i, "prepared day found in RAM")
                        else:
                            progress.day(i, "preparing the day (prepare_day)")
                    prepared = ctx.prepared(i)
                    ctx.current = (i, prepared)
                    if not (spec.has_prepare and prepared is None):
                        progress.day(i, "running the strategy (process_day)")
                        with timed("day:process"):
                            result = module.process_day(Day(ctx, i), merged)
                        with timed("engine:collect_trades"):
                            for trade in _as_trades(result, spec.name):
                                rows.append((date, trade))
                        days_run += 1
                        ctx.remeasure_fresh(i)
                except KeyError as e:
                    raise _explain_key_error(e, spec, f"the day {date}") from e
                finally:
                    ctx.current = None

            ctx.pending.pop(i, None)
            progress.done += 1
            progress.trades = len(rows)
            progress.processed = days_run
    finally:
        # cancelled / failed: drop the queued reads instead of finishing them
        pool.shutdown(wait=True, cancel_futures=True)

    progress.ahead = []
    progress.step(f"Building the trades table from {len(rows)} trades…", current=total)
    with timed("build_output_df"):
        trades = trades_to_frame(rows)

    warnings = skipped_warnings(skipped, {s: f.name for s, f in folders.items()})

    rejected = cache.rejected - rejected0
    cache_note = None
    if rejected:
        cache_note = (f"The RAM cache ({cache.budget_bytes / GB:.2f} GB) was too small for "
                      f"this run: {rejected} item(s) could not be kept and will be read or "
                      f"prepared again next run. Raise the budget (Settings -> Engine day "
                      f"cache; the Optimizer's Memory budget for parallel workers) or "
                      f"shorten the date range.")
        print(f"[engine] NOTE: {cache_note}", flush=True)

    elapsed = time.perf_counter() - t0
    st = cache.stats()
    reads = ", ".join(f"{s} {n}" for s, n in ctx.reads_by_slot.items())
    summary = (f"{spec.name}: {total} days in range, {days_run} processed"
               + (f", {sum(len(v) for v in skipped.values())} skipped" if skipped else "")
               + f", {len(rows)} trades in {_duration(elapsed)}. Files read from disk: "
               f"{ctx.raw_reads} ({reads}; {ctx.read_seconds:.2f} s of reading on "
               f"{ctx.read_threads} threads), from RAM: {ctx.raw_hits}.")
    if spec.has_prepare:
        summary += f" Prepared days: {ctx.prep_builds} built, {ctx.prep_hits} from RAM."
    summary += f" RAM cache {st['bytes'] / GB:.2f} / {st['budget'] / GB:.2f} GB."
    table = timing.report(spec.name, elapsed, echo=False)
    if verbose:
        print(f"[engine] {summary}", flush=True)
        for w in warnings:
            print(f"[engine] WARNING: {w}", flush=True)
        if table:
            print(table + "\n", flush=True)
    return RunResult(trades=trades, warnings=warnings, days_in_range=total,
                     days_run=days_run, skipped=skipped,
                     files_read=ctx.raw_reads, days_prepared=ctx.prep_builds,
                     cache_rejected=rejected, cache_note=cache_note,
                     elapsed=elapsed, timing=table, summary=summary, rows=rows)
