"""
1m_ohlcv_globex_single_asset.py — 1-minute OHLCV candles for ONE asset from daily Databento
ohlcv-1m DBN files. Single-asset twin of 1m_ohlcv_globex_mixed_assets.py: same candles,
same file format, but the input is one asset's daily files (like 1m_advanced.py) instead of
monthly all-asset files.

Input  : raw_dbn/{type}/{ASSET}/{dataset}/ — one .dbn.zst per UTC calendar day, any asset
         (e.g. parent symbology NNQ.FUT). Nothing is hard-coded: whatever outright contracts
         the files contain are used; spreads ('-') and options (' ') are dropped.
         Expected filename format: glbx-mdp3-YYYYMMDD.ohlcv-1m.dbn.zst
Output : one Parquet per trading day, written straight into the output folder picked in the
         Data Formatter:  parquet/{type}/{ASSET}/{name}/YYYY-MM-DD.parquet

Columns : open, high, low, close (float64) + volume (int64).
Metadata: file-level keys "symbol" / "front_month" (full front contract, e.g. NNQU6),
          "trade_date", "is_roll_day".

Session : 18:00 NY (prev evening) → 17:00 NY (current day) = full Globex session; the
          17:00-18:00 maintenance hour is excluded. Built by splicing the previous file and
          the current file, and cut on absolute NY times — so a missing previous day
          (holiday gap) just leaves the ETH head empty instead of mislabeling bars.
Date label: trade_date = the current file's date (= date of the RTH session, NY).
Gap filling: full 1380-bar 1-minute grid (18:00→17:00 NY), OHLC forward/back-filled, volume=0.
Front month: the symbol with the highest total session volume; roll day when the 2nd-highest
          volume > 0.2 × the highest.
Weekends / holidays: Sat/Sun files are never a trade_date (still used as the next day's prev);
          sessions without RTH bars (09:30-16:00 NY) are skipped.
First file: only used as the prev of the second (no earlier file to complete its ETH).
Skip existing: checked from the filename first — an existing day costs no decode at all.
          Files are decoded on demand and cached forward, so each is decoded at most once.
Timing  : every processing block is timed (accumulated per run) and a full breakdown table
          (seconds + % of wall) is emitted to the log at the end of each run.

Performance notes: decoding uses store.to_ndarray() (raw structured array) and resolves the
handful of instrument_ids to symbols once, instead of to_df(map_symbols=True); per-day
Parquets are built as pyarrow tables from numpy arrays against a schema template captured
once via Table.from_pandas (keeps the b'pandas' metadata so pd.read_parquet reconstructs the
tz-aware index/dtypes). Gap-filling scatters bars onto the grid by integer minute offset
(no DST transition can fall inside a session: transitions are Sun 02:00 NY, Globex closed).
"""

import datetime
import gc
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import databento as db
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

NY        = "America/New_York"
GRID_BARS = 1380          # 18:00 → 17:00 NY, one minute each

_NS_MIN = 60_000_000_000

_UNDEF_PRICE = np.iinfo(np.int64).max   # databento UNDEF_PRICE sentinel
_PX_SCALE    = 1e9                      # databento FIXED_PRICE_SCALE


# ---------------------------------------------------------------------------
# Timing framework — accumulating micro-timers, full table printed per run
# ---------------------------------------------------------------------------

_TIMES: dict[str, float] = defaultdict(float)
_CALLS: dict[str, int]   = defaultdict(int)


class _t:
    """Accumulating timer: `with _t("stage.block"): ...`"""
    __slots__ = ("name", "t0")

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = perf_counter()

    def __exit__(self, *exc):
        _TIMES[self.name] += perf_counter() - self.t0
        _CALLS[self.name] += 1
        return False


def _timing_reset() -> None:
    _TIMES.clear()
    _CALLS.clear()


def _timing_report(wall: float) -> str:
    wall    = max(wall, 1e-9)
    rows    = sorted(_TIMES.items(), key=lambda kv: kv[1], reverse=True)
    tracked = sum(_TIMES.values())
    other   = max(wall - tracked, 0.0)
    lines   = ["── timing breakdown ──────────────────────────────────────────"]
    for name, secs in rows:
        lines.append(f"{name:<36}{secs:9.2f}s  {100 * secs / wall:5.1f}%   {_CALLS[name]:>6}x")
    lines.append(f"{'(untracked: loop/interpreter overhead)':<36}{other:9.2f}s  {100 * other / wall:5.1f}%")
    lines.append(f"{'TOTAL wall':<36}{wall:9.2f}s  100.0%")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------

def _file_date(path: Path) -> datetime.date:
    """'glbx-mdp3-20260825.ohlcv-1m.dbn.zst' -> date(2026, 8, 25)."""
    ymd = path.name.split(".")[0].split("-")[-1]
    return datetime.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))


# ---------------------------------------------------------------------------
# Load (one pass per file)
# ---------------------------------------------------------------------------

_LOAD_COLS = ["open", "high", "low", "close", "volume", "symbol"]


def _px_float(ints: np.ndarray) -> np.ndarray:
    """Fixed-precision int64 prices → float64, exactly like databento's _format_px
    (UNDEF_PRICE → NaN, then divide by FIXED_PRICE_SCALE)."""
    out   = ints.astype(np.float64)
    undef = ints == _UNDEF_PRICE
    if undef.any():
        out[undef] = np.nan
    out /= _PX_SCALE
    return out


def _instrument_symbols(store, unique_ids) -> dict:
    """Resolve the handful of instrument_ids in a daily file to raw symbols once."""
    imap = db.common.symbology.InstrumentMap()
    imap.insert_metadata(store.metadata)
    date = pd.Timestamp(store.metadata.start, unit="ns").date()
    out = {}
    for iid in unique_ids:
        try:
            sym = imap.resolve(int(iid), date)
        except Exception:
            sym = None
        out[int(iid)] = sym
    return out


def _is_outright(sym) -> bool:
    return isinstance(sym, str) and "-" not in sym and " " not in sym


def _load(path: Path) -> pd.DataFrame:
    """Decode one daily DBN file into a NY-indexed frame of outright-contract OHLCV bars."""
    with _t("decode.from_file"):
        store = db.DBNStore.from_file(path)
    with _t("decode.to_ndarray"):
        arr = store.to_ndarray()
    if not len(arr):
        return pd.DataFrame(columns=_LOAD_COLS)

    with _t("decode.symbols"):
        codes, uniq = pd.factorize(arr["instrument_id"])
        id2sym  = _instrument_symbols(store, uniq)
        u_sym   = np.array([id2sym[int(u)] for u in uniq], dtype=object)
        u_ok    = np.array([_is_outright(s) for s in u_sym], dtype=bool)
        keep    = u_ok[codes]
    if not keep.any():
        return pd.DataFrame(columns=_LOAD_COLS)

    with _t("decode.build_frame"):
        idx = (
            pd.DatetimeIndex(arr["ts_event"][keep].view(np.int64).view("M8[ns]"))
            .tz_localize("UTC")
            .tz_convert(NY)
        )
        out = pd.DataFrame(
            {
                "open":   _px_float(arr["open"][keep]),
                "high":   _px_float(arr["high"][keep]),
                "low":    _px_float(arr["low"][keep]),
                "close":  _px_float(arr["close"][keep]),
                "volume": arr["volume"][keep],
                "symbol": u_sym[codes][keep],
            },
            index=idx,
        )
    return out


# ---------------------------------------------------------------------------
# Session / front month
# ---------------------------------------------------------------------------

def _session(prev_df: pd.DataFrame | None, curr_df: pd.DataFrame,
             start_ns: int) -> pd.DataFrame:
    """Rows in [18:00 NY prev day, 17:00 NY trade day) from prev + curr files."""
    end_ns = start_ns + GRID_BARS * _NS_MIN
    parts = []
    for df in (prev_df, curr_df):
        if df is None or df.empty:
            continue
        i8 = df.index.values.view("int64")
        parts.append(df[(i8 >= start_ns) & (i8 < end_ns)])
    if not parts:
        return pd.DataFrame(columns=_LOAD_COLS)
    return pd.concat(parts).sort_index(kind="stable")


def _front_month(sess: pd.DataFrame) -> tuple[str | None, bool]:
    """(front symbol = max session volume, is_roll = 2nd-highest > 0.2 × highest)."""
    vol = sess.groupby("symbol")["volume"].sum().sort_values(ascending=False, kind="stable")
    if vol.empty:
        return None, False
    is_roll = len(vol) > 1 and vol.iloc[1] > 0.2 * vol.iloc[0]
    return vol.index[0], bool(is_roll)


# ---------------------------------------------------------------------------
# Gap filling (numpy) + arrow write with cached schema template
# ---------------------------------------------------------------------------

def _fill_arrays(o, h, l, c, v, pos: np.ndarray):
    """
    Scatter one day's bars onto the 1380-slot grid and gap-fill: close ffill→bfill;
    open/high/low take the filled close on missing slots; volume 0.
    Returns (open, high, low, close, volume) full-grid arrays.
    """
    n = GRID_BARS
    with _t("fill.scatter"):
        close = np.full(n, np.nan)
        close[pos] = c
        mask = np.zeros(n, dtype=bool)
        mask[pos] = True
    with _t("fill.ffill"):
        idxs = np.where(mask, np.arange(n), 0)
        np.maximum.accumulate(idxs, out=idxs)
        close_f = close[idxs]
        first = int(pos.min())
        if first > 0:                       # bfill the head before the first bar
            close_f[:first] = close_f[first]
    with _t("fill.finish"):
        open_f = close_f.copy(); open_f[pos] = o
        high_f = close_f.copy(); high_f[pos] = h
        low_f  = close_f.copy(); low_f[pos]  = l
        vol    = np.zeros(n, dtype=np.int64); vol[pos] = v
    return open_f, high_f, low_f, close_f, vol


_TEMPLATE: dict = {}     # "schema" -> pa.Schema, "meta" -> base metadata dict (per run)


def _grid_for(td: datetime.date):
    with _t("write.grid"):
        sess_start = pd.Timestamp(f"{(td - datetime.timedelta(days=1)).isoformat()} 18:00:00", tz=NY)
        full_index = pd.date_range(sess_start, periods=GRID_BARS, freq="1min", tz=NY)
    return full_index, full_index[0].value


def _ensure_template(full_index: pd.DatetimeIndex) -> None:
    """Capture the arrow schema (incl. b'pandas' metadata) once via from_pandas, so the
    direct-from-arrays fast path writes files that read back identically."""
    if _TEMPLATE:
        return
    zeros = np.zeros(GRID_BARS)
    probe = pd.DataFrame(
        {"open": zeros, "high": zeros, "low": zeros, "close": zeros,
         "volume": np.zeros(GRID_BARS, dtype=np.int64)},
        index=full_index,
    )
    table = pa.Table.from_pandas(probe)
    _TEMPLATE["schema"] = table.schema
    _TEMPLATE["meta"]   = dict(table.schema.metadata or {})


def _write_day(arrays, full_index: pd.DatetimeIndex, out_file: Path, front_symbol: str,
               trade_date: datetime.date, is_roll: bool) -> None:
    with _t("wp.table"):
        o, h, l, c, v = arrays
        table = pa.Table.from_arrays(
            [pa.array(o), pa.array(h), pa.array(l), pa.array(c), pa.array(v),
             pa.array(full_index)],
            schema=_TEMPLATE["schema"],
        )
    with _t("wp.metadata"):
        meta = dict(_TEMPLATE["meta"])
        meta[b"symbol"]      = str(front_symbol).encode()
        meta[b"front_month"] = str(front_symbol).encode()
        meta[b"trade_date"]  = str(trade_date).encode()
        meta[b"is_roll_day"] = str(bool(is_roll)).encode()
        table = table.replace_schema_metadata(meta)
    with _t("wp.write_table"):
        pq.write_table(table, out_file)


# ---------------------------------------------------------------------------
# Day builder
# ---------------------------------------------------------------------------

def _build_day(prev_df: pd.DataFrame | None, curr_df: pd.DataFrame,
               td: datetime.date, out_file: Path, log) -> bool:
    """Build + write one trading day. Returns True when a file was written."""
    full_index, start_ns = _grid_for(td)

    with _t("day.session"):
        sess = _session(prev_df, curr_df, start_ns)
    if sess.empty:
        log(f"↷ {td} — no bars in session, skipped")
        return False

    # skip days without RTH bars (holidays) — checked on all contracts, before front filter
    with _t("day.rth_check"):
        rth_lo = pd.Timestamp(f"{td.isoformat()} 09:30", tz=NY).value
        rth_hi = pd.Timestamp(f"{td.isoformat()} 16:00", tz=NY).value
        i8 = sess.index.values.view("int64")
        has_rth = bool(((i8 >= rth_lo) & (i8 <= rth_hi)).any())
    if not has_rth:
        log(f"↷ {td} — no RTH bars, skipped")
        return False

    with _t("day.front_month"):
        front_symbol, is_roll = _front_month(sess)
        g = sess[sess["symbol"].to_numpy() == front_symbol]
    if front_symbol is None or g.empty:
        return False

    with _t("day.pos"):
        off  = g.index.values.view("int64") - start_ns
        pos  = off // _NS_MIN
        ok   = (pos >= 0) & (pos < GRID_BARS) & (off % _NS_MIN == 0)
        vals = [g[col].to_numpy() for col in ("open", "high", "low", "close", "volume")]
        if not ok.all():                    # drop non-grid rows
            pos  = pos[ok]
            vals = [a[ok] for a in vals]
    if not len(pos):
        return False

    _ensure_template(full_index)
    arrays = _fill_arrays(*vals, pos)
    _write_day(arrays, full_index, out_file, front_symbol, td, is_roll)
    log(f"✓ {td} — {front_symbol}{' ROLL' if is_roll else ''}")
    return True


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_all(
    input_folder: str,
    output_folder: str,
    skip_existing: bool = True,
    on_progress: callable = None,
) -> None:
    _timing_reset()
    _TEMPLATE.clear()
    t_run0 = perf_counter()

    input_path  = Path(input_folder)
    output_path = Path(output_folder)

    with _t("run.glob"):
        files = sorted(input_path.glob("*.dbn.zst"))
    if len(files) < 2:
        if on_progress:
            on_progress(1, 1, "ERROR: Need at least 2 files to build a session.")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    total = len(files) - 1

    # Deferred load + forward cache (as in 1m_advanced): a skipped day costs no decode;
    # prev_df is loaded on demand and the current frame is carried forward as next prev.
    prev_df   = None
    prev_file = None   # which Path prev_df currently holds
    written   = 0

    for i in range(total):
        current_file = files[i + 1]

        def log(msg: str, _i=i):
            if on_progress:
                on_progress(_i + 1, total, msg)

        try:
            td = _file_date(current_file)
        except (ValueError, IndexError):
            log(f"ERROR: cannot parse date from {current_file.name}")
            continue
        out_file = output_path / f"{td.isoformat()}.parquet"

        # 0) WEEKEND — a Sat/Sun file is never a trade_date (still loaded as next prev).
        if td.weekday() >= 5:
            log(f"↷ {td} — weekend (no session)")
            continue

        # 1) SKIP FIRST — zero work; drop any cached prev so we don't hold memory.
        with _t("run.exists_check"):
            exists = skip_existing and out_file.exists()
        if exists:
            log(f"↷ {td} — already processed")
            prev_df, prev_file = None, None
            continue

        # 2) Need to process — ensure prev_df holds files[i] (load on demand, reuse cache).
        try:
            if prev_file != files[i]:
                prev_df, prev_file = None, None
                prev_df, prev_file = _load(files[i]), files[i]
            curr_df = _load(current_file)
        except Exception as e:
            log(f"ERROR loading {current_file.name}: {e}")
            prev_df, prev_file = None, None
            gc.collect()
            continue

        try:
            if _build_day(prev_df, curr_df, td, out_file, log):
                written += 1
        except Exception as e:
            log(f"ERROR {current_file.name}: {e}")

        # 3) Cache curr forward as next prev.
        # (no per-day gc.collect: daily ohlcv-1m frames are tiny and it cost ~80% of wall)
        prev_df, prev_file = curr_df, current_file

    wall = perf_counter() - t_run0
    if on_progress:
        on_progress(total, total, f"Done — {written} day-files written")
        on_progress(total, total, _timing_report(wall))
