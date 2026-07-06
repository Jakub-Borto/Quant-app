"""
1m_ohlcv_globex_mixed_assets.py — 1-minute OHLCV candles from monthly Databento
ohlcv-1m DBN files that each contain *all* ~30 assets for one calendar month.

Input  : folder of monthly .dbn.zst files.
         Expected filename format: glbx-mdp3-YYYYMMDD-YYYYMMDD.ohlcv-1m.dbn.zst
Output : one Parquet per asset per trading day, auto-routed to each asset's own folder.
         The Data Formatter passes output_folder = data/parquet/{type}/{placeholder}/{name}.
         We derive  base = output_folder.parent.parent  (= data/parquet/{type})  and route
         every asset to   base/{SYMBOL}/{folder}/YYYY-MM-DD.parquet   where {folder} is the
         typed name with the first uppercase "X" replaced by the symbol (fallback
         "{SYMBOL}_{name}"). E.g. typed name "X_1m_ohlcv_globex_15y" ->
         ES/ES_1m_ohlcv_globex_15y/, 6B/6B_1m_ohlcv_globex_15y/. Folders are created on demand.

Columns : open, high, low, close (float64) + volume (int64).
Metadata: each Parquet carries file-level key "symbol" = full front-month contract (e.g. ESM6).

Session : 18:00 NY (prev evening) → 17:00 NY (current day) = full Globex session; the
          17:00-18:00 maintenance hour is excluded. Defined in NY, DST handled via a fixed
          1380-bar grid.
Date label: trade_date = date of the RTH session (America/New_York).
Gap filling: full 1380-bar 1-minute grid (18:00→17:00 NY), OHLC forward/back-filled, volume=0.
Front month: per (trade_date, asset) the symbol with the highest total session volume.
Weekends / holidays: no RTH bars → silently skipped.
First session of the first file: skipped (no previous file to complete the ETH portion).
Month boundaries: the last session of each month (RTH lands in the next file) is completed by
          splicing that trade_date's rows from the current and next files. Each file is loaded
          and prepared exactly once.
Last file: fully processed from its own data (its final incomplete day is skipped for lack of RTH).
Skip existing: coarse month-level skip — if the reference asset (ES) already has this month's
          recent days *and* the month's boundary day, the whole month is skipped without decoding.
          Individual day-files that already exist are also skipped before writing.
Timing  : every processing block is timed (accumulated per run) and a full breakdown table
          (seconds + % of wall) is emitted to the log at the end of each run.

Performance notes (why the decode/write paths look unusual):
- Databento's to_df(map_symbols=True) resolves symbols per-row via np.unique over the whole
  file (~2.5s per monthly file). We decode via store.to_ndarray() (the raw structured array —
  no DataFrame, no pd.to_datetime, no pandas price scaling) and resolve instrument_id →
  raw symbol ourselves from store.metadata.mappings at *unique-id* level (a few hundred ids),
  reproducing resolve()'s [start_date, end_date) UTC-date semantics exactly. ts_event int64
  nanoseconds are used directly; fixed-precision int prices are scaled to float64 only for
  rows that survive the keep-mask.
- NY wall-clock is derived from UTC int64 nanoseconds + the UTC offset (≤1 DST transition per
  monthly file, located by binary search) instead of per-row tz materialisation.
- Per-day Parquets are built as pyarrow tables directly from numpy arrays against a schema
  template captured once via Table.from_pandas (so the b'pandas' metadata that lets
  pd.read_parquet reconstruct the tz-aware index/dtypes is preserved bit-for-bit).
- Gap-filling scatters bars onto the 1380-slot grid by integer minute offset from the session
  start (grid slots are absolute 60s steps, and no DST transition can fall inside a session:
  transitions happen Sun 02:00 NY, when Globex is closed).
"""

import re
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
REF_ASSET = "ES"          # reference asset used for the coarse skip check

_NS_MIN = 60_000_000_000
_NS_DAY = 86_400_000_000_000

_UNDEF_PRICE = np.iinfo(np.int64).max   # databento UNDEF_PRICE sentinel
_PX_SCALE    = 1e9                      # databento FIXED_PRICE_SCALE

# ---------------------------------------------------------------------------
# Asset registry — prefix → output folder name
# Longer prefixes before shorter ones to avoid shadowing (MES before ES, MGC before GC)
# ---------------------------------------------------------------------------
ASSETS: dict[str, str] = {
    # Equity Index
    "MES": "MES",
    "MNQ": "MNQ",
    "M2K": "M2K",
    "MYM": "MYM",
    "ES":  "ES",
    "NQ":  "NQ",
    "RTY": "RTY",
    "YM":  "YM",
    # Rates
    "ZN":  "ZN",
    "ZB":  "ZB",
    "ZF":  "ZF",
    "ZT":  "ZT",
    "SR3": "SR3",
    # Energy
    "QM":  "QM",
    "CL":  "CL",
    "NG":  "NG",
    "RB":  "RB",
    "HO":  "HO",
    # Metals
    "MGC": "MGC",
    "GC":  "GC",
    "SI":  "SI",
    "HG":  "HG",
    # Grains
    "ZC":  "ZC",
    "ZS":  "ZS",
    "ZW":  "ZW",
    # FX
    "6E":  "6E",
    "6J":  "6J",
    "6B":  "6B",
    "6C":  "6C",
    # Crypto
    "BTC": "BTC",
}

# prefixes sorted longest-first for unambiguous longest-prefix matching
_PREFIXES_BY_LEN = sorted(ASSETS, key=len, reverse=True)


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

def _parse_date_range(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    match = re.search(r"(\d{8})-(\d{8})", path.name)
    if not match:
        raise ValueError(f"Cannot parse date range from filename: {path.name}")
    return pd.Timestamp(match.group(1)), pd.Timestamp(match.group(2))


# ---------------------------------------------------------------------------
# Output-path helpers
# ---------------------------------------------------------------------------

def _folder_for(typed_name: str, symbol: str) -> str:
    """
    Per-symbol dataset folder name. The typed name carries a single uppercase 'X'
    placeholder that is replaced by the symbol (the lowercase 'x' in e.g. "globex" is
    untouched). If there is no 'X', fall back to '{symbol}_{typed_name}'.
    """
    if "X" in typed_name:
        return typed_name.replace("X", symbol, 1)
    return f"{symbol}_{typed_name}"


def _asset_for(sym) -> str | None:
    """Asset folder for a raw contract symbol; None for spreads/unmapped/missing."""
    if not isinstance(sym, str) or "-" in sym or " " in sym:
        return None
    for prefix in _PREFIXES_BY_LEN:
        if sym.startswith(prefix):
            return ASSETS[prefix]
    return None


# ---------------------------------------------------------------------------
# NY wall-clock from UTC nanoseconds
# ---------------------------------------------------------------------------

def _ny_offset_ns(ts_ns: int) -> int:
    ts = pd.Timestamp(ts_ns, unit="ns", tz="UTC").tz_convert(NY)
    return int(ts.utcoffset().total_seconds()) * 1_000_000_000


def _wall_ns(utc_ns: np.ndarray) -> np.ndarray:
    """
    NY wall-clock as int64 ns for each UTC int64 ns timestamp. Valid for spans with at
    most one DST transition (monthly files qualify). The transition instant is located
    by binary search on the offset, then applied with two vectorised adds — this matches
    pandas' tz_localize(None) output exactly at ~1/20th the cost.
    """
    lo, hi = int(utc_ns.min()), int(utc_ns.max())
    o_lo, o_hi = _ny_offset_ns(lo), _ny_offset_ns(hi)
    if o_lo == o_hi:
        return utc_ns + o_lo
    while hi - lo > 1:                      # find first ns with the new offset
        mid = (lo + hi) // 2
        if _ny_offset_ns(mid) == o_lo:
            lo = mid
        else:
            hi = mid
    wall = utc_ns + o_lo
    late = utc_ns >= hi
    wall[late] = utc_ns[late] + o_hi
    return wall


# ---------------------------------------------------------------------------
# Fast symbol resolution from DBN metadata (bypasses databento's per-row resolve)
# ---------------------------------------------------------------------------

def _id_intervals(store) -> dict[int, list[tuple[int, int, str]]] | None:
    """
    instrument_id → [(start_day, end_day, raw_symbol), ...] from store.metadata.mappings,
    with days as int days-since-epoch (UTC). Requires stype_out == instrument_id (the
    standard layout for GLBX files); returns None otherwise so the caller can fall back
    to databento's own (slow) mapping.
    """
    meta = store.metadata
    if str(meta.stype_out) != "instrument_id":
        return None
    out: dict[int, list[tuple[int, int, str]]] = {}
    epoch = datetime.date(1970, 1, 1)
    try:
        for raw_sym, entries in meta.mappings.items():
            for e in entries:
                s = e["symbol"]
                if not s:
                    continue
                iid   = int(s)
                start = (e["start_date"] - epoch).days
                end   = (e["end_date"]   - epoch).days
                out.setdefault(iid, []).append((start, end, str(raw_sym)))
    except (ValueError, KeyError, TypeError):
        return None
    for ivs in out.values():
        ivs.sort()
    return out


def _resolve_symbols_fast(store, iid: np.ndarray, days: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Per-row (symbol, asset, valid) arrays from the raw instrument_id field — all mapping and
    spread/asset classification done at unique-id level. Replicates databento resolve()
    semantics: a row is valid only when its UTC date lies in [start_date, end_date).
    """
    with _t("decode.map_build"):
        intervals = _id_intervals(store)
    if intervals is None:
        return None

    with _t("decode.map_uniques"):
        codes, uniq = pd.factorize(iid)
        n_u     = len(uniq)
        u_sym   = np.empty(n_u, dtype=object)
        u_asset = np.empty(n_u, dtype=object)
        u_start = np.full(n_u, np.iinfo(np.int64).max, dtype=np.int64)
        u_end   = np.full(n_u, np.iinfo(np.int64).min, dtype=np.int64)
        multi: list[int] = []
        for k, iid in enumerate(uniq):
            ivs = intervals.get(int(iid))
            if not ivs:
                u_sym[k] = None
                u_asset[k] = None
            elif len(ivs) == 1:
                start, end, sym = ivs[0]
                u_sym[k], u_asset[k] = sym, _asset_for(sym)
                u_start[k], u_end[k] = start, end
            else:                          # rare: id remaps mid-file — resolve per row below
                u_sym[k] = None
                u_asset[k] = None
                multi.append(k)

    with _t("decode.map_assign"):
        sym   = u_sym[codes]
        asset = u_asset[codes]
        valid = (days >= u_start[codes]) & (days < u_end[codes]) & (asset != None)  # noqa: E711
        for k in multi:
            rows = np.flatnonzero(codes == k)
            for r in rows:
                for start, end, s in intervals[int(uniq[k])]:
                    if start <= days[r] < end:
                        a = _asset_for(s)
                        if a is not None:
                            sym[r], asset[r], valid[r] = s, a, True
                        break
    return sym, asset, valid


# ---------------------------------------------------------------------------
# Load + prepare (one pass per file)
# ---------------------------------------------------------------------------

_PREP_COLS = ["open", "high", "low", "close", "volume", "symbol", "asset", "trade_date", "is_rth"]


def _px_float(ints: np.ndarray) -> np.ndarray:
    """Fixed-precision int64 prices → float64, exactly like databento's _format_px
    (UNDEF_PRICE → NaN, then divide by FIXED_PRICE_SCALE)."""
    out   = ints.astype(np.float64)
    undef = ints == _UNDEF_PRICE
    if undef.any():
        out[undef] = np.nan
    out /= _PX_SCALE
    return out


def _load_prepared(path: Path) -> pd.DataFrame:
    """
    Decode one monthly DBN file into the NY-indexed working frame: OHLCV + symbol + asset +
    trade_date (calendar RTH date, tz-naive midnight) + is_rth. Rows in the 17:00-17:59 NY
    maintenance hour, spread/unmapped symbols, and out-of-interval ids are dropped, so each
    session window is exactly [18:00, 17:00) NY.

    Fast path works on store.to_ndarray()'s structured array — no intermediate DataFrame,
    no pd.to_datetime, prices scaled only for rows that survive the keep-mask.
    """
    with _t("decode.from_file"):
        store = db.DBNStore.from_file(path)
    with _t("decode.to_ndarray"):
        arr = store.to_ndarray()

    fields = set(arr.dtype.names or ())
    fast   = {"ts_event", "instrument_id", "open", "high", "low", "close", "volume"} <= fields

    resolved = None
    if fast and len(arr):
        utc_ns   = arr["ts_event"].view(np.int64)
        resolved = _resolve_symbols_fast(store, arr["instrument_id"], utc_ns // _NS_DAY)

    if resolved is not None:
        sym, asset, valid = resolved
        raw = None
    else:                                   # unexpected schema/metadata — slow but safe
        with _t("decode.to_df_fallback"):
            raw = store.to_df()
            if raw.index.tz is None:
                raw.index = raw.index.tz_localize("UTC")
            sym = raw["symbol"].to_numpy()
            codes, uniq = pd.factorize(sym, use_na_sentinel=False)
            u_asset = np.array([_asset_for(s) for s in uniq], dtype=object)
            asset   = u_asset[codes]
            valid   = asset != None         # noqa: E711
            utc_ns  = raw.index.values.view("int64")

    if not len(utc_ns):
        return pd.DataFrame(columns=_PREP_COLS)

    with _t("prep.wall_ns"):
        wall = _wall_ns(utc_ns)
        hour = (wall // 3_600_000_000_000) % 24

    with _t("prep.keep_mask"):
        keep = valid & (hour != 17)         # drop maintenance hour + invalid symbols

    if not keep.any():
        return pd.DataFrame(columns=_PREP_COLS)

    with _t("prep.take"):
        wall_k = wall[keep]
        hour_k = hour[keep]
        sym_k  = sym[keep]
        ast_k  = asset[keep]
        if raw is None:
            cols = {c: _px_float(arr[c][keep]) for c in ("open", "high", "low", "close")}
            cols["volume"] = arr["volume"][keep]
            idx_ny = (
                pd.DatetimeIndex(utc_ns[keep].view("M8[ns]"))
                .tz_localize("UTC")
                .tz_convert(NY)
            )
        else:
            cols   = {c: raw[c].to_numpy()[keep] for c in ("open", "high", "low", "close", "volume")}
            idx_ny = raw.index.tz_convert(NY)[keep]

    with _t("prep.trade_date"):
        day_ns     = wall_k - wall_k % _NS_DAY
        trade_date = (day_ns + (hour_k >= 18) * _NS_DAY).view("M8[ns]")

    with _t("prep.rth_mask"):
        minutes = (wall_k // _NS_MIN) % 1440
        is_rth  = (minutes >= 9 * 60 + 30) & (minutes <= 16 * 60)   # 09:30 .. 16:00 inclusive

    with _t("prep.build_frame"):
        out = pd.DataFrame(
            {
                "open":       cols["open"],
                "high":       cols["high"],
                "low":        cols["low"],
                "close":      cols["close"],
                "volume":     cols["volume"],
                "symbol":     sym_k,
                "asset":      ast_k,
                "trade_date": trade_date,
                "is_rth":     is_rth,
            },
            index=idx_ny,
        )
    return out


# ---------------------------------------------------------------------------
# Front-month resolution
# ---------------------------------------------------------------------------

def _resolve_fronts(prep: pd.DataFrame) -> pd.DataFrame:
    """
    Per (trade_date, asset): front-month symbol (max total volume) + is_roll flag
    (2nd-highest volume > 0.2 × highest). One groupby pass over the whole month.
    Returns a DataFrame indexed by (trade_date, asset) with columns front_symbol, is_roll.
    """
    with _t("fronts.groupby_sum"):
        vol = (
            prep.groupby(["trade_date", "asset", "symbol"], sort=False)["volume"]
            .sum()
            .reset_index()
        )
    with _t("fronts.sort"):
        vol = vol.sort_values(
            ["trade_date", "asset", "volume"],
            ascending=[True, True, False],
            kind="stable",
        )
    with _t("fronts.top2"):
        vol["rank"] = vol.groupby(["trade_date", "asset"], sort=False).cumcount()

        r0 = vol[vol["rank"] == 0].set_index(["trade_date", "asset"])
        r1 = vol[vol["rank"] == 1].set_index(["trade_date", "asset"])["volume"]

        front   = r0["symbol"].rename("front_symbol")
        v0      = r0["volume"]
        v1      = r1.reindex(front.index)
        is_roll = (v1 > 0.2 * v0).fillna(False)

    return pd.DataFrame({"front_symbol": front, "is_roll": is_roll})


# ---------------------------------------------------------------------------
# Gap filling (numpy) + arrow write with cached schema template
# ---------------------------------------------------------------------------

def _fill_arrays(o, h, l, c, v, pos: np.ndarray):
    """
    Scatter one day's bars onto the 1380-slot grid and gap-fill exactly like the original
    reindex+ffill/bfill: close ffill→bfill; open/high/low take the filled close on missing
    slots; volume 0. Returns (open, high, low, close, volume) full-grid arrays.
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


# per-run caches (reset in run_all / per file where noted)
_GRID:      dict = {}     # trade_date -> (full_index, start_ns, pa_index_array)
_DIRS:      set  = set()  # already-ensured output dirs
_TEMPLATE:  dict = {}     # "schema" -> pa.Schema, "meta" -> base metadata dict


def _grid_for(td: pd.Timestamp):
    hit = _GRID.get(td)
    if hit is not None:
        return hit
    with _t("write.grid"):
        sess_start_date = td.date() - datetime.timedelta(days=1)
        sess_start = pd.Timestamp(f"{sess_start_date.isoformat()} 18:00:00", tz=NY)
        full_index = pd.date_range(sess_start, periods=GRID_BARS, freq="1min", tz=NY)
        entry = (full_index, full_index[0].value, pa.array(full_index))
    _GRID[td] = entry
    return entry


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


def _write_day(arrays, pa_index, out_file: Path, front_symbol: str,
               trade_date: datetime.date, is_roll: bool) -> None:
    with _t("wp.mkdir"):
        out_dir = out_file.parent
        if out_dir not in _DIRS:
            out_dir.mkdir(parents=True, exist_ok=True)
            _DIRS.add(out_dir)
    with _t("wp.table"):
        o, h, l, c, v = arrays
        table = pa.Table.from_arrays(
            [pa.array(o), pa.array(h), pa.array(l), pa.array(c), pa.array(v), pa_index],
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
# Day writer
# ---------------------------------------------------------------------------

def _write_prepared(prep: pd.DataFrame, base: Path, typed_name: str,
                    skip_existing: bool, i: int, log) -> int:
    """Front-resolve and write every trade_date present in `prep`. Returns files written."""
    if prep.empty:
        return 0

    # keep only trade_dates that actually have RTH bars (skip weekends/holidays)
    with _t("write.has_rth"):
        has_rth = prep.groupby("trade_date")["is_rth"].any()
        valid   = has_rth.index[has_rth.to_numpy()]
    with _t("write.filter_rth"):
        prep = prep[prep["trade_date"].isin(valid)]
    if prep.empty:
        return 0

    fronts = _resolve_fronts(prep)
    with _t("write.front_maps"):
        front_map = fronts["front_symbol"].to_dict()
        roll_map  = fronts["is_roll"].to_dict()

    # single vectorised pass: keep only each (trade_date, asset)'s front-symbol rows,
    # so the per-day loop below never string-filters inside groups
    with _t("write.front_filter"):
        td_days = prep["trade_date"].to_numpy().view("int64") // _NS_DAY
        a_codes, a_uniq = pd.factorize(prep["asset"].to_numpy())
        combined = td_days * 64 + a_codes                     # ≤30 assets → 6 bits is plenty
        acode = {a: k for k, a in enumerate(a_uniq)}
        front_by_key = {}
        for (td, asset), s in front_map.items():
            k = acode.get(asset)
            if k is not None:
                front_by_key[(td.value // _NS_DAY) * 64 + k] = s
        c_codes, c_uniq = pd.factorize(combined)
        u_front   = np.array([front_by_key.get(int(x)) for x in c_uniq], dtype=object)
        front_row = u_front[c_codes]
        prep_f    = prep[prep["symbol"].to_numpy() == front_row]

    with _t("write.groupby_split"):
        groups = list(prep_f.groupby(["trade_date", "asset"], sort=False))

    written = 0
    parts: list[str] = []

    for (td, asset), g in groups:
        key          = (td, asset)
        front_symbol = front_map.get(key)
        if front_symbol is None or not len(g):
            continue

        rth_date = td.date()
        out_dir  = base / asset / _folder_for(typed_name, asset)
        out_file = out_dir / f"{rth_date.isoformat()}.parquet"

        with _t("write.exists_check"):
            skip = skip_existing and out_file.exists()
        if skip:
            continue

        full_index, start_ns, pa_index = _grid_for(td)
        _ensure_template(full_index)

        with _t("write.pos"):
            i8  = g.index.values.view("int64")
            off = i8 - start_ns
            pos = off // _NS_MIN
            ok  = (pos >= 0) & (pos < GRID_BARS) & (off % _NS_MIN == 0)
            vals = [g[col].to_numpy() for col in ("open", "high", "low", "close", "volume")]
            if not ok.all():                # replicate reindex semantics: drop non-grid rows
                pos  = pos[ok]
                vals = [a[ok] for a in vals]
            if not len(pos):
                continue

        arrays = _fill_arrays(*vals, pos)
        _write_day(arrays, pa_index, out_file, front_symbol, rth_date,
                   roll_map.get(key, False))

        written += 1
        parts.append(f"{asset}:{front_symbol}{' ROLL' if roll_map.get(key) else ''}")

    if parts:
        shown = "  ".join(parts[:8]) + (" …" if len(parts) > 8 else "")
        log(i, f"  {shown}")
    return written


# ---------------------------------------------------------------------------
# Coarse skip
# ---------------------------------------------------------------------------

def _reference_complete(base: Path, typed_name: str,
                        d_start: pd.Timestamp, d_end: pd.Timestamp,
                        is_last: bool) -> bool:
    """
    Best-effort month-level skip: True only if the reference asset (ES) shows this month
    was already processed AND its boundary day (1st of next month) was already produced —
    so nothing this file would emit is missing. Holidays make this a safe false-negative
    (it simply re-processes rather than wrongly skipping).
    """
    ref_dir = base / REF_ASSET / _folder_for(typed_name, REF_ASSET)
    if not ref_dir.exists():
        return False

    # (a) month processed — some ES day exists in the last 5 calendar days of the range
    processed = any(
        (ref_dir / f"{(d_end.date() - datetime.timedelta(days=k)).isoformat()}.parquet").exists()
        for k in range(5)
    )
    if not processed:
        return False

    # (b) boundary day (1st of next month) produced, or N/A
    if is_last:
        return True
    bday = d_end.date() + datetime.timedelta(days=1)
    if bday.weekday() >= 5:          # Sat/Sun — no trading, nothing to produce
        return True
    return (ref_dir / f"{bday.isoformat()}.parquet").exists()


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
    _GRID.clear()
    _DIRS.clear()
    _TEMPLATE.clear()
    t_run0 = perf_counter()

    input_path  = Path(input_folder)
    output_path = Path(output_folder)
    typed_name  = output_path.name              # e.g. "X_1m_ohlcv_globex_15y"
    base        = output_path.parent.parent     # e.g. data/parquet/Futures

    with _t("run.glob"):
        files = sorted(input_path.glob("*.dbn.zst"))
    if len(files) < 2:
        if on_progress:
            on_progress(1, 1, "ERROR: Need at least 2 monthly files.")
        return

    n = len(files)

    def log(cur: int, msg: str):
        if on_progress:
            on_progress(min(cur + 1, n), n, msg)

    cur_prep = None      # prepared frame for files[cur_idx]
    cur_idx  = -1

    for i in range(n):
        is_first = (i == 0)
        is_last  = (i == n - 1)

        try:
            d_start, d_end = _parse_date_range(files[i])
        except ValueError as e:
            log(i, f"ERROR parsing {files[i].name}: {e}")
            cur_prep, cur_idx = None, -1
            gc.collect()
            continue

        # ── coarse skip ───────────────────────────────────────────────────────
        with _t("run.coarse_skip"):
            skip_month = skip_existing and _reference_complete(base, typed_name, d_start, d_end, is_last)
        if skip_month:
            log(i, f"↷ {d_start.strftime('%Y-%m')} — already complete (ES), skipped")
            if cur_idx == i:
                cur_prep, cur_idx = None, -1
            with _t("run.gc"):
                gc.collect()
            continue

        # ── ensure cur_prep holds files[i] (may be carried from prev boundary) ──
        if cur_idx != i:
            log(i, f"Loading {files[i].name} …")
            try:
                cur_prep, cur_idx = _load_prepared(files[i]), i
            except Exception as e:
                log(i, f"ERROR loading {files[i].name}: {e}")
                cur_prep, cur_idx = None, -1
                gc.collect()
                continue

        written = 0

        # ── owned days: trade_date in (lo, d_end] from this file's own data ──────
        # first file additionally skips its very first session (no prior file for ETH)
        lo = d_start + pd.Timedelta(days=1) if is_first else d_start
        try:
            with _t("run.owned_filter"):
                td = cur_prep["trade_date"]
                owned = cur_prep[(td > lo) & (td <= d_end)]
            written += _write_prepared(owned, base, typed_name, skip_existing, i, log)
            del owned
        except Exception as e:
            log(i, f"ERROR processing {files[i].name}: {e}")

        # ── boundary day (d_end + 1): this file's tail + next file's head ────────
        if not is_last:
            try:
                next_prep = _load_prepared(files[i + 1])
            except Exception as e:
                log(i, f"ERROR loading next {files[i + 1].name}: {e}")
                next_prep = None

            if next_prep is not None:
                try:
                    with _t("run.boundary_select"):
                        boundary = d_end + pd.Timedelta(days=1)
                        parts_b  = [
                            cur_prep[cur_prep["trade_date"] == boundary],
                            next_prep[next_prep["trade_date"] == boundary],
                        ]
                        prep_b = pd.concat(parts_b)
                    written += _write_prepared(prep_b, base, typed_name, skip_existing, i, log)
                    del prep_b, parts_b
                except Exception as e:
                    log(i, f"ERROR boundary {files[i].name}: {e}")

                # carry next file's prepared frame forward — each file decoded exactly once
                del cur_prep
                cur_prep, cur_idx = next_prep, i + 1
            else:
                del cur_prep
                cur_prep, cur_idx = None, -1
        else:
            del cur_prep
            cur_prep, cur_idx = None, -1

        _GRID.clear()      # grids are only shared within one month — cap memory

        suffix = " (last file — last day ETH may be incomplete)" if is_last else ""
        if on_progress:
            on_progress(min(i + 1, n), n,
                        f"✓ {d_start.strftime('%Y-%m')} — {written} day-files written{suffix}")
        with _t("run.gc"):
            gc.collect()

    wall = perf_counter() - t_run0
    if on_progress:
        on_progress(n, n, _timing_report(wall))
