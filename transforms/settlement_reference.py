"""
settlement_reference.py — daily settlement & session reference table for one asset.

Input  : folder of Databento STATISTICS .dbn.zst files
         (e.g. data/raw_dbn/Futures/ES/ES_2010-06-06_2026-05-18_STATISTICS_monthly).
Derived: the asset's enriched OHLCV dataset — same asset_type/asset with
         file_type raw_dbn -> parquet, folder name containing "ohlcv"
         (expected {ASSET}_1m_ohlcv_globex). One parquet per trading day,
         full 1380-bar 18:00->17:00 NY grid, front-month symbol + roll flag
         in the parquet key-value metadata.
Output : ONE parquet, output_folder/settlement_reference.parquet — one row per
         trading session over the full history:

         date, symbol, settlement, settle_is_actual, is_roll_day,
         globex_open, globex_high, globex_low, globex_close,
         rth_open, rth_high, rth_low, rth_close,
         session_volume, rth_volume

         globex_high/low ARE the whole-session extremes (no separate columns).
         Raw levels only — no derived analytics.

Settlements come from the statistics schema, stat_type = 3. stat_flags is the
CME tag 731-SettlPriceType bitfield:  1 = final (vs preliminary), 2 = actual
(vs theoretical), 4 = settling at trading tick, 8 = intraday settlement.
Kept: final AND not-intraday AND update_action == 1 AND defined price AND
non-spread symbol (no "-"). The "actual" bit is NOT required — theoretical
finals are kept as fallback and flagged via settle_is_actual.

Session date key: ts_ref (the settlement's reference date, midnight-UTC stamps
of the trade date). ts_recv keying was tried first and FAILED validation V1 on
ES 2010-2026 (83.9% match): post-2016 Friday finals are disseminated after
midnight NY / on Sunday, landing on the wrong NY calendar date. ts_ref keying
matches 96.7% — remaining misses are exchange holidays and OHLCV days past the
statistics dataset's end. Dedup: one settlement per (symbol, session_date),
last record by ts_recv wins.

Unmatched sessions are RETAINED with NaN settlement and counted — never
dropped. Rows whose front-month metadata symbol contains "-" are skipped and
counted (upstream front-month bug).

The statistics pipeline is fully vectorized; the only loop is the unavoidable
per-day OHLCV file read (O(1) column reductions inside).
"""

from pathlib import Path

import databento as db
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NY = "America/New_York"

OUTPUT_FILENAME = "statistics.parquet"

# stat_flags bits (CME tag 731-SettlPriceType, per Databento GLBX docs)
FLAG_FINAL    = 1 << 0   # final (vs preliminary)
FLAG_ACTUAL   = 1 << 1   # actual (vs theoretical/computed)
FLAG_INTRADAY = 1 << 3   # intraday settlement, disseminated before official EOD

STAT_TYPE_SETTLEMENT = 3
UPDATE_ACTION_ADDED  = 1

# ---------------------------------------------------------------------------
# Session times — structured for the per-asset-session roadmap item.
# Bars are minute-start-labeled. RTH window = bar labels in [rth_start, rth_end)
# -> rth_open comes from the rth_start bar, rth_close from the last bar before
# rth_end (the 15:59 bar for a 16:00 end). Flip RTH_END_INCLUSIVE to make the
# upper bound inclusive (window becomes [rth_start, rth_end]) without a rewrite.
# ---------------------------------------------------------------------------
SESSION_TIMES: dict[str, dict[str, str]] = {
    "ES": {"rth_start": "09:30", "rth_end": "16:00"},
    # future assets: CL, ZN, 6E, GC ...
}
DEFAULT_RTH = {"rth_start": "09:30", "rth_end": "16:00"}

RTH_END_INCLUSIVE = False


# ---------------------------------------------------------------------------
# Path derivation
# ---------------------------------------------------------------------------

def _parse_path(input_folder: str) -> tuple[str, str, str, str]:
    """(file_type, asset_type, asset, name) from .../{file_type}/{asset_type}/{asset}/{name}."""
    parts = Path(input_folder).parts
    if len(parts) < 4:
        raise ValueError(f"Input path too shallow to parse: {input_folder}")
    file_type, asset_type, asset, name = parts[-4], parts[-3], parts[-2], parts[-1]
    if file_type != "raw_dbn":
        raise ValueError(
            f"Expected a raw_dbn statistics folder, got file_type={file_type!r} "
            f"({input_folder})"
        )
    return file_type, asset_type, asset, name


def _resolve_ohlcv_folder(input_folder: str, asset_type: str, asset: str) -> Path:
    """The asset's OHLCV parquet dataset: swap raw_dbn -> parquet, glob for *ohlcv*."""
    parts = list(Path(input_folder).parts)
    parts[-4] = "parquet"
    asset_dir = Path(*parts[:-1])
    if not asset_dir.exists():
        raise FileNotFoundError(f"Parquet asset folder not found: {asset_dir}")
    matches = sorted(
        d for d in asset_dir.iterdir() if d.is_dir() and "ohlcv" in d.name.lower()
    )
    if len(matches) == 0:
        raise FileNotFoundError(f"No *ohlcv* dataset folder under {asset_dir}")
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous OHLCV dataset — {len(matches)} matches under {asset_dir}: "
            + ", ".join(d.name for d in matches)
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Statistics -> settlement lookup (fully vectorized)
# ---------------------------------------------------------------------------

def _load_settlements(input_folder: str, log) -> pd.DataFrame:
    """
    Load ALL statistics files, filter to final EOD settlements, and dedup to one
    row per (symbol, session_date). Returns columns:
    date (tz-naive midnight), symbol, settlement, settle_is_actual.
    """
    files = sorted(Path(input_folder).glob("*.dbn.zst"))
    if not files:
        raise FileNotFoundError(f"No .dbn.zst files in {input_folder}")

    frames = []
    for k, f in enumerate(files):
        # to_df: prices already float-scaled (UNDEF_PRICE -> NaN), symbols mapped
        df = db.DBNStore.from_file(f).to_df()
        df = df[df["stat_type"] == STAT_TYPE_SETTLEMENT]
        if len(df):
            frames.append(
                df[["ts_ref", "price", "stat_flags", "update_action", "symbol"]].reset_index()
            )
        log(k + 1, f"Loaded {f.name} — {len(df)} settlement records")

    if not frames:
        raise ValueError(f"No stat_type={STAT_TYPE_SETTLEMENT} rows in {input_folder}")

    stats = pd.concat(frames, ignore_index=True)

    # V4 observability: flag distribution before filtering
    flag_dist = stats["stat_flags"].value_counts().sort_index()
    log(len(files), "stat_flags distribution (stat_type=3): "
        + ", ".join(f"{k}:{v}" for k, v in flag_dist.items()))

    flags = stats["stat_flags"].to_numpy()
    keep = (
        (stats["update_action"].to_numpy() == UPDATE_ACTION_ADDED)
        & (flags & FLAG_FINAL != 0)
        & (flags & FLAG_INTRADAY == 0)
        & stats["price"].notna().to_numpy()          # UNDEF_PRICE decoded as NaN
        & ~stats["symbol"].str.contains("-", na=True).to_numpy()  # drop spreads
    )
    stats = stats[keep]
    if stats.empty:
        raise ValueError("No final EOD settlements survived filtering")

    # session date: ts_ref carries the settlement's trade date as a midnight-UTC
    # stamp — take its UTC calendar date (tz-naive midnight for the join).
    # NOT ts_recv: dissemination time trails the session (Friday finals arrive
    # Sat/Sun post-2016), failing date alignment — see module docstring.
    ts_ref = pd.DatetimeIndex(stats["ts_ref"])
    stats = stats.assign(
        date=ts_ref.tz_localize(None).normalize(),
        settle_is_actual=(stats["stat_flags"] & FLAG_ACTUAL != 0),
    )

    # dedup: last record by ts_recv per (symbol, date) — safety net vs revisions
    stats = (
        stats.sort_values("ts_recv", kind="stable")
        .drop_duplicates(subset=["symbol", "date"], keep="last")
    )

    out = stats[["date", "symbol", "price", "settle_is_actual"]].rename(
        columns={"price": "settlement"}
    )
    out["settle_is_actual"] = out["settle_is_actual"].astype("boolean")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-day OHLCV extraction (the thin loop)
# ---------------------------------------------------------------------------

def _rth_window_minutes(asset: str, log) -> tuple[int, int]:
    """RTH window as minutes-of-day (NY wall clock) from the session config."""
    times = SESSION_TIMES.get(asset)
    if times is None:
        times = DEFAULT_RTH
        log(None, f"WARNING: no SESSION_TIMES entry for {asset!r} — "
                  f"using default RTH {DEFAULT_RTH['rth_start']}-{DEFAULT_RTH['rth_end']}")
    h1, m1 = map(int, times["rth_start"].split(":"))
    h2, m2 = map(int, times["rth_end"].split(":"))
    return h1 * 60 + m1, h2 * 60 + m2


def _extract_ohlcv_day(path: Path, rth_start: int, rth_end: int) -> dict | None:
    """
    One OHLCV day-file -> one record dict, or None if the front-month symbol is
    missing/spread (caller counts and warns). O(1) column reductions only.
    """
    table = pq.read_table(path)
    meta = table.schema.metadata or {}

    sym_raw = meta.get(b"front_month") or meta.get(b"symbol")
    symbol = sym_raw.decode() if sym_raw else None
    if not symbol or "-" in symbol:
        return None

    roll_raw = meta.get(b"is_roll_day")
    is_roll = (roll_raw.decode() == "True") if roll_raw is not None else pd.NA

    date_raw = meta.get(b"trade_date")
    date = pd.Timestamp(date_raw.decode() if date_raw else path.stem)

    day = table.to_pandas()
    o = day["open"].to_numpy()
    h = day["high"].to_numpy()
    l = day["low"].to_numpy()
    c = day["close"].to_numpy()
    v = day["volume"].to_numpy()

    idx = day.index  # tz-aware NY DatetimeIndex, minute-start labels
    minutes = idx.hour * 60 + idx.minute
    if RTH_END_INCLUSIVE:
        rth = (minutes >= rth_start) & (minutes <= rth_end)
    else:
        rth = (minutes >= rth_start) & (minutes < rth_end)

    rec = {
        "date":           date,
        "symbol":         symbol,
        "is_roll_day":    is_roll,
        "globex_open":    o[0],
        "globex_high":    h.max(),
        "globex_low":     l.min(),
        "globex_close":   c[-1],
        "session_volume": int(v.sum()),
    }
    if rth.any():
        pos = np.flatnonzero(rth)
        rec.update(
            rth_open=o[pos[0]], rth_high=h[rth].max(), rth_low=l[rth].min(),
            rth_close=c[pos[-1]], rth_volume=int(v[rth].sum()),
        )
    else:  # half-day / outage: no bars inside the window
        rec.update(rth_open=np.nan, rth_high=np.nan, rth_low=np.nan,
                   rth_close=np.nan, rth_volume=0)
    return rec


# ---------------------------------------------------------------------------
# Assemble & join
# ---------------------------------------------------------------------------

_COLUMNS = [
    "date", "symbol", "settlement", "settle_is_actual", "is_roll_day",
    "globex_open", "globex_high", "globex_low", "globex_close",
    "rth_open", "rth_high", "rth_low", "rth_close",
    "session_volume", "rth_volume",
]


def _build_table(records: list[dict], settlements: pd.DataFrame) -> pd.DataFrame:
    """Concat day records, LEFT-join settlements on (date, symbol). Never drops rows."""
    days = pd.DataFrame(records)
    out = days.merge(settlements, on=["date", "symbol"], how="left", validate="1:1")
    out["settle_is_actual"] = out["settle_is_actual"].astype("boolean")
    out["is_roll_day"] = out["is_roll_day"].astype("boolean")
    return out[_COLUMNS].sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_all(
    input_folder: str,
    output_folder: str,
    skip_existing: bool = True,
    on_progress: callable = None,
) -> None:
    _, asset_type, asset, _ = _parse_path(input_folder)
    ohlcv_dir = _resolve_ohlcv_folder(input_folder, asset_type, asset)

    out_file = Path(output_folder) / OUTPUT_FILENAME
    if skip_existing and out_file.exists():
        if on_progress:
            on_progress(1, 1, f"↷ {OUTPUT_FILENAME} already exists — skipped")
        return

    stats_files = sorted(Path(input_folder).glob("*.dbn.zst"))
    day_files   = sorted(ohlcv_dir.glob("*.parquet"))
    if not day_files:
        raise FileNotFoundError(f"No day parquets in {ohlcv_dir}")

    # progress: stats files, then day files, then join+write
    total = max(len(stats_files) + len(day_files) + 2, 1)

    def log(cur: int | None, msg: str):
        if on_progress:
            on_progress(min(cur, total) if cur is not None else 1, total, msg)

    # ── phase 1: settlements (vectorized) ────────────────────────────────────
    settlements = _load_settlements(input_folder, log)
    log(len(stats_files),
        f"Settlement lookup: {len(settlements)} (symbol, date) finals")

    # ── phase 2: per-day OHLCV extraction (thin loop) ────────────────────────
    rth_start, rth_end = _rth_window_minutes(asset, log)

    records: list[dict] = []
    skipped_symbols = 0
    base = len(stats_files)
    for k, f in enumerate(day_files):
        rec = _extract_ohlcv_day(f, rth_start, rth_end)
        if rec is None:
            skipped_symbols += 1
            log(base + k + 1, f"WARNING: {f.name} — front-month symbol missing "
                              f"or spread, row skipped")
            continue
        records.append(rec)
        if (k + 1) % 250 == 0 or k + 1 == len(day_files):
            log(base + k + 1, f"OHLCV days: {k + 1}/{len(day_files)}")

    if not records:
        raise ValueError(f"No usable OHLCV day files in {ohlcv_dir}")

    # ── phase 3: join + write ────────────────────────────────────────────────
    table = _build_table(records, settlements)
    misses = int(table["settlement"].isna().sum())
    log(total - 1, f"Joined {len(table)} sessions — "
                   f"{misses} without settlement, {skipped_symbols} skipped (bad symbol)")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_file, index=False)
    log(total, f"✓ Wrote {out_file.name}: {len(table)} rows "
               f"({table['date'].iloc[0].date()} → {table['date'].iloc[-1].date()}), "
               f"{misses} settlement misses, {skipped_symbols} skipped symbols")
