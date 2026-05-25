"""
candles_1m_simple.py — 1-minute OHLCV candles from monthly Databento ohlcv-1m DBN files.

Input  : folder of monthly .dbn.zst files, each covering all assets for one month.
         Expected filename format: glbx-mdp3-YYYYMMDD-YYYYMMDD.ohlcv-1m.dbn.zst
Output : one Parquet per asset per trading day.
         {output_folder}/{asset_name}/{dataset_name}/YYYY-MM-DD.parquet

Columns: open, high, low, close, volume  (float64 OHLC, int64 volume)

Session : 18:00 NY (prev evening) → 17:00 NY (current day) = full Globex session.
          Defined in NY time, converted to UTC — DST handled automatically.
Date label: trade_date = date of the RTH session (America/New_York).
Gap filling: full 1380-bar 1-minute grid (18:00→17:00 NY), OHLC forward-filled, volume=0.
Weekends / holidays: no RTH bars → silently skipped.
First session of first file: skipped (no previous file to complete the ETH portion).
Last file: processed as primary using only its own data — last day ETH may be incomplete.
"""

import re
import databento as db
import pandas as pd
from pathlib import Path
import gc

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


# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------

def _parse_date_range(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    match = re.search(r"(\d{8})-(\d{8})", path.name)
    if not match:
        raise ValueError(f"Cannot parse date range from filename: {path.name}")
    return pd.Timestamp(match.group(1)), pd.Timestamp(match.group(2))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_file(path: Path) -> pd.DataFrame:
    df = db.DBNStore.from_file(path).to_df()
    df = df[["open", "high", "low", "close", "volume", "symbol"]]
    df = df[~df["symbol"].str.contains("-", na=False)]
    df = df[~df["symbol"].str.contains(" ", na=False)]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_front_month(df: pd.DataFrame) -> tuple[str | None, bool]:
    vol = df.groupby("symbol")["volume"].sum()
    if vol.empty:
        return None, False
    sorted_vol = vol.sort_values(ascending=False)
    front   = sorted_vol.index[0]
    is_roll = len(sorted_vol) > 1 and sorted_vol.iloc[1] > 0.2 * sorted_vol.iloc[0]
    return front, is_roll


def _fill_gaps(candles: pd.DataFrame, full_index: pd.DatetimeIndex) -> pd.DataFrame:
    candles = candles.reindex(full_index)
    candles["close"]  = candles["close"].ffill().bfill()
    candles["open"]   = candles["open"].fillna(candles["close"])
    candles["high"]   = candles["high"].fillna(candles["close"])
    candles["low"]    = candles["low"].fillna(candles["close"])
    candles["volume"] = candles["volume"].fillna(0).astype("int64")
    return candles


# ---------------------------------------------------------------------------
# Session processor
# ---------------------------------------------------------------------------

def _process_session(
    full_df: pd.DataFrame,
    session_start_ny: pd.Timestamp,
    output_path: Path,
    dataset_name: str,
    skip_existing: bool,
    log: callable,
) -> bool:
    """
    Returns True if a trading day was processed, False if skipped (weekend/holiday).
    Logs day-level info (front month, saved/skipped) via log callback.
    Progress bar updated only at month level in run_all.
    """
    session_end_ny    = session_start_ny + pd.Timedelta(hours=23)
    session_start_utc = session_start_ny.tz_convert("UTC")
    session_end_utc   = session_end_ny.tz_convert("UTC")

    session_utc = full_df[
        (full_df.index >= session_start_utc) &
        (full_df.index <  session_end_utc)
    ]

    if session_utc.empty:
        return False

    session_ny = session_utc.tz_convert("America/New_York")

    rth_mask = (
        (session_ny.index.time >= pd.Timestamp("09:30").time()) &
        (session_ny.index.time <= pd.Timestamp("16:00").time())
    )
    rth_bars = session_ny[rth_mask]

    if rth_bars.empty:
        return False

    trade_date = rth_bars.index[0].date().isoformat()

    # pre-compute full 1380-bar index once — shared across all assets
    full_index = pd.date_range(
        start   = session_start_ny,
        periods = 1380,
        freq    = "1min",
        tz      = "America/New_York",
    )

    # pre-compute per-asset masks once — avoids 30x str.startswith on full session
    symbols     = session_ny["symbol"]
    asset_masks = {prefix: symbols.str.startswith(prefix) for prefix in ASSETS}

    asset_log_parts = []

    for prefix, folder_name in ASSETS.items():
        asset_df = session_ny[asset_masks[prefix]]

        if asset_df.empty:
            continue

        front_month, is_roll = _get_front_month(asset_df)
        if front_month is None:
            continue

        roll_tag = " ROLL" if is_roll else ""

        candles = (
            asset_df[asset_df["symbol"] == front_month]
            [["open", "high", "low", "close", "volume"]]
        )

        candles = _fill_gaps(candles, full_index)

        out_dir  = output_path / folder_name / dataset_name
        out_file = out_dir / f"{trade_date}.parquet"

        if skip_existing and out_file.exists():
            asset_log_parts.append(f"{prefix}:↷")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        candles.to_parquet(out_file)
        asset_log_parts.append(f"{prefix}:{front_month}{roll_tag}")

    log(f"  {trade_date}  {'  '.join(asset_log_parts)}")
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
    input_path   = Path(input_folder)
    output_path  = Path(output_folder)
    dataset_name = input_path.name

    files = sorted(input_path.glob("*.dbn.zst"))

    if len(files) < 2:
        if on_progress:
            on_progress(1, 1, "ERROR: Need at least 2 monthly files.")
        return

    total_files = len(files)
    total_pairs = total_files - 1

    def log(msg: str):
        if on_progress:
            on_progress(min(i + 1, total_pairs), total_pairs, msg)

    for i in range(total_files):
        primary_file = files[i]
        is_last_file = (i == total_files - 1)

        try:
            date_start, date_end = _parse_date_range(primary_file)
        except ValueError as e:
            log(f"ERROR parsing filename: {e}")
            continue

        log(f"Loading {primary_file.name} …")

        try:
            primary_df = _load_file(primary_file)
        except Exception as e:
            log(f"ERROR loading {primary_file.name}: {e}")
            continue

        session_dates  = pd.date_range(start=date_start, end=date_end, freq="D")
        session_starts = [
            pd.Timestamp(f"{d.date()} 18:00:00", tz="America/New_York")
            for d in session_dates
        ]

        if i == 0:
            session_starts = session_starts[1:]

        if not session_starts:
            del primary_df
            gc.collect()
            continue

        if is_last_file:
            regular_sessions = session_starts
            last_session     = None
        else:
            regular_sessions = session_starts[:-1]
            last_session     = session_starts[-1]

        days_processed = 0
        days_skipped   = 0

        # ── regular sessions ──────────────────────────────────────────────────
        for session_start in regular_sessions:
            try:
                processed = _process_session(
                    full_df          = primary_df,
                    session_start_ny = session_start,
                    output_path      = output_path,
                    dataset_name     = dataset_name,
                    skip_existing    = skip_existing,
                    log              = log,
                )
                if processed:
                    days_processed += 1
                else:
                    days_skipped += 1
            except Exception as e:
                log(f"ERROR session {session_start.date()}: {e}")
                continue

        # ── last session — splice with next file ──────────────────────────────
        if last_session is not None:
            next_file = files[i + 1]
            try:
                next_df = _load_file(next_file)

                last_session_end_utc = (
                    last_session + pd.Timedelta(hours=23)
                ).tz_convert("UTC")

                next_df_trimmed = next_df[next_df.index < last_session_end_utc]
                combined        = pd.concat([primary_df, next_df_trimmed]).sort_index()

                del next_df, next_df_trimmed
                gc.collect()

                processed = _process_session(
                    full_df          = combined,
                    session_start_ny = last_session,
                    output_path      = output_path,
                    dataset_name     = dataset_name,
                    skip_existing    = skip_existing,
                    log              = log,
                )
                if processed:
                    days_processed += 1
                else:
                    days_skipped += 1

                del combined
                gc.collect()

            except Exception as e:
                log(f"ERROR last session {last_session.date()}: {e}")

        # ── monthly summary — only this call updates the progress bar ─────────
        suffix = " (last file — last day ETH may be incomplete)" if is_last_file else ""
        summary = f"✓ {date_start.strftime('%Y-%m')} — {days_processed} days processed, {days_skipped} skipped{suffix}"
        if on_progress:
            on_progress(min(i + 1, total_pairs), total_pairs, summary)

        del primary_df
        gc.collect()