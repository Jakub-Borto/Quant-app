"""
Joining regime label files to trades — the ONE implementation, as mandated by
modules/regime_detector/__init__.py ("build that helper ONCE in
modules/common/backend/ and share it — do not hand-roll the merge per
consumer"). Pure and Qt-free: safe in worker threads, pool workers and tests.

Why this is not a one-liner:

LOOKAHEAD. A regime file's FINAL row knows how the day ended. Joining on date
alone hands that row to every trade and turns a backtest into fiction. The
default mode is MODE_ASOF — pd.merge_asof(direction="backward") on the trade's
entry timestamp, so each trade sees the latest snapshot at or before it.
allow_exact_matches=True is correct: a snapshot stamped T is built only from
bars strictly before T (runner.slice(before=ts)), so an entry exactly at T
sees no future.

CROSS-SESSION CARRY. A regime file for RTH date D spans D-1 18:30 → D 17:00,
so the files are disjoint but adjacent. A naive backward join stamps day D's
17:00 row onto an 18:05 entry the FOLLOWING evening — a stale label from the
previous session. The guard is an equality test, not arithmetic: the matched
snapshot's own RTH date must equal the trade's date, else the label is
UNKNOWN. (A `tolerance` also happens to work on a 30-minute grid, but it
silently stops working once snapshot_minutes exceeds the 17:00→18:30 break,
so it is only a prefilter here.)

DTYPE DRIFT. Verified across real trades files: entry_time is always tz-aware
America/New_York but its RESOLUTION varies by strategy (datetime64[ns] from
ivb_model, datetime64[us] from orb), and the regime parquet index is [us].
pandas raises MergeError on mismatched merge-key dtypes, so both sides are
normalized to ns here. The `date` column drifts too — datetime.date objects in
one file, plain strings in another.

Trades whose entry has no usable snapshot get UNKNOWN, matching the regime
module's convention: an explicit "no answer" that consumers must EXCLUDE
rather than bucket as a real regime.
"""

import numpy as np
import pandas as pd

NY = "America/New_York"

# mirrors regime_detector.backend.schema.UNKNOWN_STATE (kept as a literal so
# this module stays importable without the regime_detector package)
UNKNOWN = "unknown"

MODE_ASOF = "asof"      # latest snapshot at or before the trade's entry
MODE_EXACT = "exact"    # that day's snapshot at a chosen wall-clock time
MODE_FINAL = "final"    # the day's last row — HINDSIGHT, opt-in only

MODES = (MODE_ASOF, MODE_EXACT, MODE_FINAL)
MODE_LABELS = {
    MODE_ASOF: "As of entry",
    MODE_EXACT: "Exact time",
    MODE_FINAL: "Final (hindsight)",
}


# ── normalization (the dtype-drift guards) ───────────────────────────────────

def normalize_entry_times(trades: pd.DataFrame,
                          column: str = "entry_time") -> pd.Series:
    """Trade entry timestamps as tz-aware NY at NANOSECOND resolution.

    Both halves matter: a tz-naive column is localized to NY (trades are
    always NY wall clock in this repo), and the unit is forced to ns so the
    merge key matches the regime index whatever the strategy wrote."""
    values = pd.to_datetime(trades[column])
    if values.dt.tz is None:
        values = values.dt.tz_localize(NY)
    else:
        values = values.dt.tz_convert(NY)
    return values.dt.as_unit("ns")


def normalize_trade_dates(trades: pd.DataFrame,
                          column: str = "date") -> pd.Series:
    """Trade RTH dates as tz-naive midnight Timestamps. Absorbs the observed
    datetime.date / str / Timestamp drift across trades files."""
    return pd.to_datetime(trades[column]).dt.tz_localize(None).dt.normalize()


# ── snapshot assembly ────────────────────────────────────────────────────────

def regime_snapshots(frames: dict, columns: list[str]) -> pd.DataFrame:
    """One time-sorted frame from {date: daily regime frame}.

    Adds `regime_date` — the file's own RTH date — which is what the
    cross-session guard tests against. Index is the snapshot timestamp at ns
    resolution.
    """
    keep = list(columns)
    parts = []
    for date, df in sorted(frames.items()):
        if df is None or df.empty:
            continue
        present = [c for c in keep if c in df.columns]
        if not present:
            continue
        part = df[present].copy()
        part["regime_date"] = pd.Timestamp(date)
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=keep + ["regime_date"],
                            index=pd.DatetimeIndex([], tz=NY, name="ts"))

    out = pd.concat(parts)
    idx = pd.DatetimeIndex(out.index)
    idx = idx.tz_localize(NY) if idx.tz is None else idx.tz_convert(NY)
    out.index = idx.as_unit("ns")
    return out.sort_index()


# ── the join ─────────────────────────────────────────────────────────────────

def attach_regime(trades: pd.DataFrame, frames: dict, column: str,
                  mode: str = MODE_ASOF, *, at: str | None = None,
                  session_start: str = "18:00", snapshot_minutes: int = 30,
                  out_col: str = "regime") -> pd.DataFrame:
    """
    A COPY of `trades` with `out_col` added — one label per trade, UNKNOWN
    where none applies. Never mutates the input and never reorders it.

    mode:
      MODE_ASOF  — latest snapshot at or before entry_time (no lookahead)
      MODE_EXACT — that day's snapshot at `at` ("HH:MM"), the "what was
                   knowable at 10:00" view
      MODE_FINAL — the day's last row: HINDSIGHT, for measuring how much of a
                   signal is only visible after the fact
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
    out = trades.copy()
    if out.empty:
        out[out_col] = pd.Series(dtype=object)
        return out
    if mode == MODE_EXACT and not at:
        raise ValueError("mode 'exact' needs a wall-clock time via at='HH:MM'")

    labels = (_asof_labels(out, frames, column, snapshot_minutes)
              if mode == MODE_ASOF
              else _daily_labels(out, frames, column, mode, at, session_start))
    out[out_col] = labels
    return out


def _asof_labels(trades: pd.DataFrame, frames: dict, column: str,
                 snapshot_minutes: int) -> pd.Series:
    snaps = regime_snapshots(frames, [column])
    labels = pd.Series(UNKNOWN, index=trades.index, dtype=object)
    if snaps.empty or "entry_time" not in trades.columns:
        return labels

    key = normalize_entry_times(trades).to_numpy()
    # merge_asof demands a sorted left side; filtered trade frames need not be.
    # Sort positionally, then scatter the results back so the caller's row
    # order and index survive untouched.
    order = np.argsort(key, kind="stable")
    left = pd.DataFrame({"_key": key[order]})
    merged = pd.merge_asof(
        left, snaps, left_on="_key", right_index=True, direction="backward",
        tolerance=pd.Timedelta(minutes=int(snapshot_minutes)),
        allow_exact_matches=True)

    matched = merged[column].to_numpy(dtype=object)
    # the cross-session guard: the snapshot's own RTH date must be the trade's
    same_session = np.zeros(len(matched), dtype=bool)
    if "date" in trades.columns:
        trade_dates = normalize_trade_dates(trades).to_numpy()[order]
        same_session = merged["regime_date"].to_numpy() == trade_dates
    values = np.where(pd.notna(matched) & same_session, matched, UNKNOWN)

    result = np.empty(len(trades), dtype=object)
    result[order] = values
    return pd.Series(result, index=trades.index, dtype=object)


def _daily_labels(trades: pd.DataFrame, frames: dict, column: str, mode: str,
                  at: str | None, session_start: str) -> pd.Series:
    """One row per DAY (exact / final), broadcast onto that day's trades.

    io.pick_rows drops days with no snapshot at or before the target, while
    every trade must still get a value — hence the left-join + UNKNOWN fill.
    """
    from modules.regime_detector.backend import io as rio

    labels = pd.Series(UNKNOWN, index=trades.index, dtype=object)
    if not frames or "date" not in trades.columns:
        return labels

    picked = rio.pick_rows(frames, rio.FINAL if mode == MODE_FINAL else at,
                           session_start)
    if picked.empty or column not in picked.columns:
        return labels

    per_day = picked[column]
    per_day.index = pd.DatetimeIndex(per_day.index).tz_localize(None).normalize()
    mapped = normalize_trade_dates(trades).map(per_day)
    return mapped.where(mapped.notna(), UNKNOWN).astype(object)
