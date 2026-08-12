"""
The canonical trades frame, the primitives every filter stage is built from,
and the small value types a caller hands the report.

Pure pandas, no Qt.

═══════════════════════════════════════════════════════════════════════════
 THE CANONICAL SHAPE
═══════════════════════════════════════════════════════════════════════════

Required    entry_time, ticks, day_type, date
Derived     cumulative_ticks, regime, regime_filter   — by the report, never
            by the caller
Optional    trade_type, notes, exit_reason, pnl_points, direction,
            entry_price, exit_price, sl, tp           — never synthesised;
            sections that need them hide themselves when they are absent
Index       a unique RangeIndex, produced ONCE, at the entrance

The two producers disagree, which is why this file exists:

  * the Backtester emits `ticks` / `cumulative_ticks` / `day_type` already,
    with `date` as a STRING (strategies emit date_str);
  * an optimizer run emits `pnl_ticks` / `day_bucket` (including the
    `other_high_impact` bucket) with `date` normalized to datetime64[ns].

So the optimizer's callers wrap their frame in `from_optimizer_rows()` at the
call site. That is deliberate rather than sniffing `"pnl_ticks" in columns` —
a future frame carrying both would be silently mis-handled.

Three details are load-bearing:

1. `date` IS NEVER TOUCHED. Every reader already coerces defensively
   (trade_stats.compute_metrics does pd.to_datetime itself), and coercing it
   here would change the schema of every Backtester trades file — which
   save_trades dedups by comparing `pd.read_parquet(f).equals(trades)`, so
   every pre-existing file would stop matching.
2. The sort is `kind="stable"`. The default quicksort reorders trades that
   share an entry_time, which changes cumulative_ticks and the equity path.
3. `reset_index(drop=True)` happens exactly ONCE, here — before the regime
   join and before the notes section sees the frame. Every later stage
   preserves index labels (attach_regime scatters back through argsort,
   `narrow` is boolean masking), which is what TradeNotesSection.mask_for's
   reindex contract depends on.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("entry_time", "ticks", "day_type", "date")


class TradeFrameError(ValueError):
    """A frame handed to the report is missing something it cannot work
    without. Raised in place of a bare KeyError so the message names the
    column and the caller that owes it."""


class _Keep:
    """Sentinel: 'leave this filter exactly as the user left it'."""

    def __repr__(self) -> str:      # a stray print should be legible
        return "KEEP"


KEEP = _Keep()


@dataclass(frozen=True)
class ReportContext:
    """Asset + data-root context for one selection. Destructive for the
    regime section (it rescans and drops the chosen run), so hand it over
    once per selection — never per filter toggle."""
    ticker: str
    tick_size: float
    ticks_per_point: float
    root: Path                       # data root: parquet/ for the benchmark
                                     # regression, trades/ + temp/ for saves
    candles_folder: Path             # the dataset's per-day parquet folder,
                                     # for the single-trade candlestick chart
    regime_start: str | None = None  # "YYYY-MM-DD" — narrows the regime runs
    regime_end: str | None = None


@dataclass(frozen=True)
class SaveTarget:
    """What Save Trades / Go to Analytics / Go to Monte Carlo write. Passed
    per selection, because that is the only time it changes."""
    name_parts: tuple = ()           # sanitised + "_"-joined; part 0 becomes
                                     # the filename's first underscore token,
                                     # which asset lookups downstream key off
    drop_columns: tuple = ()         # module-private columns that must not
                                     # reach a trades file (day_bucket,
                                     # combine_vid); the actions row strips
                                     # the report's own derived ones itself

    def __post_init__(self):
        object.__setattr__(self, "name_parts", tuple(self.name_parts))
        object.__setattr__(self, "drop_columns", tuple(self.drop_columns))


# ══ the canonical shape ══════════════════════════════════════════════════════
def canonical_trades(df: pd.DataFrame, *, ticks_column: str = "ticks",
                     day_type_column: str = "day_type",
                     day_type_renames: dict | None = None) -> pd.DataFrame:
    """
    A copy of `df` in the report's canonical shape. Idempotent, so a caller
    that is already canonical pays a stable sort and a cumsum and nothing
    else. See the module docstring for what is and is not touched.
    """
    if df is None:
        raise TradeFrameError("no trades frame was given to the report")
    out = df.copy()

    if ticks_column != "ticks":
        if ticks_column not in out.columns:
            raise TradeFrameError(
                f"trades frame has no '{ticks_column}' column to read ticks "
                f"from (columns: {sorted(map(str, out.columns))})")
        out["ticks"] = out[ticks_column]
    if day_type_column != "day_type":
        if day_type_column not in out.columns:
            raise TradeFrameError(
                f"trades frame has no '{day_type_column}' column to read the "
                f"day type from (columns: {sorted(map(str, out.columns))})")
        out["day_type"] = out[day_type_column]
    if day_type_renames:
        out["day_type"] = out["day_type"].replace(day_type_renames)

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise TradeFrameError(
            f"trades frame is missing {missing} — the report needs "
            f"{list(REQUIRED_COLUMNS)}")

    out = out.sort_values("entry_time", kind="stable").reset_index(drop=True)
    out["cumulative_ticks"] = out["ticks"].cumsum()
    return out


def from_optimizer_rows(df: pd.DataFrame) -> pd.DataFrame:
    """An optimizer run's rows (pnl_ticks / day_bucket) in canonical shape."""
    return canonical_trades(df, ticks_column="pnl_ticks",
                            day_type_column="day_bucket",
                            day_type_renames={"other_high_impact":
                                              "high_impact"})


# ══ filter-stage primitives ══════════════════════════════════════════════════
def narrow(df: pd.DataFrame, mask) -> pd.DataFrame:
    """
    One narrowing step: slice, copy, and recompute cumulative_ticks.

    The recompute is not optional — the equity curve reads that column, and a
    gapped cumsum draws the pre-filter path. Having exactly one primitive is
    what stopped the Backtester's and the Optimizer's chains drifting on this
    (the Optimizer's used to skip it after the trade-type stage).
    """
    out = df[mask].copy()
    out["cumulative_ticks"] = out["ticks"].cumsum()
    return out


def entry_frame(frame: pd.DataFrame, column: str, selected) -> pd.DataFrame:
    """Apply one filter to the entry-breakdown frame, which is everything the
    report is showing EXCEPT the trade-type filter (entry types have to stay
    comparable, so filtering to one of them would defeat the table)."""
    if selected is None or column not in frame.columns:
        return frame
    return frame[frame[column].isin(selected)]


def apply_mask(frame: pd.DataFrame, mask) -> pd.DataFrame:
    """The same idea for a predicate that isn't column+isin — the trade-notes
    query. `mask` is None when the query has nothing to say."""
    return frame if mask is None else frame[mask]


# ══ save names ═══════════════════════════════════════════════════════════════
_UNSAFE = re.compile(r"[^A-Za-z0-9._\-]+")


def save_name(parts) -> str:
    """
    Filename stem for a saved trades file: falsy parts dropped, each part
    made filename-safe, joined with "_".

    Part 0 stays the first underscore token on purpose — Analytics and Monte
    Carlo derive the asset from it (see the note in trade_files).
    """
    return "_".join(_UNSAFE.sub("-", str(p)).strip("-") for p in parts if p)
