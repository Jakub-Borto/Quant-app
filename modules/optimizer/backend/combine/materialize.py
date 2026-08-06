"""
Rebuild a chosen combine set's MERGED trades, on demand.

Nothing ever persists them: a combine run stores the path, the member
identities and the meta — the merged rows exist only for the microseconds
`evaluate_set` spends walking them. Rebuilding is exact rather than
approximate because the identity of a member is its `vid`, the algorithm's own
string, and because the merge is a pure function of the pooled trades. Nothing
is re-backtested here.

Two rules this module must not drift from:

  * Grouping goes through `pool.iter_variant_cells` — the same generator
    `build_pool` uses. Re-deriving the groupby here would be the one way a
    saved run's vids could stop resolving.
  * `"all"` (100% of data) is the IN-SAMPLE merge stitched onto the
    OUT-OF-SAMPLE merge, NOT one continuous walk across the boundary.
    Selection sealed the split, so the two halves were merged independently;
    stitching is what makes the report's totals equal the `is_ticks` +
    `oos_ticks` a user reads off the path table. The cost is that a position
    held across the boundary day can leave one overlapping pair at the seam.
"""

import pandas as pd

from .merge import _epoch_ns, merge_streams, no_overlap_walk, trades_to_tuples
from .pool import (_POOL_COLUMNS, boundary_from_dates, iter_variant_cells,
                   load_entry_runs)

SCOPES = ("all", "is", "oos")
SCOPE_LABELS = {
    "all": "100% of data",
    "is":  "1st split (in-sample)",
    "oos": "2nd split (out-of-sample)",
}

VID_COLUMN = "combine_vid"      # provenance: which member each row came from
_ORDER_COLUMN = "_merge_order"  # scratch, dropped before returning

_MAX_NOTES = 5


def build_cells(runs: dict, combine_meta: dict) -> dict:
    """
    {vid: full-column trades frame} for the pooled runs, under the SAVED
    combine run's day-bucket and shared-window filters.

    This is the cold half of the work (parquet + groupby); the merge itself is
    microseconds, so callers cache the result and re-merge per k / per scope.
    """
    buckets = set(combine_meta.get("enabled_day_buckets") or []) or None
    cells, dupes = {}, []
    for vid, _run, _tt, _params, cell in iter_variant_cells(
            runs, buckets, combine_meta.get("shared_start"),
            combine_meta.get("shared_end")):
        if vid in cells:
            dupes.append(vid)
        _assert_normalized_dates(cell, vid)
        cells[vid] = cell
    if dupes:
        raise ValueError(
            "ambiguous variant ids — two parameter cells format to the same "
            "name (float params colliding under '%g'): "
            + ", ".join(sorted(set(dupes))))
    return cells


def split_boundary_ns(cells: dict, combine_meta: dict):
    """
    The run's own IS/OOS cut, RE-DERIVED from the rebuilt pool with the saved
    is_fraction rather than parsed out of meta["split_boundary"].

    Two reasons: it is the algorithm's own function (`boundary_from_dates`,
    fed the same pre-floor pool `split_date_boundary` saw), and combine runs
    saved before the epoch-unit fix in merge.py carry a meta boundary of
    1970-01-01. None when the pool has fewer than two trading dates.
    """
    return boundary_from_dates(
        (value for cell in cells.values() if "date" in cell.columns
         for value in _epoch_ns(cell["date"])),
        float(combine_meta.get("is_fraction", 0.7)))


def load_cells(container: str, combine_meta: dict, *, root) -> tuple:
    """(cells, runs) for a saved combine run — reads meta['runs'] off disk."""
    runs = load_entry_runs(container, list(combine_meta.get("runs") or []),
                           root=root)
    return build_cells(runs, combine_meta), runs


def verify_cells(cells: dict, members_df: pd.DataFrame) -> list:
    """
    Human-readable notes when the entry runs on disk no longer match what the
    combine run was saved against (a run re-optimized in place, say). Empty
    list = the rebuild is faithful.
    """
    notes, seen = [], set()
    for row in members_df.itertuples():
        if row.vid in seen:
            continue
        seen.add(row.vid)
        cell = cells.get(row.vid)
        saved = int(row.n_is) + int(row.n_oos)
        if cell is None:
            notes.append(f"member variant missing from the entry runs: {row.vid}")
        elif len(cell) != saved:
            notes.append(f"{row.vid}: {len(cell)} trades on disk, {saved} when "
                         "this combine run was saved")
    if len(notes) > _MAX_NOTES:
        notes = notes[:_MAX_NOTES] + [f"… and {len(notes) - _MAX_NOTES} more"]
    return notes


def combined_trades(cells: dict, combine_meta: dict, member_vids: list,
                    *, scope: str = "all", boundary=None) -> pd.DataFrame:
    """
    One path point's merged trades as FULL trade rows (every column the entry
    run had, plus VID_COLUMN), in merged order.

    scope: "is" / "oos" apply the same date predicate `split_pool` does, so
    their totals reproduce the path table's is_ticks / oos_ticks exactly;
    "all" concatenates the two (see the module docstring). `boundary` is the
    epoch-ns cut — computed from the pool when the caller doesn't pass one.
    """
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r} — expected one of {SCOPES}")
    vids = list(dict.fromkeys(member_vids))
    missing = [v for v in vids if v not in cells]
    if missing:
        raise ValueError("member variants no longer resolve against the entry "
                         "runs on disk: " + "; ".join(missing))

    boundary_ns = split_boundary_ns(cells, combine_meta) \
        if boundary is None else boundary
    streams = {vid: trades_to_tuples(
        cells[vid][[c for c in _POOL_COLUMNS if c in cells[vid].columns]], vid)
        for vid in vids}

    def in_sample(date_ns):
        return boundary_ns is not None and date_ns <= boundary_ns

    def out_of_sample(date_ns):
        return boundary_ns is None or date_ns > boundary_ns

    if scope == "is":
        return _walk(cells, streams, in_sample)
    if scope == "oos":
        return _walk(cells, streams, out_of_sample)
    halves = [_walk(cells, streams, in_sample),
              _walk(cells, streams, out_of_sample)]
    return pd.concat(halves, ignore_index=True)


def report_context(runs: dict, combine_meta: dict) -> dict:
    """
    What the shared trade report needs but the combine meta doesn't carry:
    tick_size and the strategy name live only in the pooled ENTRY runs' metas
    (the compatibility gate never compares tick_size, hence the note).
    """
    metas = [meta for _name, (meta, _trades) in sorted(runs.items())]
    first = metas[0] if metas else {}
    notes = []
    tick_sizes = {m.get("tick_size") for m in metas if m.get("tick_size")}
    if len(tick_sizes) > 1:
        notes.append(f"pooled entry runs disagree on tick_size "
                     f"({', '.join(f'{t:g}' for t in sorted(tick_sizes))}) — "
                     f"using {first.get('tick_size'):g}")
    return {
        "ticker": combine_meta.get("ticker") or first.get("ticker"),
        "tick_size": first.get("tick_size"),
        "ticks_per_point": combine_meta.get("ticks_per_point")
                           or first.get("ticks_per_point"),
        "strategy": first.get("strategy"),
        "notes": notes,
    }


# ══ internals ═══════════════════════════════════════════════════════════════
def _assert_normalized_dates(cell: pd.DataFrame, vid: str) -> None:
    """
    The IS/OOS boundary is a calendar date compared against the tuples' raw
    int64 `date`; that only lines up because the optimizer engine normalizes
    `date` to midnight. A time component would drift the split by up to a day.
    """
    if "date" not in cell.columns or cell.empty:
        return
    dates = pd.to_datetime(cell["date"])
    if not (dates == dates.dt.normalize()).all():
        raise ValueError(f"variant {vid!r} has trade dates carrying a time "
                         "component — the IS/OOS boundary cannot be applied")


def _walk(cells: dict, streams: dict, keep) -> pd.DataFrame:
    """No-overlap merge of the members' tuples on one side of the boundary."""
    parts = []
    for stream in streams.values():
        side = [t for t in stream if keep(t[5])]
        if side:
            parts.append(side)
    kept = no_overlap_walk(merge_streams(*parts)) if parts else []
    if not kept:
        return _empty_like(cells, streams.keys())

    # (vid, seq) -> the full row: trades_to_tuples enumerates BEFORE it sorts,
    # so seq is the row's position in the cell frame it was built from
    by_vid = {}
    for position, trade in enumerate(kept):
        by_vid.setdefault(trade[2], []).append((trade[3], position))

    frames = []
    for vid, pairs in by_vid.items():
        part = cells[vid].iloc[[seq for seq, _pos in pairs]].copy()
        part[VID_COLUMN] = vid
        part[_ORDER_COLUMN] = [pos for _seq, pos in pairs]
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    return (out.sort_values(_ORDER_COLUMN, kind="mergesort")
               .drop(columns=_ORDER_COLUMN)
               .reset_index(drop=True))


def _empty_like(cells: dict, vids) -> pd.DataFrame:
    """Zero rows, but the union of the members' columns — an empty scope still
    has to be a well-formed frame for the report's filters."""
    frames = [cells[v].iloc[0:0] for v in vids if v in cells]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out[VID_COLUMN] = pd.Series(dtype=object)
    return out
