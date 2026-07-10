"""
The no-overlap merge primitive: one open position at a time.

Trades are plain tuples (fast to sort and walk in pure Python):

    (entry_ns, exit_ns, vid, seq, pnl_ticks, date_ns)

entry/exit are int64 epoch-nanoseconds (tz-aware timestamps convert
losslessly), `vid` is the variant id and `seq` the row's position within its
variant — together the deterministic tie-break the spec requires: ties on
entry_time break by earlier exit_time, then (vid, seq). Tuple comparison on
the first four fields implements exactly that ordering.

The walk keeps a trade only if its entry_time >= the last kept trade's
exit_time. Multiple non-overlapping trades per day all survive; a trade
overlapping a kept one is skipped.

Incremental merging: greedy repeatedly scores "S ∪ {candidate}". Re-walking
is unavoidable (removing one kept trade can free a slot for a previously
skipped one — merging into the KEPT stream would be wrong), but re-SORTING is
not: members' tuple lists are already sorted, so heapq.merge streams the
union in order with no sort. The tested invariant: heapq.merge over sorted
member lists == full sort of the concatenation.
"""

from heapq import merge as _heap_merge


def trades_to_tuples(trades, vid: str) -> list:
    """One variant's DataFrame -> sorted trade tuples (see module doc)."""
    entry = trades["entry_time"].astype("int64").to_list()
    exit_ = trades["exit_time"].astype("int64").to_list()
    pnl   = trades["pnl_ticks"].astype(float).to_list()
    date  = trades["date"].astype("int64").to_list()
    rows  = [(e, x, vid, i, p, d)
             for i, (e, x, p, d) in enumerate(zip(entry, exit_, pnl, date))]
    rows.sort()
    return rows


def merge_streams(*sorted_streams) -> "iterator":
    """Time-ordered union of already-sorted tuple lists (no re-sort)."""
    if len(sorted_streams) == 1:
        return iter(sorted_streams[0])
    return _heap_merge(*sorted_streams)


def no_overlap_walk(ordered_trades) -> list:
    """
    The core rule: walk trades in time order keeping a last-exit cursor;
    keep a trade only if entry >= last kept exit. Returns the kept tuples.
    """
    kept = []
    last_exit = None
    for t in ordered_trades:
        if last_exit is None or t[0] >= last_exit:
            kept.append(t)
            last_exit = t[1]
    return kept


def merged_total(ordered_trades) -> float:
    """sum(pnl_ticks) of the kept trades — the greedy scoring hot path."""
    total = 0.0
    last_exit = None
    for t in ordered_trades:
        if last_exit is None or t[0] >= last_exit:
            total += t[4]
            last_exit = t[1]
    return total
