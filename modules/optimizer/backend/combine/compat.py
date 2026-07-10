"""
Compatibility gate over the selected entry runs' meta.json.

Merged pnl_ticks are only commensurable when every run traded the same asset
on the same dataset with the same ticks_per_point — those mismatches are hard
stops (naming the offending runs). Date-range differences are recovered, not
rejected: the shared window is the intersection, and the pool is trimmed to
it with a note.
"""

import pandas as pd


def check_compatibility(metas: dict) -> dict:
    """
    metas: {run_name: meta_dict}. Returns
      {ok, errors, warnings, ticker, dataset, ticks_per_point,
       shared_start, shared_end}
    """
    errors, warnings = [], []

    def _values(key):
        return {name: meta.get(key) for name, meta in metas.items()}

    for key in ("ticker", "dataset", "ticks_per_point"):
        values = _values(key)
        missing = [n for n, v in values.items() if v is None]
        if missing:
            errors.append(f"{key} missing from meta.json of: {', '.join(missing)}")
            continue
        if len(set(values.values())) > 1:
            detail = ", ".join(f"{n}={v}" for n, v in values.items())
            errors.append(f"{key} differs across runs ({detail})")

    tpp = _values("ticks_per_point")
    bad_tpp = [n for n, v in tpp.items()
               if isinstance(v, (int, float)) and v <= 0]
    if bad_tpp:
        errors.append(f"ticks_per_point not positive in: {', '.join(bad_tpp)}")

    shared_start = shared_end = None
    try:
        starts = {n: pd.Timestamp(m["start_date"]) for n, m in metas.items()}
        ends   = {n: pd.Timestamp(m["end_date"]) for n, m in metas.items()}
        shared_start, shared_end = max(starts.values()), min(ends.values())
        if shared_start > shared_end:
            errors.append("date ranges do not overlap across the selected runs")
        elif len(set(starts.values())) > 1 or len(set(ends.values())) > 1:
            warnings.append(
                f"date ranges differ — using shared window "
                f"{shared_start.date()} → {shared_end.date()}"
            )
    except (KeyError, TypeError, ValueError):
        errors.append("start_date/end_date missing or unparsable in a meta.json")

    first = next(iter(metas.values()), {})
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "ticker": first.get("ticker"),
        "dataset": first.get("dataset"),
        "ticks_per_point": first.get("ticks_per_point"),
        "shared_start": shared_start,
        "shared_end": shared_end,
    }
