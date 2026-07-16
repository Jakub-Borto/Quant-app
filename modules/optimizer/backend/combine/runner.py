"""
The combine pipeline in the strict §6.1 order:

  load runs -> compatibility gate -> pool (variants, day filter, shared
  window) -> chronological IS/OOS split -> per-entry min-trades (IS only) ->
  greedy + swap on the IS merged stream -> OOS evaluation of the whole path
  -> path/members tables ready to persist.

Selection touches ONLY the in-sample slice; every path row's out-of-sample
number is computed afterwards, in one sealed pass.
"""

import json

import pandas as pd

from ..io import RUNS_ROOT
from .compat import check_compatibility
from .evaluate import evaluate_set
from .pool import (apply_min_trades, build_pool, load_entry_runs, split_pool,
                   split_date_boundary)
from .select import greedy_select


def run_combine(container: str, run_names: list, *, enabled_buckets: set,
                floors: dict, is_fraction: float, lam: float = 0.0,
                max_k: int = 30, n_seeds: int = 1, max_swaps: int = 50,
                log=None, root=RUNS_ROOT) -> dict:
    """
    Execute the full pipeline. Returns {path_df, members_df, meta, variants,
    boundary} — the caller persists via combine.io. Raises ValueError with a
    readable message on gate failures / empty pools.
    """
    def _log(msg):
        if log is not None:
            log(msg)

    runs = load_entry_runs(container, run_names, root)
    gate = check_compatibility({n: meta for n, (meta, _) in runs.items()})
    if not gate["ok"]:
        raise ValueError("incompatible runs — " + "; ".join(gate["errors"]))
    for w in gate["warnings"]:
        _log(f"[gate] {w}")

    variants = build_pool(runs, enabled_buckets,
                          gate["shared_start"], gate["shared_end"])
    if not variants:
        raise ValueError("empty pool — no trades survive the day-type filter "
                         "and shared date window")
    _log(f"[pool] {len(variants)} variants from {len(runs)} runs")

    boundary = split_date_boundary(variants, is_fraction)
    if boundary is None:
        raise ValueError("not enough distinct trading dates to split IS/OOS")
    split_pool(variants, boundary)
    boundary_date = pd.Timestamp(boundary).date()
    _log(f"[split] in-sample through {boundary_date} "
         f"({is_fraction:.0%} of trading dates)")

    variants = apply_min_trades(variants, floors)
    variants = [v for v in variants if v.n_is > 0]
    if not variants:
        raise ValueError("empty pool — every variant fell below its "
                         "min-trades floor on the in-sample slice")
    _log(f"[floor] {len(variants)} variants meet their min-trades floor")

    result = greedy_select(variants, lam=lam, max_k=max_k, n_seeds=n_seeds,
                           max_swaps=max_swaps, log=log)
    path = result["path"]
    if not path:
        raise ValueError("greedy selected nothing — no variant has a "
                         "positive in-sample merged contribution")

    # ── sealed OOS evaluation of every path point ─────────────────────────────
    path_rows, member_rows = [], []
    for point in path:
        members = [variants[i] for i in point["members"]]
        is_m  = evaluate_set([v.is_tuples for v in members])
        oos_m = evaluate_set([v.oos_tuples for v in members])
        path_rows.append({
            "k": point["k"],
            "stage": point["stage"],
            "member_vids": json.dumps([v.vid for v in members]),
            "is_ticks": is_m["total_ticks"],
            "oos_ticks": oos_m["total_ticks"],
            "is_sharpe": is_m["sharpe_daily"],
            "oos_sharpe": oos_m["sharpe_daily"],
            "is_max_dd": is_m["max_dd_ticks"],
            "oos_max_dd": oos_m["max_dd_ticks"],
            "is_max_dd_pct": is_m["max_dd_pct"],
            "oos_max_dd_pct": oos_m["max_dd_pct"],
            "n_trades_is": is_m["n_trades"],
            "n_trades_oos": oos_m["n_trades"],
            "oos_empty": oos_m["empty"],
        })
        for v in members:
            member_rows.append({
                "k": point["k"], "stage": point["stage"], "vid": v.vid,
                "run": v.run, "trade_type": v.trade_type,
                "params": json.dumps(v.params, default=str), "n_is": v.n_is,
                "n_oos": v.n_oos,
            })

    path_df = pd.DataFrame(path_rows)
    peak_idx = int(path_df["oos_ticks"].idxmax())
    path_df["is_oos_peak"] = False
    path_df.loc[peak_idx, "is_oos_peak"] = True
    _log(f"[oos] peak at k={path_df.loc[peak_idx, 'k']} "
         f"({path_df.loc[peak_idx, 'oos_ticks']:.0f} OOS ticks)")

    meta = {
        "container": container,
        "runs": sorted(run_names),
        "ticker": gate["ticker"],
        "dataset": gate["dataset"],
        "ticks_per_point": gate["ticks_per_point"],
        "shared_start": str(gate["shared_start"].date()),
        "shared_end": str(gate["shared_end"].date()),
        "enabled_day_buckets": sorted(enabled_buckets),
        "min_trades_floors": floors,
        "is_fraction": is_fraction,
        "split_boundary": str(boundary_date),
        "lambda": lam,
        "max_k": max_k,
        "n_seeds": n_seeds,
        "max_swaps": max_swaps,
        "pool_size": len(variants),
        "created_at": pd.Timestamp.now().isoformat(),
    }
    return {"path_df": path_df, "members_df": pd.DataFrame(member_rows),
            "meta": meta, "variants": variants, "boundary": boundary}
