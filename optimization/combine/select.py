"""
Greedy forward selection + swap step over the variant pool, scoring ONLY the
in-sample no-overlap merged stream (total pnl_ticks) minus a redundancy
penalty — never standalone metrics, because merged P&L is non-additive.

Forward step: for each remaining candidate C,
    score(C) = marginal merged-ticks gain of S ∪ {C}
               − λ · max_{s in S} corr(C_daily, s_daily)
Add argmax(score); ties break on vid (deterministic). Recording every set
size as a path point. Stops when the best candidate's score <= 0 or its raw
marginal gain < 0 (keeps the IS path monotone non-decreasing), or at max_k.

Swap step (after forward, on the final set): repeatedly try replacing each
member with each non-member; apply the best change that raises the set
objective merged_ticks − λ·mean_pairwise_corr; stop when no swap improves or
after max_swaps applied swaps. The improved set is recorded as an extra
"swap" path point at the same k (forward path points stay nested prefixes).

Multi-seed (n_seeds > 1): re-run the forward pass forcing the first pick to
each of the top-N standalone-IS variants and keep the best-final-IS path —
mitigates first-pick lock-in.

The in-sample daily-P&L correlation matrix is computed once and cached in
the returned context. Variants sharing no active days correlate ≈ 0 on the
zero-filled shared date index.
"""

import numpy as np
import pandas as pd

from .merge import merge_streams, merged_total


def correlation_matrix(variants: list) -> np.ndarray:
    """
    variants × variants Pearson correlation of in-sample daily P&L, aligned
    on the union of in-sample dates (inactive days = 0). NaN (zero-variance
    vectors) -> 0 so the penalty never poisons a score.
    """
    frame = pd.DataFrame({i: v.is_daily for i, v in enumerate(variants)}) \
        .fillna(0.0)
    if frame.empty or len(frame) < 2:
        return np.zeros((len(variants), len(variants)))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(frame.to_numpy(dtype=float).T)
    corr = np.atleast_2d(corr)
    return np.nan_to_num(corr, nan=0.0)


def _forward_path(variants, corr, lam, max_k, first_pick=None, log=None):
    """One forward pass. Returns the path (member indices per set size)."""
    n = len(variants)
    remaining = set(range(n))
    selected: list = []
    raw_union: list = []          # sorted union of selected members' tuples
    current_total = 0.0
    path = []

    for step in range(min(max_k, n)):
        best = None               # (score, gain, vid, idx)
        if first_pick is not None and step == 0:
            candidates = [first_pick]
        else:
            candidates = remaining
        for i in candidates:
            v = variants[i]
            total = merged_total(merge_streams(raw_union, v.is_tuples)) \
                if raw_union else merged_total(iter(v.is_tuples))
            gain = total - current_total
            redundancy = max((corr[i][j] for j in selected), default=0.0)
            score = gain - lam * redundancy
            if best is None or (score, gain) > (best[0], best[1]) \
                    or ((score, gain) == (best[0], best[1]) and v.vid < best[2]):
                best = (score, gain, v.vid, i)

        if best is None or best[0] <= 0 or best[1] < 0:
            break

        _, gain, _, idx = best
        selected.append(idx)
        remaining.discard(idx)
        raw_union = list(merge_streams(raw_union, variants[idx].is_tuples))
        current_total += gain
        path.append({"k": len(selected), "stage": "forward",
                     "members": list(selected), "is_ticks": current_total})
        if log is not None:
            log(f"[greedy] k={len(selected)}  +{variants[idx].vid}  "
                f"->  IS {current_total:.0f} ticks")
    return path


def _set_objective(variants, indices, corr, lam) -> float:
    """Order-free swap objective: merged ticks − λ·mean pairwise corr."""
    streams = [variants[i].is_tuples for i in indices]
    total = merged_total(merge_streams(*streams)) if streams else 0.0
    if lam and len(indices) > 1:
        pairs = [corr[a][b] for x, a in enumerate(indices)
                 for b in indices[x + 1:]]
        total -= lam * (sum(pairs) / len(pairs))
    return total


def _swap_improve(variants, selected, corr, lam, max_swaps, log=None):
    """Best-improvement swaps until none helps (or max_swaps applied)."""
    selected = list(selected)
    if not selected:
        return selected, 0
    n = len(variants)
    current = _set_objective(variants, selected, corr, lam)
    applied = 0
    while applied < max_swaps:
        best = None               # (objective, pos, candidate)
        in_set = set(selected)
        for pos in range(len(selected)):
            for c in range(n):
                if c in in_set:
                    continue
                trial = selected[:pos] + [c] + selected[pos + 1:]
                obj = _set_objective(variants, trial, corr, lam)
                if obj > current + 1e-9 and (best is None or obj > best[0]):
                    best = (obj, pos, c)
        if best is None:
            break
        current = best[0]
        selected[best[1]] = best[2]
        applied += 1
        if log is not None:
            log(f"[swap] {applied}: IS objective -> {current:.0f}")
    return selected, applied


def greedy_select(variants: list, lam: float = 0.0, max_k: int = 30,
                  n_seeds: int = 1, max_swaps: int = 50, log=None) -> dict:
    """
    Full selection: (multi-seed) forward path + swap step on the final set.
    Returns {"path": [...], "corr": matrix}; path rows carry member INDICES
    into `variants` (stage "forward" rows are nested prefixes; an extra
    "swap" row is appended when the swap step improved the final set).
    """
    if not variants:
        return {"path": [], "corr": np.zeros((0, 0))}

    corr = correlation_matrix(variants)

    seeds = [None]
    if n_seeds > 1:
        standalone = sorted(
            range(len(variants)),
            key=lambda i: (-merged_total(iter(variants[i].is_tuples)),
                           variants[i].vid),
        )
        seeds = standalone[:n_seeds]

    best_path, best_final = None, -float("inf")
    for s_i, seed in enumerate(seeds):
        if log is not None and len(seeds) > 1:
            log(f"[seed {s_i + 1}/{len(seeds)}] first pick forced: "
                f"{variants[seed].vid}")
        path = _forward_path(variants, corr, lam, max_k, first_pick=seed,
                             log=log)
        final = path[-1]["is_ticks"] if path else -float("inf")
        if final > best_final:
            best_path, best_final = path, final

    path = best_path or []
    if path:
        selected = path[-1]["members"]
        swapped, applied = _swap_improve(variants, selected, corr, lam,
                                         max_swaps, log)
        if applied and set(swapped) != set(selected):
            streams = [variants[i].is_tuples for i in swapped]
            path.append({
                "k": len(swapped), "stage": "swap", "members": swapped,
                "is_ticks": merged_total(merge_streams(*streams)),
            })
    return {"path": path, "corr": corr}
