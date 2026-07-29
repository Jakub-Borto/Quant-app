"""
Regime-switching Monte Carlo — the pure estimation + simulation engine.

NAMING: this is a Markov REGIME-SWITCHING simulation, not "MCMC". Markov Chain
Monte Carlo is a Bayesian sampling technique and solves a different problem.
Both use a Markov chain; do not let the names merge.

WHY IT EXISTS. A plain bootstrap draws trades independently, which assumes
trades are independent. They are not: volatility clusters, so losers arrive in
runs, and independent resampling scatters those runs apart and systematically
UNDERSTATES drawdown. Here the chain stays in a regime for a realistic stretch
and keeps drawing that regime's trades while it does, so the bad outcomes bunch
the way they really do. That clustering is the entire point of the method;
every design decision below protects it.

THE TWO POPULATIONS (§6 of the build plan) — the crux, and the easiest thing to
get wrong:

  ALL days define how regimes MOVE.   -> transition matrix
  TRADE days define what happens WHEN YOU TRADE in each. -> trade-probability + pools

Estimating the matrix from trade days only would delete the flat days and with
them the clustering. Given real states H H H N N L L N H H and trades on days
Y - Y Y - - Y Y - Y, an all-days walk counts H->H three times; a trade-only walk
jumps H(1)->H(3)->N(4)->L(7)->N(8)->H(10) and barely sees H->H at all. It would
understate persistence — the one thing this method exists to capture.

DAY LABELLING. Every day in the estimation window needs exactly one state.
Under tag=final that is the day's is_final row throughout. Under tag=entry a
TRADE day is labelled by the snapshot at-or-before its (first) trade's entry —
genuinely point-in-time — but a NO-TRADE day has no entry timestamp to join on,
so `no_trade_snapshot` picks it:

  NO_TRADE_FINAL     (default) — the day's is_final row. Chosen by the user.
      HONESTY NOTE: is_final knows how the day ended, so the trade-probability
      DENOMINATOR becomes partly retrospective. The trade POOLS stay strictly
      point-in-time either way; only the "how often does this regime trade"
      estimate inherits the hindsight.
  NO_TRADE_DECISION  — the snapshot at-or-before a fixed wall-clock decision
      time (the UI defaults it to the median entry time of the trades file).
      Fully point-in-time, and measured at roughly the hour a trade day is.

CHAIN BREAKS. `unknown` is not a state — it is an absence of answer. An unknown
day contributes no transition either into or out of itself, and critically does
NOT let its neighbours join across it. Calendar gaps larger than `max_gap_days`
break the chain too, so a non-contiguous estimation range never manufactures a
transition across the hole.

SIMPLIFICATIONS, stated rather than hidden:
  - No intra-trade regime change. A trade keeps the tag it was given at entry
    even if it spans hours and crosses a boundary.
  - Thin-pool tails. With ~65 trades in a state the sim redraws the same handful
    of extremes across thousands of paths; the fan chart's tails look more
    precise than the data supports. `pool_size` is surfaced so this is visible.
  - The matrix is treated as EXACT. ~300 days gives a wobbly 0.70. The raw count
    matrix is the honesty mechanism — a 0.70 from 8 transitions is not a 0.70
    from 90.

Pure and Qt-free (tests/test_qt_smoke.py enforces it): safe in worker threads,
pool workers and tests.
"""

import numpy as np
import pandas as pd

from modules.common.backend.regime_join import (MODE_ASOF, UNKNOWN,
                                                attach_regime,
                                                normalize_entry_times,
                                                normalize_trade_dates)
from modules.monte_carlo.methods.base import _step_commission
from modules.regime_detector.backend import io as rio

# How each historical trade AND each day of the matrix is labelled.
TAG_ENTRY = "entry"   # as-of the trade's entry — point-in-time
TAG_FINAL = "final"   # the is_final row — retrospective, structural research only
TAGS = (TAG_ENTRY, TAG_FINAL)
TAG_LABELS = {TAG_ENTRY: "As of entry (point-in-time)",
              TAG_FINAL: "Final row (retrospective)"}

# Which snapshot labels a day the strategy did not trade (tag=entry only).
NO_TRADE_FINAL = "final"
NO_TRADE_DECISION = "decision"
NO_TRADE_MODES = (NO_TRADE_FINAL, NO_TRADE_DECISION)
NO_TRADE_LABELS = {
    NO_TRADE_FINAL: "Day's final snapshot (retrospective)",
    NO_TRADE_DECISION: "Snapshot at a fixed decision time",
}

DEFAULT_SESSION_START = "18:00"
# Calendar days between two consecutive regime files that still count as
# consecutive trading days. 5 spans a weekend plus a holiday Monday; anything
# wider is a data gap and must not become a transition.
DEFAULT_MAX_GAP_DAYS = 5
DEFAULT_MIN_TRADES_PER_STATE = 15

# Simulation horizon modes.
HORIZON_MATCH = "match"   # as many trading days as the pool window holds
HORIZON_FIXED = "fixed"   # a user-set day count

# Starting-regime modes. Only `stationary` is wired up; `per_regime` (one fan
# per starting state) is the planned extension and shares this machinery.
START_STATIONARY = "stationary"


# ── day labelling ────────────────────────────────────────────────────────────

def _frame_days(frames: dict) -> pd.DatetimeIndex:
    """The run's trading-day calendar: one normalized Timestamp per daily file.

    Forced to nanosecond resolution: pandas 3 hands out [us] indexes here and
    [ns] there, and a unit mismatch makes `isin`/`intersection` quietly miss."""
    days = pd.DatetimeIndex(sorted(pd.Timestamp(d).normalize() for d in frames))
    return days.as_unit("ns")


def _trade_days(trades: pd.DataFrame) -> pd.Series:
    """Each trade's RTH date at the same resolution as _frame_days."""
    return normalize_trade_dates(trades).dt.as_unit("ns")


def _snapshot_labels(frames: dict, column: str, at: str,
                     session_start: str) -> pd.Series:
    """One label per day from a single snapshot per day (`at` = "HH:MM" or
    rio.FINAL), indexed by the RTH date. Days with no snapshot yet at that
    time are simply absent — the caller fills them with UNKNOWN."""
    picked = rio.pick_rows(frames, at, session_start)
    if picked.empty or column not in picked.columns:
        return pd.Series(dtype=object)
    out = picked[column].astype(object)
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    return out


def day_labels(trades: pd.DataFrame, frames: dict, column: str, *,
               tag: str = TAG_ENTRY,
               no_trade_snapshot: str = NO_TRADE_FINAL,
               decision_time: str | None = None,
               session_start: str = DEFAULT_SESSION_START,
               snapshot_minutes: int = 30) -> tuple[pd.Series, pd.Series]:
    """
    (day_labels, trade_labels) — the ONE labelling both estimation halves use.

    day_labels   : Series state-per-day, indexed by every day the regime run
                   covers (UNKNOWN where no snapshot applies).
    trade_labels : Series state-per-trade, aligned to `trades.index`.

    Both come from the same `tag`, deliberately: a matrix estimated off one
    snapshot and pools drawn off another are incoherent, and the incoherence is
    invisible in the output.
    """
    if tag not in TAGS:
        raise ValueError(f"unknown tag {tag!r}, expected one of {TAGS}")
    if no_trade_snapshot not in NO_TRADE_MODES:
        raise ValueError(f"unknown no_trade_snapshot {no_trade_snapshot!r}")

    days = _frame_days(frames)

    if tag == TAG_FINAL:
        # Retrospective throughout: the day's closing view labels the day, and
        # every trade on it inherits that label.
        labels = _snapshot_labels(frames, column, rio.FINAL, session_start)
        labels = labels.reindex(days).fillna(UNKNOWN).astype(object)
        return labels, _broadcast_to_trades(trades, labels)

    # tag == TAG_ENTRY -------------------------------------------------------
    # Per-trade: the shared as-of join (backward, cross-session guarded).
    tagged = attach_regime(trades, frames, column, MODE_ASOF,
                           session_start=session_start,
                           snapshot_minutes=snapshot_minutes,
                           out_col="_rs_state")
    trade_labels = tagged["_rs_state"].astype(object)

    # No-trade days fall back to the chosen snapshot; trade days then overwrite
    # it with their own (first) trade's label, so a trade's pool state and its
    # day's chain state are the same value by construction.
    if no_trade_snapshot == NO_TRADE_DECISION:
        if not decision_time:
            raise ValueError("no_trade_snapshot='decision' needs decision_time='HH:MM'")
        at = decision_time
    else:
        at = rio.FINAL
    labels = _snapshot_labels(frames, column, at, session_start)
    labels = labels.reindex(days).fillna(UNKNOWN).astype(object)

    if len(trades):
        first = _first_trade_label_per_day(trades, trade_labels)
        hit = labels.index.intersection(first.index)
        if len(hit):
            labels.loc[hit] = first.loc[hit]
    return labels, trade_labels


def _first_trade_label_per_day(trades: pd.DataFrame,
                               trade_labels: pd.Series) -> pd.Series:
    """The label of each day's EARLIEST trade, indexed by RTH date. Days with a
    single trade (every strategy in this repo today) reduce to that trade."""
    dates = _trade_days(trades).to_numpy()
    order = np.argsort(normalize_entry_times(trades).to_numpy(), kind="stable")
    ordered = pd.Series(trade_labels.to_numpy(dtype=object)[order],
                        index=pd.DatetimeIndex(dates[order]).as_unit("ns"))
    return ordered[~ordered.index.duplicated(keep="first")].sort_index()


def _broadcast_to_trades(trades: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """Each trade takes its day's label (tag=final). UNKNOWN off the calendar."""
    if not len(trades):
        return pd.Series(dtype=object)
    mapped = _trade_days(trades).map(labels)
    return mapped.where(mapped.notna(), UNKNOWN).astype(object)


def slice_days(labels: pd.Series, start: str | None,
               end: str | None) -> pd.Series:
    """`labels` narrowed to [start, end] (inclusive YYYY-MM-DD, None = open)."""
    out = labels
    if start:
        out = out[out.index >= pd.Timestamp(start)]
    if end:
        out = out[out.index <= pd.Timestamp(end)]
    return out


# ── estimation: the transition matrix (ALL days) ─────────────────────────────

def transition_counts(labels: pd.Series, states: list[str], *,
                      max_gap_days: int = DEFAULT_MAX_GAP_DAYS) -> np.ndarray:
    """
    RAW state->state counts over consecutive days. Counts, not probabilities:
    they are how a reader tells a trustworthy 0.70 from a fragile one, so they
    are kept and displayed rather than normalized away.

    A pair contributes nothing when either day is UNKNOWN (an absence of answer
    is not a state) or when the calendar gap exceeds `max_gap_days`. Neither
    case lets the surrounding days join across the hole.
    """
    index = {s: i for i, s in enumerate(states)}
    counts = np.zeros((len(states), len(states)), dtype=np.int64)
    if len(labels) < 2:
        return counts

    days = labels.index.to_numpy()
    vals = labels.to_numpy(dtype=object)
    gaps = (days[1:] - days[:-1]) / np.timedelta64(1, "D")
    for k in range(len(vals) - 1):
        i, j = index.get(vals[k]), index.get(vals[k + 1])
        if i is None or j is None or gaps[k] > max_gap_days:
            continue
        counts[i, j] += 1
    return counts


def transition_probs(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row-normalized probabilities, empty-row mask). A state never observed
    transitioning out of keeps an all-zero row and is flagged, never silently
    filled with a uniform guess."""
    totals = counts.sum(axis=1)
    probs = np.zeros(counts.shape, dtype=float)
    seen = totals > 0
    probs[seen] = counts[seen] / totals[seen, None]
    return probs, ~seen


# ── estimation: trade-probability + pools (TRADE days) ───────────────────────

def trade_populations(labels: pd.Series, trades: pd.DataFrame,
                      trade_labels: pd.Series,
                      states: list[str]) -> dict:
    """
    Per state: day counts, trade-day counts, trade-probability, and the pool of
    POSITIONAL trade indices.

    Indices, not P&L values: the simulation gathers whole trades so a sizer's
    mc_prepare per-trade arrays (risk_based's stop distance, etc.) ride along
    exactly as they do in the bootstrap. Storing bare P&L would silently break
    every risk-aware sizer.

    trade-probability = state-days that traded / all state-days. This is what
    lets an all-days chain produce a realistic cadence: IVB trades ~54% of days,
    and some regimes trade more than others.
    """
    if len(trades):
        day_of_trade = _trade_days(trades)
        in_window = day_of_trade.isin(labels.index)
        traded_days = set(day_of_trade[in_window])
    else:
        day_of_trade = pd.Series(dtype="datetime64[ns]")
        in_window = pd.Series(dtype=bool)
        traded_days = set()

    n_days, n_trade_days, p_trade, pools = {}, {}, {}, {}
    for s in states:
        mask = labels == s
        days_s = labels.index[mask]
        n_days[s] = int(mask.sum())
        n_trade_days[s] = int(sum(1 for d in days_s if d in traded_days))
        p_trade[s] = (n_trade_days[s] / n_days[s]) if n_days[s] else 0.0
        if len(trades):
            hit = in_window.to_numpy() & (trade_labels.to_numpy(dtype=object) == s)
            pools[s] = np.nonzero(hit)[0].astype(np.int64)
        else:
            pools[s] = np.zeros(0, dtype=np.int64)
    return {"n_days": n_days, "n_trade_days": n_trade_days,
            "p_trade": p_trade, "pools": pools}


def pool_profiles(trades: pd.DataFrame, pools: dict,
                  states: list[str]) -> dict:
    """Per state: mean / sd / win-rate / worst trade, in ticks.

    This is what tells a reader whether the regime split MATTERS for their
    strategy — the sample-size columns only say whether it is well measured.
    If every state's distribution looks the same, clustering regimes cannot
    cluster losers and the method degenerates to a slower bootstrap; that is a
    real finding about the strategy, not a failure, and it should be visible
    before spending a run on it.

    Verified on ES/IVB: the states separate on SPREAD, not on mean (sd 25/32/70
    ticks, worst -50/-56/-135, means all ~+4). So the effect lands in tail risk
    while the median path barely moves — exactly the shape the mean-only view
    would have hidden.
    """
    out = {}
    has_ticks = "ticks" in trades.columns
    for s in states:
        idx = np.asarray(pools.get(s, []), dtype=np.int64)
        if idx.size == 0 or not has_ticks:
            out[s] = {"mean": None, "sd": None, "win_rate": None, "worst": None}
            continue
        t = trades["ticks"].to_numpy(dtype=float)[idx]
        out[s] = {
            "mean": float(np.mean(t)),
            "sd": float(np.std(t, ddof=1)) if t.size > 1 else 0.0,
            "win_rate": float(np.mean(t > 0)),
            "worst": float(np.min(t)),
        }
    return out


def multi_trade_days(trades: pd.DataFrame, labels: pd.Series) -> int:
    """Days inside the pool window carrying more than one trade. The Bernoulli
    trade-probability collapses such a day to a single draw, so the count is
    surfaced rather than silently absorbed."""
    if not len(trades):
        return 0
    dates = _trade_days(trades)
    per_day = dates[dates.isin(labels.index)].value_counts()
    return int((per_day > 1).sum())


# ── the estimated model ──────────────────────────────────────────────────────

def estimate(trades: pd.DataFrame, frames: dict, column: str,
             states: list[str], *,
             tag: str = TAG_ENTRY,
             no_trade_snapshot: str = NO_TRADE_FINAL,
             decision_time: str | None = None,
             session_start: str = DEFAULT_SESSION_START,
             snapshot_minutes: int = 30,
             matrix_start: str | None = None, matrix_end: str | None = None,
             pool_start: str | None = None, pool_end: str | None = None,
             max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
             min_trades_per_state: int = DEFAULT_MIN_TRADES_PER_STATE) -> dict:
    """
    The whole model the simulation runs on, and the whole model the UI previews
    BEFORE running — one function, so what is displayed is what is simulated.

    TWO WINDOWS, deliberately separate:
      [matrix_start, matrix_end] — days feeding the transition matrix. May run
          far wider than the trades file (ES regimes go back to 2010), which is
          usually the difference between a 4000-transition estimate and a
          300-transition one.
      [pool_start, pool_end]     — days feeding trade-probability and the pools.
          Cannot usefully exceed the trades file: widening it only adds days
          that could not have traded, deflating every trade-probability toward
          zero. The UI defaults both windows to the trades span.
    """
    warnings: list[str] = []
    labels, trade_labels = day_labels(
        trades, frames, column, tag=tag, no_trade_snapshot=no_trade_snapshot,
        decision_time=decision_time, session_start=session_start,
        snapshot_minutes=snapshot_minutes)

    matrix_labels = slice_days(labels, matrix_start, matrix_end)
    pool_labels = slice_days(labels, pool_start, pool_end)

    counts = transition_counts(matrix_labels, states, max_gap_days=max_gap_days)
    probs, empty_rows = transition_probs(counts)
    pops = trade_populations(pool_labels, trades, trade_labels, states)

    # Starting distribution: the empirical state frequency over the matrix
    # window. Preferred to the chain's algebraic stationary vector because it
    # stays defined when a row is empty.
    freq = np.array([int((matrix_labels == s).sum()) for s in states], dtype=float)
    start_probs = freq / freq.sum() if freq.sum() else np.zeros(len(states))

    # A state that trades but has no pool cannot be simulated — force it flat
    # rather than crash mid-run.
    p_trade = dict(pops["p_trade"])
    for s in states:
        if p_trade[s] > 0 and len(pops["pools"][s]) == 0:
            p_trade[s] = 0.0
            warnings.append(f"State '{s}' has trade days but no usable trades "
                            f"in the pool window — simulated as non-trading.")

    thin = [s for s in states
            if 0 < len(pops["pools"][s]) < min_trades_per_state]
    if thin:
        warnings.append(
            f"Thin trade pool(s): {', '.join(thin)} — under "
            f"{min_trades_per_state} trades. The simulation redraws the same "
            f"few extremes across every path, so the fan chart's tails are "
            f"more precise-looking than the data supports.")
    starved = [s for s in states if len(pops["pools"][s]) == 0 and pops["n_days"][s]]
    if starved:
        warnings.append(f"State(s) {', '.join(starved)} were never traded in "
                        f"the pool window — simulated as flat days.")
    if empty_rows.any():
        names = [s for s, e in zip(states, empty_rows) if e]
        warnings.append(
            f"No transitions observed out of {', '.join(names)} — the chain "
            f"holds those states instead of moving (they absorb).")
    n_multi = multi_trade_days(trades, pool_labels)
    if n_multi:
        warnings.append(
            f"{n_multi} day(s) in the pool window carry more than one trade. "
            f"The trade-probability model draws at most ONE trade per "
            f"simulated day, so those days are collapsed to a single draw and "
            f"total activity is understated.")
    unknown_matrix = int((matrix_labels == UNKNOWN).sum())
    if unknown_matrix:
        warnings.append(
            f"{unknown_matrix} of {len(matrix_labels)} matrix-window days have "
            f"no usable snapshot ('unknown') — each breaks the chain rather "
            f"than contributing a transition.")
    if tag == TAG_FINAL:
        warnings.append(
            "regime_tag = final uses the day's closing row, which knows how "
            "the day ended. Legitimate for structural research; NOT a live "
            "filter.")
    elif no_trade_snapshot == NO_TRADE_FINAL:
        warnings.append(
            "No-trade days are labelled by the day's final snapshot, so the "
            "trade-probability denominator is partly retrospective. The trade "
            "pools remain point-in-time.")

    return {
        "states": list(states),
        "column": column,
        "tag": tag,
        "no_trade_snapshot": no_trade_snapshot,
        "decision_time": decision_time,
        "counts": counts,
        "probs": probs,
        "empty_rows": empty_rows,
        "n_days": pops["n_days"],
        "n_trade_days": pops["n_trade_days"],
        "p_trade": p_trade,
        "pools": pops["pools"],
        "pool_size": {s: int(len(pops["pools"][s])) for s in states},
        "pool_profile": pool_profiles(trades, pops["pools"], states),
        "start_probs": start_probs,
        "matrix_days": int(len(matrix_labels)),
        "pool_days": int(len(pool_labels)),
        "matrix_range": (matrix_start, matrix_end),
        "pool_range": (pool_start, pool_end),
        "min_trades_per_state": int(min_trades_per_state),
        "multi_trade_days": n_multi,
        "warnings": warnings,
    }


# ── simulation ───────────────────────────────────────────────────────────────

def simulate(trades: pd.DataFrame, sizer_module, sizer_params: dict,
             model: dict, *, horizon: int, n_paths: int, seed: int,
             cost_ctx: dict | None = None,
             start_state: str | None = None) -> np.ndarray:
    """
    (n_paths, horizon + 1) equity matrix, indexed by TRADING DAY — not by trade.

    Each simulated day: the chain moves, then the day's regime decides whether a
    trade happens at all. A no-trade day contributes zero P&L but still advances
    the chain — that is precisely what preserves the clustering a bootstrap
    destroys, so it is never skipped.

    Day 1 IS the seeded state (rather than a draw out of it), so a future
    per-regime start mode means what it says: "given today opens in high vol".

    Requires a sizer with mc_prepare/mc_size. Sizing, commissions and slippage
    reuse the bootstrap's step machinery verbatim, so costs feed back into
    equity and therefore into the next day's size.
    """
    if not (hasattr(sizer_module, "mc_prepare") and hasattr(sizer_module, "mc_size")):
        raise ValueError(
            "Regime-switching MC requires a sizer with mc_prepare/mc_size "
            "hooks (fixed, kelly and risk_based all qualify).")
    if len(trades) == 0:
        raise ValueError("Trade file is empty.")
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1.")
    if horizon < 1:
        raise ValueError("Simulation horizon must be at least 1 day.")

    states = model["states"]
    n_states = len(states)
    if n_states == 0:
        raise ValueError("The regime column declares no states.")

    # Absorbing fallback for never-left states: hold rather than invent a
    # uniform transition. estimate() already warned about it.
    probs = np.array(model["probs"], dtype=float).copy()
    for i in np.nonzero(np.array(model["empty_rows"], dtype=bool))[0]:
        probs[i, i] = 1.0
    cum = probs.cumsum(axis=1)

    p_trade = np.array([model["p_trade"][s] for s in states], dtype=float)
    pools = [np.asarray(model["pools"][s], dtype=np.int64) for s in states]

    start = np.array(model["start_probs"], dtype=float)
    if start_state is not None:
        start = np.zeros(n_states)
        start[states.index(start_state)] = 1.0
    if start.sum() <= 0:
        raise ValueError("No days in the matrix window carry a usable regime "
                         "state — nothing to start the chain from.")
    start = start / start.sum()

    dollars_per_tick = float(sizer_params["dollars_per_tick"])
    account_size = float(sizer_params["account_size"])
    state_hook = sizer_module.mc_prepare(trades, sizer_params)
    per_trade = state_hook.get("per_trade", {})

    ticks = trades["ticks"].to_numpy(dtype=float)
    pnl_per_contract = ticks * dollars_per_tick
    costs_on = bool(cost_ctx and cost_ctx.get("enabled"))
    if costs_on:
        n_slip = cost_ctx["n"]
        # Sign-based on GROSS ticks: n for winners/scratch, 2n for losers —
        # mirrors the bootstrap engine exactly.
        slip_ticks = np.where(ticks > 0, n_slip,
                              np.where(ticks < 0, 2 * n_slip, n_slip)).astype(float)

    rng = np.random.default_rng(seed)
    equity = np.full(n_paths, account_size, dtype=float)
    out = np.empty((n_paths, horizon + 1), dtype=float)
    out[:, 0] = account_size

    current = np.searchsorted(start.cumsum(), rng.random(n_paths), side="right")
    current = np.clip(current, 0, n_states - 1)

    for k in range(horizon):
        if k > 0:                                    # day 1 keeps the seed
            u = rng.random(n_paths)
            current = (u[:, None] >= cum[current]).sum(axis=1)
            current = np.clip(current, 0, n_states - 1)

        trade_mask = rng.random(n_paths) < p_trade[current]
        draw = np.zeros(n_paths, dtype=np.int64)
        for s in range(n_states):
            pool = pools[s]
            if pool.size == 0:
                continue
            take = trade_mask & (current == s)
            n_take = int(take.sum())
            if n_take:
                draw[take] = pool[rng.integers(0, pool.size, size=n_take)]

        step = {name: arr[draw] for name, arr in per_trade.items()}
        # Zeroing size (not P&L) makes a no-trade day free of commission and
        # slippage too — _step_commission and the slippage term are both
        # proportional to size.
        size = sizer_module.mc_size(equity, step, state_hook, sizer_params)
        size = np.where(trade_mask, size, 0.0)

        net = size * pnl_per_contract[draw]
        if costs_on:
            net = net - _step_commission(size, cost_ctx) \
                      - slip_ticks[draw] * dollars_per_tick * size

        equity = equity + net
        out[:, k + 1] = equity

    return out


def resolve_horizon(model: dict, mode: str, fixed_days: int) -> int:
    """Trading days per simulated path. HORIZON_MATCH mirrors the POOL window —
    the days actually being simulated — so a path is directly comparable to the
    real equity curve even when the matrix was estimated over a wider span."""
    if mode == HORIZON_FIXED:
        return int(fixed_days)
    return int(model["pool_days"])
