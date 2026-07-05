"""
monte_carlo/base.py

Shared utilities for Monte Carlo simulation scripts.
Import from here to avoid duplicating logic across mc types.
"""

import numpy as np
import pandas as pd


def build_equity_curve(
    trades:       pd.DataFrame,
    sizer_module,
    sizer_params: dict,
) -> np.ndarray:
    """
    Apply sizer to a trades DataFrame and return the equity curve
    as a numpy array starting from account_size.

    Returns shape: (n_trades + 1,)  — index 0 is the starting balance.
    """
    sized       = sizer_module.apply(trades, sizer_params)
    account     = sizer_params["account_size"]
    equity      = sized["equity"].to_numpy(dtype=np.float64)

    # Prepend the starting balance so the curve starts at account_size
    return np.concatenate([[account], equity])


def run_paths(
    trades:       pd.DataFrame,
    sizer_module,
    sizer_params: dict,
    resample_fn,           # callable(trades, rng, path_index) -> pd.DataFrame
    n_paths:      int,
    seed:         int,
) -> np.ndarray:
    """
    Run n_paths simulations.

    resample_fn receives (trades, rng, path_index) and must return a
    resampled/reshuffled trades DataFrame with the same columns.

    Returns equity_matrix of shape (n_paths, n_trades + 1).
    """
    rng    = np.random.default_rng(seed)
    curves = []

    for i in range(n_paths):
        resampled = resample_fn(trades, rng, i)
        curve     = build_equity_curve(resampled, sizer_module, sizer_params)
        curves.append(curve)

    # Stack — all curves must be the same length (n_trades + 1)
    return np.vstack(curves)


def _step_commission(size: np.ndarray, cost_ctx: dict) -> np.ndarray:
    """
    Per-step, per-path commission ($), vectorized over paths.

    Amount is LIVE — it depends on each path's `size` this step — so it cannot be
    precomputed as a flat per-trade value. Rates ride in via cost_ctx. The full/
    micro decomposition and the ×2 (entry+exit) mirror the analytics cost model.
    size == 0 ⇒ commission 0 automatically.
    """
    full_comm  = cost_ctx.get("full_comm")
    if full_comm is None:
        # Asset has no commission rate → bill 0 (caller surfaces the warning).
        return np.zeros(size.shape, dtype=float)

    full_count = np.floor(size)
    if cost_ctx.get("microable"):
        micro_comm = cost_ctx["micro_comm"]
        # round(), never int() — avoid float-dust truncation (e.g. 0.3*10 = 2.9999).
        micro_count = np.round((size - full_count) * 10)
        return (full_count * full_comm + micro_count * micro_comm) * 2
    else:
        return np.round(size) * full_comm * 2


def run_paths_vectorized(
    trades:       pd.DataFrame,
    sizer_module,
    sizer_params: dict,
    idx:          np.ndarray,    # (n_paths, n_sample) bootstrap draw matrix
    dollars_per_tick: float,
    cost_ctx:     dict,          # {"enabled", "n", "full_comm", "micro_comm", "microable"}
) -> np.ndarray:
    """
    Vectorized Monte Carlo engine.

    Keeps `equity` as an (n_paths,) vector and loops over trade STEPS (n_sample),
    sizing all paths at once per step via the sizer's mc_prepare/mc_size hooks.
    Costs (commissions + slippage) are deducted per step and feed back into
    equity → into the next step's sizing.

    Returns equity_matrix of shape (n_paths, n_sample + 1); column 0 = account_size.
    """
    n_paths, n_sample = idx.shape

    state     = sizer_module.mc_prepare(trades, sizer_params)
    per_trade = state.get("per_trade", {})

    # Path-invariant per-original-trade arrays.
    ticks            = trades["ticks"].to_numpy(dtype=float)
    pnl_per_contract = ticks * dollars_per_tick

    costs_on = bool(cost_ctx and cost_ctx.get("enabled"))
    if costs_on:
        n = cost_ctx["n"]
        # Sign-based on GROSS ticks: n for winners/scratch, 2n for losers.
        # Correct because size > 0 ⇒ sign(gross) = sign(ticks); size == 0 zeroes the cost.
        slip_ticks_arr = np.where(ticks > 0, n, np.where(ticks < 0, 2 * n, n)).astype(float)

    account_size = sizer_params["account_size"]
    equity = np.full(n_paths, account_size, dtype=float)
    out    = np.empty((n_paths, n_sample + 1), dtype=float)
    out[:, 0] = account_size

    for k in range(n_sample):
        col  = idx[:, k]
        step = {name: arr[col] for name, arr in per_trade.items()}
        size = sizer_module.mc_size(equity, step, state, sizer_params)

        gross = size * pnl_per_contract[col]

        if costs_on:
            commission = _step_commission(size, cost_ctx)
            slippage   = slip_ticks_arr[col] * dollars_per_tick * size
            net = gross - commission - slippage
        else:
            net = gross

        equity = equity + net
        out[:, k + 1] = equity

    return out


# ===========================================================================
# Prop-firm stopping-time engine
# ===========================================================================
#
# Builds on run_paths_vectorized: same per-step equity vector + cost feedback,
# plus a per-path active/stopped mask, day-scoped accumulators, stacked size
# clamps, and an enabled-only breach / pass / consistency rule set.
#
# Day structure (the cheap over-build): the daily rules are day-scoped. Source
# trades are 1 trade/day, so today every step IS a day boundary and the daily
# accumulators (day_pnl reset, max_single_day_profit, peak end-of-day equity)
# collapse to per-trade. They are still maintained with full day-rollover
# structure so a future multi-trade-per-day strategy works with no breach-logic
# changes. Do NOT collapse this to pure per-trade logic.
#
# Breach checks are on CLOSED-trade equity — intra-trade excursions are
# invisible, so a trade that dipped below the floor and recovered to a green
# close is not counted as a breach. This makes every P(pass)/P(payout) an upper
# bound; the UI surfaces this as a caption.


def _rule(config: dict, name: str):
    """(enabled, value) for a {enabled, value} rule pair; (False, None) if absent."""
    r = config.get(name) or {}
    return bool(r.get("enabled", False)), r.get("value")


def run_prop_paths(
    trades:       pd.DataFrame,
    sizer_module,
    sizer_params: dict,
    config:       dict,     # resolved ruleset + start_equity, n_paths, max_trades, increment
    cost_ctx:     dict,
    seed:         int,
) -> dict:
    """
    Stopping-time Monte Carlo for a prop-firm challenge/funded account.

    Each path bootstraps trades (with replacement) and trades until it PASSES
    (target hit + consistency satisfied), FAILS (trailing EOD max-loss breach),
    or hits the `max_trades` horizon (unresolved). Only enabled rules are
    evaluated — a disabled rule is fully inert.

    config keys
    -----------
    n_paths, max_trades, increment, start_equity : scalars
    ticks_per_point                              : float | None (derived if None)
    target / max_loss_eod / daily_loss / consistency / contract_limit
                                                 : {"enabled": bool, "value": float}
        target       value = profit ($) above start_equity to pass
        max_loss_eod value = trailing drawdown ($) below the peak EOD balance,
                             locking at the starting balance (floor never exceeds start_eq)
        static_loss  value = fixed drawdown ($) below the starting balance; a floor
                             that never trails (breach when equity <= start - value)
        daily_loss   value = per-day loss budget ($, positive) — risk cap only
        consistency  value = max fraction of total profit allowed from one day
        contract_limit value = hard cap on size (full-contract units)

    Returns a dict with equity_matrix (forward-filled, (n_paths, max_trades+1)),
    per-path outcome / stop_step / final_equity, held_by_consistency, and the
    accumulators the stats need.
    """
    if not (hasattr(sizer_module, "mc_prepare") and hasattr(sizer_module, "mc_size")):
        raise ValueError(
            "Prop-firm MC requires a sizer with mc_prepare/mc_size hooks "
            "(fixed, kelly, risk_based all qualify)."
        )

    n_paths    = int(config["n_paths"])
    max_trades = int(config["max_trades"])
    start_eq   = float(config["start_equity"])
    increment  = float(config.get("increment") or 0.1)
    dollars_per_tick = float(sizer_params["dollars_per_tick"])

    target_on, target_v = _rule(config, "target")
    eod_on,    eod_v     = _rule(config, "max_loss_eod")
    static_on, static_v  = _rule(config, "static_loss")
    daily_on,  daily_v   = _rule(config, "daily_loss")
    cons_on,   cons_v    = _rule(config, "consistency")
    climit_on, climit_v  = _rule(config, "contract_limit")

    # Static loss floor: a FIXED threshold that never trails. Breach when closed
    # equity <= start_equity - static_v. Independent of the trailing EOD floor;
    # either or both may be enabled, and the higher (nearer) floor binds.
    static_floor = (start_eq - static_v) if static_on else None

    # Win cap tied to the consistency rule: model a trader who stops/reduces once
    # the day's profit hits the consistency daily threshold (cons% × target), so a
    # single day never exceeds it and the consistency recalculation is never
    # triggered. Inert unless both consistency and target are enabled.
    win_cap = (cons_v * target_v) if (config.get("cap_wins_to_consistency")
                                      and cons_on and target_on) else None

    # --- per-original-trade arrays ---
    ticks = trades["ticks"].to_numpy(dtype=float)
    n_trades = len(ticks)
    pnl_per_contract = ticks * dollars_per_tick

    # risk per contract ($) at the drawn trade's full stop distance — for the EOD
    # risk cap. Needs entry_price + sl; ticks_per_point from config or derived.
    have_stops = ("entry_price" in trades.columns) and ("sl" in trades.columns)
    risk_dollars_arr = None
    if have_stops:
        tpp = config.get("ticks_per_point")
        if tpp is None:
            nz = trades[trades["pnl_points"] != 0]
            tpp = float((nz["ticks"] / nz["pnl_points"]).iloc[0]) if not nz.empty else 0.0
        risk_dollars_arr = (
            np.abs(trades["entry_price"].to_numpy(float) - trades["sl"].to_numpy(float))
            * tpp * dollars_per_tick
        )

    costs_on = bool(cost_ctx and cost_ctx.get("enabled"))
    if costs_on:
        n_slip = cost_ctx["n"]
        slip_ticks_arr = np.where(ticks > 0, n_slip, np.where(ticks < 0, 2 * n_slip, n_slip)).astype(float)

    state     = sizer_module.mc_prepare(trades, sizer_params)
    per_trade = state.get("per_trade", {})

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_trades, size=(n_paths, max_trades))

    # --- per-path state ---
    equity         = np.full(n_paths, start_eq, dtype=float)
    active         = np.ones(n_paths, dtype=bool)
    peak_eod       = np.full(n_paths, start_eq, dtype=float)   # trailing high of EOD balance
    day_pnl        = np.zeros(n_paths, dtype=float)
    max_day_profit = np.zeros(n_paths, dtype=float)
    outcome        = np.array(["unresolved"] * n_paths, dtype=object)
    stop_step      = np.full(n_paths, max_trades, dtype=int)
    held_ever      = np.zeros(n_paths, dtype=bool)             # reached target but held by consistency

    out = np.empty((n_paths, max_trades + 1), dtype=float)
    out[:, 0] = start_eq

    for k in range(max_trades):
        out[:, k + 1] = out[:, k]                  # carry forward stopped paths
        if not active.any():
            out[:, k + 1:] = out[:, k:k + 1]
            break

        a     = active
        a_idx = np.nonzero(a)[0]
        col   = idx[a, k]

        # --- day rollover (today: every step is a new day) ---
        # Finalize the PRIOR day for active paths: ratchet the peak EOD balance
        # (trailing floor), record the prior day's profit, reset the accumulator.
        peak_eod[a]       = np.maximum(peak_eod[a], equity[a])
        max_day_profit[a] = np.maximum(max_day_profit[a], day_pnl[a])
        day_pnl[a]        = 0.0

        # --- size via the vectorized sizer hook (active subset) ---
        step = {name: arr[col] for name, arr in per_trade.items()}
        size = sizer_module.mc_size(equity[a], step, state, sizer_params)

        # --- stacked clamps ---
        if climit_on:
            size = np.minimum(size, climit_v)

        if (eod_on or static_on) and risk_dollars_arr is not None:
            rpc = risk_dollars_arr[col]                        # $ risk per contract at full stop
            # Bind on the NEAREST enabled loss floor. Trailing EOD floor LOCKS at
            # the starting balance (peak - limit, capped at start_eq); the static
            # floor is fixed. remaining = distance to whichever floor is higher.
            binding = np.full(equity[a].shape, -np.inf)
            if eod_on:
                binding = np.maximum(binding, np.minimum(peak_eod[a] - eod_v, start_eq))
            if static_on:
                binding = np.maximum(binding, static_floor)
            remaining = equity[a] - binding
            if daily_on:
                remaining = np.minimum(remaining, daily_v)     # daily budget (fresh each day)
            remaining = np.maximum(remaining, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                cap = np.floor(np.where(rpc > 0, remaining / rpc, np.inf) / increment) * increment
            cap = np.round(cap, 1)
            size = np.where(rpc > 0, np.minimum(size, cap), size)   # skip cap when no stop distance

        size = np.maximum(size, 0.0)

        # --- P&L + costs, fed back into equity before the breach check ---
        gross = size * pnl_per_contract[col]
        if costs_on:
            commission = _step_commission(size, cost_ctx)
            slippage   = slip_ticks_arr[col] * dollars_per_tick * size
            net = gross - commission - slippage
        else:
            net = gross

        # Cap the day's net so it can't exceed the consistency threshold. Only
        # trims wins (remaining_day > 0); losses pass through unchanged. Uses the
        # day's running profit so it also holds for future multi-trade days.
        if win_cap is not None:
            net = np.minimum(net, win_cap - day_pnl[a])

        equity[a]  = equity[a] + net
        day_pnl[a] = day_pnl[a] + net
        out[a, k + 1] = equity[a]

        # --- breach / pass / consistency on closed-trade equity ---
        eq_a         = equity[a]
        total_profit = eq_a - start_eq
        # consistency uses the largest single-day profit INCLUDING today so far
        eff_max_day  = np.maximum(max_day_profit[a], day_pnl[a])

        # Loss floors (−inf = rule disabled, so `eq <= −inf` is always False):
        #   EOD  — trails the peak EOD balance up, locks at the starting balance.
        #   static — fixed at start_eq − static_v.
        neg_inf = np.full(eq_a.shape, -np.inf)
        eod_floor_v    = np.minimum(peak_eod[a] - eod_v, start_eq) if eod_on else neg_inf
        static_floor_v = np.full(eq_a.shape, static_floor) if static_on else neg_inf
        eod_breach    = eq_a <= eod_floor_v
        static_breach = eq_a <= static_floor_v
        breached      = eod_breach | static_breach

        if target_on:
            hit = eq_a >= (start_eq + target_v)
            if cons_on:
                cons_ok = eff_max_day <= cons_v * np.maximum(total_profit, 1e-9)
            else:
                cons_ok = np.ones(eq_a.shape, dtype=bool)
            passed = hit & cons_ok
            held   = hit & (~cons_ok)
        else:
            passed = np.zeros(eq_a.shape, dtype=bool)
            held   = np.zeros(eq_a.shape, dtype=bool)

        # Breach resolves first (a path can't both pass and breach on one close).
        fail_mask = breached & (~passed)
        # Attribute to the binding (higher/nearer) floor when both are crossed.
        static_fail  = fail_mask & static_breach & (static_floor_v >= eod_floor_v)
        maxloss_fail = fail_mask & (~static_fail)

        held_ever[a_idx[held]]        = True
        outcome[a_idx[maxloss_fail]]  = "fail:max_loss"
        outcome[a_idx[static_fail]]   = "fail:static_loss"
        outcome[a_idx[passed]]        = "pass"
        stop_step[a_idx[fail_mask]]   = k + 1
        stop_step[a_idx[passed]]      = k + 1

        newly_stopped = fail_mask | passed
        active[a_idx[newly_stopped]] = False

    return {
        "equity_matrix":       out,
        "outcome":             outcome,
        "stop_step":           stop_step,
        "final_equity":        out[:, -1].copy(),
        "held_by_consistency": held_ever,
        "max_day_profit":      max_day_profit,
        "start_equity":        start_eq,
        "target":              target_v if target_on else None,
        "max_trades":          max_trades,
    }