"""
monte_carlo/prop_firm.py

Prop-firm (FundedNext-style) Monte Carlo as a stopping-time simulation. Each
path trades until it passes, breaches, or hits the horizon. Produces three
simulations, each with its own fan chart + stats:

    Sim 1 — Challenge:  pass vs fail.
    Sim 2 — Payout:     conditional on a fresh funded account.
    Sim 3 — Combined:   end-to-end (challenge -> funded -> paid), by composition.

Built on base.run_prop_paths (the per-step active-mask equity engine with daily
accumulators, stacked size clamps, trailing EOD floor, and enabled-only rules).

CONFIRMED design choices (see the engineering spec):
  - EOD max-loss floor is TRAILING from the highest end-of-day balance (ratchets
    up only).
  - Funded accounts reset to account_size — challenge profit is NOT carried.
  - Maximum Withdrawal caps the per-payout dollar amount:
    realized payout = min(profit_at_eligibility, maximum_withdrawal).
  - Breach checks are on closed-trade equity (optimistic; P(pass)/P(payout) are
    upper bounds — the UI surfaces this).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from base import run_prop_paths


# Marker: tells views/monte_carlo.py to render the dedicated prop-firm UI
# (per-rule checkbox + value groups, three charts) instead of the generic flow.
PROP_FIRM = True

# Structured defaults. The view reads these to seed its widgets; each rule is an
# {enabled, value} pair so it can be toggled fully inert. (Numbers are sensible
# placeholders — set them to your firm's actual terms.)
PARAMS = {
    # General
    "n_paths":    5000,
    "max_trades": 500,
    "seed":       42,

    # Challenge (passing) ruleset
    "profit_target":          {"enabled": True,  "value": 6000.0},
    "challenge_max_loss_eod": {"enabled": True,  "value": 4000.0},
    "challenge_daily_loss":   {"enabled": True,  "value": 2000.0},
    "challenge_consistency":  {"enabled": False, "value": 0.30},
    "challenge_contract_limit": {"enabled": True, "value": 3.0},

    # Payout (funded) ruleset
    "targeted_payout":      {"enabled": True,  "value": 4000.0},
    "payout_max_loss_eod":  {"enabled": True,  "value": 4000.0},
    "payout_daily_loss":    {"enabled": True,  "value": 2000.0},
    "payout_consistency":   {"enabled": True,  "value": 0.30},
    "payout_contract_limit": {"enabled": True, "value": 3.0},
    "maximum_withdrawal":   {"enabled": False, "value": 4000.0},
}


# ---------------------------------------------------------------------------
# Stats helpers (pure numpy — the view formats)
# ---------------------------------------------------------------------------

def _pctiles(arr: np.ndarray) -> dict | None:
    """median / 25th / 75th / 95th percentile of a 1-D array, or None if empty."""
    if arr is None or len(arr) == 0:
        return None
    return {
        "median": float(np.median(arr)),
        "p25":    float(np.percentile(arr, 25)),
        "p75":    float(np.percentile(arr, 75)),
        "p95":    float(np.percentile(arr, 95)),
    }


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity))


def _outcome_masks(sim: dict):
    oc = sim["outcome"]
    passed = oc == "pass"
    failed = np.char.startswith(oc.astype(str), "fail")
    unresolved = oc == "unresolved"
    return passed, failed, unresolved


def _challenge_stats(sim: dict) -> dict:
    passed, failed, unresolved = _outcome_masks(sim)
    n = len(sim["outcome"])
    eqm = sim["equity_matrix"]

    pass_dd = None
    if passed.any():
        pass_dd = max(_max_drawdown(eqm[i]) for i in np.nonzero(passed)[0])

    return {
        "n_paths":     n,
        "p_pass":      float(passed.mean()),
        "trades_to_pass": _pctiles(sim["stop_step"][passed]),
        "failure_breakdown": {
            "max_loss":   float((sim["outcome"] == "fail:max_loss").mean()),
            "unresolved": float(unresolved.mean()),
        },
        "consistency_hold_rate": float(sim["held_by_consistency"].mean()),
        "median_final_equity_passers": (
            float(np.median(sim["final_equity"][passed])) if passed.any() else None),
        "worst_peak_to_trough_passers": pass_dd,
    }


def _payout_stats(sim: dict) -> dict:
    passed, failed, unresolved = _outcome_masks(sim)
    n = len(sim["outcome"])
    held = sim["held_by_consistency"]

    return {
        "n_paths":   n,
        "p_payout":  float(passed.mean()),
        "trades_to_payout": _pctiles(sim["stop_step"][passed]),
        "breach_rate": float((sim["outcome"] == "fail:max_loss").mean()),
        "breach_breakdown": {
            "max_loss":   float((sim["outcome"] == "fail:max_loss").mean()),
            "unresolved": float(unresolved.mean()),
        },
        # The true cost of the consistency rule:
        "held_then_paid":     float((held & passed).mean()),
        "held_then_breached": float((held & failed).mean()),
        "consistency_hold_rate": float(held.mean()),
    }


def _combined_stats(sim1: dict, sim2: dict, max_withdrawal, targeted_payout, seed: int) -> dict:
    p1_pass, _, _ = _outcome_masks(sim1)
    p2_pass, _, _ = _outcome_masks(sim2)
    p_pass   = float(p1_pass.mean())
    p_payout = float(p2_pass.mean())
    p_paid   = p_pass * p_payout

    # Total trades to payout = trades-to-pass (challenge) + trades-to-payout
    # (funded). Funded is independent of how the challenge passed, so pair random
    # passers with random payers and sum to get the distribution of the sum.
    total = None
    pass_steps  = sim1["stop_step"][p1_pass]
    pay_steps   = sim2["stop_step"][p2_pass]
    if len(pass_steps) and len(pay_steps):
        rng = np.random.default_rng(seed)
        k = 20000
        sums = (rng.choice(pass_steps, size=k) + rng.choice(pay_steps, size=k))
        total = _pctiles(sums)

    # Expected payout value — the economic bottom line.
    cap = max_withdrawal if (max_withdrawal is not None) else np.inf
    realized = min(targeted_payout, cap)
    epv = p_paid * realized

    return {
        "p_pass":             p_pass,
        "p_payout_given_funded": p_payout,
        "p_paid":             p_paid,
        "total_trades_to_payout": total,
        "funnel": (p_pass, p_payout, p_paid),
        "expected_payout_value": float(epv),
        "realized_payout_per_paid": float(realized),
    }


# ---------------------------------------------------------------------------
# Sim 3 combined-curve builder (illustrative sample for the chart only)
# ---------------------------------------------------------------------------

def _build_combined_curves(sim1: dict, sim2: dict, max_sample: int, seed: int):
    """
    Concatenate challenge-pass equity with a fresh funded curve for a sample of
    passing paths. Returns (matrix, reset_x, paid_mask) for the Sim-3 chart, or
    (empty, empty, empty) if no path passed. Chart-only — stats use full arrays.
    """
    passers = np.nonzero(sim1["outcome"] == "pass")[0]
    if len(passers) == 0:
        return np.empty((0, 1)), np.empty(0, dtype=int), np.empty(0, dtype=bool)

    rng = np.random.default_rng(seed)
    if len(passers) > max_sample:
        passers = rng.choice(passers, size=max_sample, replace=False)

    n_funded = sim2["equity_matrix"].shape[0]
    funded_idx = rng.integers(0, n_funded, size=len(passers))
    funded_passed = sim2["outcome"] == "pass"

    curves, resets, paid = [], [], []
    for i, j in zip(passers, funded_idx):
        s1 = int(sim1["stop_step"][i])
        s2 = int(sim2["stop_step"][j])
        c1 = sim1["equity_matrix"][i, : s1 + 1]
        c2 = sim2["equity_matrix"][j, : s2 + 1]
        curves.append(np.concatenate([c1, c2]))
        resets.append(s1)               # x-index where the funded phase begins
        paid.append(bool(funded_passed[j]))

    width = max(len(c) for c in curves)
    mat = np.empty((len(curves), width))
    for r, c in enumerate(curves):
        mat[r, : len(c)] = c
        mat[r, len(c):]  = c[-1]        # forward-fill the flat tail
    return mat, np.array(resets, dtype=int), np.array(paid, dtype=bool)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(trades: pd.DataFrame, sizer_module, sizer_params: dict, params: dict) -> dict:
    """
    Run the three prop-firm simulations and return a result block per sim.

    params (resolved by the view) carries: n_paths, max_trades, seed,
    account_size, increment, cost_ctx, and the challenge/payout rule dicts
    (each an {enabled, value} pair), plus targeted_payout / maximum_withdrawal.
    """
    account_size = float(params["account_size"])
    n_paths      = int(params.get("n_paths", PARAMS["n_paths"]))
    max_trades   = int(params.get("max_trades", PARAMS["max_trades"]))
    seed         = int(params.get("seed", PARAMS["seed"]))
    increment    = params.get("increment") or 1.0
    cost_ctx     = params.get("cost_ctx")

    warnings = []

    base_cfg = {
        "n_paths":      n_paths,
        "max_trades":   max_trades,
        "start_equity": account_size,
        "increment":    increment,
    }

    challenge_cfg = {
        **base_cfg,
        "target":         params["profit_target"],
        "max_loss_eod":   params["challenge_max_loss_eod"],
        "daily_loss":     params["challenge_daily_loss"],
        "consistency":    params["challenge_consistency"],
        "contract_limit": params["challenge_contract_limit"],
    }
    payout_cfg = {
        **base_cfg,
        "target":         params["targeted_payout"],
        "max_loss_eod":   params["payout_max_loss_eod"],
        "daily_loss":     params["payout_daily_loss"],
        "consistency":    params["payout_consistency"],
        "contract_limit": params["payout_contract_limit"],
    }

    # Warn on configurations that make outcomes degenerate.
    if not challenge_cfg["max_loss_eod"]["enabled"]:
        warnings.append("Challenge: no active loss limit — paths can never fail.")
    if not challenge_cfg["target"]["enabled"]:
        warnings.append("Challenge: no profit target — paths can never pass.")
    if not payout_cfg["max_loss_eod"]["enabled"]:
        warnings.append("Payout: no active loss limit — funded paths can never breach.")
    if not payout_cfg["target"]["enabled"]:
        warnings.append("Payout: no targeted payout — funded paths can never get paid.")

    # Sim 1 — challenge; Sim 2 — fresh funded (different seed for independence).
    sim1 = run_prop_paths(trades, sizer_module, sizer_params, challenge_cfg, cost_ctx, seed)
    sim2 = run_prop_paths(trades, sizer_module, sizer_params, payout_cfg, cost_ctx, seed + 1)

    mw_rule = params.get("maximum_withdrawal") or {}
    max_withdrawal = mw_rule.get("value") if mw_rule.get("enabled") else None
    targeted_payout = float(params["targeted_payout"]["value"])

    comb_mat, comb_reset, comb_paid = _build_combined_curves(sim1, sim2, max_sample=400, seed=seed + 2)

    return {
        "is_prop_firm": True,
        "warnings": warnings,
        "account_size": account_size,
        "sim1": {
            "title": "Sim 1 — Challenge",
            "equity_matrix": sim1["equity_matrix"],
            "start_equity": account_size,
            "target": challenge_cfg["target"]["value"] if challenge_cfg["target"]["enabled"] else None,
            "floor_offset": challenge_cfg["max_loss_eod"]["value"] if challenge_cfg["max_loss_eod"]["enabled"] else None,
            "stats": _challenge_stats(sim1),
        },
        "sim2": {
            "title": "Sim 2 — Payout (fresh funded)",
            "equity_matrix": sim2["equity_matrix"],
            "start_equity": account_size,
            "target": payout_cfg["target"]["value"] if payout_cfg["target"]["enabled"] else None,
            "floor_offset": payout_cfg["max_loss_eod"]["value"] if payout_cfg["max_loss_eod"]["enabled"] else None,
            "stats": _payout_stats(sim2),
        },
        "sim3": {
            "title": "Sim 3 — Combined (end-to-end)",
            "equity_matrix": comb_mat,
            "reset_x": comb_reset,
            "paid_mask": comb_paid,
            "start_equity": account_size,
            "stats": _combined_stats(sim1, sim2, max_withdrawal, targeted_payout, seed + 3),
        },
    }
