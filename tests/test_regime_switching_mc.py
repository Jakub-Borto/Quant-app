"""Tests for the regime-switching Monte Carlo engine
(modules/monte_carlo/backend/regime_switching.py).

The method exists to reproduce the clustering a plain bootstrap destroys, so
the tests are aimed at the things that would silently destroy it again:

  - the matrix estimated off TRADE days instead of ALL days (it would
    understate persistence — see test_matrix_uses_all_days, which pins the
    exact fixture from the build plan);
  - a chain break quietly manufacturing a transition across a data gap;
  - matrix / trade-probability / pools drifting onto different snapshots;
  - a no-trade day failing to advance the chain, which would collapse the
    whole model back to a bootstrap.

All synthetic — no Qt, no real data, no regime run on disk.
"""

import numpy as np
import pandas as pd
import pytest

from modules.common.backend.regime_join import UNKNOWN
from modules.monte_carlo.backend import regime_switching as rs

NY = "America/New_York"

# The build plan's worked example (§6.1): all-days counts H->H three times,
# a trade-only walk would see it once.
SEQUENCE = ["H", "H", "H", "N", "N", "L", "L", "N", "H", "H"]
TRADED = [True, False, True, True, False, False, True, True, False, True]
STATES = ["H", "N", "L"]


# ── fixtures ─────────────────────────────────────────────────────────────────

def make_regime_day(date: str, states, *, start="09:30", freq="30min",
                    column="vol_state") -> pd.DataFrame:
    """One day's regime file. `states` is either a single state for the whole
    day or one state per snapshot. is_final marks the last row, as the real
    contract requires."""
    if isinstance(states, str):
        states = [states] * 14
    idx = pd.date_range(pd.Timestamp(f"{date} {start}", tz=NY),
                        periods=len(states), freq=freq)
    return pd.DataFrame(
        {column: list(states),
         "is_final": [False] * (len(states) - 1) + [True]},
        index=idx)


def business_days(n: int, start: str = "2024-01-02") -> list[str]:
    return [d.strftime("%Y-%m-%d")
            for d in pd.bdate_range(start, periods=n)]


def make_fixture(sequence=SEQUENCE, traded=TRADED, dates=None,
                 entry="10:03"):
    """(frames, trades) for a run of consecutive business days, each day flat
    in one state, with trades on the flagged days."""
    dates = dates or business_days(len(sequence))
    frames = {d: make_regime_day(d, s) for d, s in zip(dates, sequence)}
    rows = [{"date": d,
             "entry_time": pd.Timestamp(f"{d} {entry}", tz=NY),
             "ticks": 4.0, "pnl_points": 1.0}
            for d, t in zip(dates, traded) if t]
    return frames, pd.DataFrame(rows)


class StubSizer:
    """Minimal mc_prepare/mc_size sizer: a flat one contract per trade. Keeps
    the simulation tests about the chain, not about position sizing."""

    @staticmethod
    def mc_prepare(trades, params):
        return {"per_trade": {"one": np.ones(len(trades))}}

    @staticmethod
    def mc_size(equity, step, state, params):
        return np.ones(equity.shape, dtype=float)


def sizer_params(**over):
    return {"account_size": 100_000.0, "dollars_per_tick": 12.5, **over}


# ── 1. as-of join direction ──────────────────────────────────────────────────

def test_entry_tag_joins_the_snapshot_at_or_before_entry():
    """10:03 takes the 10:00 row, never the 10:30 one that has not happened."""
    day = "2024-01-02"
    states = ["low"] * 2 + ["high"] * 12          # 09:30, 10:00 low; 10:30+ high
    frames = {day: make_regime_day(day, states)}
    trades = pd.DataFrame([{"date": day,
                            "entry_time": pd.Timestamp(f"{day} 10:03", tz=NY),
                            "ticks": 4.0, "pnl_points": 1.0}])

    _, trade_labels = rs.day_labels(trades, frames, "vol_state",
                                    tag=rs.TAG_ENTRY)
    assert trade_labels.iloc[0] == "low"


def test_trade_before_the_first_snapshot_is_unknown_and_excluded():
    """No snapshot at or before entry -> unknown -> out of every pool, and the
    day contributes no transition."""
    day = "2024-01-02"
    frames = {day: make_regime_day(day, "high")}          # first row 09:30
    trades = pd.DataFrame([{"date": day,
                            "entry_time": pd.Timestamp(f"{day} 09:15", tz=NY),
                            "ticks": 4.0, "pnl_points": 1.0}])

    labels, trade_labels = rs.day_labels(trades, frames, "vol_state",
                                         tag=rs.TAG_ENTRY)
    assert trade_labels.iloc[0] == UNKNOWN
    # the trade day's own label follows its trade, so the day is unknown too
    assert labels.iloc[0] == UNKNOWN

    pops = rs.trade_populations(labels, trades, trade_labels, ["high"])
    assert pops["pools"]["high"].size == 0
    assert pops["n_days"]["high"] == 0


# ── 2. the matrix comes from ALL days, not trade days ────────────────────────

def test_matrix_uses_all_days_not_trade_days():
    """The plan's worked example. All-days counts H->H three times; a walk over
    trade days only would count it once. Anything but 3 means the flat days —
    and with them the clustering — were dropped."""
    frames, trades = make_fixture()
    labels, trade_labels = rs.day_labels(trades, frames, "vol_state",
                                         tag=rs.TAG_FINAL)
    assert list(labels) == SEQUENCE

    counts = rs.transition_counts(labels, STATES)
    h = STATES.index("H")
    assert counts[h, h] == 3

    # ...and prove the trade-only alternative really would have differed.
    traded_only = labels[labels.index.isin(
        rs._trade_days(trades).to_numpy())]
    assert rs.transition_counts(traded_only, STATES)[h, h] == 1


def test_transition_counts_match_the_full_hand_walk():
    frames, trades = make_fixture()
    labels, _ = rs.day_labels(trades, frames, "vol_state", tag=rs.TAG_FINAL)
    counts = rs.transition_counts(labels, STATES)
    i = {s: STATES.index(s) for s in STATES}
    # H H H N N L L N H H -> HH,HH,HN,NN,NL,LL,LN,NH,HH
    assert counts[i["H"], i["H"]] == 3
    assert counts[i["H"], i["N"]] == 1
    assert counts[i["N"], i["N"]] == 1
    assert counts[i["N"], i["L"]] == 1
    assert counts[i["L"], i["L"]] == 1
    assert counts[i["L"], i["N"]] == 1
    assert counts[i["N"], i["H"]] == 1
    assert counts.sum() == len(SEQUENCE) - 1


# ── 3. chain breaks ──────────────────────────────────────────────────────────

def test_unknown_day_breaks_the_chain_rather_than_joining_its_neighbours():
    """H ? H must NOT become an H->H transition — that would invent exactly the
    persistence the method is supposed to measure."""
    dates = business_days(3)
    frames = {dates[0]: make_regime_day(dates[0], "H"),
              dates[1]: make_regime_day(dates[1], UNKNOWN),
              dates[2]: make_regime_day(dates[2], "H")}
    labels, _ = rs.day_labels(pd.DataFrame(columns=["date", "entry_time"]),
                              frames, "vol_state", tag=rs.TAG_FINAL)
    counts = rs.transition_counts(labels, STATES)
    assert counts.sum() == 0


def test_calendar_gap_does_not_manufacture_a_transition():
    """Two non-contiguous sub-ranges never join across the hole."""
    early, late = business_days(2, "2024-01-02"), business_days(2, "2024-06-03")
    frames = {d: make_regime_day(d, "H") for d in early + late}
    labels, _ = rs.day_labels(pd.DataFrame(columns=["date", "entry_time"]),
                              frames, "vol_state", tag=rs.TAG_FINAL)
    counts = rs.transition_counts(labels, STATES)
    # one transition inside each sub-range, none bridging January to June
    assert counts[STATES.index("H"), STATES.index("H")] == 2


def test_weekend_still_counts_as_consecutive_trading_days():
    """The gap rule must tolerate weekends, or every Friday->Monday transition
    silently vanishes."""
    dates = ["2024-01-05", "2024-01-08"]      # Friday, Monday
    frames = {d: make_regime_day(d, "H") for d in dates}
    labels, _ = rs.day_labels(pd.DataFrame(columns=["date", "entry_time"]),
                              frames, "vol_state", tag=rs.TAG_FINAL)
    assert rs.transition_counts(labels, STATES).sum() == 1


# ── 4. trade-probability ─────────────────────────────────────────────────────

def test_trade_probability_matches_the_hand_count():
    """H: days 1,2,3,9,10 of which 1,3,10 traded -> 3/5.
       N: days 4,5,8   of which 4,8 traded       -> 2/3.
       L: days 6,7     of which 7 traded         -> 1/2."""
    frames, trades = make_fixture()
    labels, trade_labels = rs.day_labels(trades, frames, "vol_state",
                                         tag=rs.TAG_FINAL)
    pops = rs.trade_populations(labels, trades, trade_labels, STATES)

    assert pops["n_days"] == {"H": 5, "N": 3, "L": 2}
    assert pops["n_trade_days"] == {"H": 3, "N": 2, "L": 1}
    assert pops["p_trade"]["H"] == pytest.approx(3 / 5)
    assert pops["p_trade"]["N"] == pytest.approx(2 / 3)
    assert pops["p_trade"]["L"] == pytest.approx(1 / 2)
    # pools hold positional trade indices, and every trade lands in exactly one
    assert sum(len(pops["pools"][s]) for s in STATES) == len(trades)


# ── 5. row normalisation ─────────────────────────────────────────────────────

def test_rows_sum_to_one_or_are_flagged_empty():
    frames, trades = make_fixture()
    labels, _ = rs.day_labels(trades, frames, "vol_state", tag=rs.TAG_FINAL)
    probs, empty = rs.transition_probs(rs.transition_counts(labels, STATES))
    for i, state in enumerate(STATES):
        if empty[i]:
            assert probs[i].sum() == 0.0
        else:
            assert probs[i].sum() == pytest.approx(1.0)


def test_unvisited_state_is_flagged_not_silently_uniform():
    dates = business_days(3)
    frames = {d: make_regime_day(d, "H") for d in dates}
    labels, _ = rs.day_labels(pd.DataFrame(columns=["date", "entry_time"]),
                              frames, "vol_state", tag=rs.TAG_FINAL)
    probs, empty = rs.transition_probs(rs.transition_counts(labels, STATES))
    assert not empty[STATES.index("H")]
    assert empty[STATES.index("N")] and empty[STATES.index("L")]
    assert probs[STATES.index("N")].sum() == 0.0


# ── 7. no-trade days still advance the chain ─────────────────────────────────

def _alternating_model(p_trade_a=0.0, p_trade_b=1.0):
    """A deterministic 2-state chain A<->B. A never trades, B always does, so
    the only way equity can move is if the no-trade A days advanced the chain."""
    return {
        "states": ["A", "B"],
        "probs": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "empty_rows": np.array([False, False]),
        "p_trade": {"A": p_trade_a, "B": p_trade_b},
        "pools": {"A": np.array([], dtype=np.int64),
                  "B": np.array([0], dtype=np.int64)},
        "start_probs": np.array([1.0, 0.0]),      # every path starts in A
        "pool_days": 10,
    }


def test_no_trade_days_advance_the_regime():
    """Horizon 6 starting in a non-trading state: days 2, 4 and 6 are B days, so
    exactly 3 trades land. If a flat day failed to advance the chain, the paths
    would sit in A forever and equity would never move."""
    trades = pd.DataFrame({"ticks": [4.0], "pnl_points": [1.0]})
    eq = rs.simulate(trades, StubSizer, sizer_params(), _alternating_model(),
                     horizon=6, n_paths=8, seed=1)

    assert eq.shape == (8, 7)
    per_trade_dollars = 4.0 * 12.5
    assert np.allclose(eq[:, -1] - eq[:, 0], 3 * per_trade_dollars)
    # day 1 is the seeded (non-trading) state; day 2 is the first trade
    assert np.allclose(eq[:, 1], eq[:, 0])
    assert np.allclose(eq[:, 2] - eq[:, 1], per_trade_dollars)


def test_chain_length_equals_horizon_whatever_the_trade_count():
    trades = pd.DataFrame({"ticks": [4.0], "pnl_points": [1.0]})
    for p in (0.0, 0.5, 1.0):
        eq = rs.simulate(trades, StubSizer, sizer_params(),
                         _alternating_model(p_trade_a=p, p_trade_b=p),
                         horizon=25, n_paths=4, seed=7)
        assert eq.shape == (4, 26)
    # a fully non-trading model is flat, not short
    flat = rs.simulate(trades, StubSizer, sizer_params(),
                       _alternating_model(0.0, 0.0), horizon=25, n_paths=4,
                       seed=7)
    assert np.allclose(flat, 100_000.0)


# ── 8. the two ranges are independent ────────────────────────────────────────

def test_matrix_window_and_horizon_are_separate():
    """A wide matrix window with a short fixed horizon: the matrix spans every
    day, the curves are the requested length. Fusing the two controls is the
    classic error this guards."""
    frames, trades = make_fixture()
    dates = business_days(len(SEQUENCE))
    model = rs.estimate(trades, frames, "vol_state", STATES,
                        tag=rs.TAG_FINAL,
                        matrix_start=dates[0], matrix_end=dates[-1],
                        pool_start=dates[0], pool_end=dates[2])

    assert model["matrix_days"] == 10
    assert model["pool_days"] == 3
    assert model["counts"].sum() == 9          # every consecutive pair

    assert rs.resolve_horizon(model, rs.HORIZON_MATCH, 0) == 3
    assert rs.resolve_horizon(model, rs.HORIZON_FIXED, 4) == 4

    eq = rs.simulate(trades, StubSizer, sizer_params(), model,
                     horizon=4, n_paths=5, seed=3)
    assert eq.shape == (5, 5)


def test_pool_window_bounds_the_pools_but_not_the_matrix():
    frames, trades = make_fixture()
    dates = business_days(len(SEQUENCE))
    narrow = rs.estimate(trades, frames, "vol_state", STATES, tag=rs.TAG_FINAL,
                         matrix_start=dates[0], matrix_end=dates[-1],
                         pool_start=dates[0], pool_end=dates[2])
    wide = rs.estimate(trades, frames, "vol_state", STATES, tag=rs.TAG_FINAL,
                       matrix_start=dates[0], matrix_end=dates[-1],
                       pool_start=dates[0], pool_end=dates[-1])

    assert np.array_equal(narrow["counts"], wide["counts"])
    assert sum(narrow["pool_size"].values()) < sum(wide["pool_size"].values())


# ── 9. one tag drives all three estimates ────────────────────────────────────

def test_switching_the_tag_moves_matrix_probability_and_pools_together():
    """Days that open low and close high: entry-tagged trades are 'low', the
    final row is 'high'. Every downstream estimate must move as one — a matrix
    on one snapshot and pools on another is incoherent and invisible."""
    dates = business_days(6)
    states = ["low"] * 3 + ["high"] * 11        # 09:30-10:30 low, then high
    frames = {d: make_regime_day(d, states) for d in dates}
    trades = pd.DataFrame([{"date": d,
                            "entry_time": pd.Timestamp(f"{d} 10:03", tz=NY),
                            "ticks": 4.0, "pnl_points": 1.0}
                           for d in dates[:4]])
    both = ["low", "high"]

    entry = rs.estimate(trades, frames, "vol_state", both, tag=rs.TAG_ENTRY)
    final = rs.estimate(trades, frames, "vol_state", both, tag=rs.TAG_FINAL)

    # entry: every trade day is 'low'; the two flat days fall back to the
    # final snapshot ('high'). final: every day is 'high'.
    assert entry["pool_size"] == {"low": 4, "high": 0}
    assert final["pool_size"] == {"low": 0, "high": 4}
    assert entry["p_trade"]["low"] == pytest.approx(1.0)
    assert final["p_trade"]["high"] == pytest.approx(4 / 6)
    assert not np.array_equal(entry["counts"], final["counts"])


def test_no_trade_snapshot_mode_only_moves_the_flat_days():
    """The decision-time variant relabels no-trade days without touching the
    point-in-time pools."""
    dates = business_days(6)
    states = ["low"] * 3 + ["high"] * 11
    frames = {d: make_regime_day(d, states) for d in dates}
    trades = pd.DataFrame([{"date": d,
                            "entry_time": pd.Timestamp(f"{d} 10:03", tz=NY),
                            "ticks": 4.0, "pnl_points": 1.0}
                           for d in dates[:4]])
    both = ["low", "high"]

    by_final = rs.estimate(trades, frames, "vol_state", both, tag=rs.TAG_ENTRY,
                           no_trade_snapshot=rs.NO_TRADE_FINAL)
    by_decision = rs.estimate(trades, frames, "vol_state", both,
                              tag=rs.TAG_ENTRY,
                              no_trade_snapshot=rs.NO_TRADE_DECISION,
                              decision_time="10:00")

    assert by_final["pool_size"] == by_decision["pool_size"]
    # the two flat days move from 'high' (closing view) to 'low' (10:00 view)
    assert by_final["n_days"] == {"low": 4, "high": 2}
    assert by_decision["n_days"] == {"low": 6, "high": 0}
    assert by_decision["p_trade"]["low"] == pytest.approx(4 / 6)


def test_decision_mode_requires_a_time():
    frames, trades = make_fixture()
    with pytest.raises(ValueError, match="decision_time"):
        rs.day_labels(trades, frames, "vol_state", tag=rs.TAG_ENTRY,
                      no_trade_snapshot=rs.NO_TRADE_DECISION)


# ── per-state shape (does the split matter at all?) ──────────────────────────

def test_pool_profile_separates_states_by_shape_not_just_size():
    """The states here share a mean but differ in spread — the real ES/IVB
    shape. A mean-only view would call them identical; the profile must not."""
    dates = business_days(4)
    frames = {d: make_regime_day(d, s)
              for d, s in zip(dates, ["L", "L", "H", "H"])}
    ticks = {dates[0]: 10.0, dates[1]: -10.0,      # calm: +/-10
             dates[2]: 90.0, dates[3]: -90.0}      # wild: +/-90
    trades = pd.DataFrame([{"date": d,
                            "entry_time": pd.Timestamp(f"{d} 10:03", tz=NY),
                            "ticks": t, "pnl_points": t / 4}
                           for d, t in ticks.items()])

    model = rs.estimate(trades, frames, "vol_state", STATES, tag=rs.TAG_FINAL)
    prof = model["pool_profile"]
    assert prof["L"]["mean"] == pytest.approx(0.0)
    assert prof["H"]["mean"] == pytest.approx(0.0)          # identical means
    assert prof["H"]["sd"] > prof["L"]["sd"] * 5            # different spread
    assert prof["H"]["worst"] == pytest.approx(-90.0)
    assert prof["L"]["worst"] == pytest.approx(-10.0)
    assert prof["L"]["win_rate"] == pytest.approx(0.5)
    # a state with no trades reports nothing rather than a misleading zero
    assert prof["N"]["mean"] is None


# ── 10. determinism ──────────────────────────────────────────────────────────

def test_same_seed_gives_an_identical_equity_matrix():
    frames, trades = make_fixture()
    model = rs.estimate(trades, frames, "vol_state", STATES, tag=rs.TAG_FINAL)
    kwargs = dict(horizon=40, n_paths=64)
    a = rs.simulate(trades, StubSizer, sizer_params(), model, seed=11, **kwargs)
    b = rs.simulate(trades, StubSizer, sizer_params(), model, seed=11, **kwargs)
    c = rs.simulate(trades, StubSizer, sizer_params(), model, seed=12, **kwargs)

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ── surfacing the honesty warnings ───────────────────────────────────────────

def test_thin_pools_and_multi_trade_days_are_reported():
    dates = business_days(4)
    # two H days that trade twice each, then two N days that never trade
    frames = {d: make_regime_day(d, s)
              for d, s in zip(dates, ["H", "H", "N", "N"])}
    rows = []
    for d in dates[:2]:
        for t in ("10:03", "11:03"):        # two trades on the same day
            rows.append({"date": d,
                         "entry_time": pd.Timestamp(f"{d} {t}", tz=NY),
                         "ticks": 4.0, "pnl_points": 1.0})
    trades = pd.DataFrame(rows)

    model = rs.estimate(trades, frames, "vol_state", STATES, tag=rs.TAG_FINAL,
                        min_trades_per_state=15)
    assert model["multi_trade_days"] == 2
    joined = " ".join(model["warnings"])
    assert "more than one trade" in joined
    assert "Thin trade pool" in joined
    assert "never traded" in joined          # N has days but no trades
    # a state with no days at all is absent, not "starved"
    assert "L" not in joined.split("never traded in")[1].split(".")[0]


def test_costs_are_not_charged_on_no_trade_days():
    """Size is zeroed rather than P&L, so a flat day pays no commission and no
    slippage. Charging a flat day would bleed every path."""
    trades = pd.DataFrame({"ticks": [4.0], "pnl_points": [1.0]})
    cost_ctx = {"enabled": True, "n": 1, "full_comm": 4.0,
                "micro_comm": None, "microable": False}
    eq = rs.simulate(trades, StubSizer, sizer_params(),
                     _alternating_model(0.0, 0.0), horizon=8, n_paths=4,
                     seed=5, cost_ctx=cost_ctx)
    assert np.allclose(eq, 100_000.0)


def test_simulate_rejects_a_sizer_without_the_vectorized_hooks():
    class Hookless:
        @staticmethod
        def apply(trades, params):
            return trades

    trades = pd.DataFrame({"ticks": [4.0], "pnl_points": [1.0]})
    with pytest.raises(ValueError, match="mc_prepare"):
        rs.simulate(trades, Hookless, sizer_params(), _alternating_model(),
                    horizon=5, n_paths=2, seed=1)
