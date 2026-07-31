"""Tests for the shared regime→trades as-of join (modules/common/backend/
regime_join.py) and the per-regime performance table.

The three things that would silently corrupt a backtest, each with a test:
lookahead (final row leaking onto earlier trades), cross-session carry (a
stale 17:00 label on the next evening's entry), and dtype drift between
strategies (ns vs us entry_time; date as date/str/Timestamp).

All synthetic — no Qt, no real data.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from modules.common.backend import regime_join as rj
from modules.common.backend.trade_stats import regime_performance_rows

NY = "America/New_York"


# ── fixtures ─────────────────────────────────────────────────────────────────

def make_regime_day(date: str, states=None, unit: str = "us") -> pd.DataFrame:
    """One day's regime file: the 30-min grid from the prior evening 18:30
    through 17:00 on `date` (46 rows), exactly like a real run."""
    start = pd.Timestamp(f"{date} 18:30", tz=NY) - pd.Timedelta(days=1)
    idx = pd.date_range(start, periods=46, freq="30min").as_unit(unit)
    if states is None:
        states = ["normal"] * 46
    return pd.DataFrame({"vol_state": states, "price": np.arange(46.0)},
                        index=idx)


def make_trades(rows) -> pd.DataFrame:
    """rows: list of (date_str, entry_wall_str) — entry may be on the evening
    BEFORE the RTH date, exactly as an overnight strategy would emit."""
    dates, entries = [], []
    for date, entry in rows:
        dates.append(date)
        stamp = pd.Timestamp(f"{date} {entry}", tz=NY)
        if entry >= "18:00":                     # evening entry -> prior day
            stamp -= pd.Timedelta(days=1)
        entries.append(stamp)
    return pd.DataFrame({
        "date": dates,
        "entry_time": pd.DatetimeIndex(entries),
        "ticks": np.arange(len(dates), dtype=float) - 1.0,
        "direction": "long",
    })


# ── dtype drift (verified against real trades files) ─────────────────────────

def test_asof_survives_ns_us_unit_mismatch():
    """ivb trades are [ns], orb trades are [us], the regime index is [us].
    Without normalization pandas raises MergeError on the merge key."""
    frames = {"2026-01-06": make_regime_day("2026-01-06", unit="us")}
    trades = make_trades([("2026-01-06", "10:15")])

    for unit in ("ns", "us"):
        t = trades.copy()
        t["entry_time"] = t["entry_time"].dt.as_unit(unit)
        assert str(t["entry_time"].dtype) == f"datetime64[{unit}, {NY}]"
        out = rj.attach_regime(t, frames, "vol_state", rj.MODE_ASOF)
        assert out["regime"].tolist() == ["normal"]


def test_date_column_dtype_variants_all_join():
    frames = {"2026-01-06": make_regime_day("2026-01-06")}
    base = make_trades([("2026-01-06", "10:15")])
    for value in (dt.date(2026, 1, 6), "2026-01-06", pd.Timestamp("2026-01-06")):
        t = base.copy()
        t["date"] = [value]
        out = rj.attach_regime(t, frames, "vol_state", rj.MODE_ASOF)
        assert out["regime"].tolist() == ["normal"], value


def test_naive_entry_times_are_localized():
    frames = {"2026-01-06": make_regime_day("2026-01-06")}
    t = make_trades([("2026-01-06", "10:15")])
    t["entry_time"] = t["entry_time"].dt.tz_localize(None)
    out = rj.attach_regime(t, frames, "vol_state", rj.MODE_ASOF)
    assert out["regime"].tolist() == ["normal"]


# ── the as-of rule ───────────────────────────────────────────────────────────

def test_asof_picks_latest_prior_snapshot():
    states = ["normal"] * 46
    states[32] = "high"          # 10:30 on the RTH date
    states[33] = "low"           # 11:00
    frames = {"2026-01-06": make_regime_day("2026-01-06", states)}

    trades = make_trades([("2026-01-06", "10:29"),   # -> 10:00 row (normal)
                          ("2026-01-06", "10:30"),   # -> 10:30 exactly (high)
                          ("2026-01-06", "10:49"),   # -> 10:30 (high)
                          ("2026-01-06", "11:05")])  # -> 11:00 (low)
    out = rj.attach_regime(trades, frames, "vol_state", rj.MODE_ASOF)
    assert out["regime"].tolist() == ["normal", "high", "high", "low"]


def test_asof_allows_exact_match_without_lookahead():
    """A snapshot stamped T is built only from bars strictly before T, so an
    entry exactly at T may read it."""
    states = ["normal"] * 46
    states[32] = "high"
    frames = {"2026-01-06": make_regime_day("2026-01-06", states)}
    out = rj.attach_regime(make_trades([("2026-01-06", "10:30")]), frames,
                           "vol_state", rj.MODE_ASOF)
    assert out["regime"].tolist() == ["high"]


def test_asof_never_sees_the_final_row_early():
    """The whole point: a morning trade must not inherit the day's verdict."""
    states = ["normal"] * 45 + ["high"]     # only the 17:00 row is 'high'
    frames = {"2026-01-06": make_regime_day("2026-01-06", states)}
    trades = make_trades([("2026-01-06", "10:00"), ("2026-01-06", "15:00")])

    asof = rj.attach_regime(trades, frames, "vol_state", rj.MODE_ASOF)
    final = rj.attach_regime(trades, frames, "vol_state", rj.MODE_FINAL)
    assert asof["regime"].tolist() == ["normal", "normal"]
    assert final["regime"].tolist() == ["high", "high"]      # hindsight


# ── the cross-session guard ──────────────────────────────────────────────────

def test_evening_entry_does_not_inherit_previous_session():
    """An 18:05 entry belongs to the NEXT RTH date, whose file starts 18:30.
    The only earlier snapshot is the prior day's 17:00 — a stale label."""
    frames = {"2026-01-06": make_regime_day("2026-01-06", ["high"] * 46),
              "2026-01-07": make_regime_day("2026-01-07", ["low"] * 46)}
    trades = make_trades([("2026-01-07", "18:05"),   # before 18:30 -> unknown
                          ("2026-01-07", "18:45")])  # -> 18:30 row of the 7th
    out = rj.attach_regime(trades, frames, "vol_state", rj.MODE_ASOF)
    assert out["regime"].tolist() == [rj.UNKNOWN, "low"]


def test_guard_holds_for_a_wide_snapshot_grid():
    """A tolerance alone would break once snapshot_minutes exceeds the
    17:00->18:30 break; the regime_date equality guard has no such cliff."""
    frames = {"2026-01-06": make_regime_day("2026-01-06", ["high"] * 46),
              "2026-01-07": make_regime_day("2026-01-07", ["low"] * 46)}
    trades = make_trades([("2026-01-07", "18:05")])
    out = rj.attach_regime(trades, frames, "vol_state", rj.MODE_ASOF,
                           snapshot_minutes=180)
    assert out["regime"].tolist() == [rj.UNKNOWN]


def test_missing_day_is_unknown():
    frames = {"2026-01-06": make_regime_day("2026-01-06")}
    trades = make_trades([("2026-01-06", "10:00"), ("2026-01-09", "10:00")])
    for mode in (rj.MODE_ASOF, rj.MODE_FINAL):
        out = rj.attach_regime(trades, frames, "vol_state", mode)
        assert out["regime"].tolist() == ["normal", rj.UNKNOWN], mode


# ── contract: never mutate, never reorder ────────────────────────────────────

def test_preserves_row_order_and_does_not_mutate():
    states = ["normal"] * 46
    states[32] = "high"
    frames = {"2026-01-06": make_regime_day("2026-01-06", states),
              "2026-01-07": make_regime_day("2026-01-07", ["low"] * 46)}
    trades = make_trades([("2026-01-07", "10:00"), ("2026-01-06", "10:45"),
                          ("2026-01-06", "09:00")])
    trades.index = [7, 3, 5]                       # non-monotonic index too
    before = trades.copy()

    out = rj.attach_regime(trades, frames, "vol_state", rj.MODE_ASOF)
    pd.testing.assert_frame_equal(trades, before)  # input untouched
    assert "regime" not in trades.columns
    assert out.index.tolist() == [7, 3, 5]
    assert out["entry_time"].tolist() == before["entry_time"].tolist()
    assert out["regime"].tolist() == ["low", "high", "normal"]


def test_empty_trades_and_empty_frames():
    frames = {"2026-01-06": make_regime_day("2026-01-06")}
    empty = make_trades([]).iloc[:0]
    assert "regime" in rj.attach_regime(empty, frames, "vol_state",
                                        rj.MODE_ASOF).columns
    trades = make_trades([("2026-01-06", "10:00")])
    for mode in (rj.MODE_ASOF, rj.MODE_FINAL):
        out = rj.attach_regime(trades, {}, "vol_state", mode)
        assert out["regime"].tolist() == [rj.UNKNOWN], mode


# ── exact / final modes ──────────────────────────────────────────────────────

def test_exact_mode_uses_the_chosen_clock_time():
    states = ["normal"] * 46
    states[32] = "high"                       # 10:30
    frames = {"2026-01-06": make_regime_day("2026-01-06", states)}
    # every trade of the day gets the 10:30 label, whenever it entered
    trades = make_trades([("2026-01-06", "09:00"), ("2026-01-06", "15:00")])
    out = rj.attach_regime(trades, frames, "vol_state", rj.MODE_EXACT,
                           at="10:30")
    assert out["regime"].tolist() == ["high", "high"]

    earlier = rj.attach_regime(trades, frames, "vol_state", rj.MODE_EXACT,
                               at="10:00")
    assert earlier["regime"].tolist() == ["normal", "normal"]


def test_exact_mode_requires_a_time():
    frames = {"2026-01-06": make_regime_day("2026-01-06")}
    with pytest.raises(ValueError, match="at="):
        rj.attach_regime(make_trades([("2026-01-06", "10:00")]), frames,
                         "vol_state", rj.MODE_EXACT)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        rj.attach_regime(make_trades([("2026-01-06", "10:00")]), {},
                         "vol_state", "whenever")


# ── per-regime performance rows ──────────────────────────────────────────────

def test_regime_rows_order_declared_states_with_unknown_last():
    trades = pd.DataFrame({
        "regime": ["high", "low", "unknown", "low", "normal"],
        "ticks": [10.0, -4.0, 3.0, 8.0, 0.0],
    })
    rows = regime_performance_rows(trades, states=["low", "normal", "high"])
    assert [r["Regime"] for r in rows] == ["low", "normal", "high", "unknown"]
    low = rows[0]
    assert low["Trades"] == 2 and low["Wins"] == 1 and low["Losses"] == 1
    assert low["Total Ticks"] == 4
    # a declared state with no trades still gets a row (stable table shape)
    empty = regime_performance_rows(
        trades[trades["regime"] == "high"], states=["low", "normal", "high"])
    assert [r["Trades"] for r in empty] == [0, 0, 1, 0]


def test_regime_rows_without_declared_states():
    trades = pd.DataFrame({"regime": ["b", "a", "unknown"],
                           "ticks": [1.0, 2.0, 3.0]})
    rows = regime_performance_rows(trades)
    assert [r["Regime"] for r in rows] == ["a", "b", "unknown"]


def test_entry_breakdown_rows():
    from modules.common.backend.trade_stats import (compute_metrics,
                                                    entry_breakdown_rows)

    n = 24
    dates = pd.date_range("2026-01-05", periods=n, freq="D")
    trades = pd.DataFrame({
        "date": dates,
        "direction": ["long", "short"] * (n // 2),
        "trade_type": (["breakout"] * 12) + (["fade"] * 12),
        "ticks": ([10.0, -5.0, 0.0] * 4) + ([20.0, -4.0] * 6),
    })
    trades["cumulative_ticks"] = trades["ticks"].cumsum()

    rows = entry_breakdown_rows(trades)
    # the whole-strategy benchmark comes FIRST — the table pins it there
    assert [r["Entry"] for r in rows] == ["All entries", "breakout", "fade"]

    # every figure must match compute_metrics on that subset, with
    # cumulative_ticks recomputed per entry (each type is its own curve)
    sub = trades[trades["trade_type"] == "breakout"].copy()
    sub["cumulative_ticks"] = sub["ticks"].cumsum()
    m = compute_metrics(sub)
    row = next(r for r in rows if r["Entry"] == "breakout")
    assert row["Trades"] == m["total_trades"] == 12
    assert row["Wins"] == 4 and row["Losses"] == 4
    assert row["BE Rate"] == f"{m['breakeven_rate']:.1%}"    # the 0.0 ticks
    assert row["Profit Factor"] == f"{m['profit_factor']:.2f}"
    assert row["Sharpe (daily)"] == f"{m['sharpe_daily']:.2f}"
    assert row["Sharpe (traded days)"] == f"{m['sharpe_trade']:.2f}"
    assert row["Total Ticks"] == int(round(m["total_ticks"]))

    # a single entry type needs no "All entries" comparison row
    one = trades[trades["trade_type"] == "fade"]
    assert [r["Entry"] for r in entry_breakdown_rows(one)] == ["fade"]

    # no trade_type column at all -> the section hides
    assert entry_breakdown_rows(trades.drop(columns=["trade_type"])) is None


def test_entry_breakdown_handles_an_all_breakeven_entry():
    """compute_metrics divides by len(trades); an entry whose every trade is
    flat must not blow up or report a bogus profit factor."""
    from modules.common.backend.trade_stats import entry_breakdown_rows

    trades = pd.DataFrame({
        "date": pd.date_range("2026-01-05", periods=4, freq="D"),
        "direction": "long", "trade_type": ["flat"] * 4,
        "ticks": [0.0, 0.0, 0.0, 0.0],
    })
    trades["cumulative_ticks"] = trades["ticks"].cumsum()
    row = entry_breakdown_rows(trades)[0]
    assert row["Trades"] == 4 and row["BE Rate"] == "100.0%"
    assert row["Profit Factor"] == "0.00"


def test_regime_rows_none_without_column():
    assert regime_performance_rows(pd.DataFrame({"ticks": [1.0]})) is None


# ── memory estimate (the guard against loading a 16-year run blindly) ────────

def test_estimate_load_size_scales_with_range(tmp_path):
    from modules.regime_detector.backend import io as rio

    days = [f"2026-01-{d:02d}" for d in range(1, 21)]
    for date in days:
        make_regime_day(date).to_parquet(tmp_path / f"{date}.parquet")
    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")

    n_all, gb_all = rio.estimate_load_size(tmp_path)
    assert n_all == 20                      # meta.json is not a day file
    assert gb_all > 0

    n_half, gb_half = rio.estimate_load_size(tmp_path, start="2026-01-01",
                                             end="2026-01-10")
    assert n_half == 10
    assert gb_half == pytest.approx(gb_all / 2, rel=0.01)

    assert rio.estimate_load_size(tmp_path, start="2030-01-01") == (0, 0.0)


def test_estimate_is_close_to_the_real_loaded_size(tmp_path):
    """The estimate measures ONE file and scales it — verify that against
    what actually lands in memory, or the budget check is theatre."""
    from modules.regime_detector.backend import io as rio

    days = [f"2026-02-{d:02d}" for d in range(1, 13)]
    for date in days:
        make_regime_day(date).to_parquet(tmp_path / f"{date}.parquet")

    _n, estimated_gb = rio.estimate_load_size(tmp_path)
    frames = rio.load_day_frames(tmp_path)
    actual_gb = sum(f.memory_usage(deep=True).sum()
                    for f in frames.values()) / (1024 ** 3)
    assert estimated_gb == pytest.approx(actual_gb, rel=0.05)
