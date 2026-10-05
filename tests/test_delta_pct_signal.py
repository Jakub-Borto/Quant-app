"""
delta_pct_signal (strategies/delta_pct_signal/) against the acceptance
criteria of its spec (v1, section 10): synthetic days with hand-computed
trades (1-13), then invariants and engine fit on the real data (14-15, skipped
when no data root has the datasets).
"""

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules.engine import DayCache, EngineError, run_strategy
from modules.optimizer.backend.loader import load_strategy

TZ = "America/New_York"
D = "2026-03-10"                                    # a Tuesday; the evening part is 03-09
STRAT = load_strategy("delta_pct_signal")


def ts(hhmm: str, date: str = D) -> pd.Timestamp:
    day = pd.Timestamp(date).date()
    if int(hhmm[:2]) >= 18:
        day -= datetime.timedelta(days=1)
    return pd.Timestamp(f"{day} {hhmm}", tz=TZ)


def make_day(folder: Path, date: str = D, bars: dict | None = None, last: str = "16:59",
             volume: int = 10):
    """One session 18:00 (D-1) -> `last` (D): flat at 100, volume `volume`, delta 0.
    bars = {"HH:MM": {column: value}} overrides (>= 18:00 = the evening of D-1)."""
    idx = pd.date_range(ts("18:00", date), ts(last, date), freq="1min")
    df = pd.DataFrame({"open": 100.0, "high": 100.25, "low": 99.75, "close": 100.0,
                       "volume": np.full(len(idx), volume, dtype="uint32"),
                       "volume_delta_pct": 0.0}, index=idx)
    for hhmm, values in (bars or {}).items():
        for col, v in values.items():
            df.loc[ts(hhmm, date), col] = v
    folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(folder / f"{date}.parquet")
    return folder


def run(folder, **params):
    res = run_strategy(STRAT, folder, "2026-01-01", "2026-12-31", params, tick_size=0.25,
                       cache=DayCache(), verbose=False)
    return res.trades


def notes(row) -> dict:
    return json.loads(row["notes"])


SIG = {"volume_delta_pct": 50.0}                    # a buy-side signal bar


# ── 1-2 basic follow / fade ──────────────────────────────────────────────────
def test_basic_follow_long(tmp_path):
    f = make_day(tmp_path / "ES", bars={"10:00": SIG, "10:01": {"open": 101.0},
                                       "10:05": {"close": 103.0}})
    t = run(f)
    assert len(t) == 1
    r = t.iloc[0]
    assert r["direction"] == "long" and r["exit_reason"] == "time"
    assert (r["entry_time"], r["exit_time"]) == (ts("10:01"), ts("10:05"))
    assert (r["entry_price"], r["exit_price"], r["pnl_points"]) == (101.0, 103.0, 2.0)
    assert np.isnan(r["sl"]) and np.isnan(r["tp"]) and r["trade_type"] == "follow_any"
    assert notes(r) == {"signal_time": "10:00", "signal_delta_pct": 50.0, "signal_volume": 10,
                        "volume_ratio": 1.0, "bars_held": 5, "extensions": 0, "session": "rth"}


def test_fade_is_the_mirror(tmp_path):
    f = make_day(tmp_path / "ES", bars={"10:00": SIG, "10:01": {"open": 101.0},
                                       "10:05": {"close": 103.0}})
    r = run(f, direction_mode="fade").iloc[0]
    assert r["direction"] == "short" and r["pnl_points"] == -2.0
    assert (r["entry_time"], r["exit_time"]) == (ts("10:01"), ts("10:05"))
    assert r["trade_type"] == "fade_any"


# ── 3-5 signal filters ───────────────────────────────────────────────────────
def test_threshold_is_inclusive_and_symmetric(tmp_path):
    f = make_day(tmp_path / "ES", bars={"10:00": {"volume_delta_pct": 40.0},
                                       "11:00": {"volume_delta_pct": 39.9},
                                       "12:00": {"volume_delta_pct": -40.0}})
    t = run(f)
    assert list(t["entry_time"]) == [ts("10:01"), ts("12:01")]
    assert list(t["direction"]) == ["long", "short"]


def test_price_filter(tmp_path):
    f = make_day(tmp_path / "ES", bars={
        "10:00": {**SIG, "open": 100.0, "close": 100.5},      # candle agrees
        "12:00": {**SIG, "open": 100.0, "close": 100.0}})     # doji: "against"
    signal_times = lambda **p: [notes(r)["signal_time"] for _, r in run(f, **p).iterrows()]
    assert signal_times(price_filter="with") == ["10:00"]
    assert signal_times(price_filter="against") == ["12:00"]
    assert signal_times(price_filter="any") == ["10:00", "12:00"]
    assert run(f, price_filter="against")["trade_type"].tolist() == ["follow_against"]


def test_volume_filter_warmup_and_zero_volume(tmp_path):
    f = make_day(tmp_path / "ES", bars={
        "10:00": {"volume_delta_pct": 80.0, "volume": 5},     # below 1 x median 10
        "18:05": {"volume_delta_pct": 80.0},                  # in the 20-bar warm-up
        "12:00": {"volume_delta_pct": 80.0, "volume": 0}})    # zero volume never signals
    common = dict(entry_start="18:00", entry_end="15:45")
    assert run(f, **common).empty
    t = run(f, min_volume_mult=0.0, **common)
    assert [notes(r)["signal_time"] for _, r in t.iterrows()] == ["18:05", "10:00"]
    assert all(notes(r)["volume_ratio"] is None for _, r in t.iterrows())
    assert t.iloc[0]["entry_time"] == ts("18:06") and notes(t.iloc[0])["session"] == "eth"


# ── 6-9 signals during a trade ───────────────────────────────────────────────
def two_signals(tmp_path, second=50.0):
    return make_day(tmp_path / "ES", bars={
        "10:00": SIG, "10:01": {"open": 101.0},
        "10:02": {"volume_delta_pct": second}, "10:03": {"open": 102.0},
        "10:05": {"close": 103.0}, "10:07": {"close": 104.0}})


def test_ignore(tmp_path):
    t = run(two_signals(tmp_path))
    assert len(t) == 1 and t.iloc[0]["exit_time"] == ts("10:05")
    assert notes(t.iloc[0])["extensions"] == 0


def test_close_and_reopen(tmp_path):
    t = run(two_signals(tmp_path), on_signal_in_trade="close_and_reopen")
    assert len(t) == 2
    a, b = t.iloc[0], t.iloc[1]
    assert a["exit_reason"] == "restart" and a["exit_time"] == b["entry_time"] == ts("10:03")
    assert a["exit_price"] == b["entry_price"] == 102.0 and notes(a)["bars_held"] == 2
    assert b["direction"] == "long" and b["exit_time"] == ts("10:07") and b["exit_price"] == 104.0
    rev = run(two_signals(tmp_path, second=-50.0), on_signal_in_trade="close_and_reopen")
    assert rev.iloc[0]["exit_reason"] == "reverse" and rev.iloc[1]["direction"] == "short"


def test_extend_or_reverse(tmp_path):
    t = run(two_signals(tmp_path), on_signal_in_trade="extend_or_reverse")
    assert len(t) == 1
    r = t.iloc[0]
    assert (r["entry_time"], r["entry_price"]) == (ts("10:01"), 101.0)
    assert (r["exit_time"], r["exit_price"], r["exit_reason"]) == (ts("10:07"), 104.0, "time")
    assert notes(r)["extensions"] == 1 and notes(r)["bars_held"] == 7
    rev = run(two_signals(tmp_path, second=-50.0), on_signal_in_trade="extend_or_reverse")
    assert list(rev["exit_reason"]) == ["reverse", "time"] and rev.iloc[1]["direction"] == "short"


@pytest.mark.parametrize("mode", ["ignore", "close_and_reopen", "extend_or_reverse"])
def test_signal_on_the_exit_bar_opens_a_new_trade(tmp_path, mode):
    f = make_day(tmp_path / "ES", bars={"10:00": SIG, "10:05": SIG})
    t = run(f, on_signal_in_trade=mode)
    assert list(t["entry_time"]) == [ts("10:01"), ts("10:06")]
    assert t.iloc[0]["exit_time"] == ts("10:05") and t.iloc[0]["exit_reason"] == "time"


# ── 10-11 flat time and window anchoring ─────────────────────────────────────
def test_flat_exit_and_half_day(tmp_path):
    f = make_day(tmp_path / "ES", bars={"15:45": SIG, "15:55": {"close": 99.0}})
    r = run(f, hold_bars=20).iloc[0]
    assert (r["exit_time"], r["exit_price"], r["exit_reason"]) == (ts("15:55"), 99.0, "flat")
    half = make_day(tmp_path / "HALF", last="12:59", bars={"12:56": SIG})
    r = run(half, hold_bars=5).iloc[0]
    assert (r["exit_time"], r["exit_reason"]) == (ts("12:59"), "flat")
    # a signal on the flat bar (or the file's last bar) never trades
    assert run(make_day(tmp_path / "LAST", last="12:59", bars={"12:59": SIG})).empty


def test_window_anchoring(tmp_path):
    f = make_day(tmp_path / "ES", bars={h: SIG for h in
                                       ("19:00", "09:00", "09:29", "09:30", "12:00", "15:45",
                                        "15:46")})
    overnight = run(f, entry_start="18:00", entry_end="09:29", hold_bars=1, min_volume_mult=0.0)
    assert [notes(r)["signal_time"] for _, r in overnight.iterrows()] == ["19:00", "09:00", "09:29"]
    assert overnight.iloc[0]["entry_time"] == ts("19:01") == pd.Timestamp("2026-03-09 19:01", tz=TZ)
    assert (overnight["entry_time"] < ts("09:31")).all()
    rth = run(f, hold_bars=1)
    assert [notes(r)["signal_time"] for _, r in rth.iterrows()] == ["09:30", "12:00", "15:45"]


# ── 12 validation ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("bad, name", [
    ({"entry_start": "15:50", "entry_end": "15:45"}, "entry_start"),
    ({"entry_end": "15:55", "flat_time": "15:55"}, "entry_end"),
    ({"flat_time": "17:30"}, "flat_time"),
    ({"entry_start": "17:00"}, "entry_start"),
    ({"entry_end": "9h30"}, "entry_end"),
    ({"delta_threshold": 0.0}, "delta_threshold"),
    ({"delta_threshold": 100.5}, "delta_threshold"),
    ({"hold_bars": 0}, "hold_bars"),
    ({"min_volume_mult": -1.0}, "min_volume_mult"),
    ({"volume_lookback": 0}, "volume_lookback"),
    ({"direction_mode": "both"}, "direction_mode"),
])
def test_validation(tmp_path, bad, name):
    f = make_day(tmp_path / "ES", bars={"10:00": SIG})
    with pytest.raises(ValueError, match=name):
        run(f, **bad)


# ── 13 no look-ahead ─────────────────────────────────────────────────────────
def test_no_look_ahead(tmp_path):
    base = {"10:00": SIG, "10:01": {"open": 101.0}}
    first = run(make_day(tmp_path / "A", bars=base)).iloc[0]
    rng = np.random.default_rng(7)
    noisy = dict(base)
    noisy["10:01"] = {"open": 101.0, "high": 109.0, "low": 90.0, "close": 95.0,
                      "volume": 999, "volume_delta_pct": -100.0}
    for m in range(2, 300):                                   # every later bar changes
        t = (pd.Timestamp("2026-03-10 10:00") + pd.Timedelta(minutes=m)).strftime("%H:%M")
        noisy[t] = {"open": float(rng.uniform(90, 110)), "close": float(rng.uniform(90, 110)),
                    "volume": int(rng.integers(0, 50)),
                    "volume_delta_pct": float(rng.uniform(-100, 100))}
    other = run(make_day(tmp_path / "B", bars=noisy)).iloc[0]
    for col in ("direction", "entry_time", "entry_price", "trade_type"):
        assert first[col] == other[col], col
    assert notes(first)["signal_time"] == notes(other)["signal_time"] == "10:00"


# ── 14-15 real data ──────────────────────────────────────────────────────────
def _dataset(asset: str, name: str) -> Path | None:
    from modules.common.backend.settings import load_settings
    for root in load_settings().data_roots:
        p = Path(root) / "parquet" / "Futures" / asset / f"{asset}_{name}"
        if p.is_dir():
            return p
    return None


ES = _dataset("ES", "1m_advanced")
real = pytest.mark.skipif(ES is None, reason="ES_1m_advanced not in any data root")


@real
@pytest.mark.parametrize("mode", ["ignore", "close_and_reopen", "extend_or_reverse"])
def test_invariants_on_real_es(mode):
    res = run_strategy(STRAT, ES, "2000-01-01", "2100-01-01", {"on_signal_in_trade": mode},
                       tick_size=0.25, verbose=False)
    t = res.trades
    assert len(t) > 100
    for _, day in t.groupby("date"):
        entries, exits = day["entry_time"].tolist(), day["exit_time"].tolist()
        assert all(e2 >= x1 for x1, e2 in zip(exits, entries[1:]))           # no overlap
        flat = ts("15:55", str(day["date"].iloc[0]))
        assert (day["exit_time"] <= flat).all()
        for _, r in day.iterrows():
            sig = ts(notes(r)["signal_time"], str(r["date"]))
            assert r["entry_time"] > sig
    assert set(t["exit_reason"]) <= {"time", "flat", "restart", "reverse"}


@real
def test_parallel_optimizer_matches_serial_and_other_assets():
    from modules.optimizer.backend.engine import run_grid
    axes = [{"param": "delta_threshold", "values": [30.0, 50.0], "role": "x"},
            {"param": "direction_mode", "values": ["follow", "fade"], "role": "y"}]
    kw = dict(base_params={}, axes=axes, tick_size=0.25, ticks_per_point=4, bucket_map={})
    serial = run_grid(STRAT, ES, "2026-01-01", "2026-03-31", **kw)
    parallel = run_grid(None, ES, "2026-01-01", "2026-03-31", n_workers=3,
                        strategy_name="delta_pct_signal", **kw)
    pd.testing.assert_frame_equal(serial, parallel)
    assert len(serial) > 0
    for asset, tick in (("NQ", 0.25), ("ZN", 0.015625), ("CL", 0.01)):
        folder = _dataset(asset, "1m_advanced")
        if folder is not None:
            res = run_strategy(STRAT, folder, "2026-01-01", "2026-02-28", {}, tick_size=tick,
                               verbose=False)
            assert res.days_run > 0, asset
    ohlcv = _dataset("ES", "1m_ohlcv_globex")
    if ohlcv is not None:
        with pytest.raises(EngineError, match="volume_delta_pct"):
            run_strategy(STRAT, ohlcv, "2026-01-01", "2026-01-31", {}, tick_size=0.25,
                         verbose=False)
