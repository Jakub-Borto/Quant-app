"""
The two worked examples from STRATEGY_GUIDE.md run exactly as the guide
describes (hand-computed trades on synthetic days), and load through the same
plugin discovery the Backtester uses. If these fail, the guide is wrong.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules.common.backend.plugins import list_strategies
from modules.engine import DayCache, run_strategy
from modules.optimizer.backend.loader import load_strategy

TZ = "America/New_York"
REPO = Path(__file__).resolve().parents[1]


def make_day(folder: Path, date: str, base: float = 100.0, bars: dict | None = None,
             indicators: Path | None = None, vwap: float = 100.0, cvd_step: float = 0.0):
    """RTH 1-minute bars 09:30..15:59 flat at `base` (+/-0.1 wicks); `bars`
    overrides {"HH:MM": (open, high, low, close)}. Optionally writes an
    indicators file (constant vwap_bar_rth, cumulative_delta rising by
    cvd_step per bar)."""
    idx = pd.date_range(f"{date} 09:30", f"{date} 15:59", freq="1min", tz=TZ)
    df = pd.DataFrame({"open": base, "high": base + 0.1, "low": base - 0.1,
                       "close": base}, index=idx, dtype=float)
    for hhmm, (o, h, l, c) in (bars or {}).items():
        df.loc[pd.Timestamp(f"{date} {hhmm}", tz=TZ)] = [o, h, l, c]
    folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(folder / f"{date}.parquet")
    if indicators is not None:
        indicators.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"vwap_bar_rth": vwap,
                      "cumulative_delta": np.arange(len(idx)) * cvd_step},
                     index=idx, dtype=float).to_parquet(indicators / f"{date}.parquet")


def run(name, folder, extra=None, **params):
    return run_strategy(load_strategy(name), folder, "2026-01-01", "2026-12-31", params,
                        tick_size=0.25, extra_folders=extra or {}, cache=DayCache(),
                        verbose=False)


def test_examples_are_discovered_like_any_strategy():
    names = [r.name for r in list_strategies([REPO / "strategies"])]
    assert "example_first_hour_breakout" in names
    assert "example_prev_day_levels" in names


def test_first_hour_breakout_takes_the_hand_computed_trade(tmp_path):
    folder = tmp_path / "ES_1m"
    # range 09:30-10:29: high 101, low 99; 10:40 closes at 101.5 -> long signal;
    # entry = 10:41 open 101.5; stop = 99 - 2 ticks = 98.5; risk 3.0;
    # target = 101.5 + 3.0 * 1.5 = 106.0, hit by the 11:00 bar
    make_day(folder, "2026-03-02", bars={
        "09:45": (100, 101, 100, 100), "10:10": (100, 100, 99, 100),
        "10:40": (100, 101.6, 100, 101.5), "10:41": (101.5, 101.6, 101.4, 101.5),
        "11:00": (101.5, 106.2, 101.4, 106.0)})
    res = run("example_first_hour_breakout", folder)
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t["direction"] == "long" and t["exit_reason"] == "tp"
    assert t["entry_time"] == pd.Timestamp("2026-03-02 10:41", tz=TZ)
    assert (t["entry_price"], t["sl"], t["tp"], t["exit_price"]) == (101.5, 98.5, 106.0, 106.0)
    assert t["pnl_points"] == 4.5
    assert json.loads(t["notes"]) == {"range_high": 101.0, "range_low": 99.0, "bars_held": 20}


def test_first_hour_breakout_time_exit(tmp_path):
    folder = tmp_path / "ES_1m"
    make_day(folder, "2026-03-02", bars={
        "09:45": (100, 101, 100, 100), "10:10": (100, 100, 99, 100),
        "10:40": (100, 101.6, 100, 101.5), "10:41": (101.5, 101.6, 101.4, 101.5)})
    t = run("example_first_hour_breakout", folder).trades.iloc[0]
    assert t["exit_reason"] == "time_exit"
    assert t["exit_time"] == pd.Timestamp("2026-03-02 15:55", tz=TZ)
    assert t["exit_price"] == 100.0


def test_prev_day_levels_uses_previous_days_and_the_indicators_slot(tmp_path):
    main, ind = tmp_path / "ES_1m_advanced", tmp_path / "ES_1m_indicators"
    days = ["2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06"]
    for d in days:      # five history days: RTH range 100..110 (range 10)
        make_day(main, d, base=105, indicators=ind, vwap=105,
                 bars={"10:00": (105, 110, 105, 105), "12:00": (105, 105, 100, 105)})
    # trade day: 10:30 closes at 111 (> yesterday's high 110), CVD rising,
    # above VWAP 105 -> long; entry = 10:31 open 111.0
    # stop = floor((111 - 10*0.25)/0.25)*0.25 = 108.5; target = floor((111+5)/0.25)*0.25 = 116.0
    # (price then holds at 111 until the 11:00 bar reaches the target)
    hold = {f"10:{m:02d}": (111.0, 111.2, 110.9, 111.0) for m in range(31, 60)}
    make_day(main, "2026-03-09", base=105, indicators=ind, vwap=105, cvd_step=1.0,
             bars={"10:30": (105, 111.2, 105, 111.0), **hold,
                   "11:00": (111.0, 116.5, 110.9, 116.0)})
    res = run("example_prev_day_levels", main, extra={"indicators": ind})
    assert len(res.trades) == 1                       # history days lack 5 prior days
    t = res.trades.iloc[0]
    assert t["date"] == pd.Timestamp("2026-03-09").date()
    assert t["direction"] == "long" and t["trade_type"] == "prev_day_high_break"
    assert (t["entry_price"], t["sl"], t["tp"]) == (111.0, 108.5, 116.0)
    assert t["exit_reason"] == "tp" and t["exit_price"] == 116.0
    notes = json.loads(t["notes"])
    assert notes["prev_date"] == "2026-03-06" and notes["prev_high"] == 110.0
    assert notes["avg_range"] == 10.0 and notes["signal_time"] == "10:30"


def test_prev_day_levels_needs_the_indicators_folder(tmp_path):
    from modules.engine import EngineError
    make_day(tmp_path / "ES_1m_advanced", "2026-03-02")
    with pytest.raises(EngineError, match="indicators"):
        run("example_prev_day_levels", tmp_path / "ES_1m_advanced")
