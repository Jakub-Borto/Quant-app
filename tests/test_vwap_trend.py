"""
vwap_trend strategy: signal classification, fill timing, contiguity, EOD /
exclusion / zero-volume / NaN-VWAP handling, output schema, param validation,
day-cache param-independence (spec §14/§15).

All data is synthetic and written to tmp_path as candle + indicator parquet
siblings, mirroring parquet/{type}/{asset}/{dataset}/YYYY-MM-DD.parquet.
"""

import json

import numpy as np
import pandas as pd
import pytest

from modules.optimizer.backend.loader import load_strategy
from modules.optimizer.backend.param_space import sweep_kind

CANDLES_DS = "TEST_1m"
IND_DS     = "TEST_1m_indicators"
DATE       = "2026-01-05"
TZ         = "America/New_York"

BASE_PARAMS = {
    "indicators_dataset": IND_DS,
    "vwap_anchor":        "rth",
    "vwap_band_ticks":    0.0,
    "band_rule":          "carry_forward",
    "trade_start_time":   "09:31",
    "trade_end_time":     "16:00",
    "exclude_start":      "",
    "exclude_end":        "",
    "skip_zero_volume":   True,
    "sl_convention":      "vwap_at_entry",
    "tick_size":          0.25,
}

START, END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31")


@pytest.fixture(scope="module")
def strat():
    return load_strategy("vwap_trend")


def write_day(root, opens, closes, vwaps, volumes=None, date=DATE,
              first_bar="09:30", vwap_globex=None, ind_columns=None):
    """One synthetic session: candles + indicator files with matching names."""
    n = len(opens)
    idx = pd.date_range(f"{date} {first_bar}", periods=n, freq="1min", tz=TZ)
    volumes = [100] * n if volumes is None else volumes
    candles = pd.DataFrame({
        "open":   np.asarray(opens,  dtype=np.float64),
        "high":   np.maximum(opens, closes) + 0.25,
        "low":    np.minimum(opens, closes) - 0.25,
        "close":  np.asarray(closes, dtype=np.float64),
        "volume": np.asarray(volumes, dtype=np.float64),
    }, index=idx)
    ind = pd.DataFrame({
        "vwap_bar_rth":    np.asarray(vwaps, dtype=np.float64),
        "vwap_bar_globex": np.asarray(vwap_globex if vwap_globex is not None else vwaps,
                                      dtype=np.float64),
    }, index=idx)
    if ind_columns is not None:
        ind = ind[ind_columns]
    (root / CANDLES_DS).mkdir(exist_ok=True, parents=True)
    (root / IND_DS).mkdir(exist_ok=True, parents=True)
    candles.to_parquet(root / CANDLES_DS / f"{date}.parquet")
    ind.to_parquet(root / IND_DS / f"{date}.parquet")
    return root / CANDLES_DS


def run(strat, folder, **overrides):
    return strat.run(folder, START, END, {**BASE_PARAMS, **overrides})


# ── golden session (hand-computed) ───────────────────────────────────────────
# VWAP flat at 100, band 0: long from bar0's close, flip short on bar2's close,
# NEUTRAL (==vwap) on bar4 carries the short, flip long on bar5's close, long
# held to the EOD flat. 3 trades, all boundaries at next-bar opens.

G_OPENS  = [100.0, 100.5, 101.2, 99.5, 98.8, 99.9, 100.8, 101.0, 101.3, 101.1,
            101.4, 101.6, 101.5, 101.8, 102.0, 101.9, 102.2, 102.4, 102.3, 102.6]
G_CLOSES = [101.0, 101.5, 99.0, 98.5, 100.0, 101.0, 101.2, 101.4, 101.0, 101.3,
            101.5, 101.4, 101.7, 101.9, 101.8, 102.1, 102.3, 102.2, 102.5, 102.7]
G_VWAP   = [100.0] * 20


def ts(hhmm, date=DATE):
    return pd.Timestamp(f"{date} {hhmm}", tz=TZ)


def test_golden_session(strat, tmp_path):
    folder = write_day(tmp_path, G_OPENS, G_CLOSES, G_VWAP)
    df = run(strat, folder)

    assert list(df.columns) == [
        "date", "direction", "trade_type", "entry_time", "exit_time",
        "entry_price", "exit_price", "sl", "tp", "exit_reason", "pnl_points", "notes"]
    assert len(df) == 3
    assert list(df["direction"]) == ["long", "short", "long"]
    assert set(df["trade_type"]) == {"vwap_trend"}

    t1, t2, t3 = df.iloc[0], df.iloc[1], df.iloc[2]
    # fills at next-bar opens, never at the signal bar's close
    assert t1["entry_time"] == ts("09:31") and t1["entry_price"] == 100.5
    assert t1["exit_time"]  == ts("09:33") and t1["exit_price"]  == 99.5
    assert t1["exit_reason"] == "vwap_flip"
    assert t2["entry_time"] == ts("09:33") and t2["entry_price"] == 99.5
    assert t2["exit_time"]  == ts("09:36") and t2["exit_price"]  == pytest.approx(100.8)
    assert t3["entry_time"] == ts("09:36")
    assert t3["exit_time"]  == ts("09:49") and t3["exit_price"] == 102.7
    assert t3["exit_reason"] == "eod"

    assert t1["pnl_points"] == pytest.approx(-1.0)
    assert t2["pnl_points"] == pytest.approx(-1.3)   # short: entry - exit
    assert t3["pnl_points"] == pytest.approx(102.7 - 100.8)

    # sl_convention='vwap_at_entry', band 0 -> sl == vwap, tp always NaN
    assert list(df["sl"]) == [100.0, 100.0, 100.0]
    assert df["tp"].isna().all()

    notes = json.loads(t3["notes"])
    assert notes["vwap_at_entry"] == 100.0
    assert notes["bars_held"] == 14
    assert notes["band_ticks"] == 0.0
    assert notes["filled_on_zero_volume_bar"] is False


def test_contiguity_and_flat_at_eod(strat, tmp_path):
    folder = write_day(tmp_path, G_OPENS, G_CLOSES, G_VWAP)
    df = run(strat, folder)
    # stop-and-reverse: exit of N == entry of N+1 (time AND price)
    for i in range(len(df) - 1):
        assert df.iloc[i]["exit_time"] == df.iloc[i + 1]["entry_time"]
        assert df.iloc[i]["exit_price"] == df.iloc[i + 1]["entry_price"]
    # session ends flat, nothing past the window
    assert df.iloc[-1]["exit_reason"] == "eod"
    assert (df["exit_time"] <= ts("16:00")).all()


def test_trade_end_forces_eod_inside_data(strat, tmp_path):
    folder = write_day(tmp_path, G_OPENS, G_CLOSES, G_VWAP)
    df = run(strat, folder, trade_end_time="09:40")
    assert len(df) > 0
    assert (df["exit_time"] <= ts("09:40")).all()
    assert df.iloc[-1]["exit_reason"] == "eod"
    assert df.iloc[-1]["exit_time"] == ts("09:40")
    assert df.iloc[-1]["exit_price"] == G_CLOSES[10]     # close of the 09:40 bar


def test_band_zero_equality_is_neutral(strat, tmp_path):
    # close == vwap exactly on the second bar: under carry_forward the long
    # holds; the paper's strict >/< rule is reproduced (no flip on equality)
    opens  = [100.0, 100.4, 100.6, 100.7]
    closes = [100.5, 100.0, 100.8, 100.9]
    folder = write_day(tmp_path, opens, closes, [100.0] * 4)
    df = run(strat, folder)
    assert len(df) == 1 and df.iloc[0]["direction"] == "long"


def test_band_carry_forward_holds_through_band(strat, tmp_path):
    # band = 2 ticks * 0.25 = 0.5: closes at 99.7 / 100.4 sit inside the band
    # -> the long survives; only a close < 99.5 would flip it
    opens  = [100.0, 100.9, 100.5, 100.2, 100.4]
    closes = [100.8, 99.7, 100.4, 100.1, 100.6]
    folder = write_day(tmp_path, opens, closes, [100.0] * 5)
    df = run(strat, folder, vwap_band_ticks=2.0)
    assert len(df) == 1
    assert df.iloc[0]["direction"] == "long"
    assert df.iloc[0]["exit_reason"] == "eod"
    # sl = flip level at entry = vwap - band
    assert df.iloc[0]["sl"] == pytest.approx(99.5)


def test_band_flat_stands_aside_inside_band(strat, tmp_path):
    # same tape under band_rule='flat': the 99.7 close (inside band) closes the
    # long at the next open; re-entry on the first clean close outside the band
    opens  = [100.0, 100.9, 100.5, 100.2, 100.4, 100.7]
    closes = [100.8, 99.7, 100.4, 100.1, 100.6, 100.9]
    folder = write_day(tmp_path, opens, closes, [100.0] * 6)
    df = run(strat, folder, vwap_band_ticks=2.0, band_rule="flat")
    assert list(df["exit_reason"])[0] == "band_flat"
    assert df.iloc[0]["exit_time"] == ts("09:32")
    assert df.iloc[0]["exit_price"] == 100.5
    # flat through the in-band bars; re-entry fill after the 100.6 close (09:34)
    assert df.iloc[1]["entry_time"] == ts("09:35")


def test_zero_volume_bar_produces_no_flip(strat, tmp_path):
    # bar1 is a forward-filled zero-volume bar whose close sits below VWAP: with
    # skip_zero_volume on it must NOT flip the long; with it off it must.
    # Deliberately passes legacy 0/1 ints (default is now a bool) — back-compat.
    opens  = [100.0, 100.6, 100.7, 100.8, 100.9]
    closes = [100.5, 98.0, 100.9, 101.0, 101.1]
    vols   = [100, 0, 100, 100, 100]
    folder = write_day(tmp_path, opens, closes, [100.0] * 5, volumes=vols)

    df_skip = run(strat, folder, skip_zero_volume=1)
    assert len(df_skip) == 1 and df_skip.iloc[0]["direction"] == "long"
    # the long's entry fill landed on the zero-volume bar -> flagged
    assert json.loads(df_skip.iloc[0]["notes"])["filled_on_zero_volume_bar"] is True

    df_all = run(strat, folder, skip_zero_volume=0)
    assert list(df_all["direction"])[:2] == ["long", "short"]


def test_nan_vwap_prefix_no_signal_no_crash(strat, tmp_path):
    # rth anchor with a pre-09:30 window: NaN VWAP -> flat until 09:30; the
    # first signal is the 09:30 close, first fill the 09:31 open
    n = 65   # 08:30 .. 09:34
    opens  = [100.0 + i * 0.01 for i in range(n)]
    closes = [100.5 + i * 0.01 for i in range(n)]
    vwaps  = [np.nan] * 60 + [100.0] * 5
    folder = write_day(tmp_path, opens, closes, vwaps, first_bar="08:30")
    df = run(strat, folder, trade_start_time="08:30")
    assert len(df) == 1
    assert df.iloc[0]["entry_time"] == ts("09:31")


def test_all_nan_vwap_day_is_empty(strat, tmp_path):
    folder = write_day(tmp_path, G_OPENS, G_CLOSES, [np.nan] * 20)
    df = run(strat, folder)
    assert df.empty
    assert list(df.columns) == ["date", "direction", "trade_type", "entry_time",
                                "exit_time", "entry_price", "exit_price", "sl",
                                "tp", "exit_reason", "pnl_points", "notes"]


def test_exclusion_window(strat, tmp_path):
    folder = write_day(tmp_path, G_OPENS, G_CLOSES, G_VWAP)
    df = run(strat, folder, exclude_start="09:40", exclude_end="09:45")
    excl0, excl1 = ts("09:40"), ts("09:45")
    # the position open at 09:40 is force-closed at that bar's OPEN
    hit = df[df["exit_reason"] == "exclusion"]
    assert len(hit) == 1
    assert hit.iloc[0]["exit_time"] == excl0
    assert hit.iloc[0]["exit_price"] == G_OPENS[10]
    # no entries inside the exclusion window; signals resume at/after 09:45
    assert not ((df["entry_time"] >= excl0) & (df["entry_time"] < excl1)).any()
    after = df[df["entry_time"] >= excl1]
    assert len(after) >= 1 and after.iloc[0]["entry_time"] == ts("09:46")


def test_one_bar_trade_on_penultimate_flip(strat, tmp_path):
    # long all day, the second-to-last close flips short -> 1-bar trade:
    # entry at the last bar's open, immediate EOD exit at its close
    opens  = [100.0, 100.6, 100.8, 100.7, 100.5]
    closes = [100.5, 100.9, 101.0, 99.0, 99.2]
    folder = write_day(tmp_path, opens, closes, [100.0] * 5)
    df = run(strat, folder)
    last = df.iloc[-1]
    assert last["direction"] == "short"
    assert last["entry_time"] == ts("09:34") and last["entry_price"] == 100.5
    assert last["exit_reason"] == "eod" and last["exit_price"] == 99.2
    assert json.loads(last["notes"])["bars_held"] == 1


def test_single_bar_window_no_trades(strat, tmp_path):
    folder = write_day(tmp_path, [100.0], [101.0], [100.0])
    df = run(strat, folder, trade_start_time="09:30")
    assert df.empty


def test_globex_anchor_selects_other_column(strat, tmp_path):
    # rth vwap would say long the whole session; globex vwap says short
    folder = write_day(tmp_path, G_OPENS, G_CLOSES, [90.0] * 20,
                       vwap_globex=[110.0] * 20)
    assert set(run(strat, folder, vwap_anchor="rth")["direction"]) == {"long"}
    assert set(run(strat, folder, vwap_anchor="globex")["direction"]) == {"short"}


def test_sl_conventions(strat, tmp_path):
    folder = write_day(tmp_path, G_OPENS, G_CLOSES, G_VWAP)
    df = run(strat, folder, sl_convention="realized_exit")
    losers, winners = df[df["pnl_points"] < 0], df[df["pnl_points"] > 0]
    assert (losers["sl"] == losers["exit_price"]).all() and losers["tp"].isna().all()
    assert (winners["tp"] == winners["exit_price"]).all() and winners["sl"].isna().all()

    df_none = run(strat, folder, sl_convention="none")
    assert df_none["sl"].isna().all() and df_none["tp"].isna().all()


def test_param_validation_errors(strat, tmp_path):
    folder = write_day(tmp_path, G_OPENS, G_CLOSES, G_VWAP)
    for bad in [
        {"vwap_anchor": "weekly"},
        {"band_rule": "hold"},
        {"sl_convention": "magic"},
        {"vwap_band_ticks": -1.0},
        {"trade_start_time": "16:00", "trade_end_time": "09:31"},
        {"trade_start_time": "9am"},
        {"exclude_start": "12:00"},                          # one-sided
        {"exclude_start": "15:00", "exclude_end": "12:00"},  # inverted
        {"exclude_start": "08:00", "exclude_end": "12:00"},  # outside window
        {"indicators_dataset": ""},
    ]:
        with pytest.raises(ValueError):
            run(strat, folder, **bad)


def test_missing_indicators_folder_errors(strat, tmp_path):
    folder = write_day(tmp_path, G_OPENS, G_CLOSES, G_VWAP)
    with pytest.raises(FileNotFoundError, match="indicators folder"):
        run(strat, folder, indicators_dataset="DOES_NOT_EXIST")


def test_missing_anchor_column_errors(strat, tmp_path):
    folder = write_day(tmp_path, G_OPENS, G_CLOSES, G_VWAP,
                       ind_columns=["vwap_bar_globex"])
    with pytest.raises(ValueError, match="vwap_bar_rth"):
        run(strat, folder, vwap_anchor="rth")


def test_missing_indicator_file_skips_day(strat, tmp_path):
    folder = write_day(tmp_path, G_OPENS, G_CLOSES, G_VWAP, date="2026-01-05")
    write_day(tmp_path, G_OPENS, G_CLOSES, G_VWAP, date="2026-01-06")
    (tmp_path / IND_DS / "2026-01-06.parquet").unlink()
    df = run(strat, folder)
    assert set(df["date"]) == {pd.Timestamp("2026-01-05").date()}


def test_day_cache_is_param_independent(strat, tmp_path, monkeypatch):
    folder = write_day(tmp_path, G_OPENS, G_CLOSES, G_VWAP)
    calls = []
    orig = strat.data.read_candles
    monkeypatch.setattr(strat.data, "read_candles",
                        lambda f: (calls.append(f), orig(f))[1])
    a = run(strat, folder, vwap_band_ticks=0.0)
    n_first = len(calls)
    b = run(strat, folder, vwap_band_ticks=3.0, band_rule="flat", vwap_anchor="globex")
    assert n_first == 1
    assert len(calls) == n_first        # second run: zero reads, params differ
    assert len(a) > 0 and len(b) > 0


def test_optimizer_sweepability():
    from strategies.vwap_trend.params import PARAMS, PARAMS_OPTIONS
    assert sweep_kind(PARAMS["vwap_band_ticks"]) == "float"
    assert sweep_kind(PARAMS["trade_start_time"]) == "categorical"
    assert sweep_kind(PARAMS["trade_end_time"]) == "categorical"
    assert sweep_kind(PARAMS["skip_zero_volume"]) == "bool"
    # dropdown params: choice sweep over the declared options
    for key in ("vwap_anchor", "band_rule", "sl_convention"):
        assert PARAMS[key] in PARAMS_OPTIONS[key]
        assert sweep_kind(PARAMS[key], PARAMS_OPTIONS[key]) == "choice"
